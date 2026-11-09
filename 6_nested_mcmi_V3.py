"""Run nested configuration screening and retain inner-ranked candidates."""

from __future__ import annotations

import math
import os
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_backend
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold
from threadpoolctl import threadpool_limits

os.environ.setdefault("NEUROPSYCHE_STAGE_NUMBER", "6")

from V3_config_loader import cfg
import V3_model_core as core
from V3_runtime import (
    CheckpointStore,
    GracefulStop,
    JsonlStore,
    RunControl,
    env_bool,
    file_sha256,
    make_stage_run_dir,
    read_jsonl,
    stable_hash,
    write_manifest,
)

try:
    from scipy.stats import spearmanr
except Exception:
    spearmanr = None


RESUME = env_bool("NEUROPSYCHE_RESUME", True)
RUN_SCRIPT_NAME = cfg.ordinal_stage_script_name(Path(__file__).name)
OUT_DIR = make_stage_run_dir(cfg.REPORTS_DIR, RUN_SCRIPT_NAME, resume=RESUME)
INNER_RESULTS_JSONL = OUT_DIR / "nested_v3_inner_config_summary.jsonl"
SELECTED_JSONL = OUT_DIR / "nested_v3_selected_configs.jsonl"
TOPK_JSONL = OUT_DIR / "nested_v3_topk_candidates.jsonl"
OUTER_PREDICTIONS_JSONL = OUT_DIR / "nested_v3_outer_subject_predictions.jsonl"
ERRORS_JSONL = OUT_DIR / "nested_v3_errors.jsonl"
CONFIG_TABLE_CSV = OUT_DIR / "nested_v3_feature_configurations.csv"
ALL_CONFIG_RESULTS_CSV = OUT_DIR / "nested_v3_all_configuration_results.csv"
TOPK_CANDIDATES_CSV = OUT_DIR / "nested_v3_topk_candidates.csv"
FINAL_METRICS_CSV = OUT_DIR / "nested_v3_final_metrics_rank1.csv"
SELECTION_STABILITY_CSV = OUT_DIR / "nested_v3_selection_stability_rank1.csv"
INNER_SPLIT_DIR = OUT_DIR / "inner_splits"
CSV_CELL_CHECKPOINT = OUT_DIR / "_checkpoints" / "all_config_csv_cells.completed.txt"

TASKS = ["binary_BR60", "binary_BR75", "ordinal", "continuous"]

# A fixed superset of columns lets each completed cell be appended to one
# ordinary CSV during the run. The final CSV is rebuilt from the durable
# JSONL stores at successful completion, removing any accidental duplicates.
ALL_CONFIG_COLUMNS = [
    "status", "analysis_key", "cell_key", "outer_fold", "task",
    "ordinal_scheme", "scale",
    "config_id", "inner_rank", "retained_top_k", "feature_set",
    "duration_sec", "duration_label", "n_features", "n_inner_subjects",
    "inner_class_counts", "inner_n_folds", "inner_n_repeats",
    "inner_total_splits", "inner_metric_mean", "inner_metric_sd",
    "inner_metric_min", "inner_metric_max", "valid_repetitions",
    "inner_selected_threshold", "inner_roc_auc_mean", "inner_pr_auc_mean",
    "inner_balanced_accuracy_mean", "inner_accuracy_mean",
    "inner_sensitivity_mean", "inner_specificity_mean",
    "inner_precision_mean", "inner_f1_mean", "inner_qwk_mean",
    "inner_mae_mean", "inner_rmse_mean", "inner_pearson_r_mean",
    "inner_spearman_rho_mean", "error",
]


def top_k_to_retain() -> int:
    value = int(getattr(cfg, "TOP_K_CONFIGS", 20))
    if value < 1:
        raise ValueError("TOP_K_CONFIGS must be at least 1.")
    return value


def latest_fold_file() -> Path:
    raw = cfg.runtime_input("fold_file")
    path = Path(raw) if raw else None
    if path is None or not path.exists():
        raise FileNotFoundError("No fixed outer-fold file was found. Run Stage 5 first.")
    return path


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    fold_path = latest_fold_file()
    folds = pd.read_csv(fold_path, dtype={"subject_id": str})
    if not {"subject_id", "outer_fold"}.issubset(folds.columns):
        raise ValueError("fixed_outer_folds.csv must contain subject_id and outer_fold.")
    folds["subject_id"] = folds["subject_id"].astype(str).str.strip()
    folds["outer_fold"] = pd.to_numeric(folds["outer_fold"], errors="raise").astype(int)
    expected = set(range(1, int(cfg.OUTER_N_FOLDS) + 1))
    actual = set(folds["outer_fold"].unique())
    if actual != expected:
        raise ValueError(f"Outer fold labels are {sorted(actual)}, expected {sorted(expected)}")

    mcmi = pd.read_csv(cfg.MCMI_CSV, dtype={cfg.SUBJECT_ID_COLUMN: str})
    mcmi[cfg.SUBJECT_ID_COLUMN] = mcmi[cfg.SUBJECT_ID_COLUMN].astype(str).str.strip()
    missing = [s for s in cfg.MCMI_SCALE_ORDER if s not in mcmi.columns]
    if missing:
        raise ValueError(f"MCMI CSV is missing scales: {missing}")
    data = folds.merge(
        mcmi[[cfg.SUBJECT_ID_COLUMN] + list(cfg.MCMI_SCALE_ORDER)],
        left_on="subject_id",
        right_on=cfg.SUBJECT_ID_COLUMN,
        how="left",
        validate="one_to_one",
    )
    if cfg.SUBJECT_ID_COLUMN != "subject_id" and cfg.SUBJECT_ID_COLUMN in data.columns:
        data = data.drop(columns=[cfg.SUBJECT_ID_COLUMN])

    permutation_raw = os.environ.get("NEUROPSYCHE_PERMUTATION_INDEX", "").strip()
    if permutation_raw:
        permutation_index = int(permutation_raw)
        base_seed = int(cfg.RANDOM_SEED) + permutation_index * 100003
        permute_rows_together = env_bool(
            "NEUROPSYCHE_PERMUTE_MCMI_ROWS_TOGETHER",
            True,
        )
        if permute_rows_together:
            order = np.random.default_rng(base_seed).permutation(len(data))
            data.loc[:, list(cfg.MCMI_SCALE_ORDER)] = data.loc[
                order, list(cfg.MCMI_SCALE_ORDER)
            ].to_numpy()
        else:
            for index, scale in enumerate(cfg.MCMI_SCALE_ORDER):
                order = np.random.default_rng(base_seed + (index + 1) * 1009).permutation(len(data))
                data.loc[:, scale] = data.loc[order, scale].to_numpy()
    return data, folds, fold_path


def tasks_to_run() -> list[str]:
    out = []
    if bool(getattr(cfg, "RUN_BR60", True)):
        out.append("binary_BR60")
    if bool(getattr(cfg, "RUN_BR75", True)):
        out.append("binary_BR75")
    if bool(getattr(cfg, "RUN_ORDINAL", True)):
        out.append("ordinal")
    if bool(getattr(cfg, "RUN_CONTINUOUS", True)):
        out.append("continuous")
    return out


def task_labels(scores: np.ndarray, task: str) -> np.ndarray:
    if task == "binary_BR60":
        return core.make_binary(scores, "BR60")
    if task == "binary_BR75":
        return core.make_binary(scores, "BR75")
    if task == "ordinal":
        return core.make_brackets(scores, cutoffs=cfg.ordinal_cutoffs())
    if task == "continuous":
        return np.asarray(scores, dtype=float)
    raise ValueError(task)


def valid_label_mask(y: np.ndarray, task: str) -> np.ndarray:
    if task == "continuous":
        return np.isfinite(y)
    return np.isfinite(y) & (y >= 0)


def split_seed(outer_fold: int, task: str, scale: str) -> int:
    token = stable_hash(
        {"outer": outer_fold, "task": task, "scale": scale, "base": int(cfg.INNER_BASE_SEED)},
        length=8,
    )
    return int(token, 16) % (2**31 - 1)


def regression_bins(y: np.ndarray, max_bins: int) -> Optional[np.ndarray]:
    y = np.asarray(y, dtype=float)
    for bins in range(min(int(max_bins), len(np.unique(y))), 1, -1):
        try:
            labels = pd.qcut(y, q=bins, labels=False, duplicates="drop")
            labels = np.asarray(labels, dtype=int)
            if len(np.unique(labels)) >= 2 and np.min(np.bincount(labels)) >= 2:
                return labels
        except Exception:
            continue
    return None


def create_inner_splits(
    y: np.ndarray,
    task: str,
    seed: int,
) -> list[tuple[int, int, np.ndarray, np.ndarray]]:
    n_folds = int(cfg.INNER_N_FOLDS)
    n_repeats = int(cfg.INNER_N_REPEATS)
    splits: list[tuple[int, int, np.ndarray, np.ndarray]] = []
    y = np.asarray(y)

    strata = regression_bins(y, int(cfg.REGRESSION_STRATIFICATION_BINS)) if task == "continuous" else y.astype(int)

    for rep in range(n_repeats):
        rep_seed = seed + rep * 10007
        if strata is not None and len(np.unique(strata)) >= 2:
            splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=rep_seed)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fold_pairs = list(splitter.split(np.zeros(len(y)), strata))
        else:
            splitter = KFold(n_splits=n_folds, shuffle=True, random_state=rep_seed)
            fold_pairs = list(splitter.split(np.zeros(len(y))))
        for fold_idx, (train_idx, val_idx) in enumerate(fold_pairs):
            splits.append((rep, fold_idx, np.asarray(train_idx, dtype=int), np.asarray(val_idx, dtype=int)))
    return splits


def inner_split_file(outer_fold: int, task: str, scale: str) -> Path:
    safe = f"outer{outer_fold}__{task}__{scale}".replace("/", "_")
    return INNER_SPLIT_DIR / f"{safe}.npz"


def load_or_create_inner_splits(
    subject_ids: Sequence[str],
    y: np.ndarray,
    outer_fold: int,
    task: str,
    scale: str,
) -> list[tuple[int, int, np.ndarray, np.ndarray]]:
    path = inner_split_file(outer_fold, task, scale)
    path.parent.mkdir(parents=True, exist_ok=True)
    subject_hash = stable_hash(list(subject_ids))
    label_hash = stable_hash(np.asarray(y).tolist())
    if path.exists():
        z = np.load(path, allow_pickle=True)
        if str(z["subject_hash"].item()) != subject_hash or str(z["label_hash"].item()) != label_hash:
            raise ValueError(f"Existing inner split file is incompatible: {path}")
        reps = z["repetition"].astype(int)
        folds = z["fold"].astype(int)
        trains = z["train_indices"]
        tests = z["test_indices"]
        return [
            (int(r), int(f), np.asarray(tr, dtype=int), np.asarray(te, dtype=int))
            for r, f, tr, te in zip(reps, folds, trains, tests)
        ]

    splits = create_inner_splits(y, task, split_seed(outer_fold, task, scale))
    np.savez_compressed(
        path,
        repetition=np.asarray([s[0] for s in splits], dtype=int),
        fold=np.asarray([s[1] for s in splits], dtype=int),
        train_indices=np.asarray([s[2] for s in splits], dtype=object),
        test_indices=np.asarray([s[3] for s in splits], dtype=object),
        subject_hash=np.asarray(subject_hash),
        label_hash=np.asarray(label_hash),
        subject_ids=np.asarray(subject_ids, dtype=str),
    )
    return splits


def _fit_predict_one_split(
    task: str,
    X: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
) -> dict[str, Any]:
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    with threadpool_limits(limits=int(cfg.NUMERICAL_THREADS_PER_WORKER)):
        if task.startswith("binary_"):
            if len(np.unique(y_train)) < 2:
                score = np.full(len(val_idx), float(np.mean(y_train)))
                pred = np.full(len(val_idx), int(np.mean(y_train) >= 0.5))
            else:
                model = core.make_binary_pipeline(str(cfg.BINARY_MODEL_NAME), float(cfg.LOGISTIC_C))
                model.fit(X_train, y_train.astype(int))
                score = core.binary_score(model, X_val)
                pred = (score >= 0.0).astype(int)
            return {"indices": val_idx, "score": np.asarray(score), "pred": np.asarray(pred)}
        if task == "ordinal":
            model = core.CumulativeOrdinalLogit(
                C=float(cfg.LOGISTIC_C),
                n_classes=cfg.ordinal_n_classes(),
            )
            model.fit(X_train, y_train.astype(int))
            prob = model.predict_proba(X_val)
            pred = np.argmax(prob, axis=1).astype(int)
            expected = (prob * np.arange(prob.shape[1])[None, :]).sum(axis=1)
            return {"indices": val_idx, "score": expected, "pred": pred, "prob": prob}
        model = core.make_ridge_pipeline(float(cfg.RIDGE_ALPHA))
        model.fit(X_train, y_train.astype(float))
        pred = np.asarray(model.predict(X_val), dtype=float)
        return {"indices": val_idx, "score": pred, "pred": pred}


def classifier_workers() -> int:
    total = os.cpu_count() or 1
    by_percent = max(1, int(total * int(cfg.CPU_USAGE_PERCENT) / 100))
    return max(1, min(int(cfg.MAX_CLASSIFIER_WORKERS), by_percent))


def metric_for_task(
    task: str,
    y_true: np.ndarray,
    score: np.ndarray,
    pred: np.ndarray,
) -> dict[str, float]:
    result: dict[str, float] = {}
    if task.startswith("binary_"):
        result["primary"] = float(roc_auc_score(y_true, score)) if len(np.unique(y_true)) == 2 else np.nan
        result["roc_auc"] = result["primary"]
        result["pr_auc"] = float(average_precision_score(y_true, score)) if len(np.unique(y_true)) == 2 else np.nan
        result["balanced_accuracy"] = float(balanced_accuracy_score(y_true, pred))
        result["accuracy"] = float(accuracy_score(y_true, pred))
        result["sensitivity"] = float(recall_score(y_true, pred, pos_label=1, zero_division=0))
        result["specificity"] = float(recall_score(y_true, pred, pos_label=0, zero_division=0))
        result["precision"] = float(precision_score(y_true, pred, zero_division=0))
        result["f1"] = float(f1_score(y_true, pred, zero_division=0))
        return result
    if task == "ordinal":
        result["primary"] = float(cohen_kappa_score(y_true, pred, weights="quadratic"))
        result["qwk"] = result["primary"]
        result["accuracy"] = float(accuracy_score(y_true, pred))
        result["mae"] = float(mean_absolute_error(y_true, pred))
        return result
    if len(y_true) < 3 or np.std(y_true) == 0 or np.std(score) == 0:
        r = np.nan
    else:
        r = float(np.corrcoef(y_true, score)[0, 1])
    result["primary"] = r
    result["pearson_r"] = r
    result["mae"] = float(mean_absolute_error(y_true, score))
    result["rmse"] = float(math.sqrt(mean_squared_error(y_true, score)))
    if spearmanr is not None and len(y_true) >= 3:
        result["spearman_rho"] = float(spearmanr(y_true, score).statistic)
    else:
        result["spearman_rho"] = np.nan
    return result


def optimal_binary_threshold(y: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(y, dtype=int)
    scores = np.asarray(scores, dtype=float)
    valid = np.isfinite(scores)
    y, scores = y[valid], scores[valid]
    if len(y) == 0 or len(np.unique(y)) < 2:
        return 0.0
    candidates = np.unique(np.quantile(scores, np.linspace(0.02, 0.98, 97)))
    best_threshold, best_value = 0.0, -np.inf
    for threshold in candidates:
        pred = (scores >= threshold).astype(int)
        value = balanced_accuracy_score(y, pred)
        if value > best_value:
            best_threshold, best_value = float(threshold), float(value)
    return best_threshold


def evaluate_config(
    X: np.ndarray,
    y: np.ndarray,
    task: str,
    splits: Sequence[tuple[int, int, np.ndarray, np.ndarray]],
    return_oof: bool = False,
) -> tuple[dict[str, Any], Optional[pd.DataFrame]]:
    jobs = [delayed(_fit_predict_one_split)(task, X, y, tr, te) for _, _, tr, te in splits]
    with parallel_backend("loky", inner_max_num_threads=int(cfg.NUMERICAL_THREADS_PER_WORKER)):
        outputs = Parallel(
            n_jobs=classifier_workers(),
            max_nbytes="20M",
            mmap_mode="r",
        )(jobs)

    n_repeats = int(cfg.INNER_N_REPEATS)
    metrics = []
    oof_rows = []
    score_sum = np.zeros(len(y), dtype=float)
    score_count = np.zeros(len(y), dtype=int)

    for rep in range(n_repeats):
        score = np.full(len(y), np.nan, dtype=float)
        pred = np.full(len(y), np.nan, dtype=float)
        for split_info, output in zip(splits, outputs):
            split_rep, split_fold, _, val_idx = split_info
            if split_rep != rep:
                continue
            score[val_idx] = np.asarray(output["score"], dtype=float)
            pred[val_idx] = np.asarray(output["pred"], dtype=float)
        valid = np.isfinite(score) & np.isfinite(pred)
        if valid.sum() != len(y):
            raise RuntimeError(f"Inner repetition {rep} did not predict every training subject.")
        score_sum[valid] += score[valid]
        score_count[valid] += 1
        rep_metrics = metric_for_task(
            task,
            y,
            score,
            pred.astype(int) if task != "continuous" else pred,
        )
        rep_metrics["repetition"] = rep
        metrics.append(rep_metrics)
        if return_oof:
            for idx in range(len(y)):
                oof_rows.append({
                    "repetition": rep,
                    "index": idx,
                    "y_true": float(y[idx]),
                    "score": float(score[idx]),
                    "pred": float(pred[idx]),
                })

    metric_df = pd.DataFrame(metrics)
    primary = pd.to_numeric(metric_df["primary"], errors="coerce")
    summary: dict[str, Any] = {
        "inner_metric_mean": float(primary.mean()),
        "inner_metric_sd": float(primary.std(ddof=1)) if primary.notna().sum() > 1 else 0.0,
        "inner_metric_min": float(primary.min()),
        "inner_metric_max": float(primary.max()),
        "valid_repetitions": int(primary.notna().sum()),
    }
    for col in metric_df.columns:
        if col in {"primary", "repetition"}:
            continue
        values = pd.to_numeric(metric_df[col], errors="coerce")
        summary[f"inner_{col}_mean"] = float(values.mean())

    if task.startswith("binary_"):
        mean_scores = np.divide(
            score_sum,
            score_count,
            out=np.full(len(y), np.nan, dtype=float),
            where=score_count > 0,
        )
        summary["inner_selected_threshold"] = optimal_binary_threshold(y, mean_scores)
    else:
        summary["inner_selected_threshold"] = np.nan

    return summary, pd.DataFrame(oof_rows) if return_oof else None


def analysis_key(outer_fold: int, task: str, scale: str, config_id: int) -> str:
    return f"outer={outer_fold}|task={task}|scale={scale}|config={config_id}"


def cell_key(outer_fold: int, task: str, scale: str) -> str:
    return f"outer={outer_fold}|task={task}|scale={scale}"


def rank_cell_results(rows: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    if not rows:
        return []
    df = pd.DataFrame(rows).copy()
    df["inner_metric_mean"] = pd.to_numeric(df.get("inner_metric_mean"), errors="coerce")
    df["n_features"] = pd.to_numeric(df.get("n_features"), errors="coerce")
    df["config_id"] = pd.to_numeric(df.get("config_id"), errors="coerce").astype("Int64")
    df = df[np.isfinite(df["inner_metric_mean"])].copy()
    if df.empty:
        return []
    df = df.sort_values(
        ["inner_metric_mean"],
        ascending=[False],
        kind="mergesort",
    ).reset_index(drop=True)
    tolerance = float(getattr(cfg, "SELECTION_TIE_TOLERANCE", 0.0))
    tie_groups = []
    group = -1
    group_best = np.nan
    for value in df["inner_metric_mean"].to_numpy(dtype=float):
        if group < 0 or group_best - value > tolerance:
            group += 1
            group_best = value
        tie_groups.append(group)
    df["_tie_group"] = tie_groups
    df = df.sort_values(
        ["_tie_group", "n_features", "config_id"],
        ascending=[True, True, True],
        kind="mergesort",
    ).drop(columns="_tie_group").reset_index(drop=True)
    df["inner_rank"] = np.arange(1, len(df) + 1, dtype=int)
    df["retained_top_k"] = df["inner_rank"] <= int(top_k)
    return df.to_dict("records")


def append_ranked_cell_to_csv(
    ranked_rows: list[dict[str, Any]],
    checkpoint: CheckpointStore,
) -> None:
    if not ranked_rows:
        return
    ck = str(ranked_rows[0]["cell_key"])
    if checkpoint.contains(ck):
        return
    frame = pd.DataFrame(ranked_rows)
    frame["status"] = "ok"
    for column in ALL_CONFIG_COLUMNS:
        if column not in frame.columns:
            frame[column] = np.nan
    frame = frame[ALL_CONFIG_COLUMNS]
    ALL_CONFIG_RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(
        ALL_CONFIG_RESULTS_CSV,
        mode="a",
        header=not ALL_CONFIG_RESULTS_CSV.exists(),
        index=False,
    )
    checkpoint.add(ck)


def fit_outer_selected(
    selected_config: core.FeatureConfig,
    train_subjects: list[str],
    test_subjects: list[str],
    score_map: dict[str, float],
    task: str,
    threshold: Optional[float],
) -> list[dict[str, Any]]:
    all_subjects = train_subjects + test_subjects
    X, names, used, missing = core.assemble_X(all_subjects, selected_config)
    if used != all_subjects or missing:
        raise RuntimeError(f"Selected config has missing feature subjects: {missing[:3]}")
    n_train = len(train_subjects)
    X_train, X_test = X[:n_train], X[n_train:]
    y_train = task_labels(np.asarray([score_map[s] for s in train_subjects], dtype=float), task)
    y_test = task_labels(np.asarray([score_map[s] for s in test_subjects], dtype=float), task)

    if task.startswith("binary_"):
        if len(np.unique(y_train)) < 2:
            scores = np.full(len(test_subjects), float(np.mean(y_train)))
        else:
            model = core.make_binary_pipeline(str(cfg.BINARY_MODEL_NAME), float(cfg.LOGISTIC_C))
            model.fit(X_train, y_train.astype(int))
            scores = core.binary_score(model, X_test)
        use_threshold = 0.0 if threshold is None or not np.isfinite(float(threshold)) else float(threshold)
        preds = (scores >= use_threshold).astype(int)
    elif task == "ordinal":
        model = core.CumulativeOrdinalLogit(
            C=float(cfg.LOGISTIC_C),
            n_classes=cfg.ordinal_n_classes(),
        )
        model.fit(X_train, y_train.astype(int))
        probs = model.predict_proba(X_test)
        preds = np.argmax(probs, axis=1).astype(int)
        scores = (probs * np.arange(probs.shape[1])[None, :]).sum(axis=1)
    else:
        model = core.make_ridge_pipeline(float(cfg.RIDGE_ALPHA))
        model.fit(X_train, y_train.astype(float))
        scores = np.asarray(model.predict(X_test), dtype=float)
        preds = scores

    rows = []
    for sid, raw_score, y_true, score, pred in zip(
        test_subjects,
        [score_map[s] for s in test_subjects],
        y_test,
        scores,
        preds,
    ):
        rows.append({
            "subject_id": sid,
            "raw_br_score": float(raw_score),
            "y_true": float(y_true),
            "prediction_score": float(score),
            "prediction_label": float(pred),
            "decision_threshold": threshold,
        })
    return rows


def compute_final_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if predictions.empty:
        return pd.DataFrame()
    for (task, scale), group in predictions.groupby(["task", "scale"]):
        y = group["y_true"].to_numpy()
        score = group["prediction_score"].to_numpy(dtype=float)
        pred = group["prediction_label"].to_numpy()
        metrics = metric_for_task(
            str(task),
            y.astype(int) if task != "continuous" else y.astype(float),
            score,
            pred.astype(int) if task != "continuous" else pred.astype(float),
        )
        row = {
            "task": task,
            "scale": scale,
            "n_subjects": int(len(group)),
            "n_outer_folds": int(group["outer_fold"].nunique()),
            "class_counts": core.class_count_text(y.astype(int)) if task != "continuous" else "",
            **metrics,
        }
        if str(task).startswith("binary_"):
            prevalence = float(np.mean(y))
            row["prevalence"] = prevalence
            row["dummy_roc_auc"] = 0.5
            row["dummy_pr_auc"] = prevalence
        elif task == "ordinal":
            dummy = np.full(len(y), int(np.median(y)))
            row["dummy_qwk"] = float(cohen_kappa_score(y.astype(int), dummy, weights="quadratic"))
        else:
            dummy = np.full(len(y), float(np.mean(y)))
            row["dummy_mae"] = float(mean_absolute_error(y, dummy))
            row["dummy_rmse"] = float(math.sqrt(mean_squared_error(y, dummy)))
        rows.append(row)
    return pd.DataFrame(rows)


def rebuild_all_configuration_csv(top_k: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    success = pd.DataFrame(read_jsonl(INNER_RESULTS_JSONL))
    errors = pd.DataFrame(read_jsonl(ERRORS_JSONL))

    ranked_groups = []
    if not success.empty:
        success = success.drop_duplicates(subset=["analysis_key"], keep="last")
        for _, group in success.groupby("cell_key", sort=False):
            ranked_groups.extend(rank_cell_results(group.to_dict("records"), top_k))
    ranked = pd.DataFrame(ranked_groups)
    if not ranked.empty:
        ranked["status"] = "ok"
        if "ordinal_scheme" not in ranked.columns:
            ranked["ordinal_scheme"] = ""
        ordinal_rows = ranked["task"].astype(str).eq("ordinal")
        missing_scheme = ranked["ordinal_scheme"].isna() | ranked[
            "ordinal_scheme"
        ].astype(str).str.strip().eq("")
        ranked.loc[ordinal_rows & missing_scheme, "ordinal_scheme"] = (
            cfg.ordinal_scheme_name()
        )

    # Add configurations that were attempted but did not yield a valid result.
    failed = pd.DataFrame()
    if not errors.empty and "analysis_key" in errors.columns:
        errors = errors.dropna(subset=["analysis_key"]).drop_duplicates(subset=["analysis_key"], keep="last")
        successful_keys = set(ranked["analysis_key"].astype(str)) if not ranked.empty else set()
        failed = errors[~errors["analysis_key"].astype(str).isin(successful_keys)].copy()
        if not failed.empty:
            failed["status"] = "error"
            failed["inner_rank"] = np.nan
            failed["retained_top_k"] = False

    combined = pd.concat([ranked, failed], ignore_index=True, sort=False)
    for column in ALL_CONFIG_COLUMNS:
        if column not in combined.columns:
            combined[column] = np.nan
    combined = combined[ALL_CONFIG_COLUMNS]
    if not combined.empty:
        ordinal_rows = combined["task"].astype(str).eq("ordinal")
        missing_scheme = combined["ordinal_scheme"].isna() | combined[
            "ordinal_scheme"
        ].astype(str).str.strip().eq("")
        combined.loc[ordinal_rows & missing_scheme, "ordinal_scheme"] = (
            cfg.ordinal_scheme_name()
        )
        combined = combined.sort_values(
            ["outer_fold", "task", "scale", "status", "inner_rank", "config_id"],
            na_position="last",
        )
    combined.to_csv(ALL_CONFIG_RESULTS_CSV, index=False)

    topk = ranked[ranked["retained_top_k"] == True].copy() if not ranked.empty else pd.DataFrame()
    stored_topk = pd.DataFrame(read_jsonl(TOPK_JSONL))
    if not topk.empty and not stored_topk.empty:
        # Candidate checkpoints may contain older rankings from an interrupted
        # run. Keep the CURRENT ranking from the complete inner-result table,
        # and import only candidate-specific metadata such as a rebuilt binary
        # threshold from the newest matching cell/config checkpoint.
        stored_topk = stored_topk.drop_duplicates(["cell_key", "config_id"], keep="last")
        metadata_columns = [
            column for column in [
                "cell_key", "config_id", "inner_selected_threshold",
                "selection_rule", "top_k_requested", "outer_train_n",
                "outer_test_n", "fold_assignment_hash",
                "outer_test_used_for_ranking",
            ] if column in stored_topk.columns
        ]
        topk = topk.drop(columns=[c for c in metadata_columns if c in topk.columns and c not in {"cell_key", "config_id"}], errors="ignore")
        topk = topk.merge(
            stored_topk[metadata_columns],
            on=["cell_key", "config_id"],
            how="left",
            validate="one_to_one",
        )
    if not topk.empty:
        topk = topk.sort_values(["outer_fold", "task", "scale", "inner_rank"])
    topk.to_csv(TOPK_CANDIDATES_CSV, index=False)
    return ranked, topk


def main() -> None:
    cfg.apply_persisted_settings()
    if str(cfg.CONFIG_SELECTION_RULE) != "top_k":
        raise ValueError(f"Unsupported configuration-selection rule: {cfg.CONFIG_SELECTION_RULE}")
    cfg.validate_config()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    INNER_SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    control = RunControl.from_environment(OUT_DIR / "_control")
    top_k = top_k_to_retain()

    data, folds_df, fold_path = load_data()
    configs = core.build_feature_configs()
    top_k = min(top_k, len(configs))

    scales = list(cfg.MCMI_SCALE_ORDER)
    run_tasks = tasks_to_run()
    if set(run_tasks) != set(TASKS):
        print(f"Tasks enabled: {run_tasks}")

    pd.DataFrame([
        {
            "config_id": c.config_id,
            "feature_set": c.feature_set,
            "duration_sec": c.duration_sec,
            "duration_label": c.duration_label,
            "continuous_families": "+".join(c.continuous_families),
            "epoch_families": "+".join(c.epoch_families),
        }
        for c in configs
    ]).to_csv(CONFIG_TABLE_CSV, index=False)

    inner_store = JsonlStore(INNER_RESULTS_JSONL, key_field="analysis_key")
    selected_store = JsonlStore(SELECTED_JSONL, key_field="cell_key")
    topk_store = JsonlStore(TOPK_JSONL, key_field="candidate_key")
    prediction_store = JsonlStore(OUTER_PREDICTIONS_JSONL, key_field="prediction_key")
    error_store = JsonlStore(ERRORS_JSONL)
    csv_checkpoint = CheckpointStore(CSV_CELL_CHECKPOINT)

    inner_rows = read_jsonl(INNER_RESULTS_JSONL)
    rows_by_cell: dict[str, list[dict[str, Any]]] = {}
    for row in inner_rows:
        rows_by_cell.setdefault(str(row.get("cell_key")), []).append(row)
    selected_existing = {str(r.get("cell_key")): r for r in read_jsonl(SELECTED_JSONL)}
    prediction_keys = prediction_store.keys

    score_maps = {
        scale: dict(zip(data["subject_id"].astype(str), pd.to_numeric(data[scale], errors="coerce")))
        for scale in scales
    }
    fold_assignment_hash = stable_hash(
        folds_df[["subject_id", "outer_fold"]].sort_values("subject_id").to_dict("records")
    )

    total_cells = int(cfg.OUTER_N_FOLDS) * len(run_tasks) * len(scales)
    cell_counter = 0
    try:
        for outer_fold in range(1, int(cfg.OUTER_N_FOLDS) + 1):
            for task in run_tasks:
                for scale in scales:
                    cell_counter += 1
                    control.checkpoint(
                        f"before cell {cell_counter}/{total_cells}: outer {outer_fold}, {task}, {scale}"
                    )
                    ck = cell_key(outer_fold, task, scale)
                    scores_all = np.asarray(
                        [score_maps[scale].get(sid, np.nan) for sid in data["subject_id"].astype(str)],
                        dtype=float,
                    )
                    labels_all = task_labels(scores_all, task)
                    valid = valid_label_mask(labels_all, task)
                    cell_data = data.loc[valid, ["subject_id", "outer_fold"]].copy()
                    cell_data["label"] = labels_all[valid]
                    train_df = cell_data[cell_data["outer_fold"] != outer_fold].copy()
                    test_df = cell_data[cell_data["outer_fold"] == outer_fold].copy()
                    train_subjects = train_df["subject_id"].astype(str).tolist()
                    test_subjects = test_df["subject_id"].astype(str).tolist()
                    y_train = train_df["label"].to_numpy(dtype=float if task == "continuous" else int)
                    if len(train_subjects) < int(cfg.INNER_N_FOLDS) or not test_subjects:
                        error_store.append({"cell_key": ck, "error": "insufficient train/test subjects"})
                        continue
                    splits = load_or_create_inner_splits(
                        train_subjects, y_train, outer_fold, task, scale
                    )

                    cell_rows = rows_by_cell.setdefault(ck, [])
                    completed_config_ids = {
                        int(r["config_id"]) for r in cell_rows if "config_id" in r
                    }
                    for config_index, config in enumerate(configs, start=1):
                        key = analysis_key(outer_fold, task, scale, config.config_id)
                        if config.config_id in completed_config_ids or inner_store.contains(key):
                            continue
                        control.checkpoint(f"{ck}; configuration {config_index}/{len(configs)}")
                        try:
                            X, feature_names, used_subjects, missing = core.assemble_X(
                                train_subjects, config
                            )
                            if used_subjects != train_subjects or missing:
                                raise RuntimeError(f"Missing feature subjects: {missing[:3]}")
                            summary, _ = evaluate_config(
                                X, y_train, task, splits, return_oof=False
                            )
                            row = {
                                "analysis_key": key,
                                "cell_key": ck,
                                "outer_fold": outer_fold,
                                "task": task,
                                "ordinal_scheme": (
                                    cfg.ordinal_scheme_name()
                                    if task == "ordinal"
                                    else ""
                                ),
                                "scale": scale,
                                "config_id": config.config_id,
                                "feature_set": config.feature_set,
                                "duration_sec": config.duration_sec,
                                "duration_label": config.duration_label,
                                "n_features": int(X.shape[1]),
                                "n_inner_subjects": int(len(train_subjects)),
                                "inner_class_counts": core.class_count_text(y_train.astype(int)) if task != "continuous" else "",
                                "inner_n_folds": int(cfg.INNER_N_FOLDS),
                                "inner_n_repeats": int(cfg.INNER_N_REPEATS),
                                "inner_total_splits": len(splits),
                                **summary,
                            }
                            inner_store.append(row)
                            cell_rows.append(row)
                            completed_config_ids.add(config.config_id)
                        except Exception as error:
                            error_store.append({
                                "analysis_key": key,
                                "cell_key": ck,
                                "outer_fold": outer_fold,
                                "task": task,
                                "ordinal_scheme": (
                                    cfg.ordinal_scheme_name()
                                    if task == "ordinal"
                                    else ""
                                ),
                                "scale": scale,
                                "config_id": config.config_id,
                                "feature_set": config.feature_set,
                                "duration_sec": config.duration_sec,
                                "duration_label": config.duration_label,
                                "error": repr(error),
                            })
                        if config_index % 25 == 0:
                            print(
                                f"{ck}: {config_index}/{len(configs)} configurations",
                                flush=True,
                            )

                    if not cell_rows:
                        error_store.append({"cell_key": ck, "error": "No valid configuration results"})
                        continue
                    ranked_rows = rank_cell_results(cell_rows, top_k)
                    if not ranked_rows:
                        error_store.append({"cell_key": ck, "error": "No finite configuration metrics"})
                        continue
                    append_ranked_cell_to_csv(ranked_rows, csv_checkpoint)

                    retained = [r for r in ranked_rows if bool(r["retained_top_k"])]

                    # Compatibility with an older/partial Stage 6 run: earlier
                    # rows may not contain a training-only binary threshold for
                    # every retained candidate. Recompute it only when missing.
                    if task.startswith("binary_"):
                        refreshed_retained = []
                        for candidate in retained:
                            candidate = dict(candidate)
                            threshold = candidate.get("inner_selected_threshold", np.nan)
                            if threshold is None or not np.isfinite(float(threshold)):
                                candidate_config = next(
                                    c for c in configs if c.config_id == int(candidate["config_id"])
                                )
                                X_candidate, _, used_candidate, missing_candidate = core.assemble_X(
                                    train_subjects, candidate_config
                                )
                                if used_candidate != train_subjects or missing_candidate:
                                    raise RuntimeError(
                                        f"Cannot rebuild threshold for config {candidate_config.config_id}: "
                                        f"{missing_candidate[:3]}"
                                    )
                                threshold_summary, _ = evaluate_config(
                                    X_candidate, y_train, task, splits, return_oof=False
                                )
                                candidate["inner_selected_threshold"] = threshold_summary.get(
                                    "inner_selected_threshold", 0.0
                                )
                            refreshed_retained.append(candidate)
                        retained = refreshed_retained

                    for candidate in retained:
                        candidate = dict(candidate)
                        candidate.update({
                            "candidate_key": f"{ck}|rank={int(candidate['inner_rank'])}|config={int(candidate['config_id'])}",
                            "selection_rule": "inner_rank_top_k",
                            "top_k_requested": int(top_k),
                            "outer_train_n": len(train_subjects),
                            "outer_test_n": len(test_subjects),
                            "fold_assignment_hash": fold_assignment_hash,
                            "outer_test_used_for_ranking": False,
                        })
                        topk_store.append(candidate)

                    # Keep rank 1 outer predictions for backward compatibility and
                    # a direct comparison with the final Stage-8 top-K result.
                    rank1 = retained[0]
                    selected = selected_existing.get(ck)
                    if selected is None:
                        selected = dict(rank1)
                        selected.update({
                            "cell_key": ck,
                            "outer_fold": outer_fold,
                            "task": task,
                            "scale": scale,
                            "selection_rule": "inner_rank_1",
                            "top_k_requested": int(top_k),
                            "outer_train_n": len(train_subjects),
                            "outer_test_n": len(test_subjects),
                            "fold_assignment_hash": fold_assignment_hash,
                            "outer_test_used_for_selection": False,
                        })
                        selected_store.append(selected)
                        selected_existing[ck] = selected

                    selected_config = next(
                        c for c in configs if c.config_id == int(selected["config_id"])
                    )
                    threshold = selected.get("inner_selected_threshold")
                    missing_prediction_keys = [
                        f"{ck}|subject={sid}"
                        for sid in test_subjects
                        if f"{ck}|subject={sid}" not in prediction_keys
                    ]
                    if missing_prediction_keys:
                        outer_rows = fit_outer_selected(
                            selected_config,
                            train_subjects,
                            test_subjects,
                            score_maps[scale],
                            task,
                            threshold,
                        )
                        for row in outer_rows:
                            row.update({
                                "prediction_key": f"{ck}|subject={row['subject_id']}",
                                "cell_key": ck,
                                "outer_fold": outer_fold,
                                "task": task,
                                "ordinal_scheme": (
                                    cfg.ordinal_scheme_name()
                                    if task == "ordinal"
                                    else ""
                                ),
                                "scale": scale,
                                "inner_rank": 1,
                                "selected_config_id": selected_config.config_id,
                                "selected_feature_set": selected_config.feature_set,
                                "selected_duration_sec": selected_config.duration_sec,
                                "selected_duration_label": selected_config.duration_label,
                            })
                            prediction_store.append(row)
                            prediction_keys.add(row["prediction_key"])
                    print(
                        f"Completed {cell_counter}/{total_cells}: {ck}; retained top {len(retained)}",
                        flush=True,
                    )
    except GracefulStop as stop:
        print(f"Graceful stop: {stop}")
        # The JSONL stores and incrementally appended all-configuration CSV remain resumable.
        return

    ranked_df, topk_df = rebuild_all_configuration_csv(top_k)
    expected_cells = {
        (outer_fold, task, scale)
        for outer_fold in range(1, int(cfg.OUTER_N_FOLDS) + 1)
        for task in run_tasks
        for scale in scales
    }
    completed_cells: set[tuple[int, str, str]] = set()
    if not topk_df.empty:
        for (outer_fold, task, scale), group in topk_df.groupby(
            ["outer_fold", "task", "scale"], sort=False
        ):
            ranks = sorted(pd.to_numeric(group["inner_rank"], errors="coerce").dropna().astype(int).unique())
            if ranks == list(range(1, top_k + 1)):
                completed_cells.add((int(outer_fold), str(task), str(scale)))
    incomplete_cells = sorted(expected_cells - completed_cells)
    if incomplete_cells:
        preview = ", ".join(map(str, incomplete_cells[:5]))
        raise RuntimeError(
            f"Stage 6 did not retain Top {top_k} for {len(incomplete_cells)} analysis cells. "
            f"First incomplete cells: {preview}"
        )
    if not ranked_df.empty:
        ranked_df.to_csv(
            OUT_DIR / "nested_v3_inner_config_summary.csv.gz",
            index=False,
            compression="gzip",
        )
    selected_df = pd.DataFrame(read_jsonl(SELECTED_JSONL))
    pred_df = pd.DataFrame(read_jsonl(OUTER_PREDICTIONS_JSONL))

    if not selected_df.empty:
        selected_df.to_csv(OUT_DIR / "nested_v3_selected_configs_rank1.csv", index=False)
        stability = (
            selected_df.groupby(
                ["task", "scale", "config_id", "feature_set", "duration_label"],
                dropna=False,
            )
            .size()
            .reset_index(name="outer_folds_selected")
            .sort_values(
                ["task", "scale", "outer_folds_selected"],
                ascending=[True, True, False],
            )
        )
        stability.to_csv(SELECTION_STABILITY_CSV, index=False)
    if not pred_df.empty:
        pred_df.to_csv(
            OUT_DIR / "nested_v3_outer_subject_predictions_rank1.csv.gz",
            index=False,
            compression="gzip",
        )
        final = compute_final_metrics(pred_df)
        final.to_csv(FINAL_METRICS_CSV, index=False)

    # Compatibility aliases used by older Stage 7/8 versions.
    if not selected_df.empty:
        selected_df.to_csv(OUT_DIR / "nested_v3_selected_configs.csv", index=False)
    if not pred_df.empty:
        pred_df.to_csv(
            OUT_DIR / "nested_v3_outer_subject_predictions.csv.gz",
            index=False,
            compression="gzip",
        )
        compute_final_metrics(pred_df).to_csv(
            OUT_DIR / "nested_v3_final_metrics.csv", index=False
        )

    write_manifest(OUT_DIR, {
        "script": Path(__file__).name,
        "stage": 6,
        "fold_file": str(fold_path),
        "fold_assignment_hash": fold_assignment_hash,
        "mcmi_csv": str(cfg.MCMI_CSV),
        "mcmi_sha256": file_sha256(cfg.MCMI_CSV),
        "n_subjects": int(len(data)),
        "outer_folds": int(cfg.OUTER_N_FOLDS),
        "inner_folds": int(cfg.INNER_N_FOLDS),
        "inner_repeats": int(cfg.INNER_N_REPEATS),
        "inner_splits_per_outer_cell": int(cfg.INNER_N_FOLDS) * int(cfg.INNER_N_REPEATS),
        "tasks": run_tasks,
        "ordinal_scheme": cfg.ordinal_scheme_name(),
        "ordinal_cutoffs": list(cfg.ordinal_cutoffs()),
        "ordinal_labels": cfg.ordinal_labels(),
        "ordinal_n_classes": cfg.ordinal_n_classes(),
        "scales": scales,
        "n_feature_configurations": len(configs),
        "n_successful_configuration_rows": int(len(ranked_df)),
        "top_k_configurations_retained_per_cell": int(top_k),
        "n_topk_candidate_rows": int(len(topk_df)),
        "n_expected_analysis_cells": int(len(expected_cells)),
        "n_completed_analysis_cells": int(len(completed_cells)),
        "n_error_rows": int(len(read_jsonl(ERRORS_JSONL))),
        "feature_config_hash": stable_hash([asdict(c) for c in configs]),
        "selection_rule": "inner_rank_top_k",
        "outer_test_used_for_candidate_ranking": False,
        "rank1_outer_evaluation_is_compatibility_output": True,
        "all_configuration_results_csv": str(ALL_CONFIG_RESULTS_CSV),
        "topk_candidates_csv": str(TOPK_CANDIDATES_CSV),
        "settings_hash": cfg.settings_hash_for(6),
        "permutation_index": os.environ.get("NEUROPSYCHE_PERMUTATION_INDEX") or None,
        "mcmi_rows_permuted_together": env_bool(
            "NEUROPSYCHE_PERMUTE_MCMI_ROWS_TOGETHER",
            True,
        ) if os.environ.get("NEUROPSYCHE_PERMUTATION_INDEX") else None,
    })
    (OUT_DIR / "NESTED_COMPLETE.flag").write_text("complete", encoding="utf-8")
    print(f"Nested V3 screening finished. Results: {OUT_DIR}")
    print(f"All tested configurations: {ALL_CONFIG_RESULTS_CSV}")
    print(f"Top-{top_k} candidates for Stage 8: {TOPK_CANDIDATES_CSV}")


if __name__ == "__main__":
    main()
