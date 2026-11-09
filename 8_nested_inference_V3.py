"""Evaluate retained candidates on outer folds and run Stage 8 inference."""

from __future__ import annotations

import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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

os.environ.setdefault("NEUROPSYCHE_STAGE_NUMBER", "8")

from V3_config_loader import cfg
import V3_model_core as core
from V3_runtime import (
    GracefulStop,
    JsonlStore,
    RunControl,
    env_bool,
    file_sha256,
    load_json,
    make_stage_run_dir,
    read_jsonl,
    stable_hash,
    write_manifest,
)


RESUME = env_bool("NEUROPSYCHE_RESUME", True)
RUN_SCRIPT_NAME = cfg.ordinal_stage_script_name(Path(__file__).name)
OUT_DIR = make_stage_run_dir(cfg.REPORTS_DIR, RUN_SCRIPT_NAME, resume=RESUME)

TOPK_PREDICTIONS_JSONL = OUT_DIR / "topk_outer_subject_predictions.jsonl"
TOPK_PREDICTIONS_CSV = OUT_DIR / "topk_outer_subject_predictions.csv.gz"
TOPK_FOLD_METRICS_CSV = OUT_DIR / "topk_outer_metrics_by_fold_and_rank.csv"
TOPK_RANK_METRICS_CSV = OUT_DIR / "topk_outer_metrics_pooled_by_rank.csv"
TOPK_MEAN_METRICS_CSV = OUT_DIR / "topk_mean_metrics_by_k.csv"
TOPK_ENSEMBLE_METRICS_CSV = OUT_DIR / "topk_ensemble_metrics_by_k.csv"
TOPK_COMBINED_METRICS_CSV = OUT_DIR / "topk_combined_metrics_by_k.csv"
FINAL_TOPK_METRICS_CSV = OUT_DIR / "topk_final_prespecified_k_metrics.csv"

BOOTSTRAP_JSONL = OUT_DIR / "topk_bootstrap_metrics.jsonl"
BOOTSTRAP_SUMMARY_CSV = OUT_DIR / "topk_bootstrap_confidence_intervals.csv"
PERMUTATION_RESULTS_CSV = OUT_DIR / "full_nested_topk_permutation_metrics.csv"
PERMUTATION_PVALUES_CSV = OUT_DIR / "full_nested_topk_permutation_pvalues.csv"
FIXED_PERMUTATION_RESULTS_CSV = OUT_DIR / "fixed_candidate_permutation_metrics.csv"
FIXED_PERMUTATION_PVALUES_CSV = OUT_DIR / "fixed_candidate_permutation_pvalues.csv"

TASK_ORDER = ["binary_BR60", "binary_BR75", "ordinal", "continuous"]
TASK_TITLES = {
    "binary_BR60": "Binary BR60",
    "binary_BR75": "Binary BR75",
    "ordinal": f"Ordinal ({cfg.ordinal_n_classes()} classes)",
    "continuous": "Continuous ridge",
}
TASK_YLABELS = {
    "binary_BR60": "AUC",
    "binary_BR75": "AUC",
    "ordinal": "Quadratic weighted kappa",
    "continuous": "Pearson r",
}
TASK_CHANCE = {
    "binary_BR60": 0.5,
    "binary_BR75": 0.5,
    "ordinal": 0.0,
    "continuous": 0.0,
}


def configured_top_k() -> int:
    value = int(getattr(cfg, "TOP_K_CONFIGS", 20))
    if value < 1:
        raise ValueError("TOP_K_CONFIGS must be at least 1.")
    return value


def nested_input_dir() -> Path:
    raw = cfg.runtime_input("stage6_run")
    run = Path(raw) if raw else None
    if run is None or not run.exists():
        raise FileNotFoundError(
            "No Stage 6 nested-analysis run was found for the selected "
            "ordinal scheme."
        )
    return run


def latest_fold_file() -> Path:
    raw = cfg.runtime_input("fold_file")
    path = Path(raw) if raw else None
    if path is None or not path.exists():
        raise FileNotFoundError("No fixed outer-fold file was found. Run Stage 5 first.")
    return path


def load_analysis_data(permutation_index: Optional[int] = None) -> pd.DataFrame:
    folds = pd.read_csv(latest_fold_file(), dtype={"subject_id": str})
    folds["subject_id"] = folds["subject_id"].astype(str).str.strip()
    folds["outer_fold"] = pd.to_numeric(folds["outer_fold"], errors="raise").astype(int)

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

    if permutation_index is not None:
        base_seed = int(cfg.RANDOM_SEED) + int(permutation_index) * 100003
        if bool(getattr(cfg, "PERMUTE_MCMI_ROWS_TOGETHER", True)):
            order = np.random.default_rng(base_seed).permutation(len(data))
            data.loc[:, list(cfg.MCMI_SCALE_ORDER)] = data.loc[
                order, list(cfg.MCMI_SCALE_ORDER)
            ].to_numpy()
        else:
            for index, scale in enumerate(cfg.MCMI_SCALE_ORDER):
                order = np.random.default_rng(base_seed + (index + 1) * 1009).permutation(len(data))
                data.loc[:, scale] = data.loc[order, scale].to_numpy()
    return data


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
    return np.isfinite(y) if task == "continuous" else (np.isfinite(y) & (y >= 0))


def primary_metric_name(task: str) -> str:
    if task.startswith("binary_"):
        return "roc_auc"
    if task == "ordinal":
        return "qwk"
    return "pearson_r"


def metric_for_task(
    task: str,
    y_true: np.ndarray,
    score: np.ndarray,
    pred: np.ndarray,
) -> dict[str, float]:
    if task.startswith("binary_"):
        roc = float(roc_auc_score(y_true, score)) if len(np.unique(y_true)) == 2 else np.nan
        return {
            "primary": roc,
            "roc_auc": roc,
            "pr_auc": float(average_precision_score(y_true, score)) if len(np.unique(y_true)) == 2 else np.nan,
            "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
            "accuracy": float(accuracy_score(y_true, pred)),
            "sensitivity": float(recall_score(y_true, pred, pos_label=1, zero_division=0)),
            "specificity": float(recall_score(y_true, pred, pos_label=0, zero_division=0)),
            "precision": float(precision_score(y_true, pred, zero_division=0)),
            "f1": float(f1_score(y_true, pred, zero_division=0)),
        }
    if task == "ordinal":
        qwk = float(cohen_kappa_score(y_true.astype(int), pred.astype(int), weights="quadratic"))
        return {
            "primary": qwk,
            "qwk": qwk,
            "accuracy": float(accuracy_score(y_true.astype(int), pred.astype(int))),
            "mae": float(mean_absolute_error(y_true, pred)),
        }
    r = np.nan
    if len(y_true) >= 3 and np.std(y_true) > 0 and np.std(score) > 0:
        r = float(np.corrcoef(y_true, score)[0, 1])
    return {
        "primary": r,
        "pearson_r": r,
        "mae": float(mean_absolute_error(y_true, score)),
        "rmse": float(math.sqrt(mean_squared_error(y_true, score))),
    }


def load_topk_candidates(source_run: Path, requested_k: int) -> pd.DataFrame:
    candidate_path = source_run / "nested_v3_topk_candidates.csv"
    if candidate_path.exists():
        candidates = pd.read_csv(candidate_path)
    else:
        all_path = source_run / "nested_v3_all_configuration_results.csv"
        if not all_path.exists():
            raise FileNotFoundError(
                f"Stage 6 contains neither nested_v3_topk_candidates.csv nor {all_path.name}"
            )
        candidates = pd.read_csv(all_path)
        if "status" in candidates.columns:
            candidates = candidates[candidates["status"].astype(str).str.lower().eq("ok")]
        candidates["inner_metric_mean"] = pd.to_numeric(candidates["inner_metric_mean"], errors="coerce")
        parts = []
        for _, group in candidates.groupby(["outer_fold", "task", "scale"], sort=False):
            group = group.sort_values(
                ["inner_metric_mean", "n_features", "config_id"],
                ascending=[False, True, True],
            ).copy()
            group["inner_rank"] = np.arange(1, len(group) + 1)
            parts.append(group[group["inner_rank"] <= requested_k])
        candidates = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

    required = {"outer_fold", "task", "scale", "config_id", "inner_rank"}
    missing = required - set(candidates.columns)
    if missing:
        raise ValueError(f"Top-K candidate file is missing columns: {sorted(missing)}")
    candidates["outer_fold"] = pd.to_numeric(candidates["outer_fold"], errors="raise").astype(int)
    candidates["config_id"] = pd.to_numeric(candidates["config_id"], errors="raise").astype(int)
    candidates["inner_rank"] = pd.to_numeric(candidates["inner_rank"], errors="raise").astype(int)
    candidates = candidates[candidates["inner_rank"] <= requested_k].copy()
    candidates = candidates.sort_values(["outer_fold", "task", "scale", "inner_rank"])
    incomplete = []
    expected_ranks = list(range(1, requested_k + 1))
    for cell, group in candidates.groupby(["outer_fold", "task", "scale"], sort=False):
        ranks = sorted(group["inner_rank"].unique().tolist())
        if ranks != expected_ranks or len(group) != requested_k:
            incomplete.append(cell)
    if incomplete:
        preview = ", ".join(map(str, incomplete[:5]))
        raise ValueError(
            f"The selected Stage 6 run does not contain exactly Top {requested_k} for "
            f"{len(incomplete)} analysis cells. First incomplete cells: {preview}"
        )
    manifest = load_json(source_run / "manifest.json", {}) or {}
    expected_cells = {
        (outer_fold, str(task), str(scale))
        for outer_fold in range(1, int(manifest.get("outer_folds", 0)) + 1)
        for task in manifest.get("tasks", [])
        for scale in manifest.get("scales", [])
    }
    actual_cells = {
        (int(outer_fold), str(task), str(scale))
        for outer_fold, task, scale in candidates[["outer_fold", "task", "scale"]]
        .drop_duplicates()
        .itertuples(index=False)
    }
    missing_cells = sorted(expected_cells - actual_cells)
    if missing_cells:
        preview = ", ".join(map(str, missing_cells[:5]))
        raise ValueError(
            f"The selected Stage 6 run is missing {len(missing_cells)} analysis cells. "
            f"First missing cells: {preview}"
        )
    return candidates


def config_lookup() -> dict[int, core.FeatureConfig]:
    return {int(c.config_id): c for c in core.build_feature_configs()}


def fit_predict_candidate(
    config: core.FeatureConfig,
    train_subjects: list[str],
    test_subjects: list[str],
    score_map: dict[str, float],
    task: str,
    threshold: Optional[float],
) -> list[dict[str, Any]]:
    all_subjects = train_subjects + test_subjects
    X, names, used, missing = core.assemble_X(all_subjects, config)
    if used != all_subjects or missing:
        raise RuntimeError(f"Candidate configuration has missing subjects: {missing[:3]}")
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
            scores = np.asarray(core.binary_score(model, X_test), dtype=float)
        use_threshold = 0.0 if threshold is None or not np.isfinite(float(threshold)) else float(threshold)
        preds = (scores >= use_threshold).astype(int)
        centered = scores - use_threshold
    elif task == "ordinal":
        model = core.CumulativeOrdinalLogit(
            C=float(cfg.LOGISTIC_C),
            n_classes=cfg.ordinal_n_classes(),
        )
        model.fit(X_train, y_train.astype(int))
        probs = model.predict_proba(X_test)
        preds = np.argmax(probs, axis=1).astype(int)
        scores = (probs * np.arange(probs.shape[1])[None, :]).sum(axis=1)
        centered = scores
        use_threshold = np.nan
    else:
        model = core.make_ridge_pipeline(float(cfg.RIDGE_ALPHA))
        model.fit(X_train, y_train.astype(float))
        scores = np.asarray(model.predict(X_test), dtype=float)
        preds = scores
        centered = scores
        use_threshold = np.nan

    rows = []
    for sid, raw_score, y_true, score, centered_score, pred in zip(
        test_subjects,
        [score_map[s] for s in test_subjects],
        y_test,
        scores,
        centered,
        preds,
    ):
        rows.append({
            "subject_id": sid,
            "raw_br_score": float(raw_score),
            "y_true": float(y_true),
            "prediction_score": float(score),
            "prediction_score_centered": float(centered_score),
            "prediction_label": float(pred),
            "decision_threshold": float(use_threshold) if np.isfinite(use_threshold) else np.nan,
        })
    return rows


def evaluate_topk_candidates(
    source_run: Path,
    evaluation_dir: Path,
    control: RunControl,
    permutation_index: Optional[int] = None,
) -> pd.DataFrame:
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    predictions_jsonl = evaluation_dir / "topk_outer_subject_predictions.jsonl"
    predictions_csv = evaluation_dir / "topk_outer_subject_predictions.csv.gz"
    error_store = JsonlStore(evaluation_dir / "topk_outer_errors.jsonl")
    prediction_store = JsonlStore(predictions_jsonl, key_field="prediction_key")
    completed = prediction_store.keys
    expected_prediction_keys: set[str] = set()

    requested_k = configured_top_k()
    candidates = load_topk_candidates(source_run, requested_k)
    if candidates.empty:
        raise RuntimeError("No Top-K candidates were found in the Stage 6 run.")
    data = load_analysis_data(permutation_index=permutation_index)
    configs = config_lookup()

    score_maps = {
        scale: dict(zip(data["subject_id"].astype(str), pd.to_numeric(data[scale], errors="coerce")))
        for scale in candidates["scale"].astype(str).unique()
    }

    total = len(candidates)
    for index, candidate in enumerate(candidates.itertuples(index=False), start=1):
        outer_fold = int(candidate.outer_fold)
        task = str(candidate.task)
        scale = str(candidate.scale)
        config_id = int(candidate.config_id)
        rank = int(candidate.inner_rank)
        candidate_key = f"outer={outer_fold}|task={task}|scale={scale}|rank={rank}|config={config_id}"
        control.checkpoint(f"Top-K outer evaluation {index}/{total}: {candidate_key}")

        scores_all = np.asarray(
            [score_maps[scale].get(sid, np.nan) for sid in data["subject_id"].astype(str)],
            dtype=float,
        )
        labels_all = task_labels(scores_all, task)
        valid = valid_label_mask(labels_all, task)
        cell = data.loc[valid, ["subject_id", "outer_fold"]].copy()
        train_subjects = cell[cell["outer_fold"] != outer_fold]["subject_id"].astype(str).tolist()
        test_subjects = cell[cell["outer_fold"] == outer_fold]["subject_id"].astype(str).tolist()
        if not train_subjects or not test_subjects:
            raise RuntimeError(f"Insufficient train/test subjects for {candidate_key}.")

        missing_keys = [
            f"{candidate_key}|subject={sid}"
            for sid in test_subjects
            if f"{candidate_key}|subject={sid}" not in completed
        ]
        expected_prediction_keys.update(
            f"{candidate_key}|subject={sid}" for sid in test_subjects
        )
        if not missing_keys:
            continue
        if config_id not in configs:
            raise RuntimeError(f"Configuration ID {config_id} is not available under the current Stage 4/6 settings.")

        threshold = getattr(candidate, "inner_selected_threshold", np.nan)
        try:
            outer_rows = fit_predict_candidate(
                configs[config_id],
                train_subjects,
                test_subjects,
                score_maps[scale],
                task,
                threshold,
            )
            for row in outer_rows:
                row.update({
                    "prediction_key": f"{candidate_key}|subject={row['subject_id']}",
                    "candidate_key": candidate_key,
                    "outer_fold": outer_fold,
                    "task": task,
                    "ordinal_scheme": (
                        cfg.ordinal_scheme_name() if task == "ordinal" else ""
                    ),
                    "scale": scale,
                    "inner_rank": rank,
                    "config_id": config_id,
                    "feature_set": getattr(candidate, "feature_set", configs[config_id].feature_set),
                    "duration_sec": getattr(candidate, "duration_sec", configs[config_id].duration_sec),
                    "duration_label": getattr(candidate, "duration_label", configs[config_id].duration_label),
                    "inner_metric_mean": getattr(candidate, "inner_metric_mean", np.nan),
                    "outer_test_used_for_ranking": False,
                })
                prediction_store.append(row)
                completed.add(row["prediction_key"])
        except Exception as error:
            error_store.append({
                "candidate_key": candidate_key,
                "outer_fold": outer_fold,
                "task": task,
                "scale": scale,
                "inner_rank": rank,
                "config_id": config_id,
                "error": repr(error),
            })
        if index % 20 == 0 or index == total:
            print(f"Top-K outer evaluation: {index}/{total} candidates", flush=True)

    predictions = pd.DataFrame(read_jsonl(predictions_jsonl))
    actual_prediction_keys = set(predictions.get("prediction_key", pd.Series(dtype=str)).astype(str))
    missing_predictions = sorted(expected_prediction_keys - actual_prediction_keys)
    if missing_predictions:
        raise RuntimeError(
            f"Top-K evaluation is incomplete: {len(missing_predictions)} subject predictions are missing. "
            f"First missing key: {missing_predictions[0]}"
        )
    if not predictions.empty:
        predictions.to_csv(predictions_csv, index=False, compression="gzip")
    return predictions


def pooled_rank_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (task, scale, rank), group in predictions.groupby(["task", "scale", "inner_rank"], sort=True):
        y = group["y_true"].to_numpy(dtype=float)
        score = group["prediction_score"].to_numpy(dtype=float)
        pred = group["prediction_label"].to_numpy(dtype=float)
        metrics = metric_for_task(
            str(task),
            y.astype(int) if task != "continuous" else y,
            score,
            pred.astype(int) if task != "continuous" else pred,
        )
        rows.append({
            "task": task,
            "scale": scale,
            "inner_rank": int(rank),
            "n_subjects": int(group["subject_id"].astype(str).nunique()),
            "n_outer_folds": int(group["outer_fold"].nunique()),
            "config_ids_by_outer_fold": ";".join(
                f"{int(fold)}:{int(config)}"
                for fold, config in group[["outer_fold", "config_id"]].drop_duplicates().sort_values("outer_fold").itertuples(index=False)
            ),
            **metrics,
        })
    return pd.DataFrame(rows)


def fold_rank_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (task, scale, outer_fold, rank), group in predictions.groupby(
        ["task", "scale", "outer_fold", "inner_rank"], sort=True
    ):
        y = group["y_true"].to_numpy(dtype=float)
        score = group["prediction_score"].to_numpy(dtype=float)
        pred = group["prediction_label"].to_numpy(dtype=float)
        metrics = metric_for_task(
            str(task),
            y.astype(int) if task != "continuous" else y,
            score,
            pred.astype(int) if task != "continuous" else pred,
        )
        rows.append({
            "task": task,
            "scale": scale,
            "outer_fold": int(outer_fold),
            "inner_rank": int(rank),
            "config_id": int(group["config_id"].iloc[0]),
            "feature_set": str(group["feature_set"].iloc[0]),
            "duration_label": str(group["duration_label"].iloc[0]),
            "n_subjects": int(len(group)),
            **metrics,
        })
    return pd.DataFrame(rows)


def topk_mean_metrics(rank_metrics: pd.DataFrame, max_k: int) -> pd.DataFrame:
    rows = []
    metric_columns = [
        c for c in ["primary", "roc_auc", "pr_auc", "balanced_accuracy", "accuracy", "sensitivity", "specificity", "precision", "f1", "qwk", "mae", "rmse", "pearson_r"]
        if c in rank_metrics.columns
    ]
    for (task, scale), group in rank_metrics.groupby(["task", "scale"], sort=True):
        group = group.sort_values("inner_rank")
        for k in range(1, min(max_k, int(group["inner_rank"].max())) + 1):
            selected = group[group["inner_rank"] <= k]
            if len(selected) < k:
                continue
            row = {
                "task": task,
                "scale": scale,
                "top_k": int(k),
                "n_candidate_ranks": int(len(selected)),
                "primary_metric_name": primary_metric_name(str(task)),
            }
            for col in metric_columns:
                values = pd.to_numeric(selected[col], errors="coerce")
                row[f"mean_of_config_{col}"] = float(values.mean())
                row[f"sd_of_config_{col}"] = float(values.std(ddof=1)) if values.notna().sum() > 1 else 0.0
                row[f"min_of_config_{col}"] = float(values.min()) if values.notna().any() else np.nan
                row[f"max_of_config_{col}"] = float(values.max()) if values.notna().any() else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def ensemble_arrays(group: pd.DataFrame, task: str, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected = group[group["inner_rank"] <= int(k)].copy()
    subject_groups = []
    for subject_id, subject_rows in selected.groupby("subject_id", sort=False):
        if subject_rows["inner_rank"].nunique() < k:
            continue
        subject_groups.append(subject_rows.sort_values("inner_rank"))
    if not subject_groups:
        return np.asarray([]), np.asarray([]), np.asarray([])

    y = np.asarray([rows["y_true"].iloc[0] for rows in subject_groups], dtype=float)
    if task.startswith("binary_"):
        score = np.asarray([
            pd.to_numeric(rows["prediction_score_centered"], errors="coerce").mean()
            for rows in subject_groups
        ], dtype=float)
        pred = (score >= 0.0).astype(int)
    elif task == "ordinal":
        score = np.asarray([
            pd.to_numeric(rows["prediction_score"], errors="coerce").mean()
            for rows in subject_groups
        ], dtype=float)
        pred = np.clip(
            np.rint(score),
            0,
            cfg.ordinal_n_classes() - 1,
        ).astype(int)
    else:
        score = np.asarray([
            pd.to_numeric(rows["prediction_score"], errors="coerce").mean()
            for rows in subject_groups
        ], dtype=float)
        pred = score
    return y, score, pred


def topk_ensemble_metrics(predictions: pd.DataFrame, max_k: int) -> pd.DataFrame:
    rows = []
    for (task, scale), group in predictions.groupby(["task", "scale"], sort=True):
        available_k = int(group["inner_rank"].max())
        for k in range(1, min(max_k, available_k) + 1):
            y, score, pred = ensemble_arrays(group, str(task), k)
            if len(y) == 0:
                continue
            metrics = metric_for_task(
                str(task),
                y.astype(int) if task != "continuous" else y,
                score,
                pred.astype(int) if task != "continuous" else pred,
            )
            rows.append({
                "task": task,
                "scale": scale,
                "top_k": int(k),
                "n_subjects": int(len(y)),
                "primary_metric_name": primary_metric_name(str(task)),
                **{f"ensemble_{key}": value for key, value in metrics.items()},
            })
    return pd.DataFrame(rows)


def summarize_topk(
    predictions: pd.DataFrame,
    output_dir: Path,
    final_k: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if predictions.empty:
        raise RuntimeError("No Top-K outer predictions were produced.")
    fold_metrics = fold_rank_metrics(predictions)
    rank_metrics = pooled_rank_metrics(predictions)
    max_available = int(rank_metrics["inner_rank"].max())
    if max_available < int(final_k):
        raise RuntimeError(
            f"Top-K summary is incomplete: requested {int(final_k)}, available {max_available}."
        )
    actual_final_k = int(final_k)
    mean_metrics = topk_mean_metrics(rank_metrics, actual_final_k)
    ensemble_metrics = topk_ensemble_metrics(predictions, actual_final_k)
    combined = mean_metrics.merge(ensemble_metrics, on=["task", "scale", "top_k", "primary_metric_name"], how="outer")
    combined["prespecified_top_k"] = int(final_k)
    combined["actual_maximum_validated_k"] = int(actual_final_k)
    combined["is_prespecified_final_k"] = combined["top_k"].astype(int) == int(actual_final_k)
    combined["primary_topk_result"] = combined["mean_of_config_primary"]
    combined["secondary_ensemble_result"] = combined.get("ensemble_primary")

    fold_metrics.to_csv(output_dir / "topk_outer_metrics_by_fold_and_rank.csv", index=False)
    rank_metrics.to_csv(output_dir / "topk_outer_metrics_pooled_by_rank.csv", index=False)
    mean_metrics.to_csv(output_dir / "topk_mean_metrics_by_k.csv", index=False)
    ensemble_metrics.to_csv(output_dir / "topk_ensemble_metrics_by_k.csv", index=False)
    combined.to_csv(output_dir / "topk_combined_metrics_by_k.csv", index=False)
    final = combined[combined["is_prespecified_final_k"] == True].copy()
    final.to_csv(output_dir / "topk_final_prespecified_k_metrics.csv", index=False)
    return fold_metrics, rank_metrics, combined, final


def requested_plot_k_values(final_k: int) -> list[int]:
    values = [int(v) for v in cfg.PERMUTATION_K_VALUES if int(v) >= 1]
    values = sorted({v for v in values if v <= final_k} | {int(final_k)})
    return values


def create_outer_topk_plot(combined: pd.DataFrame, output: Path, final_k: int) -> None:
    if combined.empty:
        return
    k_values = requested_plot_k_values(final_k)
    scale_order = [s for s in cfg.MCMI_SCALE_ORDER if s in set(combined["scale"].astype(str))]
    remaining = [s for s in combined["scale"].astype(str).unique() if s not in scale_order]
    scale_order.extend(remaining)

    fig, axes = plt.subplots(2, 2, figsize=(16, 9))
    axes = axes.ravel()
    handles, labels = [], []
    for ax, task in zip(axes, TASK_ORDER):
        sub = combined[combined["task"].astype(str) == task]
        ax.set_title(TASK_TITLES[task], fontweight="bold")
        ax.set_ylabel(TASK_YLABELS[task])
        ax.set_xticks(np.arange(len(scale_order)))
        ax.set_xticklabels(scale_order, rotation=45, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.25)
        for k in k_values:
            line = sub[sub["top_k"] == int(k)].set_index("scale")
            y = [
                float(line.loc[scale, "mean_of_config_primary"])
                if scale in line.index else np.nan
                for scale in scale_order
            ]
            plotted, = ax.plot(np.arange(len(scale_order)), y, marker="o", linewidth=1.6, markersize=4, label=f"Top {k}")
            if ax is axes[0]:
                handles.append(plotted)
                labels.append(f"Top {k}")
        chance_line = ax.axhline(TASK_CHANCE[task], linestyle="--", linewidth=1.0, label="Chance")
        if ax is axes[0]:
            handles.append(chance_line)
            labels.append("Chance")

    fig.suptitle("Outer-test mean performance of inner-ranked Top-k EEG configurations", fontsize=15, fontweight="bold", y=0.98)
    fig.text(
        0.5,
        0.035,
        "Candidates were ranked only inside each outer-training sample. Each line is the mean pooled outer-test metric of inner ranks 1 through k.",
        ha="center",
        fontsize=9,
    )
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False, bbox_to_anchor=(0.5, 0.005))
    fig.tight_layout(rect=(0.02, 0.08, 1, 0.95))
    if bool(getattr(cfg, "MAKE_PNG", True)):
        fig.savefig(output / "outer_test_topk_mean_performance.png", dpi=200)
    if bool(getattr(cfg, "MAKE_PDF", True)):
        fig.savefig(output / "outer_test_topk_mean_performance.pdf")
    plt.close(fig)


def bootstrap_group_arrays(group: pd.DataFrame, final_k: int):
    ranks = sorted(int(r) for r in group["inner_rank"].unique() if int(r) <= final_k)
    subject_ids = sorted(group["subject_id"].astype(str).unique())
    by_rank = {}
    for rank in ranks:
        rank_df = group[group["inner_rank"] == rank].copy()
        rank_df["subject_id"] = rank_df["subject_id"].astype(str)
        rank_df = rank_df.drop_duplicates("subject_id").set_index("subject_id")
        if not set(subject_ids).issubset(set(rank_df.index)):
            continue
        by_rank[rank] = rank_df.loc[subject_ids]
    ranks = sorted(by_rank)
    return subject_ids, ranks, by_rank


def bootstrap_statistics_for_indices(
    task: str,
    subject_ids: list[str],
    ranks: list[int],
    by_rank: dict[int, pd.DataFrame],
    indices: np.ndarray,
) -> tuple[float, float, int]:
    rank_values = []
    y_reference = None
    ensemble_scores = []
    for rank in ranks:
        df = by_rank[rank]
        y = df["y_true"].to_numpy(dtype=float)[indices]
        score = df["prediction_score"].to_numpy(dtype=float)[indices]
        pred = df["prediction_label"].to_numpy(dtype=float)[indices]
        metrics = metric_for_task(
            task,
            y.astype(int) if task != "continuous" else y,
            score,
            pred.astype(int) if task != "continuous" else pred,
        )
        rank_values.append(metrics["primary"])
        if y_reference is None:
            y_reference = y
        if task.startswith("binary_"):
            ensemble_scores.append(df["prediction_score_centered"].to_numpy(dtype=float)[indices])
        else:
            ensemble_scores.append(score)

    topk_mean = float(np.nanmean(rank_values)) if rank_values else np.nan
    if y_reference is None or not ensemble_scores:
        return topk_mean, np.nan, len(ranks)
    ensemble_score = np.nanmean(np.vstack(ensemble_scores), axis=0)
    if task.startswith("binary_"):
        ensemble_pred = (ensemble_score >= 0).astype(int)
    elif task == "ordinal":
        ensemble_pred = np.clip(
            np.rint(ensemble_score),
            0,
            cfg.ordinal_n_classes() - 1,
        ).astype(int)
    else:
        ensemble_pred = ensemble_score
    ensemble_primary = metric_for_task(
        task,
        y_reference.astype(int) if task != "continuous" else y_reference,
        ensemble_score,
        ensemble_pred.astype(int) if task != "continuous" else ensemble_pred,
    )["primary"]
    return topk_mean, float(ensemble_primary), len(ranks)


def run_bootstrap(predictions: pd.DataFrame, final_k: int, control: RunControl) -> pd.DataFrame:
    n_boot = int(cfg.N_BOOTSTRAPS)
    store = JsonlStore(BOOTSTRAP_JSONL, key_field="bootstrap_key")
    completed = store.keys
    rng_base = int(cfg.RANDOM_SEED) + 800000

    for (task, scale), group in predictions.groupby(["task", "scale"], sort=True):
        subject_ids, ranks, by_rank = bootstrap_group_arrays(group, final_k)
        if not subject_ids or not ranks:
            continue
        token = int(stable_hash({"task": str(task), "scale": str(scale)}, length=8), 16)
        for b in range(n_boot):
            key = f"{task}|{scale}|{b}"
            if key in completed:
                continue
            if b % 50 == 0:
                control.checkpoint(f"Top-K bootstrap {task}/{scale}: {b}/{n_boot}")
            rng = np.random.default_rng(rng_base + token + b)
            indices = rng.integers(0, len(subject_ids), size=len(subject_ids))
            mean_value, ensemble_value, actual_k = bootstrap_statistics_for_indices(
                str(task), subject_ids, ranks, by_rank, indices
            )
            store.append({
                "bootstrap_key": key,
                "task": task,
                "scale": scale,
                "bootstrap": b,
                "requested_top_k": int(final_k),
                "actual_top_k": int(actual_k),
                "topk_mean_primary": mean_value,
                "topk_ensemble_primary": ensemble_value,
            })
            completed.add(key)

    df = pd.DataFrame(read_jsonl(BOOTSTRAP_JSONL))
    rows = []
    for (task, scale), group in df.groupby(["task", "scale"]):
        for statistic in ["topk_mean_primary", "topk_ensemble_primary"]:
            values = pd.to_numeric(group[statistic], errors="coerce").dropna().to_numpy()
            rows.append({
                "task": task,
                "scale": scale,
                "statistic": statistic,
                "top_k": int(final_k),
                "n_valid_bootstraps": int(len(values)),
                "bootstrap_mean": float(np.mean(values)) if len(values) else np.nan,
                "bootstrap_sd": float(np.std(values, ddof=1)) if len(values) > 1 else np.nan,
                "ci_2_5": float(np.quantile(values, 0.025)) if len(values) else np.nan,
                "ci_97_5": float(np.quantile(values, 0.975)) if len(values) else np.nan,
            })
    summary = pd.DataFrame(rows)
    summary.to_csv(BOOTSTRAP_SUMMARY_CSV, index=False)
    return summary


def bh_adjust(pvalues: pd.Series) -> pd.Series:
    p = pd.to_numeric(pvalues, errors="coerce").to_numpy(dtype=float)
    out = np.full(len(p), np.nan)
    valid_idx = np.where(np.isfinite(p))[0]
    if len(valid_idx) == 0:
        return pd.Series(out, index=pvalues.index)
    order = valid_idx[np.argsort(p[valid_idx])]
    m = len(order)
    adjusted = np.empty(m)
    running = 1.0
    for rank_from_end, idx in enumerate(order[::-1], start=1):
        rank = m - rank_from_end + 1
        value = min(running, p[idx] * m / rank)
        adjusted[m - rank_from_end] = value
        running = value
    for pos, idx in enumerate(order):
        out[idx] = min(1.0, adjusted[pos])
    return pd.Series(out, index=pvalues.index)


def permutation_k_values(final_k: int) -> list[int]:
    values = sorted({int(k) for k in cfg.PERMUTATION_K_VALUES if 1 <= int(k) <= int(final_k)})
    if not values:
        raise ValueError("No permutation Top-k value is valid for the selected Stage 6 run.")
    return values


def run_fixed_candidate_permutations(
    source_run: Path,
    observed_combined: pd.DataFrame,
    final_k: int,
    control: RunControl,
) -> pd.DataFrame:
    n_perm = int(cfg.N_PERMUTATIONS)
    k_values = permutation_k_values(final_k)
    perm_root = OUT_DIR / "fixed_candidate_permutations"
    perm_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    completed: set[int] = set()
    if FIXED_PERMUTATION_RESULTS_CSV.exists():
        old = pd.read_csv(FIXED_PERMUTATION_RESULTS_CSV)
        if not old.empty and "permutation" in old.columns:
            rows = old.to_dict("records")
            completed = set(pd.to_numeric(old["permutation"], errors="coerce").dropna().astype(int))

    for perm in range(1, n_perm + 1):
        control.checkpoint(f"before fixed-candidate permutation {perm}/{n_perm}")
        eval_dir = perm_root / f"perm_{perm:04d}"
        combined_path = eval_dir / "topk_combined_metrics_by_k.csv"
        if perm not in completed or not combined_path.exists():
            predictions = evaluate_topk_candidates(
                source_run,
                eval_dir,
                control,
                permutation_index=perm,
            )
            if predictions.empty:
                raise RuntimeError(f"No predictions were produced for permutation {perm}.")
            _, _, combined, _ = summarize_topk(predictions, eval_dir, final_k)
        else:
            combined = pd.read_csv(combined_path)

        keep = combined[pd.to_numeric(combined["top_k"], errors="coerce").isin(k_values)].copy()
        if keep.empty:
            raise RuntimeError(f"Permutation {perm} produced no requested Top-k results.")
        keep["permutation"] = perm
        rows = [row for row in rows if int(row.get("permutation", -1)) != perm]
        rows.extend(keep.to_dict("records"))
        pd.DataFrame(rows).to_csv(FIXED_PERMUTATION_RESULTS_CSV, index=False)
        completed.add(perm)
        print(f"Completed fixed-candidate permutation {perm}/{n_perm}", flush=True)

    null_df = pd.DataFrame(rows)
    observed = observed_combined[
        pd.to_numeric(observed_combined["top_k"], errors="coerce").isin(k_values)
    ].copy()
    statistic_columns = {
        "topk_mean_primary": "primary_topk_result",
        "topk_ensemble_primary": "secondary_ensemble_result",
    }
    pvalue_rows: list[dict[str, Any]] = []

    for _, real in observed.iterrows():
        task = str(real["task"])
        scale = str(real["scale"])
        top_k = int(real["top_k"])
        same_cell = null_df[
            null_df["task"].astype(str).eq(task)
            & null_df["scale"].astype(str).eq(scale)
            & pd.to_numeric(null_df["top_k"], errors="coerce").eq(top_k)
        ]
        same_task_k = null_df[
            null_df["task"].astype(str).eq(task)
            & pd.to_numeric(null_df["top_k"], errors="coerce").eq(top_k)
        ]
        for statistic, column in statistic_columns.items():
            real_value = pd.to_numeric(pd.Series([real[column]]), errors="coerce").iloc[0]
            if not np.isfinite(real_value):
                continue
            values = pd.to_numeric(same_cell[column], errors="coerce").dropna().to_numpy(dtype=float)
            raw_p = (1 + int(np.sum(values >= real_value))) / (1 + len(values)) if len(values) else np.nan
            max_values = []
            for _, group in same_task_k.groupby("permutation"):
                group_values = pd.to_numeric(group[column], errors="coerce").dropna()
                if len(group_values):
                    max_values.append(float(group_values.max()))
            max_values = np.asarray(max_values, dtype=float)
            fwer_p = (
                (1 + int(np.sum(max_values >= real_value))) / (1 + len(max_values))
                if len(max_values) else np.nan
            )
            pvalue_rows.append({
                "task": task,
                "scale": scale,
                "statistic": statistic,
                "top_k": top_k,
                "real_metric": float(real_value),
                "n_permutations": int(len(values)),
                "raw_fixed_candidate_permutation_p": raw_p,
                "max_stat_fwer_p": fwer_p,
            })

    pvalues = pd.DataFrame(pvalue_rows)
    if pvalues.empty:
        raise RuntimeError("No fixed-candidate permutation p-values were produced.")
    pvalues["fdr_bh_q"] = pvalues.groupby(
        ["task", "statistic", "top_k"]
    )["raw_fixed_candidate_permutation_p"].transform(bh_adjust)
    pvalues = pvalues.sort_values(["task", "statistic", "top_k", "raw_fixed_candidate_permutation_p"])
    pvalues.to_csv(FIXED_PERMUTATION_PVALUES_CSV, index=False)
    return pvalues


def run_full_nested_permutations(
    real_final: pd.DataFrame,
    final_k: int,
    control: RunControl,
) -> pd.DataFrame:
    n_perm = int(cfg.N_PERMUTATIONS)
    perm_root = OUT_DIR / "full_nested_topk_permutations"
    perm_root.mkdir(parents=True, exist_ok=True)
    script6 = Path(__file__).resolve().parent / cfg.SCRIPT_NESTED_MCMI

    result_rows = []
    completed_perm = set()
    if PERMUTATION_RESULTS_CSV.exists():
        old = pd.read_csv(PERMUTATION_RESULTS_CSV)
        result_rows = old.to_dict("records")
        completed_perm = set(old["permutation"].astype(int).unique())

    for perm in range(1, n_perm + 1):
        control.checkpoint(f"before full nested Top-K permutation {perm}/{n_perm}")
        perm_dir = perm_root / f"perm_{perm:04d}" / "stage6"
        final_path = perm_root / f"perm_{perm:04d}" / "topk_evaluation" / "topk_final_prespecified_k_metrics.csv"
        if perm not in completed_perm or not final_path.exists():
            env = os.environ.copy()
            env["NEUROPSYCHE_SETTINGS_FILE"] = os.environ.get(
                "NEUROPSYCHE_SETTINGS_FILE",
                str(Path(__file__).resolve().parent / "V3_user_settings.json"),
            )
            env["NEUROPSYCHE_STAGE_RUN_DIR"] = str(perm_dir)
            env["NEUROPSYCHE_CONTROL_DIR"] = str(control.control_dir)
            env["NEUROPSYCHE_RESUME"] = "1"
            env["NEUROPSYCHE_PERMUTATION_INDEX"] = str(perm)
            env["NEUROPSYCHE_PERMUTE_MCMI_ROWS_TOGETHER"] = (
                "1" if bool(cfg.PERMUTE_MCMI_ROWS_TOGETHER) else "0"
            )
            env["NEUROPSYCHE_STAGE_NUMBER"] = "6"
            code = subprocess.call([sys.executable, "-u", str(script6)], cwd=str(script6.parent), env=env)
            if code != 0:
                raise RuntimeError(f"Nested permutation {perm} failed with exit code {code}")

            eval_dir = final_path.parent
            perm_predictions = evaluate_topk_candidates(
                perm_dir,
                eval_dir,
                control,
                permutation_index=perm,
            )
            summarize_topk(perm_predictions, eval_dir, final_k)

        null_final = pd.read_csv(final_path)
        null_final["permutation"] = perm
        result_rows = [r for r in result_rows if int(r.get("permutation", -1)) != perm]
        result_rows.extend(null_final.to_dict("records"))
        pd.DataFrame(result_rows).to_csv(PERMUTATION_RESULTS_CSV, index=False)
        completed_perm.add(perm)
        print(f"Completed full nested Top-K permutation {perm}/{n_perm}", flush=True)

    null_df = pd.DataFrame(result_rows)
    rows = []
    statistic_columns = {
        "topk_mean_primary": "primary_topk_result",
        "topk_ensemble_primary": "secondary_ensemble_result",
    }
    for _, real in real_final.iterrows():
        task, scale = str(real["task"]), str(real["scale"])
        for statistic, column in statistic_columns.items():
            real_value = float(real[column])
            null = null_df[(null_df["task"].astype(str) == task) & (null_df["scale"].astype(str) == scale)]
            values = pd.to_numeric(null[column], errors="coerce").dropna().to_numpy()
            raw_p = (1 + int(np.sum(values >= real_value))) / (1 + len(values)) if len(values) else np.nan

            max_values = []
            for _, pg in null_df[null_df["task"].astype(str) == task].groupby("permutation"):
                vals = pd.to_numeric(pg[column], errors="coerce").dropna()
                if len(vals):
                    max_values.append(float(vals.max()))
            max_values = np.asarray(max_values, dtype=float)
            fwer_p = (1 + int(np.sum(max_values >= real_value))) / (1 + len(max_values)) if len(max_values) else np.nan
            rows.append({
                "task": task,
                "scale": scale,
                "statistic": statistic,
                "top_k": int(final_k),
                "real_metric": real_value,
                "n_permutations": int(len(values)),
                "raw_full_nested_permutation_p": raw_p,
                "max_stat_fwer_p": fwer_p,
            })
    pvalues = pd.DataFrame(rows)
    pvalues["fdr_bh_q"] = pvalues.groupby(["task", "statistic"])["raw_full_nested_permutation_p"].transform(bh_adjust)
    pvalues.to_csv(PERMUTATION_PVALUES_CSV, index=False)
    return pvalues


def main() -> None:
    cfg.apply_persisted_settings()
    if str(cfg.MULTIPLICITY_METHOD) != "fdr_bh":
        raise ValueError(f"Unsupported multiplicity method: {cfg.MULTIPLICITY_METHOD}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    control = RunControl.from_environment(OUT_DIR / "_control")
    source = nested_input_dir()
    final_k = configured_top_k()

    try:
        predictions = evaluate_topk_candidates(source, OUT_DIR, control)
        if predictions.empty:
            raise RuntimeError("No Top-K outer-test predictions were produced.")
        predictions.to_csv(TOPK_PREDICTIONS_CSV, index=False, compression="gzip")

        fold_metrics, rank_metrics, combined, real_final = summarize_topk(
            predictions,
            OUT_DIR,
            final_k,
        )
        create_outer_topk_plot(combined, OUT_DIR, int(real_final["top_k"].max()))

        bootstrap_summary = pd.DataFrame()
        permutation_pvalues = pd.DataFrame()
        if bool(getattr(cfg, "RUN_BOOTSTRAP", True)):
            bootstrap_summary = run_bootstrap(predictions, final_k, control)
            print(f"Top-K bootstrap confidence intervals: {BOOTSTRAP_SUMMARY_CSV}")
        permutation_method = str(getattr(cfg, "PERMUTATION_METHOD", "none"))
        if permutation_method not in {"none", "fixed_candidate", "full_nested"}:
            raise ValueError(f"Unknown permutation method: {permutation_method}")
        if permutation_method == "fixed_candidate":
            permutation_pvalues = run_fixed_candidate_permutations(
                source, combined, final_k, control
            )
            print(f"Fixed-candidate permutation p-values: {FIXED_PERMUTATION_PVALUES_CSV}")
        elif permutation_method == "full_nested":
            permutation_pvalues = run_full_nested_permutations(real_final, final_k, control)
            print(f"Full nested Top-K permutation p-values: {PERMUTATION_PVALUES_CSV}")

        lines = [
            "MAPPER Top-K outer validation",
            f"Source Stage 6 run: {source}",
            f"Prespecified Top K: {final_k}",
            f"Outer prediction rows: {len(predictions):,}",
            f"Ordinal scheme: {cfg.ordinal_scheme_name()}",
            f"Ordinal classes: {' | '.join(cfg.ordinal_labels())}",
            "",
            "Primary final result: arithmetic mean of the pooled outer-test primary metrics for inner ranks 1 through K.",
            "Secondary result: metric calculated after averaging the K candidate predictions for each subject.",
            "Candidates were ranked only by inner-CV performance. Outer-test results were never used to reorder or select them.",
        ]
        (OUT_DIR / "README_topk_results.txt").write_text("\n".join(lines), encoding="utf-8")

        permutation_results_path = (
            FIXED_PERMUTATION_RESULTS_CSV
            if permutation_method == "fixed_candidate"
            else PERMUTATION_RESULTS_CSV
        )
        completed_permutations = 0
        if permutation_method != "none" and permutation_results_path.exists():
            permutation_frame = pd.read_csv(permutation_results_path)
            completed_permutations = int(permutation_frame["permutation"].nunique())
        requested_permutations = int(cfg.N_PERMUTATIONS) if permutation_method != "none" else 0
        if completed_permutations != requested_permutations:
            raise RuntimeError(
                f"Permutation run is incomplete: {completed_permutations}/{requested_permutations}."
            )
        if requested_permutations:
            expected_null = (
                combined[pd.to_numeric(combined["top_k"], errors="coerce").isin(permutation_k_values(final_k))]
                if permutation_method == "fixed_candidate"
                else real_final
            )[["task", "scale", "top_k"]].drop_duplicates()
            null_counts = permutation_frame.groupby(["task", "scale", "top_k"])["permutation"].nunique()
            incomplete_null_cells = []
            for task, scale, top_k in expected_null.itertuples(index=False):
                count = int(null_counts.get((task, scale, top_k), 0))
                if count != requested_permutations:
                    incomplete_null_cells.append((task, scale, int(top_k), count))
            if incomplete_null_cells:
                raise RuntimeError(
                    "Permutation output is incomplete for one or more analysis cells. "
                    f"First incomplete cell: {incomplete_null_cells[0]}"
                )

        requested_bootstraps = int(cfg.N_BOOTSTRAPS) if bool(getattr(cfg, "RUN_BOOTSTRAP", True)) else 0
        completed_bootstraps = 0
        bootstrap_records = read_jsonl(BOOTSTRAP_JSONL) if requested_bootstraps else []
        expected_bootstrap_cells = {
            (str(task), str(scale))
            for task, scale in predictions[["task", "scale"]].drop_duplicates().itertuples(index=False)
        }
        if requested_bootstraps and bootstrap_records:
            bootstrap_frame = pd.DataFrame(bootstrap_records)
            bootstrap_counts = bootstrap_frame.groupby(["task", "scale"])["bootstrap"].nunique()
            counts_by_cell = {
                cell: int(bootstrap_counts.get(cell, 0))
                for cell in expected_bootstrap_cells
            }
            completed_bootstraps = min(counts_by_cell.values()) if counts_by_cell else 0
            incomplete_bootstrap_cells = [
                (*cell, count)
                for cell, count in sorted(counts_by_cell.items())
                if count != requested_bootstraps
            ]
            if incomplete_bootstrap_cells:
                raise RuntimeError(
                    "Bootstrap output is incomplete for one or more analysis cells. "
                    f"First incomplete cell: {incomplete_bootstrap_cells[0]}"
                )
        elif requested_bootstraps:
            raise RuntimeError("Bootstrap output is empty.")

        write_manifest(OUT_DIR, {
            "script": Path(__file__).name,
            "stage": 8,
            "source_nested_run": str(source),
            "mcmi_csv": str(cfg.MCMI_CSV),
            "mcmi_sha256": file_sha256(cfg.MCMI_CSV),
            "top_k_requested": int(final_k),
            "ordinal_scheme": cfg.ordinal_scheme_name(),
            "ordinal_cutoffs": list(cfg.ordinal_cutoffs()),
            "ordinal_labels": cfg.ordinal_labels(),
            "ordinal_n_classes": cfg.ordinal_n_classes(),
            "n_outer_prediction_rows": int(len(predictions)),
            "n_rank_metric_rows": int(len(rank_metrics)),
            "n_final_topk_rows": int(len(real_final)),
            "primary_final_statistic": "mean of pooled outer-test primary metrics for inner ranks 1..K",
            "secondary_final_statistic": "prediction-level Top-K ensemble primary metric",
            "outer_test_used_for_candidate_ranking": False,
            "bootstrap_enabled": bool(getattr(cfg, "RUN_BOOTSTRAP", True)),
            "n_bootstraps_requested": requested_bootstraps,
            "n_bootstraps_completed_per_cell": completed_bootstraps,
            "n_bootstrap_analysis_cells": int(len(expected_bootstrap_cells)) if requested_bootstraps else 0,
            "permutation_method": permutation_method,
            "n_permutations_requested": requested_permutations,
            "n_permutations_completed": completed_permutations,
            "permutation_k_values": permutation_k_values(final_k) if permutation_method == "fixed_candidate" else [int(final_k)] if permutation_method == "full_nested" else [],
            "permutation_candidate_selection_frozen": permutation_method == "fixed_candidate",
            "permutation_reruns_stage6": permutation_method == "full_nested",
            "permute_mcmi_rows_together": bool(cfg.PERMUTE_MCMI_ROWS_TOGETHER) if permutation_method != "none" else None,
            "settings_hash": cfg.settings_hash_for(8),
        })
        (OUT_DIR / "INFERENCE_COMPLETE.flag").write_text("complete", encoding="utf-8")
        print(f"Top-K outer validation and inference finished: {OUT_DIR}")
    except GracefulStop as stop:
        print(f"Graceful stop: {stop}")


if __name__ == "__main__":
    main()
