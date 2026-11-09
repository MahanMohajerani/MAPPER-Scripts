"""Create and lock the five outer folds used throughout MAPPER."""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from V3_config_loader import cfg
from V3_runtime import (
    GracefulStop,
    RunControl,
    env_bool,
    file_sha256,
    make_stage_run_dir,
    manifest_complete,
    run_dirs,
    stable_hash,
    write_manifest,
)


RESUME = env_bool("NEUROPSYCHE_RESUME", True)
OUT_DIR = make_stage_run_dir(cfg.REPORTS_DIR, Path(__file__).name, resume=RESUME)
FOLDS_CSV = OUT_DIR / "fixed_outer_folds.csv"
BALANCE_CSV = OUT_DIR / "fixed_outer_fold_balance.csv"
CANDIDATES_CSV = OUT_DIR / "fold_assignment_candidate_scores.csv"


def latest_cohort_file() -> Path:
    raw = cfg.runtime_input("qc_cohort")
    path = Path(raw) if raw else None
    if path is None or not path.exists():
        raise FileNotFoundError("No Stage 3 QC cohort file was found.")
    return path


def latest_feature_preflight() -> Path | None:
    raw = cfg.runtime_input("feature_preflight")
    path = Path(raw) if raw else cfg.REPORTS_DIR / "feature_cache_preflight_V3.csv"
    return path if path.exists() else None


def normalize_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def load_analysis_data() -> tuple[pd.DataFrame, Path, Path | None]:
    cohort_path = latest_cohort_file()
    cohort = pd.read_csv(cohort_path, dtype={"subject_id": str})
    include_col = "final_include" if "final_include" in cohort.columns else str(getattr(cfg, "FEATURE_INCLUSION_COLUMN", "minimum_include"))
    if include_col not in cohort.columns:
        raise ValueError(f"Cohort file lacks inclusion column: {include_col}")
    cohort = cohort[cohort[include_col].apply(normalize_bool)].copy()
    cohort["subject_id"] = cohort["subject_id"].astype(str).str.strip()

    preflight_path = latest_feature_preflight()
    if preflight_path is not None:
        preflight = pd.read_csv(preflight_path, dtype={"subject_id": str})
        if "complete_feature_cache" in preflight.columns:
            complete = set(preflight.loc[preflight["complete_feature_cache"].apply(normalize_bool), "subject_id"].astype(str))
            cohort = cohort[cohort["subject_id"].isin(complete)].copy()

    mcmi = pd.read_csv(cfg.MCMI_CSV, dtype={cfg.SUBJECT_ID_COLUMN: str})
    mcmi[cfg.SUBJECT_ID_COLUMN] = mcmi[cfg.SUBJECT_ID_COLUMN].astype(str).str.strip()
    scales = [s for s in cfg.MCMI_SCALE_ORDER if s in mcmi.columns]
    if len(scales) != len(cfg.MCMI_SCALE_ORDER):
        missing = sorted(set(cfg.MCMI_SCALE_ORDER).difference(scales))
        raise ValueError(f"Missing MCMI scale columns: {missing}")
    data = cohort[["subject_id"]].merge(
        mcmi[[cfg.SUBJECT_ID_COLUMN] + scales],
        left_on="subject_id",
        right_on=cfg.SUBJECT_ID_COLUMN,
        how="inner",
        validate="one_to_one",
    )
    if cfg.SUBJECT_ID_COLUMN != "subject_id" and cfg.SUBJECT_ID_COLUMN in data.columns:
        data = data.drop(columns=[cfg.SUBJECT_ID_COLUMN])
    data = data.dropna(subset=scales, how="all").reset_index(drop=True)
    if len(data) < int(cfg.OUTER_N_FOLDS):
        raise ValueError("Too few included subjects for five outer folds.")
    return data, cohort_path, preflight_path


def make_outcome_matrix(data: pd.DataFrame) -> tuple[np.ndarray, list[str], np.ndarray, list[str]]:
    """Return categorical multi-hot labels plus continuous score matrix."""
    categorical: list[np.ndarray] = []
    names: list[str] = []
    continuous: list[np.ndarray] = []
    continuous_names: list[str] = []

    for scale in cfg.MCMI_SCALE_ORDER:
        score = pd.to_numeric(data[scale], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(score)
        score_fill = score.copy()
        if not finite.all():
            score_fill[~finite] = np.nanmedian(score[finite]) if finite.any() else 0.0
        if bool(getattr(cfg, "BALANCE_CONTINUOUS_MOMENTS", True)):
            continuous.append(score_fill)
            continuous_names.append(scale)

        br60 = (score_fill >= float(cfg.BRACKET_CUTOFF_1)).astype(int)
        br75 = (score_fill >= float(cfg.BRACKET_CUTOFF_2)).astype(int)
        if bool(getattr(cfg, "BALANCE_BR60", True)):
            categorical.append(br60)
            names.append(f"{scale}__BR60")
        if bool(getattr(cfg, "BALANCE_BR75", True)):
            categorical.append(br75)
            names.append(f"{scale}__BR75")

        bracket = np.select(
            [
                score_fill < float(cfg.BRACKET_CUTOFF_1),
                (score_fill >= float(cfg.BRACKET_CUTOFF_1)) & (score_fill < float(cfg.BRACKET_CUTOFF_2)),
                (score_fill >= float(cfg.BRACKET_CUTOFF_2)) & (score_fill < float(cfg.BRACKET_CUTOFF_3)),
                score_fill >= float(cfg.BRACKET_CUTOFF_3),
            ],
            [0, 1, 2, 3],
            default=0,
        )
        if bool(getattr(cfg, "BALANCE_ORDINAL", True)):
            for cls in range(4):
                categorical.append((bracket == cls).astype(int))
                names.append(f"{scale}__ordinal_{cls}")

        if bool(getattr(cfg, "BALANCE_CONTINUOUS_MOMENTS", True)):
            try:
                q = pd.qcut(score_fill, q=min(5, len(np.unique(score_fill))), labels=False, duplicates="drop")
                q = np.asarray(q, dtype=int)
                for cls in np.unique(q):
                    categorical.append((q == cls).astype(int))
                    names.append(f"{scale}__continuous_q{int(cls)}")
            except Exception:
                pass

    Y = np.vstack(categorical).T.astype(float) if categorical else np.empty((len(data), 0))
    C = np.vstack(continuous).T.astype(float) if continuous else np.empty((len(data), 0))
    return Y, names, C, continuous_names


def assignment_score(folds: np.ndarray, Y: np.ndarray, C: np.ndarray, n_folds: int) -> tuple[float, dict[str, float]]:
    n = len(folds)
    sizes = np.array([(folds == f).sum() for f in range(n_folds)], dtype=float)
    size_target = n / n_folds
    size_penalty = float(np.mean(((sizes - size_target) / max(1.0, size_target)) ** 2))

    categorical_penalty = 0.0
    if Y.shape[1]:
        totals = Y.sum(axis=0)
        target = totals / n_folds
        cat_errors = []
        for f in range(n_folds):
            counts = Y[folds == f].sum(axis=0)
            denom = np.maximum(1.0, target)
            cat_errors.append(((counts - target) / denom) ** 2)
        categorical_penalty = float(np.mean(np.vstack(cat_errors)))

    continuous_penalty = 0.0
    if C.shape[1]:
        global_mean = np.nanmean(C, axis=0)
        global_sd = np.nanstd(C, axis=0)
        global_sd[global_sd < 1e-9] = 1.0
        mean_errors = []
        sd_errors = []
        for f in range(n_folds):
            fold_c = C[folds == f]
            mean_errors.append(((np.nanmean(fold_c, axis=0) - global_mean) / global_sd) ** 2)
            sd_errors.append(((np.nanstd(fold_c, axis=0) - global_sd) / global_sd) ** 2)
        continuous_penalty = float(np.mean(np.vstack(mean_errors)) + 0.25 * np.mean(np.vstack(sd_errors)))
    weight = float(getattr(cfg, "CONTINUOUS_BALANCE_WEIGHT", 0.25))
    total_score = categorical_penalty + weight * continuous_penalty + 10.0 * size_penalty
    return total_score, {
        "total_score": total_score,
        "categorical_penalty": categorical_penalty,
        "continuous_penalty": continuous_penalty,
        "size_penalty": size_penalty,
    }


def greedy_assignment(Y: np.ndarray, C: np.ndarray, n_folds: int, seed: int) -> np.ndarray:
    """Rarity-first greedy assignment with equal fold capacity."""
    rng = np.random.default_rng(seed)
    n = Y.shape[0]
    capacities = np.array([n // n_folds + (1 if f < n % n_folds else 0) for f in range(n_folds)], dtype=int)
    totals = np.maximum(1.0, Y.sum(axis=0))
    rarity = (Y / totals).sum(axis=1)
    jitter = rng.uniform(0, 1e-6, size=n)
    order = np.argsort(-(rarity + jitter))
    folds = np.full(n, -1, dtype=int)

    for idx in order:
        candidates = [f for f in range(n_folds) if (folds == f).sum() < capacities[f]]
        rng.shuffle(candidates)
        best_fold = candidates[0]
        best_score = math.inf
        for f in candidates:
            trial = folds.copy()
            trial[idx] = f
            # For unassigned rows, distribute temporary dummy folds only for scoring.
            unassigned = np.where(trial < 0)[0]
            if len(unassigned):
                temp = trial.copy()
                remaining_slots = []
                for ff in range(n_folds):
                    remaining_slots.extend([ff] * max(0, capacities[ff] - int((temp == ff).sum())))
                rng.shuffle(remaining_slots)
                temp[unassigned] = np.asarray(remaining_slots[:len(unassigned)], dtype=int)
            else:
                temp = trial
            score, _ = assignment_score(temp, Y, C, n_folds)
            if score < best_score:
                best_score = score
                best_fold = f
        folds[idx] = best_fold
    return folds


def improve_by_swaps(folds: np.ndarray, Y: np.ndarray, C: np.ndarray, n_folds: int, seed: int, attempts: int = 1500) -> np.ndarray:
    rng = np.random.default_rng(seed)
    current = folds.copy()
    current_score, _ = assignment_score(current, Y, C, n_folds)
    for _ in range(attempts):
        a, b = rng.choice(len(current), size=2, replace=False)
        if current[a] == current[b]:
            continue
        trial = current.copy()
        trial[a], trial[b] = trial[b], trial[a]
        score, _ = assignment_score(trial, Y, C, n_folds)
        if score < current_score:
            current = trial
            current_score = score
    return current


def build_balance_report(data: pd.DataFrame, folds: np.ndarray) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    cutoffs = [
        float(cfg.BRACKET_CUTOFF_1),
        float(cfg.BRACKET_CUTOFF_2),
        float(cfg.BRACKET_CUTOFF_3),
    ]
    for fold in range(int(cfg.OUTER_N_FOLDS)):
        sub = data.iloc[np.where(folds == fold)[0]]
        for scale in cfg.MCMI_SCALE_ORDER:
            score = pd.to_numeric(sub[scale], errors="coerce")
            brackets = pd.cut(
                score,
                bins=[-np.inf, *cutoffs, np.inf],
                right=False,
                labels=[0, 1, 2, 3],
            )
            row = {
                "outer_fold": fold + 1,
                "scale": scale,
                "n_subjects": int(len(sub)),
                "mean": float(score.mean()),
                "sd": float(score.std(ddof=0)),
                "cutoff_1": cutoffs[0],
                "cutoff_2": cutoffs[1],
                "cutoff_3": cutoffs[2],
                "br60_positive": int((score >= cutoffs[0]).sum()),
                "br75_positive": int((score >= cutoffs[1]).sum()),
            }
            for cls in range(4):
                row[f"ordinal_{cls}"] = int((brackets == cls).sum())
            rows.append(row)
    return pd.DataFrame(rows)


def fold_design_hash() -> str:
    return stable_hash({
        "n_folds": int(cfg.OUTER_N_FOLDS),
        "seed": int(cfg.OUTER_FOLD_SEED),
        "candidate_assignments": int(cfg.OUTER_FOLD_CANDIDATE_ASSIGNMENTS),
        "balance_br60": bool(cfg.BALANCE_BR60),
        "balance_br75": bool(cfg.BALANCE_BR75),
        "balance_ordinal": bool(cfg.BALANCE_ORDINAL),
        "balance_continuous": bool(cfg.BALANCE_CONTINUOUS_MOMENTS),
        "continuous_weight": float(cfg.CONTINUOUS_BALANCE_WEIGHT),
        "scales": list(cfg.MCMI_SCALE_ORDER),
        "ordinal_cutoffs": [
            float(cfg.BRACKET_CUTOFF_1),
            float(cfg.BRACKET_CUTOFF_2),
            float(cfg.BRACKET_CUTOFF_3),
        ],
    })


def validate_assignment(
    assignment: pd.DataFrame,
    subject_ids: list[str],
    n_folds: int,
) -> str:
    required = {"subject_id", "outer_fold"}
    if not required.issubset(assignment.columns):
        raise ValueError("Fold assignment is missing subject_id or outer_fold.")
    normalized = assignment.copy()
    normalized["subject_id"] = normalized["subject_id"].astype(str).str.strip()
    normalized["outer_fold"] = pd.to_numeric(
        normalized["outer_fold"], errors="raise"
    ).astype(int)
    if normalized["subject_id"].duplicated().any():
        raise ValueError("Fold assignment contains duplicate subject IDs.")
    if set(normalized["subject_id"]) != set(map(str, subject_ids)):
        raise ValueError("Fold assignment does not match the current analysis cohort.")
    expected_folds = set(range(1, int(n_folds) + 1))
    if set(normalized["outer_fold"]) != expected_folds:
        raise ValueError("Fold assignment does not contain every expected outer fold.")
    calculated = stable_hash(
        normalized[["subject_id", "outer_fold"]]
        .sort_values("subject_id")
        .to_dict("records")
    )
    if "fold_assignment_hash" in normalized.columns:
        recorded = {
            str(value).strip()
            for value in normalized["fold_assignment_hash"].dropna()
            if str(value).strip()
        }
        if recorded and recorded != {calculated}:
            raise ValueError("Fold assignment hash does not match its rows.")
    return calculated


def find_locked_assignment(
    cohort_hash: str,
    design_hash: str,
    subject_ids: list[str],
) -> Path | None:
    for run in run_dirs(cfg.REPORTS_DIR, Path(__file__).name):
        if run.resolve() == OUT_DIR.resolve() or not manifest_complete(run):
            continue
        manifest = run / "manifest.json"
        fold_file = run / "fixed_outer_folds.csv"
        balance_file = run / "fixed_outer_fold_balance.csv"
        candidates_file = run / "fold_assignment_candidate_scores.csv"
        if not all(path.exists() for path in [manifest, fold_file, balance_file, candidates_file]):
            continue
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            assignment = pd.read_csv(fold_file, dtype={"subject_id": str})
            assignment_hash = validate_assignment(
                assignment,
                subject_ids,
                int(cfg.OUTER_N_FOLDS),
            )
            if (
                payload.get("cohort_hash") == cohort_hash
                and payload.get("fold_design_hash") == design_hash
                and int(payload.get("n_folds", 0)) == int(cfg.OUTER_N_FOLDS)
                and payload.get("fold_assignment_hash") == assignment_hash
            ):
                return fold_file
        except Exception:
            continue
    return None


def main() -> None:
    cfg.apply_persisted_settings()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    control = RunControl.from_environment(OUT_DIR / "_control")
    data, cohort_path, preflight_path = load_analysis_data()
    subject_ids = data["subject_id"].astype(str).tolist()
    cohort_hash = stable_hash(sorted(subject_ids))
    design_hash = fold_design_hash()

    if bool(getattr(cfg, "LOCK_OUTER_FOLDS", True)):
        locked = find_locked_assignment(cohort_hash, design_hash, subject_ids)
        if locked is not None and locked.resolve() != FOLDS_CSV.resolve():
            shutil.copy2(locked, FOLDS_CSV)
            source_balance = locked.parent / "fixed_outer_fold_balance.csv"
            source_candidates = locked.parent / "fold_assignment_candidate_scores.csv"
            shutil.copy2(source_balance, BALANCE_CSV)
            shutil.copy2(source_candidates, CANDIDATES_CSV)
            source_manifest = json.loads((locked.parent / "manifest.json").read_text(encoding="utf-8"))
            write_manifest(OUT_DIR, {
                "script": Path(__file__).name,
                "stage": 5,
                "reused_locked_assignment": str(locked),
                "cohort_file": str(cohort_path),
                "feature_preflight": str(preflight_path) if preflight_path else None,
                "mcmi_csv": str(cfg.MCMI_CSV),
                "cohort_hash": cohort_hash,
                "fold_design_hash": design_hash,
                "fold_assignment_hash": source_manifest.get("fold_assignment_hash"),
                "n_subjects": len(subject_ids),
                "n_folds": int(cfg.OUTER_N_FOLDS),
                "candidate_assignments_tested": source_manifest.get("candidate_assignments_tested"),
                "mcmi_sha256": file_sha256(cfg.MCMI_CSV),
                "settings_hash": cfg.settings_hash_for(5),
            })
            (OUT_DIR / "FOLDS_COMPLETE.flag").write_text("complete", encoding="utf-8")
            print(f"Reused locked outer-fold assignment: {locked}")
            return

    resume_files = [FOLDS_CSV, BALANCE_CSV, CANDIDATES_CSV]
    if RESUME and all(path.exists() for path in resume_files):
        try:
            assignment = pd.read_csv(FOLDS_CSV, dtype={"subject_id": str})
            assignment_hash = validate_assignment(
                assignment,
                subject_ids,
                int(cfg.OUTER_N_FOLDS),
            )
            candidate_count = int(len(pd.read_csv(CANDIDATES_CSV)))
            if candidate_count != int(cfg.OUTER_FOLD_CANDIDATE_ASSIGNMENTS):
                raise ValueError("saved candidate search is incomplete")
        except (OSError, TypeError, ValueError) as error:
            print(f"Existing fold outputs are incomplete; rebuilding them ({error}).")
        else:
            write_manifest(OUT_DIR, {
                "script": Path(__file__).name,
                "stage": 5,
                "cohort_file": str(cohort_path),
                "feature_preflight": str(preflight_path) if preflight_path else None,
                "mcmi_csv": str(cfg.MCMI_CSV),
                "cohort_hash": cohort_hash,
                "fold_design_hash": design_hash,
                "fold_assignment_hash": assignment_hash,
                "n_subjects": int(len(assignment)),
                "n_folds": int(cfg.OUTER_N_FOLDS),
                "candidate_assignments_tested": candidate_count,
                "mcmi_sha256": file_sha256(cfg.MCMI_CSV),
                "settings_hash": cfg.settings_hash_for(5),
            })
            (OUT_DIR / "FOLDS_COMPLETE.flag").write_text("complete", encoding="utf-8")
            print(f"Resumed fixed outer folds: {FOLDS_CSV}")
            return

    Y, label_names, C, continuous_names = make_outcome_matrix(data)
    n_candidates = int(getattr(cfg, "OUTER_FOLD_CANDIDATE_ASSIGNMENTS", 250))
    base_seed = int(getattr(cfg, "OUTER_FOLD_SEED", 27))
    n_folds = int(getattr(cfg, "OUTER_N_FOLDS", 5))
    best_folds = None
    best_score = math.inf
    candidate_rows = []

    try:
        for candidate in range(n_candidates):
            control.checkpoint(f"before fold candidate {candidate + 1}/{n_candidates}")
            seed = base_seed + candidate * 1009
            folds = greedy_assignment(Y, C, n_folds, seed)
            folds = improve_by_swaps(folds, Y, C, n_folds, seed + 17)
            score, parts = assignment_score(folds, Y, C, n_folds)
            candidate_rows.append({"candidate": candidate + 1, "seed": seed, **parts})
            if score < best_score:
                best_score = score
                best_folds = folds.copy()
            if (candidate + 1) % 10 == 0:
                pd.DataFrame(candidate_rows).to_csv(CANDIDATES_CSV, index=False)
                print(f"Candidate {candidate + 1}/{n_candidates}; best balance score={best_score:.6g}", flush=True)
    except GracefulStop as stop:
        pd.DataFrame(candidate_rows).to_csv(CANDIDATES_CSV, index=False)
        print(f"Graceful stop: {stop}")
        return

    if best_folds is None:
        raise RuntimeError("Could not create an outer-fold assignment.")

    assignment = pd.DataFrame({"subject_id": subject_ids, "outer_fold": best_folds + 1})
    assignment["cohort_hash"] = cohort_hash
    assignment_hash = validate_assignment(assignment, subject_ids, n_folds)
    assignment["fold_assignment_hash"] = assignment_hash
    assignment.to_csv(FOLDS_CSV, index=False)
    balance = build_balance_report(data, best_folds)
    balance.to_csv(BALANCE_CSV, index=False)
    pd.DataFrame(candidate_rows).to_csv(CANDIDATES_CSV, index=False)

    write_manifest(OUT_DIR, {
        "script": Path(__file__).name,
        "stage": 5,
        "cohort_file": str(cohort_path),
        "feature_preflight": str(preflight_path) if preflight_path else None,
        "mcmi_csv": str(cfg.MCMI_CSV),
        "cohort_hash": cohort_hash,
        "fold_design_hash": design_hash,
        "fold_assignment_hash": assignment_hash,
        "n_subjects": len(subject_ids),
        "n_folds": n_folds,
        "candidate_assignments_tested": n_candidates,
        "best_balance_score": best_score,
        "categorical_labels_balanced": label_names,
        "continuous_scales_balanced": continuous_names,
        "selection_used_eeg_or_model_results": False,
        "mcmi_sha256": file_sha256(cfg.MCMI_CSV),
        "settings_hash": cfg.settings_hash_for(5),
    })
    (OUT_DIR / "FOLDS_COMPLETE.flag").write_text("complete", encoding="utf-8")
    print(f"Fixed outer folds saved to: {FOLDS_CSV}")
    print(f"Balance report saved to: {BALANCE_CSV}")
    print(f"Fold sizes: {assignment['outer_fold'].value_counts().sort_index().to_dict()}")


if __name__ == "__main__":
    main()
