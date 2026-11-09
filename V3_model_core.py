"""Shared feature-assembly and model utilities for MAPPER."""

from __future__ import annotations

import time
import datetime
import hashlib
import itertools
import json
import math
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from joblib import Parallel, delayed, parallel_backend
from threadpoolctl import threadpool_limits

import numpy as np
import pandas as pd

from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    cohen_kappa_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from scipy.stats import spearmanr
except Exception:
    spearmanr = None

from V3_config_loader import cfg


INCLUSION_COLUMN = str(
    getattr(
        cfg,
        "CLASSIFICATION_INCLUSION_COLUMN",
        getattr(cfg, "FEATURE_INCLUSION_COLUMN", "minimum_include"),
    )
).strip()

# Main Monte Carlo L2SO setting.
N_MONTE_CARLO_L2SO_SPLITS = int(getattr(cfg, "L2SO_REPEATS", 10))

USE_INNER_CV_TUNING = bool(getattr(cfg, "USE_INNER_CV_TUNING", False))
INNER_CV_FOLDS = int(getattr(cfg, "INNER_CV_FOLDS", 3))

LOGISTIC_C_GRID = list(getattr(cfg, "LOGISTIC_C_GRID", [0.01, 0.1, 1.0, 10.0]))
RIDGE_ALPHA_GRID = list(getattr(cfg, "RIDGE_ALPHA_GRID", [0.1, 1.0, 10.0, 100.0]))

FIXED_LOGISTIC_C = float(getattr(cfg, "LOGISTIC_C", 1.0))
FIXED_RIDGE_ALPHA = float(getattr(cfg, "RIDGE_ALPHA", 1.0))

BINARY_TARGETS = list(getattr(cfg, "BINARY_TARGETS", ["BR75", "BR60"]))

# Binary model choice.
# Options currently supported in this script:
#   "l2_logistic"     = probability model
#   "linear_svm_svc"  = SVC(kernel="linear"), uses decision_function for AUC
BINARY_MODEL_NAME = str(
    getattr(cfg, "BINARY_MODEL_NAME", "linear_svm_svc")
).strip()

RUN_BINARY_AUC = bool(getattr(cfg, "RUN_BINARY_AUC", True))
RUN_FOUR_BRACKETS = bool(getattr(cfg, "RUN_FOUR_BRACKETS", True))
RUN_CONTINUOUS_RIDGE = bool(getattr(cfg, "RUN_CONTINUOUS_RIDGE", True))

RUN_ALL_255_FAMILY_COMBINATIONS = bool(
    getattr(cfg, "RUN_ALL_255_FAMILY_COMBINATIONS", True)
)

EPOCH_DURATION_MODE = str(
    getattr(cfg, "EPOCH_DURATION_MODE", "all")
).strip().lower()

if EPOCH_DURATION_MODE not in ["reference", "all"]:
    raise ValueError("EPOCH_DURATION_MODE must be 'reference' or 'all'.")

SAVE_SPLIT_LEVEL_PREDICTIONS = bool(
    getattr(cfg, "SAVE_SPLIT_LEVEL_PREDICTIONS", True)
)

SAVE_MODEL_MATRICES = bool(
    getattr(cfg, "SAVE_MODEL_MATRICES", True)
)

RANDOM_SEED = int(getattr(cfg, "RANDOM_SEED", 42))

RUN_TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M")
CLASSIFICATION_ROOT = cfg.REPORTS_DIR / "classification_v2"

RESUME_FROM_DIR_RAW = getattr(cfg, "CLASSIFICATION_RESUME_FROM_DIR", None)
if RESUME_FROM_DIR_RAW:
    OUT_DIR = Path(RESUME_FROM_DIR_RAW)
    RUN_TIMESTAMP = OUT_DIR.name.replace("run_", "")
else:
    OUT_DIR = CLASSIFICATION_ROOT / f"run_{RUN_TIMESTAMP}"

MODEL_DATA_DIR = cfg.BASE_DIR / "_model_data" / "classification_v2" / OUT_DIR.name
STOP_REQUEST_FILE = OUT_DIR / "STOP_REQUESTED.txt"

# Parallel classifier settings.
# This parallelizes Monte Carlo L2SO splits inside each analysis.
USE_PARALLEL_CLASSIFIER = True


MAX_CLASSIFIER_WORKERS = int(getattr(cfg, "MAX_CLASSIFIER_WORKERS", 4))

# Important:
# each worker should use 1 numerical thread to avoid CPU oversubscription.
NUMERICAL_THREADS_PER_WORKER = int(getattr(cfg, "NUMERICAL_THREADS_PER_WORKER", 1))

CLASSIFIER_PARALLEL_BACKEND = str(
    getattr(cfg, "CLASSIFIER_PARALLEL_BACKEND", "loky")
).strip().lower()

if CLASSIFIER_PARALLEL_BACKEND not in ["loky", "threading"]:
    raise ValueError("CLASSIFIER_PARALLEL_BACKEND must be 'loky' or 'threading'")

# 1. CONSTANTS

FAMILY_FILE_STEMS = {
    "BAND_POWER": "band_power",
    "MICROSTATES": "microstates",
    "CONNECTIVITY": "connectivity",
    "SPECTRAL_PARAMETERIZATION": "spectral_parameterization",
    "HJORTH": "hjorth",
    "ENTROPY": "entropy",
    "ASYMMETRY": "asymmetry",
    "KURTOSIS": "kurtosis",
}

BRACKET_CUTOFF_1 = int(getattr(cfg, "BRACKET_CUTOFF_1", 60))
BRACKET_CUTOFF_2 = int(getattr(cfg, "BRACKET_CUTOFF_2", 75))
BRACKET_CUTOFF_3 = int(getattr(cfg, "BRACKET_CUTOFF_3", 85))

VARIANCE_THRESHOLD = 1e-12

def get_classifier_workers() -> int:
    if not USE_PARALLEL_CLASSIFIER:
        return 1

    total = os.cpu_count() or 1

    percent = int(getattr(cfg, "CPU_USAGE_PERCENT", 75))
    by_percent = max(1, int(total * percent / 100))

    requested = int(MAX_CLASSIFIER_WORKERS)

    return max(1, min(requested, by_percent))

def run_parallel_jobs(jobs):
    n_workers = get_classifier_workers()

    if CLASSIFIER_PARALLEL_BACKEND == "loky":
        with parallel_backend(
            "loky",
            inner_max_num_threads=NUMERICAL_THREADS_PER_WORKER,
        ):
            return Parallel(
                n_jobs=n_workers,
                max_nbytes="20M",
                mmap_mode="r",
            )(jobs)

    with threadpool_limits(limits=NUMERICAL_THREADS_PER_WORKER):
        return Parallel(
            n_jobs=n_workers,
            backend="threading",
        )(jobs)

def get_enabled_families() -> List[str]:
    if hasattr(cfg, "enabled_feature_families"):
        families = list(cfg.enabled_feature_families())
    else:
        ff = getattr(cfg, "FEATURE_FAMILIES", {})
        if isinstance(ff, dict):
            families = [k for k, v in ff.items() if bool(v)]
        else:
            families = list(ff)

    return [f for f in families if f in FAMILY_FILE_STEMS]


CONTINUOUS_FAMILIES = list(getattr(cfg, "CONTINUOUS_FAMILIES", []))
EPOCH_FAMILIES = list(getattr(cfg, "EPOCH_FAMILIES", []))
ENABLED_FAMILIES = get_enabled_families()


# 2. DATA CLASSES

@dataclass(frozen=True)
class FeatureConfig:
    config_id: int
    feature_set: str
    duration_sec: Optional[float]
    continuous_families: Tuple[str, ...]
    epoch_families: Tuple[str, ...]

    @property
    def duration_label(self) -> str:
        if self.duration_sec is None:
            return "continuous"
        return cfg.duration_label(float(self.duration_sec))

    @property
    def all_families(self) -> Tuple[str, ...]:
        return self.continuous_families + self.epoch_families


# 3. BASIC HELPERS

def ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DATA_DIR.mkdir(parents=True, exist_ok=True)


def clean_subject_id(x) -> str:
    return str(x).strip()


def bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    return s.astype(str).str.strip().str.lower().isin(["true", "1", "yes", "y"])


def safe_float(x, default=np.nan) -> float:
    try:
        return float(x)
    except Exception:
        return default


def stable_hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def array_checksum(X: np.ndarray) -> str:
    X2 = np.ascontiguousarray(X.astype(np.float64, copy=False))
    return hashlib.sha256(X2.tobytes()).hexdigest()[:16]


def join_ids(ids: Sequence[str]) -> str:
    return "|".join(map(str, ids))


def class_count_text(y: np.ndarray) -> str:
    vals, counts = np.unique(y, return_counts=True)
    return ";".join([f"{int(v)}:{int(c)}" for v, c in zip(vals, counts)])


def valid_two_class(y: np.ndarray) -> bool:
    return len(np.unique(y)) == 2


def get_mcmi_scales(mcmi_df: pd.DataFrame) -> List[str]:
    scales = []
    for col in mcmi_df.columns:
        col = str(col)
        if not col.startswith("BR_"):
            continue
        if "bracket" in col.lower():
            continue
        if col.endswith("_BR60") or col.endswith("_BR75") or col.endswith("_BR85"):
            continue
        scales.append(col)
    return scales


def feature_stem(family: str) -> str:
    return FAMILY_FILE_STEMS.get(family, family.lower())


def family_kind(family: str) -> str:
    if family in CONTINUOUS_FAMILIES:
        return "continuous"
    if family in EPOCH_FAMILIES:
        return "epoch"
    raise ValueError(f"Family is not listed as continuous or epoch: {family}")


def feature_path(subject_id: str, family: str, duration_sec: Optional[float]) -> Path:
    stem = feature_stem(family) + ".npz"

    if family in CONTINUOUS_FAMILIES:
        return cfg.subject_features_folder(subject_id) / "continuous" / stem

    if family in EPOCH_FAMILIES:
        if duration_sec is None:
            raise ValueError(f"Epoch family needs duration: {family}")
        return (
            cfg.subject_features_folder(subject_id)
            / "epochs"
            / cfg.duration_label(float(duration_sec))
            / stem
        )

    raise ValueError(f"Unknown feature family: {family}")


def make_brackets(
    scores: np.ndarray,
    cutoffs: Optional[
        Sequence[float]
    ] = None,
) -> np.ndarray:
    scores = np.asarray(
        scores,
        dtype=float,
    )

    out = np.full(
        scores.shape,
        -1,
        dtype=int,
    )

    valid = np.isfinite(
        scores
    )

    active_cutoffs = np.asarray(
        (
            list(cutoffs)
            if cutoffs is not None
            else [
                BRACKET_CUTOFF_1,
                BRACKET_CUTOFF_2,
                BRACKET_CUTOFF_3,
            ]
        ),
        dtype=float,
    )

    if (
        active_cutoffs.ndim != 1
        or len(active_cutoffs) < 1
    ):
        raise ValueError(
            "Ordinal cutoffs must be a non-empty "
            "one-dimensional sequence."
        )

    if (
        not np.all(
            np.isfinite(active_cutoffs)
        )
        or np.any(
            np.diff(active_cutoffs) <= 0
        )
    ):
        raise ValueError(
            "Ordinal cutoffs must be finite "
            "and strictly increasing."
        )

    out[valid] = np.searchsorted(
        active_cutoffs,
        scores[valid],
        side="right",
    )

    return out


def make_binary(scores: np.ndarray, target: str) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    y = np.full(scores.shape, -1, dtype=int)

    if target == "BR60":
        cutoff = BRACKET_CUTOFF_1
    elif target == "BR75":
        cutoff = BRACKET_CUTOFF_2
    elif target == "BR85":
        cutoff = BRACKET_CUTOFF_3
    else:
        raise ValueError(f"Unknown binary target: {target}")

    valid = np.isfinite(scores)
    y[valid] = (scores[valid] >= cutoff).astype(int)
    return y


# 4. FEATURE CONFIGURATIONS

def build_feature_configs() -> List[FeatureConfig]:
    enabled = [f for f in ENABLED_FAMILIES if f in CONTINUOUS_FAMILIES or f in EPOCH_FAMILIES]
    enabled = sorted(enabled)

    reference_duration = float(getattr(cfg, "REFERENCE_DURATION", 2.0))
    all_durations = [float(d) for d in getattr(cfg, "CACHE_DURATIONS", [1.0, 2.0, 3.0, 4.0, 5.0])]

    configs: List[FeatureConfig] = []
    config_id = 0

    if RUN_ALL_255_FAMILY_COMBINATIONS:
        combos = []
        for r in range(1, len(enabled) + 1):
            combos.extend(itertools.combinations(enabled, r))
    else:
        combos = []
        continuous = tuple([f for f in enabled if f in CONTINUOUS_FAMILIES])
        epoch = tuple([f for f in enabled if f in EPOCH_FAMILIES])

        if continuous:
            combos.append(continuous)
            for f in continuous:
                combos.append((f,))

        if epoch:
            combos.append(epoch)
            for f in epoch:
                combos.append((f,))

        if continuous and epoch:
            combos.append(continuous + epoch)

    for combo in combos:
        continuous_fams = tuple([f for f in combo if f in CONTINUOUS_FAMILIES])
        epoch_fams = tuple([f for f in combo if f in EPOCH_FAMILIES])

        feature_set = "+".join(combo)

        if epoch_fams:
            if EPOCH_DURATION_MODE.lower() == "reference":
                durations = [reference_duration]
            elif EPOCH_DURATION_MODE.lower() == "all":
                durations = all_durations
            else:
                raise ValueError("EPOCH_DURATION_MODE must be 'reference' or 'all'")
        else:
            durations = [None]

        for duration in durations:
            config_id += 1
            configs.append(
                FeatureConfig(
                    config_id=config_id,
                    feature_set=feature_set,
                    duration_sec=duration,
                    continuous_families=continuous_fams,
                    epoch_families=epoch_fams,
                )
            )

    return configs


# 5. FEATURE LOADING

_FEATURE_VECTOR_CACHE: Dict[Tuple[str, str, str], Tuple[np.ndarray, List[str]]] = {}


def load_feature_vector(subject_id: str, family: str, duration_sec: Optional[float]) -> Tuple[np.ndarray, List[str]]:
    duration_key = "continuous" if duration_sec is None else cfg.duration_label(float(duration_sec))
    key = (subject_id, family, duration_key)

    if key in _FEATURE_VECTOR_CACHE:
        return _FEATURE_VECTOR_CACHE[key]

    path = feature_path(subject_id, family, duration_sec)

    if not path.exists():
        raise FileNotFoundError(f"Missing feature file: {path}")

    z = np.load(path, allow_pickle=True)

    if "X_summary" not in z.files:
        raise ValueError(f"Missing X_summary in {path}")

    x = np.asarray(z["X_summary"], dtype=float).ravel()

    if not np.all(np.isfinite(x)):
        raise ValueError(f"NaN/inf found in X_summary: {path}")

    if "summary_feature_names" in z.files:
        names = [str(n) for n in np.asarray(z["summary_feature_names"]).ravel()]
    else:
        names = [f"feature_{i:04d}" for i in range(x.size)]

    if len(names) != x.size:
        names = [f"feature_{i:04d}" for i in range(x.size)]

    prefix = f"{family_kind(family)}__{duration_key}__{family}"
    names = [f"{prefix}__{name}" for name in names]

    _FEATURE_VECTOR_CACHE[key] = (x, names)
    return x, names


def assemble_X(
    subject_ids: Sequence[str],
    feature_config: FeatureConfig,
) -> Tuple[np.ndarray, List[str], List[str], List[str]]:
    rows = []
    used_subjects = []
    missing_subjects = []
    feature_names: Optional[List[str]] = None

    for sid in subject_ids:
        try:
            vectors = []
            names = []

            for fam in feature_config.continuous_families:
                x, n = load_feature_vector(sid, fam, None)
                vectors.append(x)
                names.extend(n)

            for fam in feature_config.epoch_families:
                x, n = load_feature_vector(sid, fam, feature_config.duration_sec)
                vectors.append(x)
                names.extend(n)

            if not vectors:
                raise ValueError("No feature families in this configuration")

            row = np.concatenate(vectors).astype(float)

            if feature_names is None:
                feature_names = names
            elif len(feature_names) != len(names):
                raise ValueError(f"Feature-name length mismatch for {sid}")

            rows.append(row)
            used_subjects.append(sid)

        except Exception as error:
            missing_subjects.append(f"{sid}: {error}")

    if not rows:
        raise RuntimeError(f"No usable subjects for feature config: {feature_config.feature_set}")

    X = np.vstack(rows)

    if feature_names is None:
        feature_names = [f"feature_{i:04d}" for i in range(X.shape[1])]

    if len(set(feature_names)) != len(feature_names):
        counts = {}
        unique_names = []
        for name in feature_names:
            counts[name] = counts.get(name, 0) + 1
            if counts[name] == 1:
                unique_names.append(name)
            else:
                unique_names.append(f"{name}__dup{counts[name]}")
        feature_names = unique_names

    if not np.all(np.isfinite(X)):
        raise ValueError(f"Non-finite values in X for {feature_config.feature_set}")

    return X, feature_names, used_subjects, missing_subjects


# 6. SPLITS

def make_mc_l2so_splits(
    y: np.ndarray,
    subject_ids: Sequence[str],
    n_splits: int,
    seed: int,
    require_label_diversity: bool = False,
    require_train_two_classes: bool = False,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)

    y = np.asarray(y)
    n = len(subject_ids)

    if n < 4:
        raise ValueError("Need at least 4 subjects for L2SO.")

    all_idx = np.arange(n)

    use_balanced = bool(
        getattr(cfg, "USE_COVERAGE_BALANCED_L2SO", True)
    )

    require_label_diversity = bool(
        getattr(cfg, "REQUIRE_LABEL_DIVERSE_TEST_PAIRS", False)
    ) and bool(require_label_diversity)

    avoid_duplicates = bool(
        getattr(cfg, "AVOID_DUPLICATE_TEST_PAIRS", True)
    )

    # Old pure-random fallback, kept for reproducibility if needed
    if not use_balanced:
        splits = []

        for _ in range(int(n_splits)):
            test_idx = np.sort(rng.choice(all_idx, size=2, replace=False))
            train_idx = np.setdiff1d(all_idx, test_idx, assume_unique=False)

            if require_train_two_classes and len(np.unique(y[train_idx])) < 2:
                for _attempt in range(100):
                    test_idx = np.sort(rng.choice(all_idx, size=2, replace=False))
                    train_idx = np.setdiff1d(all_idx, test_idx, assume_unique=False)

                    if len(np.unique(y[train_idx])) >= 2:
                        break

            splits.append((train_idx, test_idx))

        return splits

    # Coverage-balanced splitter
    if int(n_splits) * 2 < n:
        raise ValueError(
            f"Cannot test every subject at least once: "
            f"{n_splits} splits gives {n_splits * 2} test slots "
            f"for {n} subjects."
        )

    pair_i, pair_j = np.triu_indices(n, k=1)
    all_pairs = np.column_stack([pair_i, pair_j]).astype(int)

    test_counts = np.zeros(n, dtype=int)
    used_pairs = set()
    splits = []

    def pair_is_allowed(pair: np.ndarray) -> bool:
        a = int(pair[0])
        b = int(pair[1])

        if avoid_duplicates:
            key = (min(a, b), max(a, b))
            if key in used_pairs:
                return False

        if require_label_diversity:
            if y[a] == y[b]:
                return False

        if require_train_two_classes:
            train_mask = np.ones(n, dtype=bool)
            train_mask[[a, b]] = False

            if len(np.unique(y[train_mask])) < 2:
                return False

        return True

    for _split_id in range(int(n_splits)):
        pair_count_sum = test_counts[all_pairs[:, 0]] + test_counts[all_pairs[:, 1]]

        order_counts = np.unique(pair_count_sum)
        chosen_pair = None

        for count_value in order_counts:
            candidate_idx = np.where(pair_count_sum == count_value)[0]
            rng.shuffle(candidate_idx)

            for idx in candidate_idx:
                pair = all_pairs[idx]

                if pair_is_allowed(pair):
                    chosen_pair = pair
                    break

            if chosen_pair is not None:
                break

        # If duplicate-pair avoidance blocks everything, allow duplicates as fallback.
        if chosen_pair is None and avoid_duplicates:
            candidate_idx = np.argsort(pair_count_sum)

            for idx in candidate_idx:
                pair = all_pairs[idx]
                a = int(pair[0])
                b = int(pair[1])

                if require_label_diversity and y[a] == y[b]:
                    continue

                if require_train_two_classes:
                    train_mask = np.ones(n, dtype=bool)
                    train_mask[[a, b]] = False

                    if len(np.unique(y[train_mask])) < 2:
                        continue

                chosen_pair = pair
                break

        if chosen_pair is None:
            raise RuntimeError(
                "Could not create a valid L2SO split. "
                "Check class counts and split constraints."
            )

        test_idx = np.sort(chosen_pair.astype(int))
        train_idx = np.setdiff1d(all_idx, test_idx, assume_unique=False)

        test_counts[test_idx] += 1
        used_pairs.add((int(test_idx[0]), int(test_idx[1])))

        splits.append((train_idx, test_idx))

    return splits


# 6B. FIXED COMMON SPLIT SETS

_FIXED_SPLIT_CACHE = {}


def subject_list_checksum(subject_ids: Sequence[str]) -> str:
    return stable_hash_text(join_ids(subject_ids))


def label_checksum(y: np.ndarray) -> str:
    y = np.asarray(y)

    if np.issubdtype(y.dtype, np.floating):
        text = "|".join([f"{float(v):.8f}" for v in y.tolist()])
    else:
        text = "|".join(map(str, y.tolist()))

    return stable_hash_text(text)


def fixed_split_set_id(
    task: str,
    scale: str,
    target: str,
    subject_ids: Sequence[str],
    y: np.ndarray,
) -> str:
    raw = "__".join(
        [
            "fixed_l2so",
            f"task={task}",
            f"scale={scale}",
            f"target={target}",
            f"n_subjects={len(subject_ids)}",
            f"n_splits={N_MONTE_CARLO_L2SO_SPLITS}",
            f"seed={RANDOM_SEED}",
            f"subjects={subject_list_checksum(subject_ids)}",
            f"labels={label_checksum(y)}",
        ]
    )

    return stable_hash_text(raw)


def fixed_split_seed(
    task: str,
    scale: str,
    target: str,
    subject_ids: Sequence[str],
    y: np.ndarray,
) -> int:
    raw = "__".join(
        [
            "fixed_seed",
            f"task={task}",
            f"scale={scale}",
            f"target={target}",
            f"n_subjects={len(subject_ids)}",
            f"n_splits={N_MONTE_CARLO_L2SO_SPLITS}",
            f"base_seed={RANDOM_SEED}",
            f"subjects={subject_list_checksum(subject_ids)}",
            f"labels={label_checksum(y)}",
        ]
    )

    offset = int(stable_hash_text(raw), 16) % 100000
    return int(RANDOM_SEED + offset)


def fixed_split_path(split_set_id: str) -> Path:
    root = Path(
        getattr(
            cfg,
            "FIXED_SPLIT_DIR",
            CLASSIFICATION_ROOT / "_fixed_l2so_splits",
        )
    )

    root.mkdir(parents=True, exist_ok=True)

    return root / f"{split_set_id}.csv"


def save_fixed_splits(
    path: Path,
    split_set_id: str,
    task: str,
    scale: str,
    target: str,
    subject_ids: Sequence[str],
    y: np.ndarray,
    splits: List[Tuple[np.ndarray, np.ndarray]],
) -> None:
    rows = []

    subj_checksum = subject_list_checksum(subject_ids)
    lab_checksum = label_checksum(y)

    for split_id, (train_idx, test_idx) in enumerate(splits):
        if task == "ridge_regression":
            train_class_counts = "continuous"
            test_class_counts = "continuous"
            train_score_mean = float(np.mean(y[train_idx]))
            test_score_mean = float(np.mean(y[test_idx]))
        else:
            train_class_counts = class_count_text(y[train_idx])
            test_class_counts = class_count_text(y[test_idx])
            train_score_mean = np.nan
            test_score_mean = np.nan

        rows.append(
            {
                "split_set_id": split_set_id,
                "task": task,
                "scale": scale,
                "target": target,
                "split_id": int(split_id),
                "n_subjects": int(len(subject_ids)),
                "n_splits": int(N_MONTE_CARLO_L2SO_SPLITS),
                "random_seed": int(RANDOM_SEED),
                "subject_list_checksum": subj_checksum,
                "label_checksum": lab_checksum,
                "train_subject_ids": join_ids([subject_ids[i] for i in train_idx]),
                "test_subject_ids": join_ids([subject_ids[i] for i in test_idx]),
                "test_subject_1": subject_ids[int(test_idx[0])],
                "test_subject_2": subject_ids[int(test_idx[1])],
                "train_class_counts": train_class_counts,
                "test_class_counts": test_class_counts,
                "train_score_mean": train_score_mean,
                "test_score_mean": test_score_mean,
            }
        )

    pd.DataFrame(rows).to_csv(path, index=False)


def load_fixed_splits(
    path: Path,
    subject_ids: Sequence[str],
    expected_split_set_id: str,
    expected_label_checksum: str,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    df = pd.read_csv(path)

    required_cols = {
        "split_set_id",
        "split_id",
        "test_subject_ids",
        "subject_list_checksum",
        "label_checksum",
    }

    missing_cols = required_cols - set(df.columns)

    if missing_cols:
        raise RuntimeError(
            f"Fixed split file is missing columns: {sorted(missing_cols)}\n"
            f"File: {path}"
        )

    found_split_ids = sorted(df["split_set_id"].astype(str).unique().tolist())

    if found_split_ids != [str(expected_split_set_id)]:
        raise RuntimeError(
            f"Fixed split file split_set_id mismatch.\n"
            f"Expected: {expected_split_set_id}\n"
            f"Found: {found_split_ids}\n"
            f"File: {path}"
        )

    expected_subject_checksum = subject_list_checksum(subject_ids)
    found_subject_checksums = sorted(
        df["subject_list_checksum"].astype(str).unique().tolist()
    )

    if found_subject_checksums != [str(expected_subject_checksum)]:
        raise RuntimeError(
            f"Fixed split file subject-list checksum mismatch.\n"
            f"Expected: {expected_subject_checksum}\n"
            f"Found: {found_subject_checksums}\n"
            f"File: {path}"
        )

    found_label_checksums = sorted(
        df["label_checksum"].astype(str).unique().tolist()
    )

    if found_label_checksums != [str(expected_label_checksum)]:
        raise RuntimeError(
            f"Fixed split file label checksum mismatch.\n"
            f"Expected: {expected_label_checksum}\n"
            f"Found: {found_label_checksums}\n"
            f"File: {path}"
        )

    if len(df) != int(N_MONTE_CARLO_L2SO_SPLITS):
        raise RuntimeError(
            f"Fixed split file has wrong number of splits.\n"
            f"Expected: {N_MONTE_CARLO_L2SO_SPLITS}\n"
            f"Found: {len(df)}\n"
            f"File: {path}"
        )

    id_to_idx = {str(sid): i for i, sid in enumerate(subject_ids)}
    all_idx = np.arange(len(subject_ids))

    splits = []

    for _, row in df.sort_values("split_id").iterrows():
        test_subjects = str(row["test_subject_ids"]).split("|")

        missing = [sid for sid in test_subjects if sid not in id_to_idx]

        if missing:
            raise RuntimeError(
                f"Fixed split file contains subjects not in current subject list: "
                f"{missing[:10]}\nFile: {path}"
            )

        test_idx = np.sort(
            np.asarray([id_to_idx[str(sid)] for sid in test_subjects], dtype=int)
        )

        if len(test_idx) != 2:
            raise RuntimeError(
                f"Each L2SO split must contain exactly 2 test subjects. "
                f"Found {len(test_idx)} in file: {path}"
            )

        train_idx = np.setdiff1d(all_idx, test_idx, assume_unique=False)
        splits.append((train_idx, test_idx))

    return splits


def get_splits_for_analysis(
    task: str,
    scale: str,
    target: str,
    y: np.ndarray,
    subject_ids: Sequence[str],
    exploratory_analysis_id: str,
    require_label_diversity: bool = False,
    require_train_two_classes: bool = False,
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], Dict[str, object]]:
    split_mode = str(
        getattr(cfg, "VALIDATION_SPLIT_MODE", "exploratory")
    ).strip().lower()

    subj_checksum = subject_list_checksum(subject_ids)
    lab_checksum = label_checksum(y)

    if split_mode == "fixed_common":
        split_set_id = fixed_split_set_id(
            task=task,
            scale=scale,
            target=target,
            subject_ids=subject_ids,
            y=y,
        )

        path = fixed_split_path(split_set_id)
        path_str = str(path)

        if split_set_id in _FIXED_SPLIT_CACHE:
            splits = _FIXED_SPLIT_CACHE[split_set_id]
        else:
            if path.exists():
                splits = load_fixed_splits(
                    path=path,
                    subject_ids=subject_ids,
                    expected_split_set_id=split_set_id,
                    expected_label_checksum=lab_checksum,
                )
            else:
                seed = fixed_split_seed(
                    task=task,
                    scale=scale,
                    target=target,
                    subject_ids=subject_ids,
                    y=y,
                )

                splits = make_mc_l2so_splits(
                    y=y,
                    subject_ids=subject_ids,
                    n_splits=N_MONTE_CARLO_L2SO_SPLITS,
                    seed=seed,
                    require_label_diversity=require_label_diversity,
                    require_train_two_classes=require_train_two_classes,
                )

                save_fixed_splits(
                    path=path,
                    split_set_id=split_set_id,
                    task=task,
                    scale=scale,
                    target=target,
                    subject_ids=subject_ids,
                    y=y,
                    splits=splits,
                )

            _FIXED_SPLIT_CACHE[split_set_id] = splits

        info = {
            "split_mode": split_mode,
            "split_set_id": split_set_id,
            "fixed_split_path": path_str,
            "subject_list_checksum": subj_checksum,
            "label_checksum": lab_checksum,
        }

        return splits, info

    if split_mode == "exploratory":
        split_seed = RANDOM_SEED + int(stable_hash_text(exploratory_analysis_id), 16) % 100000

        splits = make_mc_l2so_splits(
            y=y,
            subject_ids=subject_ids,
            n_splits=N_MONTE_CARLO_L2SO_SPLITS,
            seed=split_seed,
            require_label_diversity=require_label_diversity,
            require_train_two_classes=require_train_two_classes,
        )

        info = {
            "split_mode": split_mode,
            "split_set_id": f"exploratory_{stable_hash_text(exploratory_analysis_id)}",
            "fixed_split_path": "",
            "subject_list_checksum": subj_checksum,
            "label_checksum": lab_checksum,
        }

        return splits, info

    raise ValueError(
        "VALIDATION_SPLIT_MODE must be 'fixed_common' or 'exploratory'."
    )


def add_split_info_to_result(
    result: Dict[str, object],
    split_info: Dict[str, object],
) -> Dict[str, object]:
    for key, value in split_info.items():
        result[key] = value

    return result


def add_split_info_to_split_df(
    split_df: pd.DataFrame,
    split_info: Dict[str, object],
) -> pd.DataFrame:
    out = split_df.copy()

    for key, value in split_info.items():
        out[key] = value

    return out


def enforce_common_subjects_for_fixed_splits(
    expected_subjects: Sequence[str],
    used_subjects: Sequence[str],
    feature_config: FeatureConfig,
    missing_subjects: Sequence[str],
) -> None:
    split_mode = str(
        getattr(cfg, "VALIDATION_SPLIT_MODE", "exploratory")
    ).strip().lower()

    strict = bool(
        getattr(cfg, "STRICT_COMMON_SUBJECTS_FOR_FIXED_SPLITS", True)
    )

    if split_mode != "fixed_common" or not strict:
        return

    expected = list(map(str, expected_subjects))
    used = list(map(str, used_subjects))

    if expected != used:
        raise RuntimeError(
            "Fixed common split mode requires every feature configuration "
            "to use the exact same subject list.\n\n"
            f"Feature config: cfg{feature_config.config_id:03d} | "
            f"{feature_config.feature_set} | {feature_config.duration_label}\n"
            f"Expected subjects: {len(expected)}\n"
            f"Used subjects: {len(used)}\n"
            f"First missing/failed feature messages:\n"
            + "\n".join(map(str, list(missing_subjects)[:20]))
        )

# 7. MODELS

def make_logistic_pipeline(C: float) -> Pipeline:
    return Pipeline(
        steps=[
            ("variance", VarianceThreshold(threshold=VARIANCE_THRESHOLD)),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=float(C),
                    class_weight=getattr(cfg, "CLASS_WEIGHT", "balanced"),
                    solver=getattr(cfg, "SOLVER", "lbfgs"),
                    max_iter=int(getattr(cfg, "MAX_ITER", 1000)),
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )


def make_linear_svm_svc_pipeline(C: float) -> Pipeline:
    return Pipeline(
        steps=[
            ("variance", VarianceThreshold(threshold=VARIANCE_THRESHOLD)),
            ("scaler", StandardScaler()),
            (
                "model",
                SVC(
                    kernel="linear",
                    C=float(C),
                    class_weight=getattr(cfg, "CLASS_WEIGHT", "balanced"),
                    probability=False,
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )


def make_binary_pipeline(model_name: str, C: float) -> Pipeline:
    model_name = str(model_name).strip()

    if model_name == "l2_logistic":
        return make_logistic_pipeline(C)

    if model_name == "linear_svm_svc":
        return make_linear_svm_svc_pipeline(C)

    raise ValueError(
        f"Unknown BINARY_MODEL_NAME={model_name!r}. "
        "Use 'l2_logistic' or 'linear_svm_svc'."
    )


def binary_score(model, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X)[:, 1], dtype=float).ravel()

    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(X), dtype=float).ravel()

    return np.asarray(model.predict(X), dtype=float).ravel()


def binary_probability(model, X: np.ndarray) -> Optional[np.ndarray]:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X)[:, 1], dtype=float).ravel()

    return None


def binary_score_type(model) -> str:
    if hasattr(model, "predict_proba"):
        return "probability"
    if hasattr(model, "decision_function"):
        return "decision_function"
    return "prediction"

def make_ridge_pipeline(alpha: float) -> Pipeline:
    return Pipeline(
        steps=[
            ("variance", VarianceThreshold(threshold=VARIANCE_THRESHOLD)),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=float(alpha))),
        ]
    )


class ConstantBinaryModel:
    def __init__(self, p_positive: float):
        self.p_positive = float(np.clip(p_positive, 0.0, 1.0))

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        n = X.shape[0]
        p1 = np.full(n, self.p_positive, dtype=float)
        return np.column_stack([1.0 - p1, p1])

    def decision_function(self, X):
        n = X.shape[0]
        p = float(np.clip(self.p_positive, 1e-12, 1.0 - 1e-12))
        logit = math.log(p / (1.0 - p))
        return np.full(n, logit, dtype=float)


class CumulativeOrdinalLogit:

    def __init__(
        self,
        C: float = 1.0,
        n_classes: int = 4,
    ):
        self.C = float(C)
        self.n_classes = int(n_classes)

        if self.n_classes < 2:
            raise ValueError(
                "CumulativeOrdinalLogit "
                "requires at least two classes."
            )

        self.thresholds = list(
            range(
                1,
                self.n_classes,
            )
        )

        self.models = {}


    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ):
        y = np.asarray(
            y,
            dtype=int,
        )

        if y.size == 0:
            raise ValueError(
                "Cannot fit an ordinal model "
                "with no observations."
            )

        if np.any(
            (y < 0)
            | (y >= self.n_classes)
        ):
            raise ValueError(
                "Ordinal labels must be integers "
                f"from 0 to {self.n_classes - 1}."
            )

        self.models = {}

        for threshold in self.thresholds:

            yt = (
                y >= threshold
            ).astype(int)

            if len(np.unique(yt)) < 2:

                self.models[
                    threshold
                ] = ConstantBinaryModel(
                    float(
                        np.mean(yt)
                    )
                )

            else:

                model = make_logistic_pipeline(
                    self.C
                )

                model.fit(
                    X,
                    yt,
                )

                self.models[
                    threshold
                ] = model

        return self


    def predict_cumulative_proba(
        self,
        X: np.ndarray,
    ) -> np.ndarray:

        cols = []

        for threshold in self.thresholds:

            p = self.models[
                threshold
            ].predict_proba(
                X
            )[:, 1]

            cols.append(
                p
            )

        cum = np.column_stack(
            cols
        )

        cum = np.clip(
            cum,
            0.0,
            1.0,
        )

        # Enforce:
        # P(y>=1) >= P(y>=2) >= ...
        cum = np.minimum.accumulate(
            cum,
            axis=1,
        )

        return cum


    def predict_proba(
        self,
        X: np.ndarray,
    ) -> np.ndarray:

        cum = (
            self.predict_cumulative_proba(
                X
            )
        )

        probs = np.empty(
            (
                len(X),
                self.n_classes,
            ),
            dtype=float,
        )

        probs[:, 0] = (
            1.0
            - cum[:, 0]
        )

        for class_index in range(
            1,
            self.n_classes - 1,
        ):

            probs[:, class_index] = (
                cum[:, class_index - 1]
                - cum[:, class_index]
            )

        probs[:, -1] = (
            cum[:, -1]
        )

        probs = np.clip(
            probs,
            0.0,
            1.0,
        )

        row_sums = probs.sum(
            axis=1,
            keepdims=True,
        )

        row_sums[
            row_sums == 0
        ] = 1.0

        probs = (
            probs
            / row_sums
        )

        return probs


    def predict(
        self,
        X: np.ndarray,
    ) -> np.ndarray:

        return np.argmax(
            self.predict_proba(X),
            axis=1,
        ).astype(int)


# 8. INNER CV TUNING

def inner_splits_for_classification(y: np.ndarray, seed: int):
    y = np.asarray(y)
    counts = pd.Series(y).value_counts()
    min_count = int(counts.min()) if len(counts) else 0
    n_splits = min(INNER_CV_FOLDS, min_count)

    if n_splits >= 2:
        return list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(np.zeros(len(y)), y))

    n_splits = min(INNER_CV_FOLDS, len(y))
    if n_splits < 2:
        return []

    return list(KFold(n_splits=n_splits, shuffle=True, random_state=seed).split(np.zeros(len(y))))


def inner_splits_for_regression(y: np.ndarray, seed: int):
    n_splits = min(INNER_CV_FOLDS, len(y))

    if n_splits < 2:
        return []

    return list(KFold(n_splits=n_splits, shuffle=True, random_state=seed).split(np.zeros(len(y))))


def tune_binary_C(X: np.ndarray, y: np.ndarray, seed: int) -> Tuple[float, float, str]:
    if not USE_INNER_CV_TUNING:
        return FIXED_LOGISTIC_C, np.nan, f"tuning_off_fixed_C={FIXED_LOGISTIC_C}"

    if not valid_two_class(y):
        return 1.0, np.nan, "one_class_training"

    splits = inner_splits_for_classification(y, seed)

    if len(splits) < 2:
        return 1.0, np.nan, "too_few_inner_splits"

    best_C = 1.0
    best_score = -np.inf
    details = []

    for C in LOGISTIC_C_GRID:
        scores = []

        for tr, va in splits:
            if not valid_two_class(y[tr]):
                continue

            model = make_binary_pipeline(BINARY_MODEL_NAME, C)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                model.fit(X[tr], y[tr])

            score_values = binary_score(model, X[va])
            prob_values = binary_probability(model, X[va])

            if valid_two_class(y[va]):
                score = roc_auc_score(y[va], score_values)
            else:
                if prob_values is not None:
                    pred = (prob_values >= 0.5).astype(int)
                else:
                    pred = (score_values >= 0.0).astype(int)
                score = balanced_accuracy_score(y[va], pred)

            scores.append(float(score))

        mean_score = float(np.mean(scores)) if scores else -np.inf
        details.append(f"C={C}:score={mean_score:.6f}")

        if mean_score > best_score:
            best_score = mean_score
            best_C = float(C)

    return best_C, best_score, ";".join(details)


def ordinal_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float))))


def tune_ordinal_C(X: np.ndarray, y: np.ndarray, seed: int) -> Tuple[float, float, str]:
    if not USE_INNER_CV_TUNING:
        return 1.0, np.nan, "tuning_off"

    if len(np.unique(y)) < 2:
        return 1.0, np.nan, "one_class_training"

    splits = inner_splits_for_classification(y, seed)

    if len(splits) < 2:
        return 1.0, np.nan, "too_few_inner_splits"

    best_C = 1.0
    best_score = np.inf
    details = []

    for C in LOGISTIC_C_GRID:
        scores = []

        for tr, va in splits:
            if len(np.unique(y[tr])) < 2:
                continue

            model = CumulativeOrdinalLogit(C=C)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                model.fit(X[tr], y[tr])

            pred = model.predict(X[va])
            scores.append(ordinal_mae(y[va], pred))

        mean_score = float(np.mean(scores)) if scores else np.inf
        details.append(f"C={C}:mae={mean_score:.6f}")

        if mean_score < best_score:
            best_score = mean_score
            best_C = float(C)

    return best_C, best_score, ";".join(details)


def tune_ridge_alpha(X: np.ndarray, y: np.ndarray, seed: int) -> Tuple[float, float, str]:
    if not USE_INNER_CV_TUNING:
        return FIXED_RIDGE_ALPHA, np.nan, f"tuning_off_fixed_alpha={FIXED_RIDGE_ALPHA}"

    splits = inner_splits_for_regression(y, seed)

    if len(splits) < 2:
        return 1.0, np.nan, "too_few_inner_splits"

    best_alpha = 1.0
    best_score = np.inf
    details = []

    for alpha in RIDGE_ALPHA_GRID:
        scores = []

        for tr, va in splits:
            model = make_ridge_pipeline(alpha)
            model.fit(X[tr], y[tr])
            pred = model.predict(X[va])
            scores.append(mean_absolute_error(y[va], pred))

        mean_score = float(np.mean(scores)) if scores else np.inf
        details.append(f"alpha={alpha}:mae={mean_score:.6f}")

        if mean_score < best_score:
            best_score = mean_score
            best_alpha = float(alpha)

    return best_alpha, best_score, ";".join(details)


# 9. METRICS

def safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    try:
        if len(np.unique(y_true)) < 2:
            return np.nan
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return np.nan


def safe_brier(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    try:
        return float(brier_score_loss(y_true, y_prob))
    except Exception:
        return np.nan


def safe_balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    try:
        return float(balanced_accuracy_score(y_true, y_pred))
    except Exception:
        return np.nan


def safe_macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    try:
        return float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    except Exception:
        return np.nan


def safe_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    try:
        return float(accuracy_score(y_true, y_pred))
    except Exception:
        return np.nan


def safe_qwk(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    try:
        return float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))
    except Exception:
        return np.nan


def safe_pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    try:
        if len(y_true) < 3:
            return np.nan
        if np.std(y_true) == 0 or np.std(y_pred) == 0:
            return np.nan
        return float(np.corrcoef(y_true, y_pred)[0, 1])
    except Exception:
        return np.nan


def safe_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    try:
        if spearmanr is None:
            return np.nan
        if len(y_true) < 3:
            return np.nan
        return float(spearmanr(y_true, y_pred).correlation)
    except Exception:
        return np.nan


def regression_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    try:
        return float(math.sqrt(mean_squared_error(y_true, y_pred)))
    except Exception:
        return np.nan


def compute_binary_metrics(df: pd.DataFrame) -> Dict[str, float]:
    y = df["true_label"].to_numpy(dtype=int)

    if "score_positive" in df.columns:
        score = df["score_positive"].to_numpy(dtype=float)
    else:
        score = df["prob_positive"].to_numpy(dtype=float)

    if "predicted_label" in df.columns:
        pred = df["predicted_label"].to_numpy(dtype=int)
    else:
        pred = (score >= 0.0).astype(int)

    if "prob_positive" in df.columns and df["prob_positive"].notna().all():
        p = df["prob_positive"].to_numpy(dtype=float)
        brier = safe_brier(y, p)
    else:
        brier = np.nan

    dummy_p = df["dummy_prob_positive"].to_numpy(dtype=float)
    dummy_pred = (dummy_p >= 0.5).astype(int)

    out = {
        "auc": safe_auc(y, score),
        "binary_balanced_accuracy": safe_balanced_accuracy(y, pred),
        "binary_accuracy": safe_accuracy(y, pred),
        "binary_macro_f1": safe_macro_f1(y, pred),
        "brier": brier,

        "dummy_auc": safe_auc(y, dummy_p),
        "dummy_binary_balanced_accuracy": safe_balanced_accuracy(y, dummy_pred),
        "dummy_binary_accuracy": safe_accuracy(y, dummy_pred),
        "dummy_binary_macro_f1": safe_macro_f1(y, dummy_pred),
        "dummy_brier": safe_brier(y, dummy_p),
    }

    try:
        recalls = recall_score(
            y,
            pred,
            average=None,
            labels=[0, 1],
            zero_division=0,
        )
        out["recall_class_0"] = float(recalls[0])
        out["recall_class_1"] = float(recalls[1])
    except Exception:
        out["recall_class_0"] = np.nan
        out["recall_class_1"] = np.nan

    return out

def compute_ordinal_metrics(df: pd.DataFrame) -> Dict[str, float]:
    y = df["true_label"].to_numpy(dtype=int)
    pred = df["predicted_bracket"].to_numpy(dtype=int)
    dummy_pred = df["dummy_predicted_bracket"].to_numpy(dtype=int)

    out = {
        "ordinal_mae": ordinal_mae(y, pred),
        "ordinal_rmse": regression_rmse(y.astype(float), pred.astype(float)),
        "quadratic_weighted_kappa": safe_qwk(y, pred),
        "ordinal_balanced_accuracy": safe_balanced_accuracy(y, pred),
        "ordinal_accuracy": safe_accuracy(y, pred),
        "ordinal_macro_f1": safe_macro_f1(y, pred),

        "dummy_ordinal_mae": ordinal_mae(y, dummy_pred),
        "dummy_ordinal_rmse": regression_rmse(y.astype(float), dummy_pred.astype(float)),
        "dummy_quadratic_weighted_kappa": safe_qwk(y, dummy_pred),
        "dummy_ordinal_balanced_accuracy": safe_balanced_accuracy(y, dummy_pred),
        "dummy_ordinal_accuracy": safe_accuracy(y, dummy_pred),
        "dummy_ordinal_macro_f1": safe_macro_f1(y, dummy_pred),
    }

    try:
        recalls = recall_score(
            y,
            pred,
            average=None,
            labels=[0, 1, 2, 3],
            zero_division=0,
        )
        out["recall_bracket_0"] = float(recalls[0])
        out["recall_bracket_1"] = float(recalls[1])
        out["recall_bracket_2"] = float(recalls[2])
        out["recall_bracket_3"] = float(recalls[3])
    except Exception:
        out["recall_bracket_0"] = np.nan
        out["recall_bracket_1"] = np.nan
        out["recall_bracket_2"] = np.nan
        out["recall_bracket_3"] = np.nan

    try:
        precisions = precision_score(
            y,
            pred,
            average=None,
            labels=[0, 1, 2, 3],
            zero_division=0,
        )
        out["precision_bracket_0"] = float(precisions[0])
        out["precision_bracket_1"] = float(precisions[1])
        out["precision_bracket_2"] = float(precisions[2])
        out["precision_bracket_3"] = float(precisions[3])
    except Exception:
        out["precision_bracket_0"] = np.nan
        out["precision_bracket_1"] = np.nan
        out["precision_bracket_2"] = np.nan
        out["precision_bracket_3"] = np.nan

    return out

def compute_ridge_metrics(df: pd.DataFrame) -> Dict[str, float]:
    y = df["true_score"].to_numpy(dtype=float)
    pred = df["predicted_score"].to_numpy(dtype=float)
    dummy_pred = df["dummy_predicted_score"].to_numpy(dtype=float)

    return {
        "pearson_r": safe_pearson(y, pred),
        "spearman_r": safe_spearman(y, pred),
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": regression_rmse(y, pred),
        "r2": float(r2_score(y, pred)) if len(y) >= 2 else np.nan,
        "dummy_pearson_r": safe_pearson(y, dummy_pred),
        "dummy_spearman_r": safe_spearman(y, dummy_pred),
        "dummy_mae": float(mean_absolute_error(y, dummy_pred)),
        "dummy_rmse": regression_rmse(y, dummy_pred),
        "dummy_r2": float(r2_score(y, dummy_pred)) if len(y) >= 2 else np.nan,
    }


# 10. AGGREGATION

def aggregate_binary_predictions(split_df: pd.DataFrame) -> pd.DataFrame:
    agg_spec = {
        "true_label": ("true_label", "first"),
        "score_positive": ("score_positive", "mean"),
        "prob_positive": ("prob_positive", "mean"),
        "dummy_prob_positive": ("dummy_prob_positive", "mean"),
        "n_test_predictions": ("score_positive", "size"),
    }

    if "score_type" in split_df.columns:
        agg_spec["score_type"] = ("score_type", "first")

    g = split_df.groupby("subject_id", as_index=False).agg(**agg_spec)

    if g["prob_positive"].notna().all():
        g["predicted_label"] = (g["prob_positive"] >= 0.5).astype(int)
    else:
        g["predicted_label"] = (g["score_positive"] >= 0.0).astype(int)

    g["dummy_predicted_label"] = (g["dummy_prob_positive"] >= 0.5).astype(int)
    return g

def aggregate_ordinal_predictions(split_df: pd.DataFrame) -> pd.DataFrame:
    prob_cols = ["prob_bracket_0", "prob_bracket_1", "prob_bracket_2", "prob_bracket_3"]
    dummy_prob_cols = [
        "dummy_prob_bracket_0",
        "dummy_prob_bracket_1",
        "dummy_prob_bracket_2",
        "dummy_prob_bracket_3",
    ]

    agg_spec = {
        "true_label": ("true_label", "first"),
        "n_test_predictions": ("true_label", "size"),
    }

    for col in prob_cols + dummy_prob_cols:
        agg_spec[col] = (col, "mean")

    g = split_df.groupby("subject_id", as_index=False).agg(**agg_spec)
    g["predicted_bracket"] = np.argmax(g[prob_cols].to_numpy(dtype=float), axis=1).astype(int)
    g["dummy_predicted_bracket"] = np.argmax(g[dummy_prob_cols].to_numpy(dtype=float), axis=1).astype(int)
    return g


def aggregate_ridge_predictions(split_df: pd.DataFrame) -> pd.DataFrame:
    g = split_df.groupby("subject_id", as_index=False).agg(
        true_score=("true_score", "first"),
        predicted_score=("predicted_score", "mean"),
        dummy_predicted_score=("dummy_predicted_score", "mean"),
        n_test_predictions=("predicted_score", "size"),
    )
    return g


# 11. INPUT LOADING

def load_subjects() -> List[str]:
    preprocessing_path = cfg.REPORTS_DIR / "preprocessing.csv"

    if not preprocessing_path.exists():
        raise FileNotFoundError(f"Missing preprocessing report: {preprocessing_path}")

    pre = pd.read_csv(preprocessing_path)

    if INCLUSION_COLUMN not in pre.columns:
        raise ValueError(f"Column {INCLUSION_COLUMN!r} not found in {preprocessing_path}")

    mask = bool_series(pre[INCLUSION_COLUMN])
    subjects = [clean_subject_id(x) for x in pre.loc[mask, "subject_id"].tolist()]
    subjects = sorted(subjects)

    if not subjects:
        raise RuntimeError(f"No subjects found with {INCLUSION_COLUMN}=True")

    return subjects


def load_mcmi() -> pd.DataFrame:
    if not cfg.MCMI_CSV.exists():
        raise FileNotFoundError(f"Missing MCMI CSV: {cfg.MCMI_CSV}")

    mcmi = pd.read_csv(cfg.MCMI_CSV)

    id_col = getattr(cfg, "SUBJECT_ID_COLUMN", "subject_id")

    if id_col not in mcmi.columns:
        raise ValueError(f"MCMI subject ID column not found: {id_col}")

    mcmi[id_col] = mcmi[id_col].map(clean_subject_id)
    mcmi = mcmi.rename(columns={id_col: "subject_id"})

    return mcmi


def align_scores(subject_ids: Sequence[str], mcmi: pd.DataFrame, scale: str) -> np.ndarray:
    m = mcmi.set_index("subject_id")
    vals = []

    for sid in subject_ids:
        if sid in m.index:
            vals.append(safe_float(m.loc[sid, scale]))
        else:
            vals.append(np.nan)

    return np.asarray(vals, dtype=float)


# 12. OUTPUT TABLE HELPERS

RESULT_COLUMNS = [
    "analysis_id",
    "task",
    "model_name",
    "scale",
    "target",
    "feature_config_id",
    "duration_label",
    "feature_set",
    "duration_sec",
    "continuous_families",
    "epoch_families",

    # Validation / split metadata
    "cv_method",
    "n_mc_l2so_splits",
    "inner_tuning",
    "inner_cv_folds",
    "random_seed",
    "split_mode",
    "split_set_id",
    "fixed_split_path",
    "subject_list_checksum",
    "label_checksum",
    "n_subjects",
    "n_features",
    "class_counts",
    "X_checksum",
    "model_matrix_path",
    "main_metric_name",
    "main_metric_value",

    # Binary classification metrics
    "auc",
    "binary_balanced_accuracy",
    "binary_accuracy",
    "binary_macro_f1",
    "brier",
    "recall_class_0",
    "recall_class_1",

    # Four-bracket ordinal metrics
    "ordinal_mae",
    "ordinal_rmse",
    "quadratic_weighted_kappa",
    "ordinal_balanced_accuracy",
    "ordinal_accuracy",
    "ordinal_macro_f1",
    "recall_bracket_0",
    "recall_bracket_1",
    "recall_bracket_2",
    "recall_bracket_3",
    "precision_bracket_0",
    "precision_bracket_1",
    "precision_bracket_2",
    "precision_bracket_3",

    # Continuous BR regression metrics
    "pearson_r",
    "spearman_r",
    "mae",
    "rmse",
    "r2",

    # Binary dummy baseline
    "dummy_auc",
    "dummy_binary_balanced_accuracy",
    "dummy_binary_accuracy",
    "dummy_binary_macro_f1",
    "dummy_brier",

    # Ordinal dummy baseline
    "dummy_ordinal_mae",
    "dummy_ordinal_rmse",
    "dummy_quadratic_weighted_kappa",
    "dummy_ordinal_balanced_accuracy",
    "dummy_ordinal_accuracy",
    "dummy_ordinal_macro_f1",

    # Regression dummy baseline
    "dummy_pearson_r",
    "dummy_spearman_r",
    "dummy_mae",
    "dummy_rmse",
    "dummy_r2",

    # L2SO prediction-count diagnostics
    "mean_n_test_predictions_per_subject",
    "min_n_test_predictions_per_subject",
    "max_n_test_predictions_per_subject",
]

PREDICTION_COLUMNS = [
    "analysis_id",
    "task",
    "scale",
    "target",
    "model_name",
    "feature_config_id",
    "feature_set",
    "duration_sec",
    "duration_label",
    "split_id",
    "subject_id",
    "true_label",
    "true_score",
    "score_positive",
    "score_type",
    "prob_positive",
    "predicted_label",
    "dummy_prob_positive",
    "dummy_predicted_label",
    "prob_bracket_0",
    "prob_bracket_1",
    "prob_bracket_2",
    "prob_bracket_3",
    "predicted_bracket",
    "dummy_prob_bracket_0",
    "dummy_prob_bracket_1",
    "dummy_prob_bracket_2",
    "dummy_prob_bracket_3",
    "dummy_predicted_bracket",
    "predicted_score",
    "dummy_predicted_score",
    "selected_C",
    "selected_alpha",
    "n_test_predictions",
]

SPLIT_COLUMNS = [
    "analysis_id",
    "task",
    "scale",
    "target",
    "model_name",
    "feature_config_id",
    "feature_set",
    "duration_label",

    # Fixed split metadata
    "split_mode",
    "split_set_id",
    "fixed_split_path",
    "subject_list_checksum",
    "label_checksum",

    "split_id",
    "train_subject_ids",
    "test_subject_ids",
    "selected_C",
    "selected_alpha",
    "inner_score",
    "inner_details",
    "train_class_counts",
    "test_class_counts",
    "train_score_mean",
    "test_score_mean",
]


def standardize_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    out = df.copy()

    for col in columns:
        if col not in out.columns:
            out[col] = np.nan

    return out[columns]


def write_csv_incremental(path: Path, df: pd.DataFrame, first_write: bool, compression=None) -> None:
    mode = "w" if first_write else "a"
    header = first_write
    df.to_csv(path, index=False, mode=mode, header=header, compression=compression)


def save_model_matrix(
    feature_config: FeatureConfig,
    X: np.ndarray,
    feature_names: List[str],
    subject_ids: List[str],
    X_checksum: str,
) -> str:
    label = f"cfg{feature_config.config_id:03d}__{feature_config.duration_label}__{X_checksum}"
    path = MODEL_DATA_DIR / f"X__{label}.npz"

    np.savez_compressed(
        path,
        X=X,
        subject_ids=np.asarray(subject_ids, dtype=object),
        feature_names=np.asarray(feature_names, dtype=object),
        feature_config_id=int(feature_config.config_id),
        feature_set=feature_config.feature_set,
        duration_sec=np.nan if feature_config.duration_sec is None else float(feature_config.duration_sec),
        duration_label=feature_config.duration_label,
        continuous_families=np.asarray(feature_config.continuous_families, dtype=object),
        epoch_families=np.asarray(feature_config.epoch_families, dtype=object),
        X_checksum=X_checksum,
    )

    return str(path)


# 13. ANALYSIS HELPERS

def make_analysis_id(
    task: str,
    model_name: str,
    scale: str,
    target: str,
    feature_config: FeatureConfig,
) -> str:
    raw = "__".join(
        [
            task,
            model_name,
            scale,
            target,
            str(feature_config.config_id),
            feature_config.feature_set,
            feature_config.duration_label,
        ]
    )
    return (
        f"{task}__{model_name}__{scale}__{target}__"
        f"cfg{feature_config.config_id:03d}__{stable_hash_text(raw)}"
    )


def base_analysis_row(
    analysis_id: str,
    task: str,
    model_name: str,
    scale: str,
    target: str,
    feature_config: FeatureConfig,
    n_subjects: int,
    n_features: int,
    y: np.ndarray,
    X_checksum: str,
    model_matrix_path: str,
) -> Dict[str, object]:
    return {
        "analysis_id": analysis_id,
        "task": task,
        "model_name": model_name,
        "scale": scale,
        "target": target,
        "feature_config_id": int(feature_config.config_id),
        "feature_set": feature_config.feature_set,
        "duration_sec": feature_config.duration_sec if feature_config.duration_sec is not None else np.nan,
        "duration_label": feature_config.duration_label,
        "continuous_families": "+".join(feature_config.continuous_families),
        "epoch_families": "+".join(feature_config.epoch_families),
        "n_subjects": int(n_subjects),
        "n_features": int(n_features),
        "class_counts": class_count_text(y) if task != "ridge_regression" else "continuous",
        "X_checksum": X_checksum,
        "model_matrix_path": model_matrix_path,
        "cv_method": "monte_carlo_l2so",
        "n_mc_l2so_splits": int(N_MONTE_CARLO_L2SO_SPLITS),
        "inner_tuning": bool(USE_INNER_CV_TUNING),
        "inner_cv_folds": int(INNER_CV_FOLDS),
        "random_seed": int(RANDOM_SEED),
    }


# 14. RUN TASKS

def _run_binary_one_split(
    split_id,
    train_idx,
    test_idx,
    X,
    y,
    subject_ids,
    scale,
    target,
    feature_config,
    analysis_id,
    model_name,
):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    selected_C, inner_score, inner_details = tune_binary_C(
        X_train,
        y_train,
        RANDOM_SEED + split_id,
    )

    model = make_binary_pipeline(model_name, selected_C)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(X_train, y_train)

    score = binary_score(model, X_test)
    prob = binary_probability(model, X_test)
    score_type = binary_score_type(model)

    if prob is not None:
        pred = (prob >= 0.5).astype(int)
    else:
        pred = (score >= 0.0).astype(int)

    dummy_p = float(np.mean(y_train))

    split_row = {
        "analysis_id": analysis_id,
        "task": "binary_auc",
        "scale": scale,
        "target": target,
        "model_name": model_name,
        "feature_config_id": feature_config.config_id,
        "feature_set": feature_config.feature_set,
        "duration_label": feature_config.duration_label,
        "split_id": split_id,
        "train_subject_ids": join_ids([subject_ids[i] for i in train_idx]),
        "test_subject_ids": join_ids([subject_ids[i] for i in test_idx]),
        "selected_C": selected_C,
        "inner_score": inner_score,
        "inner_details": inner_details,
        "train_class_counts": class_count_text(y_train),
        "test_class_counts": class_count_text(y_test),
    }

    pred_rows = []

    for local_j, idx in enumerate(test_idx):
        pred_rows.append(
            {
                "analysis_id": analysis_id,
                "task": "binary_auc",
                "scale": scale,
                "target": target,
                "model_name": model_name,
                "feature_config_id": feature_config.config_id,
                "feature_set": feature_config.feature_set,
                "duration_sec": feature_config.duration_sec if feature_config.duration_sec is not None else np.nan,
                "duration_label": feature_config.duration_label,
                "split_id": split_id,
                "subject_id": subject_ids[idx],
                "true_label": int(y_test[local_j]),
                "score_positive": float(score[local_j]),
                "score_type": score_type,
                "prob_positive": np.nan if prob is None else float(prob[local_j]),
                "predicted_label": int(pred[local_j]),
                "dummy_prob_positive": dummy_p,
                "dummy_predicted_label": int(dummy_p >= 0.5),
                "selected_C": selected_C,
            }
        )

    return split_row, pred_rows

def _run_ordinal_one_split(
    split_id,
    train_idx,
    test_idx,
    X,
    y,
    subject_ids,
    scale,
    feature_config,
    analysis_id,
    model_name,
):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    selected_C, inner_score, inner_details = tune_ordinal_C(
        X_train,
        y_train,
        RANDOM_SEED + split_id,
    )

    model = CumulativeOrdinalLogit(C=selected_C)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)
    pred = np.argmax(probs, axis=1).astype(int)

    counts = np.bincount(y_train.astype(int), minlength=4).astype(float)
    dummy_probs = counts / counts.sum() if counts.sum() else np.ones(4) / 4.0
    dummy_pred = int(np.argmax(dummy_probs))

    split_row = {
        "analysis_id": analysis_id,
        "task": "four_bracket_ordinal",
        "scale": scale,
        "target": "four_brackets",
        "model_name": model_name,
        "feature_config_id": feature_config.config_id,
        "feature_set": feature_config.feature_set,
        "duration_label": feature_config.duration_label,
        "split_id": split_id,
        "train_subject_ids": join_ids([subject_ids[i] for i in train_idx]),
        "test_subject_ids": join_ids([subject_ids[i] for i in test_idx]),
        "selected_C": selected_C,
        "inner_score": inner_score,
        "inner_details": inner_details,
        "train_class_counts": class_count_text(y_train),
        "test_class_counts": class_count_text(y_test),
    }

    pred_rows = []

    for local_j, idx in enumerate(test_idx):
        pred_rows.append(
            {
                "analysis_id": analysis_id,
                "task": "four_bracket_ordinal",
                "scale": scale,
                "target": "four_brackets",
                "model_name": model_name,
                "feature_config_id": feature_config.config_id,
                "feature_set": feature_config.feature_set,
                "duration_sec": feature_config.duration_sec if feature_config.duration_sec is not None else np.nan,
                "duration_label": feature_config.duration_label,
                "split_id": split_id,
                "subject_id": subject_ids[idx],
                "true_label": int(y_test[local_j]),
                "prob_bracket_0": float(probs[local_j, 0]),
                "prob_bracket_1": float(probs[local_j, 1]),
                "prob_bracket_2": float(probs[local_j, 2]),
                "prob_bracket_3": float(probs[local_j, 3]),
                "predicted_bracket": int(pred[local_j]),
                "dummy_prob_bracket_0": float(dummy_probs[0]),
                "dummy_prob_bracket_1": float(dummy_probs[1]),
                "dummy_prob_bracket_2": float(dummy_probs[2]),
                "dummy_prob_bracket_3": float(dummy_probs[3]),
                "dummy_predicted_bracket": int(dummy_pred),
                "selected_C": selected_C,
            }
        )

    return split_row, pred_rows

def _run_ridge_one_split(
    split_id,
    train_idx,
    test_idx,
    X,
    y,
    subject_ids,
    scale,
    feature_config,
    analysis_id,
    model_name,
):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    selected_alpha, inner_score, inner_details = tune_ridge_alpha(
        X_train,
        y_train,
        RANDOM_SEED + split_id,
    )

    model = make_ridge_pipeline(selected_alpha)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    dummy_pred_value = float(np.mean(y_train))

    split_row = {
        "analysis_id": analysis_id,
        "task": "ridge_regression",
        "scale": scale,
        "target": "continuous_BR",
        "model_name": model_name,
        "feature_config_id": feature_config.config_id,
        "feature_set": feature_config.feature_set,
        "duration_label": feature_config.duration_label,
        "split_id": split_id,
        "train_subject_ids": join_ids([subject_ids[i] for i in train_idx]),
        "test_subject_ids": join_ids([subject_ids[i] for i in test_idx]),
        "selected_alpha": selected_alpha,
        "inner_score": inner_score,
        "inner_details": inner_details,
        "train_score_mean": float(np.mean(y_train)),
        "test_score_mean": float(np.mean(y_test)),
    }

    pred_rows = []

    for local_j, idx in enumerate(test_idx):
        pred_rows.append(
            {
                "analysis_id": analysis_id,
                "task": "ridge_regression",
                "scale": scale,
                "target": "continuous_BR",
                "model_name": model_name,
                "feature_config_id": feature_config.config_id,
                "feature_set": feature_config.feature_set,
                "duration_sec": feature_config.duration_sec if feature_config.duration_sec is not None else np.nan,
                "duration_label": feature_config.duration_label,
                "split_id": split_id,
                "subject_id": subject_ids[idx],
                "true_score": float(y_test[local_j]),
                "predicted_score": float(pred[local_j]),
                "dummy_predicted_score": dummy_pred_value,
                "selected_alpha": selected_alpha,
            }
        )

    return split_row, pred_rows

def run_binary_analysis(
    X: np.ndarray,
    subject_ids: Sequence[str],
    y: np.ndarray,
    scale: str,
    target: str,
    feature_config: FeatureConfig,
    X_checksum: str,
    model_matrix_path: str,
):
    task = "binary_auc"
    model_name = BINARY_MODEL_NAME
    analysis_id = make_analysis_id(task, model_name, scale, target, feature_config)

    splits, split_info = get_splits_for_analysis(
        task=task,
        scale=scale,
        target=target,
        y=y,
        subject_ids=subject_ids,
        exploratory_analysis_id=analysis_id,
        require_label_diversity=False,
        require_train_two_classes=True,
    )

    split_rows = []
    pred_rows = []

    outputs = run_parallel_jobs(
        delayed(_run_binary_one_split)(
            split_id,
            train_idx,
            test_idx,
            X,
            y,
            subject_ids,
            scale,
            target,
            feature_config,
            analysis_id,
            model_name,
        )
        for split_id, (train_idx, test_idx) in enumerate(splits)
    )

    split_rows = []
    pred_rows = []

    for split_row, split_pred_rows in outputs:
        split_rows.append(split_row)
        pred_rows.extend(split_pred_rows)

    split_pred_df = pd.DataFrame(pred_rows)
    subject_pred_df = aggregate_binary_predictions(split_pred_df)

    metrics = compute_binary_metrics(subject_pred_df)

    result = base_analysis_row(
        analysis_id=analysis_id,
        task=task,
        model_name=model_name,
        scale=scale,
        target=target,
        feature_config=feature_config,
        n_subjects=len(subject_ids),
        n_features=X.shape[1],
        y=y,
        X_checksum=X_checksum,
        model_matrix_path=model_matrix_path,
    )

    result.update(metrics)
    result = add_split_info_to_result(result, split_info)
    result["main_metric_name"] = "auc"
    result["main_metric_value"] = result.get("auc", np.nan)
    result["mean_n_test_predictions_per_subject"] = float(subject_pred_df["n_test_predictions"].mean())
    result["min_n_test_predictions_per_subject"] = int(subject_pred_df["n_test_predictions"].min())
    result["max_n_test_predictions_per_subject"] = int(subject_pred_df["n_test_predictions"].max())

    for col, value in [
        ("analysis_id", analysis_id),
        ("task", task),
        ("scale", scale),
        ("target", target),
        ("model_name", model_name),
        ("feature_config_id", feature_config.config_id),
        ("feature_set", feature_config.feature_set),
        ("duration_sec", feature_config.duration_sec if feature_config.duration_sec is not None else np.nan),
        ("duration_label", feature_config.duration_label),
    ]:
        subject_pred_df[col] = value

    split_df = pd.DataFrame(split_rows)
    split_df = add_split_info_to_split_df(split_df, split_info)

    return result, subject_pred_df, split_pred_df, split_df


def run_ordinal_analysis(
    X: np.ndarray,
    subject_ids: Sequence[str],
    y: np.ndarray,
    scale: str,
    feature_config: FeatureConfig,
    X_checksum: str,
    model_matrix_path: str,
):
    task = "four_bracket_ordinal"
    model_name = "cumulative_l2_logistic"
    target = "four_brackets"
    analysis_id = make_analysis_id(task, model_name, scale, target, feature_config)

    splits, split_info = get_splits_for_analysis(
        task=task,
        scale=scale,
        target=target,
        y=y,
        subject_ids=subject_ids,
        exploratory_analysis_id=analysis_id,
        require_label_diversity=False,
        require_train_two_classes=False,
    )

    split_rows = []
    pred_rows = []


    outputs = run_parallel_jobs(
        delayed(_run_ordinal_one_split)(
            split_id,
            train_idx,
            test_idx,
            X,
            y,
            subject_ids,
            scale,
            feature_config,
            analysis_id,
            model_name,
        )
        for split_id, (train_idx, test_idx) in enumerate(splits)
    )

    split_rows = []
    pred_rows = []

    for split_row, split_pred_rows in outputs:
        split_rows.append(split_row)
        pred_rows.extend(split_pred_rows)

    
    split_pred_df = pd.DataFrame(pred_rows)
    subject_pred_df = aggregate_ordinal_predictions(split_pred_df)

    metrics = compute_ordinal_metrics(subject_pred_df)

    result = base_analysis_row(
        analysis_id=analysis_id,
        task=task,
        model_name=model_name,
        scale=scale,
        target=target,
        feature_config=feature_config,
        n_subjects=len(subject_ids),
        n_features=X.shape[1],
        y=y,
        X_checksum=X_checksum,
        model_matrix_path=model_matrix_path,
    )

    result.update(metrics)
    result = add_split_info_to_result(result, split_info)
    result["main_metric_name"] = "quadratic_weighted_kappa"
    result["main_metric_value"] = result.get("quadratic_weighted_kappa", np.nan)
    result["mean_n_test_predictions_per_subject"] = float(subject_pred_df["n_test_predictions"].mean())
    result["min_n_test_predictions_per_subject"] = int(subject_pred_df["n_test_predictions"].min())
    result["max_n_test_predictions_per_subject"] = int(subject_pred_df["n_test_predictions"].max())

    for col, value in [
        ("analysis_id", analysis_id),
        ("task", task),
        ("scale", scale),
        ("target", target),
        ("model_name", model_name),
        ("feature_config_id", feature_config.config_id),
        ("feature_set", feature_config.feature_set),
        ("duration_sec", feature_config.duration_sec if feature_config.duration_sec is not None else np.nan),
        ("duration_label", feature_config.duration_label),
    ]:
        subject_pred_df[col] = value

    split_df = pd.DataFrame(split_rows)
    split_df = add_split_info_to_split_df(split_df, split_info)

    return result, subject_pred_df, split_pred_df, split_df


def run_ridge_analysis(
    X: np.ndarray,
    subject_ids: Sequence[str],
    y: np.ndarray,
    scale: str,
    feature_config: FeatureConfig,
    X_checksum: str,
    model_matrix_path: str,
):
    task = "ridge_regression"
    model_name = "ridge_regression"
    target = "continuous_BR"
    analysis_id = make_analysis_id(task, model_name, scale, target, feature_config)

    splits, split_info = get_splits_for_analysis(
        task=task,
        scale=scale,
        target=target,
        y=y,
        subject_ids=subject_ids,
        exploratory_analysis_id=analysis_id,
        require_label_diversity=False,
        require_train_two_classes=False,
    )

    split_rows = []
    pred_rows = []

    outputs = run_parallel_jobs(
        delayed(_run_ridge_one_split)(
            split_id,
            train_idx,
            test_idx,
            X,
            y,
            subject_ids,
            scale,
            feature_config,
            analysis_id,
            model_name,
        )
        for split_id, (train_idx, test_idx) in enumerate(splits)
    )

    split_rows = []
    pred_rows = []

    for split_row, split_pred_rows in outputs:
        split_rows.append(split_row)
        pred_rows.extend(split_pred_rows)

    split_pred_df = pd.DataFrame(pred_rows)
    subject_pred_df = aggregate_ridge_predictions(split_pred_df)

    metrics = compute_ridge_metrics(subject_pred_df)

    result = base_analysis_row(
        analysis_id=analysis_id,
        task=task,
        model_name=model_name,
        scale=scale,
        target=target,
        feature_config=feature_config,
        n_subjects=len(subject_ids),
        n_features=X.shape[1],
        y=y,
        X_checksum=X_checksum,
        model_matrix_path=model_matrix_path,
    )

    result.update(metrics)
    result = add_split_info_to_result(result, split_info)
    result["main_metric_name"] = "pearson_r"
    result["main_metric_value"] = result.get("pearson_r", np.nan)
    result["mean_n_test_predictions_per_subject"] = float(subject_pred_df["n_test_predictions"].mean())
    result["min_n_test_predictions_per_subject"] = int(subject_pred_df["n_test_predictions"].min())
    result["max_n_test_predictions_per_subject"] = int(subject_pred_df["n_test_predictions"].max())

    for col, value in [
        ("analysis_id", analysis_id),
        ("task", task),
        ("scale", scale),
        ("target", target),
        ("model_name", model_name),
        ("feature_config_id", feature_config.config_id),
        ("feature_set", feature_config.feature_set),
        ("duration_sec", feature_config.duration_sec if feature_config.duration_sec is not None else np.nan),
        ("duration_label", feature_config.duration_label),
    ]:
        subject_pred_df[col] = value

    split_df = pd.DataFrame(split_rows)
    split_df = add_split_info_to_split_df(split_df, split_info)

    return result, subject_pred_df, split_pred_df, split_df


# 14B. RESUME / STOP HELPERS


def load_completed_analysis_ids(results_path: Path) -> set[str]:
    if not results_path.exists():
        return set()

    try:
        df = pd.read_csv(results_path, usecols=["analysis_id"])
        return set(df["analysis_id"].dropna().astype(str).tolist())
    except Exception as error:
        print(f"Warning: could not read completed analysis IDs from {results_path}: {error}")
        return set()


def append_one_result(results_path: Path, result: Dict[str, object], first_write: bool) -> bool:
    row = standardize_columns(pd.DataFrame([result]), RESULT_COLUMNS)
    write_csv_incremental(results_path, row, first_write)
    return False


def stop_requested() -> bool:
    return STOP_REQUEST_FILE.exists()

# 15. MAIN

def main() -> None:
    run_start_time = time.perf_counter()
    ensure_dirs()

    print(f"Classifier workers: {get_classifier_workers()}")
    print(f"Numerical threads per worker: {NUMERICAL_THREADS_PER_WORKER}")
    print(f"Classifier parallel backend: {CLASSIFIER_PARALLEL_BACKEND}")
    print("\n" + "=" * 80)
    print("classify_mcmi_v2.py")
    print("=" * 80)
    print(f"Run timestamp: {RUN_TIMESTAMP}")
    print(f"Output folder: {OUT_DIR}")
    print(f"Stop-request file: {STOP_REQUEST_FILE}")
    print(f"Inclusion column: {INCLUSION_COLUMN}")
    print(f"Monte Carlo L2SO splits per analysis: {N_MONTE_CARLO_L2SO_SPLITS}")
    print(f"Inner CV tuning: {USE_INNER_CV_TUNING}")
    print(f"Binary model: {BINARY_MODEL_NAME}")
    print(f"Run all 255 family combinations: {RUN_ALL_255_FAMILY_COMBINATIONS}")
    print(f"Epoch duration mode: {EPOCH_DURATION_MODE}")
    print("=" * 80 + "\n")

    subject_ids_all = load_subjects()
    mcmi = load_mcmi()

    scales = get_mcmi_scales(mcmi)

    feature_configs = build_feature_configs()

    print(f"Subjects included: {len(subject_ids_all)}")
    print(f"MCMI scales found: {len(scales)}")
    print(f"Enabled feature families: {ENABLED_FAMILIES}")
    print(f"Feature configurations: {len(feature_configs)}")

    feature_config_path = OUT_DIR / "classify_mcmi_v2_feature_configurations.csv"
    feature_config_df = pd.DataFrame(
        [
            {
                "feature_config_id": fc.config_id,
                "feature_set": fc.feature_set,
                "duration_sec": fc.duration_sec if fc.duration_sec is not None else np.nan,
                "duration_label": fc.duration_label,
                "continuous_families": "+".join(fc.continuous_families),
                "epoch_families": "+".join(fc.epoch_families),
                "n_families": len(fc.all_families),
            }
            for fc in feature_configs
        ]
    )
    feature_config_df.to_csv(feature_config_path, index=False)

    results_path = OUT_DIR / "classify_mcmi_v2_all_results.csv"
    subject_predictions_path = OUT_DIR / "classify_mcmi_v2_subject_predictions.csv.gz"
    split_predictions_path = OUT_DIR / "classify_mcmi_v2_split_predictions.csv.gz"
    splits_path = OUT_DIR / "classify_mcmi_v2_splits.csv.gz"
    feature_names_path = OUT_DIR / "classify_mcmi_v2_feature_names.csv.gz"
    missing_features_path = OUT_DIR / "classify_mcmi_v2_missing_features.csv"
    manifest_path = OUT_DIR / "classify_mcmi_v2_manifest.json"

    first_results = not results_path.exists()
    first_subject_preds = not subject_predictions_path.exists()
    first_split_preds = not split_predictions_path.exists()
    first_splits = not splits_path.exists()
    first_feature_names = not feature_names_path.exists()

    completed_analysis_ids = load_completed_analysis_ids(results_path)
    if completed_analysis_ids:
        print(f"Resume mode: found {len(completed_analysis_ids)} completed analyses in {results_path}")

    result_rows = []
    missing_feature_rows = []

    analysis_counter = len(completed_analysis_ids)
    n_completed_this_session = 0
    stopped_early = False

    for fc_idx, feature_config in enumerate(feature_configs, start=1):
        if stop_requested():
            stopped_early = True
            print(f"Stop requested via {STOP_REQUEST_FILE}. Finishing cleanly.")
            break

        print("\n" + "-" * 80)
        print(
            f"Assembling X {fc_idx}/{len(feature_configs)}: "
            f"cfg{feature_config.config_id:03d} | {feature_config.feature_set} | {feature_config.duration_label}"
        )

        try:
            X_all, feature_names, used_subjects, missing_subjects = assemble_X(subject_ids_all, feature_config)
            enforce_common_subjects_for_fixed_splits(
                expected_subjects=subject_ids_all,
                used_subjects=used_subjects,
                feature_config=feature_config,
                missing_subjects=missing_subjects,
            )
        except Exception as error:
            print(f"  FAILED assembling features: {error}")
            missing_feature_rows.append(
                {
                    "feature_config_id": feature_config.config_id,
                    "feature_set": feature_config.feature_set,
                    "duration_label": feature_config.duration_label,
                    "message": f"FAILED_ASSEMBLY: {error}",
                }
            )
            continue

        X_checksum = array_checksum(X_all)
        print(f"  X shape: {X_all.shape}; checksum={X_checksum}")

        for item in missing_subjects:
            missing_feature_rows.append(
                {
                    "feature_config_id": feature_config.config_id,
                    "feature_set": feature_config.feature_set,
                    "duration_label": feature_config.duration_label,
                    "message": item,
                }
            )

        model_matrix_path = ""
        if SAVE_MODEL_MATRICES:
            model_matrix_path = save_model_matrix(
                feature_config=feature_config,
                X=X_all,
                feature_names=feature_names,
                subject_ids=used_subjects,
                X_checksum=X_checksum,
            )

        fn_df = pd.DataFrame(
            {
                "feature_config_id": feature_config.config_id,
                "feature_set": feature_config.feature_set,
                "duration_label": feature_config.duration_label,
                "X_checksum": X_checksum,
                "feature_index": np.arange(len(feature_names)),
                "feature_name": feature_names,
            }
        )
        write_csv_incremental(
            feature_names_path,
            fn_df,
            first_feature_names,
            compression="gzip",
        )
        first_feature_names = False

        for scale in scales:
            if stop_requested():
                stopped_early = True
                print(f"Stop requested via {STOP_REQUEST_FILE}. Finishing cleanly.")
                break

            scores_all = align_scores(used_subjects, mcmi, scale)
            valid_score_mask = np.isfinite(scores_all)

            if valid_score_mask.sum() < 20:
                print(f"  Skipping {scale}: too few valid scores")
                continue

            X = X_all[valid_score_mask]
            subjects = [s for s, ok in zip(used_subjects, valid_score_mask) if ok]
            scores = scores_all[valid_score_mask].astype(float)

            # Binary AUC
            if RUN_BINARY_AUC:
                for binary_target in BINARY_TARGETS:
                    y_bin = make_binary(scores, binary_target)
                    valid_y = y_bin >= 0

                    X_task = X[valid_y]
                    subjects_task = [s for s, ok in zip(subjects, valid_y) if ok]
                    y_task = y_bin[valid_y]

                    if len(np.unique(y_task)) < 2:
                        continue

                    if pd.Series(y_task).value_counts().min() < 2:
                        continue

                    analysis_counter += 1

                    analysis_id_check = make_analysis_id(
                        "binary_auc",
                        BINARY_MODEL_NAME,
                        scale,
                        binary_target,
                        feature_config,
                    )

                    if analysis_id_check in completed_analysis_ids:
                        print(
                            f"  [skip] binary {scale} {binary_target} | "
                            f"cfg{feature_config.config_id:03d} already completed"
                        )
                        continue

                    print(
                        f"  [{analysis_counter}] binary {scale} {binary_target} | "
                        f"cfg{feature_config.config_id:03d} | n={len(y_task)} | {class_count_text(y_task)}"
                    )

                    result, subj_pred, split_pred, split_df = run_binary_analysis(
                        X=X_task,
                        subject_ids=subjects_task,
                        y=y_task,
                        scale=scale,
                        target=binary_target,
                        feature_config=feature_config,
                        X_checksum=X_checksum,
                        model_matrix_path=model_matrix_path,
                    )

                    subj_pred = standardize_columns(subj_pred, PREDICTION_COLUMNS)
                    write_csv_incremental(
                        subject_predictions_path,
                        subj_pred,
                        first_subject_preds,
                        compression="gzip",
                    )
                    first_subject_preds = False

                    if SAVE_SPLIT_LEVEL_PREDICTIONS:
                        split_pred = standardize_columns(split_pred, PREDICTION_COLUMNS)
                        write_csv_incremental(
                            split_predictions_path,
                            split_pred,
                            first_split_preds,
                            compression="gzip",
                        )
                        first_split_preds = False

                    split_df = standardize_columns(split_df, SPLIT_COLUMNS)
                    write_csv_incremental(
                        splits_path,
                        split_df,
                        first_splits,
                        compression="gzip",
                    )
                    first_splits = False

                    result_rows.append(result)
                    first_results = append_one_result(results_path, result, first_results)
                    completed_analysis_ids.add(str(result["analysis_id"]))
                    n_completed_this_session += 1

            # Four-bracket ordinal
            if RUN_FOUR_BRACKETS:
                y_ord = make_brackets(scores)
                valid_y = y_ord >= 0

                X_task = X[valid_y]
                subjects_task = [s for s, ok in zip(subjects, valid_y) if ok]
                y_task = y_ord[valid_y]

                if len(np.unique(y_task)) >= 2 and len(y_task) >= 20:
                    analysis_counter += 1

                    analysis_id_check = make_analysis_id(
                        "four_bracket_ordinal",
                        "cumulative_l2_logistic",
                        scale,
                        "four_brackets",
                        feature_config,
                    )

                    if analysis_id_check in completed_analysis_ids:
                        print(
                            f"  [skip] ordinal {scale} | "
                            f"cfg{feature_config.config_id:03d} already completed"
                        )
                        continue

                    print(
                        f"  [{analysis_counter}] ordinal {scale} | "
                        f"cfg{feature_config.config_id:03d} | n={len(y_task)} | {class_count_text(y_task)}"
                    )

                    result, subj_pred, split_pred, split_df = run_ordinal_analysis(
                        X=X_task,
                        subject_ids=subjects_task,
                        y=y_task,
                        scale=scale,
                        feature_config=feature_config,
                        X_checksum=X_checksum,
                        model_matrix_path=model_matrix_path,
                    )

                    subj_pred = standardize_columns(subj_pred, PREDICTION_COLUMNS)
                    write_csv_incremental(
                        subject_predictions_path,
                        subj_pred,
                        first_subject_preds,
                        compression="gzip",
                    )
                    first_subject_preds = False

                    if SAVE_SPLIT_LEVEL_PREDICTIONS:
                        split_pred = standardize_columns(split_pred, PREDICTION_COLUMNS)
                        write_csv_incremental(
                            split_predictions_path,
                            split_pred,
                            first_split_preds,
                            compression="gzip",
                        )
                        first_split_preds = False

                    split_df = standardize_columns(split_df, SPLIT_COLUMNS)
                    write_csv_incremental(
                        splits_path,
                        split_df,
                        first_splits,
                        compression="gzip",
                    )
                    first_splits = False

                    result_rows.append(result)
                    first_results = append_one_result(results_path, result, first_results)
                    completed_analysis_ids.add(str(result["analysis_id"]))
                    n_completed_this_session += 1

            # Continuous ridge
            if RUN_CONTINUOUS_RIDGE:
                y_reg = scores.astype(float)

                if len(y_reg) >= 20 and np.std(y_reg) > 0:
                    analysis_counter += 1

                    analysis_id_check = make_analysis_id(
                        "ridge_regression",
                        "ridge_regression",
                        scale,
                        "continuous_BR",
                        feature_config,
                    )

                    if analysis_id_check in completed_analysis_ids:
                        print(
                            f"  [skip] ridge {scale} | "
                            f"cfg{feature_config.config_id:03d} already completed"
                        )
                        continue

                    print(
                        f"  [{analysis_counter}] ridge {scale} | "
                        f"cfg{feature_config.config_id:03d} | n={len(y_reg)}"
                    )

                    result, subj_pred, split_pred, split_df = run_ridge_analysis(
                        X=X,
                        subject_ids=subjects,
                        y=y_reg,
                        scale=scale,
                        feature_config=feature_config,
                        X_checksum=X_checksum,
                        model_matrix_path=model_matrix_path,
                    )

                    subj_pred = standardize_columns(subj_pred, PREDICTION_COLUMNS)
                    write_csv_incremental(
                        subject_predictions_path,
                        subj_pred,
                        first_subject_preds,
                        compression="gzip",
                    )
                    first_subject_preds = False

                    if SAVE_SPLIT_LEVEL_PREDICTIONS:
                        split_pred = standardize_columns(split_pred, PREDICTION_COLUMNS)
                        write_csv_incremental(
                            split_predictions_path,
                            split_pred,
                            first_split_preds,
                            compression="gzip",
                        )
                        first_split_preds = False

                    split_df = standardize_columns(split_df, SPLIT_COLUMNS)
                    write_csv_incremental(
                        splits_path,
                        split_df,
                        first_splits,
                        compression="gzip",
                    )
                    first_splits = False

                    result_rows.append(result)
                    first_results = append_one_result(results_path, result, first_results)
                    completed_analysis_ids.add(str(result["analysis_id"]))
                    n_completed_this_session += 1

    # Results are written incrementally after every completed analysis.
    # This keeps the run resumable after Ctrl+C, crash, power loss, or manual stop.
    if not results_path.exists():
        pd.DataFrame(columns=RESULT_COLUMNS).to_csv(results_path, index=False)

    if missing_feature_rows:
        pd.DataFrame(missing_feature_rows).to_csv(missing_features_path, index=False)
    else:
        pd.DataFrame(columns=["feature_config_id", "feature_set", "duration_label", "message"]).to_csv(
            missing_features_path,
            index=False,
        )
    elapsed_seconds = time.perf_counter() - run_start_time
    elapsed_minutes = elapsed_seconds / 60.0
    analyses_per_minute = analysis_counter / elapsed_minutes if elapsed_minutes > 0 else np.nan

    manifest = {
        "elapsed_seconds": elapsed_seconds,
        "elapsed_minutes": elapsed_minutes,
        "analyses_per_minute": analyses_per_minute,
        "script": "classify_mcmi_v2.py",
        "run_timestamp": RUN_TIMESTAMP,
        "output_folder": str(OUT_DIR),
        "inclusion_column": INCLUSION_COLUMN,
        "n_subjects_initial": len(subject_ids_all),
        "n_scales": len(scales),
        "scales": scales,
        "enabled_feature_families": ENABLED_FAMILIES,
        "n_feature_configs": len(feature_configs),
        "run_all_255_family_combinations": RUN_ALL_255_FAMILY_COMBINATIONS,
        "epoch_duration_mode": EPOCH_DURATION_MODE,
        "n_monte_carlo_l2so_splits": N_MONTE_CARLO_L2SO_SPLITS,
        "use_inner_cv_tuning": USE_INNER_CV_TUNING,
        "inner_cv_folds": INNER_CV_FOLDS,
        "logistic_C_grid": LOGISTIC_C_GRID,
        "ridge_alpha_grid": RIDGE_ALPHA_GRID,
        "binary_targets": BINARY_TARGETS,
        "binary_model_name": BINARY_MODEL_NAME,
        "run_binary_auc": RUN_BINARY_AUC,
        "run_four_brackets": RUN_FOUR_BRACKETS,
        "run_continuous_ridge": RUN_CONTINUOUS_RIDGE,
        "random_seed": RANDOM_SEED,
        "resume_from_dir": str(RESUME_FROM_DIR_RAW) if RESUME_FROM_DIR_RAW else "",
        "stop_request_file": str(STOP_REQUEST_FILE),
        "stopped_early": bool(stopped_early),
        "n_completed_this_session": int(n_completed_this_session),
        "n_completed_total_in_results_file": int(len(completed_analysis_ids)),
        "outputs": {
            "all_results": str(results_path),
            "subject_predictions": str(subject_predictions_path),
            "split_predictions": str(split_predictions_path),
            "splits": str(splits_path),
            "feature_configurations": str(feature_config_path),
            "feature_names": str(feature_names_path),
            "missing_features": str(missing_features_path),
        },
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("\n" + "=" * 80)
    print("Classification finished.")
    print(f"Analyses completed / already completed total: {len(completed_analysis_ids)}")
    print(f"Completed this session: {n_completed_this_session}")
    if stopped_early:
        print("Stopped early by STOP_REQUESTED.txt; outputs are safe to resume.")
    print(f"Run folder: {OUT_DIR}")
    print("\nSaved:")
    print(f"  All results:          {results_path}")
    print(f"  Subject predictions:  {subject_predictions_path}")
    print(f"  Split predictions:    {split_predictions_path}")
    print(f"  Splits:               {splits_path}")
    print(f"  Feature configs:      {feature_config_path}")
    print(f"  Feature names:        {feature_names_path}")
    print(f"  Missing features:     {missing_features_path}")
    print(f"  Manifest:             {manifest_path}")
    print("=" * 80 + "\n")
    print(f"Elapsed time: {elapsed_minutes:.2f} minutes")
    print(f"Analyses per minute: {analyses_per_minute:.2f}")


if __name__ == "__main__":
    main()
