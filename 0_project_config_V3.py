"""Configuration and GUI runner for the eight-stage MAPPER pipeline."""

from __future__ import annotations

import csv
import json
import os
import queue
import subprocess
import sys
import threading
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional

from V3_runtime import (
    RunControl,
    file_sha256,
    latest_run_dir,
    load_json,
    manifest_complete,
    make_stage_run_dir,
    run_dirs,
    save_json_atomic,
    settings_path,
    stable_hash,
    update_manifest,
    utc_iso,
    write_manifest,
)

# General defaults.
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = Path("")
EEG_DIR = Path("")
CLEANED_DIR = Path("")
REPORTS_DIR = Path("")
MCMI_CSV = Path("")
SCRIPTS_DIR = SCRIPT_DIR
SUBJECT_ID_COLUMN = "subject_id"
EEG_FILE_EXTENSION = ".edf"
RUNNER_LOG_DIR = Path("")

# Numbered V3 scripts. The GUI itself is stage 0 and is not a tab.
SCRIPT_PREPROCESS_BASIC = "1_preprocess_basic_V3.py"
SCRIPT_PREPROCESS_EPOCHS = "2_preprocess_epochs_V3.py"
SCRIPT_PREPROCESSING_QC_REPORT = "3_preprocessing_qc_report_V3.py"
SCRIPT_BUILD_FEATURE_CACHE = "4_build_feature_cache_V3.py"
SCRIPT_CREATE_FIXED_OUTER_FOLDS = "5_create_fixed_outer_folds_V3.py"
SCRIPT_NESTED_MCMI = "6_nested_mcmi_V3.py"
SCRIPT_NESTED_RESULTS_REPORT = "7_nested_results_report_V3.py"
SCRIPT_NESTED_INFERENCE = "8_nested_inference_V3.py"

STAGE_SCRIPTS = {
    1: SCRIPT_PREPROCESS_BASIC,
    2: SCRIPT_PREPROCESS_EPOCHS,
    3: SCRIPT_PREPROCESSING_QC_REPORT,
    4: SCRIPT_BUILD_FEATURE_CACHE,
    5: SCRIPT_CREATE_FIXED_OUTER_FOLDS,
    6: SCRIPT_NESTED_MCMI,
    7: SCRIPT_NESTED_RESULTS_REPORT,
    8: SCRIPT_NESTED_INFERENCE,
}

STAGE_TITLES = {
    1: "Basic preprocessing",
    2: "Epoch preprocessing",
    3: "Preprocessing QC",
    4: "Feature cache",
    5: "Fixed outer folds",
    6: "Nested MCMI analysis",
    7: "Nested results report",
    8: "Nested inference",
}

STAGE_IDS = tuple(range(1, 9))

TIMESTAMPED_REPORT_STAGES = set(STAGE_IDS)

STAGE_SETTING_DEPENDENCIES = {
    1: (1,),
    2: (2,),
    3: (2, 3),
    4: (4,),
    5: (5,),
    6: (4, 6),
    7: (6, 7),
    8: (4, 6, 8),
}

# Narrow cross-stage definitions used by an upstream calculation.
STAGE_SHARED_SETTING_KEYS = {
    5: {
        4: ("FEATURE_INCLUSION_COLUMN",),
        6: (
            "MCMI_SCALE_ORDER",
            "BRACKET_CUTOFF_1",
            "BRACKET_CUTOFF_2",
            "BRACKET_CUTOFF_3",
        ),
    },
}

SOURCE_SNAPSHOT_STAGES = {
    3: (2,),
    4: (2,),
    5: (4,),
    6: (4,),
    7: (4, 6),
    8: (4, 6),
}

# Pipeline-wide scientific definitions retained from V2.
MIN_RECORDING_SECONDS = 30.0
RESAMPLE_SFREQ = None
DO_BANDPASS = True
BANDPASS_LOW = 0.5
BANDPASS_HIGH = 30.0
DO_NOTCH = True
NOTCH_FREQ = 50.0
DO_AVERAGE_REFERENCE = True
BAD_CHANNEL_VAR_LOW = 1e-10
BAD_CHANNEL_VAR_HIGH_MULT = 5
MAX_BAD_CHANNELS = 3
DO_INTERPOLATE_BAD_CHANNELS = True
DO_ICA = True
DO_ECG_REMOVAL = True
ICA_EOG_THRESHOLD = 3.0
ARTIFACT_REJECTION = "hybrid"
AUTOREJECT_TIMEOUT_SECONDS = 600
EPOCH_FLAT_THRESHOLD = 1e-6
VARIANCE_REJECT_Z = 3.5
CACHE_DURATIONS = [1.0, 2.0, 3.0, 4.0, 5.0]
REFERENCE_DURATION = 2.0
BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
}
MAX_FREQ = 30.0
FEATURE_FAMILIES = {
    "BAND_POWER": True,
    "MICROSTATES": True,
    "CONNECTIVITY": True,
    "SPECTRAL_PARAMETERIZATION": True,
    "HJORTH": True,
    "ENTROPY": True,
    "ASYMMETRY": True,
    "KURTOSIS": True,
}
FEATURE_FAMILY_ORDER = [
    "BAND_POWER", "MICROSTATES", "CONNECTIVITY",
    "SPECTRAL_PARAMETERIZATION", "HJORTH", "ENTROPY",
    "ASYMMETRY", "KURTOSIS",
]
CONTINUOUS_FAMILIES = {"MICROSTATES", "CONNECTIVITY", "ENTROPY", "ASYMMETRY"}
EPOCH_FAMILIES = {"BAND_POWER", "SPECTRAL_PARAMETERIZATION", "HJORTH", "KURTOSIS"}
CONNECTIVITY_DENSITIES = (0.10, 0.20, 0.30)
CONNECTIVITY_USE_ABSOLUTE_WEIGHTS = True
USE_GRAPH_METRICS = True
ENTROPY_MAX_SAMPLES = 20000
SAMPLE_ENTROPY_MAX_SAMPLES = 8000
HIGUCHI_KMAX = 8
USE_ADAPTIVE_WELCH = False
SPECPARAM_FREQ_RANGE = (1.0, 30.0)
SPECPARAM_PEAK_WIDTHS = (1.0, 8.0)
SPECPARAM_MAX_PEAKS = 6
SPECPARAM_MIN_PEAK_HEIGHT = 0.05
SPECPARAM_PEAK_THRESHOLD = 2.0
SPECPARAM_APERIODIC_MODE = "fixed"
MICROSTATE_N_STATES = 4
MICROSTATE_MAX_PEAKS = 2000
MCMI_SCALE_ORDER = [
    "BR_1", "BR_2A", "BR_2B", "BR_3", "BR_4A", "BR_4B",
    "BR_5", "BR_6A", "BR_6B", "BR_7", "BR_8A", "BR_8B",
    "BR_S", "BR_C", "BR_P",
]
BRACKET_CUTOFF_1 = 60
BRACKET_CUTOFF_2 = 75
BRACKET_CUTOFF_3 = 85

ORDINAL_SCHEME = "4class_original"

BRACKET_LABELS = {
    0: "0-59",
    1: "60-74",
    2: "75-84",
    3: "85+",
}
BINARY_TARGETS = ["BR75", "BR60"]
BINARY_MODEL_NAME = "linear_svm_svc"
LOGISTIC_C = 1.0
RIDGE_ALPHA = 1.0
CLASS_WEIGHT = "balanced"
MAX_ITER = 1000
SOLVER = "lbfgs"
RUN_BINARY_AUC = True
RUN_FOUR_BRACKETS = True
RUN_CONTINUOUS_RIDGE = True
RUN_ALL_255_FAMILY_COMBINATIONS = True
EPOCH_DURATION_MODE = "all"
RANDOM_SEED = 27
CPU_USAGE_PERCENT = 75
MAX_WORKERS = None
MAX_FEATURE_WORKERS = 12
MIN_WORKERS = 1
MAX_CLASSIFIER_WORKERS = 12
NUMERICAL_THREADS_PER_WORKER = 1
CLASSIFIER_PARALLEL_BACKEND = "loky"

# V3 validation defaults.
OUTER_N_FOLDS = 5
OUTER_FOLD_SEED = 27
OUTER_FOLD_CANDIDATE_ASSIGNMENTS = 250
LOCK_OUTER_FOLDS = True
INNER_N_FOLDS = 5
INNER_N_REPEATS = 20
INNER_BASE_SEED = 27
REGRESSION_STRATIFICATION_BINS = 5
CONFIG_SELECTION_RULE = "top_k"
SELECTION_TIE_TOLERANCE = 1e-6
TOP_K_CONFIGS = 20
TOP_K_PLOT_VALUES = [5, 10, 20, 50]
N_BOOTSTRAPS = 1000
N_PERMUTATIONS = 100
MULTIPLICITY_METHOD = "fdr_bh"

# Compatibility names used by inherited V2 code.
FEATURE_INCLUSION_COLUMN = "minimum_include"
CLASSIFICATION_INCLUSION_COLUMN = "minimum_include"
RESUME_CACHE = True
RESUME_PREPROCESSING = True
RESUME_EPOCH_PREPROCESSING = True
FORCE_REBUILD_CLEANED_EEG = False
FORCE_REBUILD_EPOCHS = False
OVERWRITE_FEATURE_CACHE = False
SAVE_SPLIT_LEVEL_PREDICTIONS = False
SAVE_MODEL_MATRICES = False
USE_INNER_CV_TUNING = False
INNER_CV_FOLDS = 5
LOGISTIC_C_GRID = [0.01, 0.1, 1.0, 10.0]
RIDGE_ALPHA_GRID = [0.1, 1.0, 10.0, 100.0]


STAGE_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "1": {
        "resume": True,
        "force_rerun": False,
        "MIN_RECORDING_SECONDS": MIN_RECORDING_SECONDS,
        "RESAMPLE_SFREQ": RESAMPLE_SFREQ,
        "DO_BANDPASS": DO_BANDPASS,
        "BANDPASS_LOW": BANDPASS_LOW,
        "BANDPASS_HIGH": BANDPASS_HIGH,
        "DO_NOTCH": DO_NOTCH,
        "NOTCH_FREQ": NOTCH_FREQ,
        "DO_AVERAGE_REFERENCE": DO_AVERAGE_REFERENCE,
        "BAD_CHANNEL_VAR_LOW": BAD_CHANNEL_VAR_LOW,
        "BAD_CHANNEL_VAR_HIGH_MULT": BAD_CHANNEL_VAR_HIGH_MULT,
        "MAX_BAD_CHANNELS": MAX_BAD_CHANNELS,
        "DO_INTERPOLATE_BAD_CHANNELS": DO_INTERPOLATE_BAD_CHANNELS,
        "DO_ICA": DO_ICA,
        "DO_ECG_REMOVAL": DO_ECG_REMOVAL,
        "ICA_EOG_THRESHOLD": ICA_EOG_THRESHOLD,
        "CPU_USAGE_PERCENT": CPU_USAGE_PERCENT,
    },
    "2": {
        "resume": True,
        "force_rerun": False,
        "CACHE_DURATIONS": CACHE_DURATIONS,
        "REFERENCE_DURATION": REFERENCE_DURATION,
        "ARTIFACT_REJECTION": ARTIFACT_REJECTION,
        "AUTOREJECT_TIMEOUT_SECONDS": AUTOREJECT_TIMEOUT_SECONDS,
        "EPOCH_FLAT_THRESHOLD": EPOCH_FLAT_THRESHOLD,
        "VARIANCE_REJECT_Z": VARIANCE_REJECT_Z,
        "MIN_RECORDING_SECONDS": MIN_RECORDING_SECONDS,
        "CPU_USAGE_PERCENT": CPU_USAGE_PERCENT,
    },
    "3": {
        "resume": True,
        "FEATURE_INCLUSION_COLUMN": FEATURE_INCLUSION_COLUMN,
        "HIGH_EPOCH_REMOVAL_PERCENT": 50.0,
        "VERY_HIGH_EPOCH_REMOVAL_PERCENT": 75.0,
        "LOW_CLEAN_SECONDS": 60.0,
        "NEAR_MINIMUM_CLEAN_SECONDS": 120.0,
    },
    "4": {
        "resume": True,
        "force_rerun": False,
        "FEATURE_INCLUSION_COLUMN": FEATURE_INCLUSION_COLUMN,
        "FEATURE_FAMILIES": FEATURE_FAMILIES,
        "CACHE_DURATIONS": CACHE_DURATIONS,
        "REFERENCE_DURATION": REFERENCE_DURATION,
        "MAX_FEATURE_WORKERS": MAX_FEATURE_WORKERS,
        "CONNECTIVITY_DENSITIES": list(CONNECTIVITY_DENSITIES),
        "CONNECTIVITY_USE_ABSOLUTE_WEIGHTS": CONNECTIVITY_USE_ABSOLUTE_WEIGHTS,
        "USE_GRAPH_METRICS": USE_GRAPH_METRICS,
        "ENTROPY_MAX_SAMPLES": ENTROPY_MAX_SAMPLES,
        "SAMPLE_ENTROPY_MAX_SAMPLES": SAMPLE_ENTROPY_MAX_SAMPLES,
        "HIGUCHI_KMAX": HIGUCHI_KMAX,
        "USE_ADAPTIVE_WELCH": USE_ADAPTIVE_WELCH,
        "SPECPARAM_FREQ_RANGE": list(SPECPARAM_FREQ_RANGE),
        "SPECPARAM_PEAK_WIDTHS": list(SPECPARAM_PEAK_WIDTHS),
        "SPECPARAM_MAX_PEAKS": SPECPARAM_MAX_PEAKS,
        "SPECPARAM_MIN_PEAK_HEIGHT": SPECPARAM_MIN_PEAK_HEIGHT,
        "SPECPARAM_PEAK_THRESHOLD": SPECPARAM_PEAK_THRESHOLD,
        "SPECPARAM_APERIODIC_MODE": SPECPARAM_APERIODIC_MODE,
        "MICROSTATE_N_STATES": MICROSTATE_N_STATES,
        "MICROSTATE_MAX_PEAKS": MICROSTATE_MAX_PEAKS,
        "BANDS": BANDS,
        "MAX_FREQ": MAX_FREQ,
    },
    "5": {
        "resume": True,
        "OUTER_N_FOLDS": OUTER_N_FOLDS,
        "OUTER_FOLD_SEED": OUTER_FOLD_SEED,
        "OUTER_FOLD_CANDIDATE_ASSIGNMENTS": OUTER_FOLD_CANDIDATE_ASSIGNMENTS,
        "LOCK_OUTER_FOLDS": LOCK_OUTER_FOLDS,
        "BALANCE_BR60": True,
        "BALANCE_BR75": True,
        "BALANCE_ORDINAL": True,
        "BALANCE_CONTINUOUS_MOMENTS": True,
        "CONTINUOUS_BALANCE_WEIGHT": 0.25,
    },
    "6": {
        "resume": True,
        "OUTER_N_FOLDS": OUTER_N_FOLDS,
        "INNER_N_FOLDS": INNER_N_FOLDS,
        "INNER_N_REPEATS": INNER_N_REPEATS,
        "INNER_BASE_SEED": INNER_BASE_SEED,
        "REGRESSION_STRATIFICATION_BINS": REGRESSION_STRATIFICATION_BINS,
        "CONFIG_SELECTION_RULE": CONFIG_SELECTION_RULE,
        "SELECTION_TIE_TOLERANCE": SELECTION_TIE_TOLERANCE,
        "TOP_K_CONFIGS": TOP_K_CONFIGS,
        "RUN_BR60": True,
        "RUN_BR75": True,
        "RUN_ORDINAL": True,
        "RUN_CONTINUOUS": True,
        "ORDINAL_SCHEME": ORDINAL_SCHEME,
        "MCMI_SCALE_ORDER": MCMI_SCALE_ORDER,
        "RUN_ALL_255_FAMILY_COMBINATIONS": RUN_ALL_255_FAMILY_COMBINATIONS,
        "EPOCH_DURATION_MODE": EPOCH_DURATION_MODE,
        "BINARY_MODEL_NAME": BINARY_MODEL_NAME,
        "LOGISTIC_C": LOGISTIC_C,
        "RIDGE_ALPHA": RIDGE_ALPHA,
        "CLASS_WEIGHT": CLASS_WEIGHT,
        "MAX_ITER": MAX_ITER,
        "MAX_CLASSIFIER_WORKERS": MAX_CLASSIFIER_WORKERS,
        "NUMERICAL_THREADS_PER_WORKER": NUMERICAL_THREADS_PER_WORKER,
        "CLASSIFIER_PARALLEL_BACKEND": CLASSIFIER_PARALLEL_BACKEND,
        "CPU_USAGE_PERCENT": CPU_USAGE_PERCENT,
        "BRACKET_CUTOFF_1": BRACKET_CUTOFF_1,
        "BRACKET_CUTOFF_2": BRACKET_CUTOFF_2,
        "BRACKET_CUTOFF_3": BRACKET_CUTOFF_3,
    },
    "7": {
        "resume": True,
        "MAKE_PNG": True,
        "MAKE_PDF": True,
        "REPORT_TOP_CONFIGS": 20,
        "TOP_K_PLOT_VALUES": TOP_K_PLOT_VALUES,
        "INCLUDE_FOLD_BALANCE": True,
        "INCLUDE_SELECTION_STABILITY": True,
    },
    "8": {
        "resume": True,
        "ORDINAL_SCHEME": ORDINAL_SCHEME,
        "MAKE_PNG": True,
        "MAKE_PDF": True,
        "N_BOOTSTRAPS": N_BOOTSTRAPS,
        "RUN_BOOTSTRAP": True,
        "PERMUTATION_METHOD": "none",
        "N_PERMUTATIONS": N_PERMUTATIONS,
        "PERMUTATION_K_VALUES": [5, 20],
        "MULTIPLICITY_METHOD": MULTIPLICITY_METHOD,
        "PERMUTE_MCMI_ROWS_TOGETHER": True,
    },

}

GENERAL_DEFAULTS = {
    "BASE_DIR": "",
    "EEG_DIR": "",
    "CLEANED_DIR": "",
    "REPORTS_DIR": "",
    "MCMI_CSV": "",
    "SUBJECT_ID_COLUMN": SUBJECT_ID_COLUMN,
    "EEG_FILE_EXTENSION": EEG_FILE_EXTENSION,
}


def default_settings_payload() -> Dict[str, Any]:
    return {
        "general": deepcopy(GENERAL_DEFAULTS),
        "stages": deepcopy(STAGE_DEFAULTS),
        "inputs": {
            str(stage): {"mode": "auto", "path": ""}
            for stage in STAGE_IDS if stage > 1
        },
        "gui": {"selected_stages": [1, 2, 3, 4, 5, 6, 7, 8], "window_geometry": "1320x850"},
    }


def normalize_settings_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    general = payload.get("general", {})
    payload["general"] = {
        key: deepcopy(general.get(key, value))
        for key, value in GENERAL_DEFAULTS.items()
    }
    stages = payload.setdefault("stages", {})
    old_stage8 = stages.get("8", {})
    if old_stage8.get("RUN_FULL_NESTED_PERMUTATION") and old_stage8.get("PERMUTATION_METHOD", "none") == "none":
        old_stage8["PERMUTATION_METHOD"] = "full_nested"
    payload["stages"] = {
        str(stage): {
            key: deepcopy(stages.get(str(stage), {}).get(key, value))
            for key, value in STAGE_DEFAULTS[str(stage)].items()
        }
        for stage in STAGE_IDS
    }
    payload["inputs"] = {
        str(stage): payload.get("inputs", {}).get(str(stage), {"mode": "auto", "path": ""})
        for stage in STAGE_IDS if stage > 1
    }
    payload.setdefault("gui", {})["selected_stages"] = [
        int(stage) for stage in payload.get("gui", {}).get("selected_stages", list(STAGE_IDS))
        if int(stage) in STAGE_IDS
    ]
    return payload


def _path(value: Any) -> Path:
    text = str(value or "").strip()
    return Path(text).expanduser() if text else Path("")


def active_stage_number(payload: Optional[Dict[str, Any]] = None) -> int:
    payload = payload or globals().get("_EFFECTIVE_SETTINGS", {})
    runtime = payload.get("runtime", {}) if isinstance(payload, dict) else {}
    raw = os.environ.get("NEUROPSYCHE_STAGE_NUMBER", runtime.get("stage", ""))
    try:
        stage = int(raw)
    except (TypeError, ValueError):
        name = Path(sys.argv[0]).name
        stage = int(name.split("_", 1)[0]) if name[:1].isdigit() else 0
    return stage if stage in STAGE_IDS else 0


def effective_settings_for_stage(payload: Dict[str, Any], stage: int) -> Dict[str, Dict[str, Any]]:
    settings = {
        str(source_stage): deepcopy(payload["stages"][str(source_stage)])
        for source_stage in STAGE_SETTING_DEPENDENCIES.get(stage, ())
    }
    for source_stage, keys in STAGE_SHARED_SETTING_KEYS.get(stage, {}).items():
        settings[f"shared_from_stage_{source_stage}"] = {
            key: deepcopy(payload["stages"][str(source_stage)][key])
            for key in keys
        }
    return settings


VALID_ORDINAL_SCHEMES = {
    "4class_original",
    "3class_posthoc",
}


def ordinal_scheme_name() -> str:
    payload = globals().get("_EFFECTIVE_SETTINGS", {})
    active_stage = active_stage_number(payload)
    settings_stage = "8" if active_stage == 8 else "6"
    scheme = str(
        payload.get("stages", {}).get(settings_stage, {}).get(
            "ORDINAL_SCHEME", globals().get("ORDINAL_SCHEME", "4class_original")
        )
    ).strip()

    if scheme not in VALID_ORDINAL_SCHEMES:
        raise ValueError(
            f"Unknown ORDINAL_SCHEME={scheme!r}. "
            "Use '4class_original' or '3class_posthoc'."
        )

    return scheme


def ordinal_cutoffs() -> tuple[float, ...]:
    payload = globals().get("_EFFECTIVE_SETTINGS", {})
    stage6 = payload.get("stages", {}).get("6", {}) if isinstance(payload, dict) else {}
    cutoffs = (
        float(stage6.get("BRACKET_CUTOFF_1", globals().get("BRACKET_CUTOFF_1", 60))),
        float(stage6.get("BRACKET_CUTOFF_2", globals().get("BRACKET_CUTOFF_2", 75))),
        float(stage6.get("BRACKET_CUTOFF_3", globals().get("BRACKET_CUTOFF_3", 85))),
    )

    if ordinal_scheme_name() == "3class_posthoc":
        return cutoffs[:2]

    return cutoffs


def ordinal_labels() -> list[str]:
    cutoffs = ordinal_cutoffs()

    if all(
        float(value).is_integer()
        for value in cutoffs
    ):
        integer_cutoffs = [
            int(value)
            for value in cutoffs
        ]

        labels = [
            f"0-{integer_cutoffs[0] - 1}"
        ]

        labels.extend(
            f"{left}-{right - 1}"
            for left, right in zip(
                integer_cutoffs,
                integer_cutoffs[1:],
            )
        )

        labels.append(
            f"{integer_cutoffs[-1]}+"
        )

        return labels

    labels = [
        f"<{cutoffs[0]:g}"
    ]

    labels.extend(
        f"{left:g}-<{right:g}"
        for left, right in zip(
            cutoffs,
            cutoffs[1:],
        )
    )

    labels.append(
        f"{cutoffs[-1]:g}+"
    )

    return labels


def ordinal_n_classes() -> int:
    return len(ordinal_cutoffs()) + 1


def ordinal_stage_script_name(
    script_name: str,
) -> str:
    return Path(script_name).name


def apply_persisted_settings() -> Dict[str, Any]:
    global BASE_DIR, EEG_DIR, CLEANED_DIR, REPORTS_DIR, MCMI_CSV
    global SUBJECT_ID_COLUMN, EEG_FILE_EXTENSION
    global SCRIPTS_DIR, RUNNER_LOG_DIR
    global RESUME_CACHE, RESUME_PREPROCESSING, RESUME_EPOCH_PREPROCESSING
    global FORCE_REBUILD_CLEANED_EEG, FORCE_REBUILD_EPOCHS, OVERWRITE_FEATURE_CACHE

    payload = default_settings_payload()
    stored = load_json(settings_path(), {}) or {}
    if isinstance(stored, dict):
        from V3_runtime import deep_update
        deep_update(payload, stored)
    normalize_settings_payload(payload)
    globals()["_EFFECTIVE_SETTINGS"] = payload

    general = payload["general"]
    BASE_DIR = _path(general.get("BASE_DIR"))
    EEG_DIR = _path(general.get("EEG_DIR"))
    CLEANED_DIR = _path(general.get("CLEANED_DIR"))
    REPORTS_DIR = _path(general.get("REPORTS_DIR"))
    MCMI_CSV = _path(general.get("MCMI_CSV"))
    SUBJECT_ID_COLUMN = str(general.get("SUBJECT_ID_COLUMN", "subject_id")).strip() or "subject_id"
    EEG_FILE_EXTENSION = str(general.get("EEG_FILE_EXTENSION", ".edf")).strip() or ".edf"
    SCRIPTS_DIR = SCRIPT_DIR
    RUNNER_LOG_DIR = REPORTS_DIR / "_runner_logs" if str(REPORTS_DIR) else Path("")

    active_stage = active_stage_number(payload)
    for stage in STAGE_SETTING_DEPENDENCIES.get(active_stage, ()):
        stage_data = payload["stages"][str(stage)]

        for key, value in stage_data.items():
            if key in {"resume", "force_rerun"}:
                continue

            globals()[key] = value

    for source_stage, keys in STAGE_SHARED_SETTING_KEYS.get(active_stage, {}).items():
        stage_data = payload["stages"][str(source_stage)]
        for key in keys:
            globals()[key] = stage_data[key]

    globals()["BRACKET_LABELS"] = {
        index: label
        for index, label in enumerate(
            ordinal_labels()
        )
    }

    s1 = payload["stages"]["1"]
    s2 = payload["stages"]["2"]
    s4 = payload["stages"]["4"]
    RESUME_PREPROCESSING = bool(s1.get("resume", True))
    FORCE_REBUILD_CLEANED_EEG = bool(s1.get("force_rerun", False))
    RESUME_EPOCH_PREPROCESSING = bool(s2.get("resume", True))
    FORCE_REBUILD_EPOCHS = bool(s2.get("force_rerun", False))
    RESUME_CACHE = bool(s4.get("resume", True))
    OVERWRITE_FEATURE_CACHE = bool(s4.get("force_rerun", False))

    return payload


def runtime_input(name: str, default: Any = None) -> Any:
    payload = globals().get("_EFFECTIVE_SETTINGS", {})
    return payload.get("runtime", {}).get("resolved_inputs", {}).get(name, default)


_EFFECTIVE_SETTINGS = apply_persisted_settings()


def duration_label(duration: float) -> str:
    return str(float(duration)).replace(".", "p") + "s"


def subject_cleaned_folder(subject_id: str) -> Path:
    return CLEANED_DIR / str(subject_id)


def subject_basic_cleaned_folder(subject_id: str) -> Path:
    return subject_cleaned_folder(subject_id) / "BasicCleanedRaw"


def subject_clean_raw_path(subject_id: str) -> Path:
    return subject_basic_cleaned_folder(subject_id) / "clean_raw.fif"


def subject_clean_raw_metadata_path(subject_id: str) -> Path:
    return subject_basic_cleaned_folder(subject_id) / "clean_raw_metadata.json"


def subject_cleaned_epochs_folder(subject_id: str) -> Path:
    return subject_cleaned_folder(subject_id) / "CleanedEpochs"


def subject_epoch_path(subject_id: str, duration: float) -> Path:
    return subject_cleaned_epochs_folder(subject_id) / f"{duration_label(duration)}_epo.fif"


def subject_stitched_clean_raw_path(subject_id: str) -> Path:
    return subject_cleaned_epochs_folder(subject_id) / "stitched_clean_raw.fif"


def subject_epoch_metadata_path(subject_id: str) -> Path:
    return subject_cleaned_epochs_folder(subject_id) / "epoch_preprocessing_metadata.json"


def subject_features_folder(subject_id: str) -> Path:
    return subject_cleaned_folder(subject_id) / "Features"


def subject_preprocessing_qc_path(subject_id: str) -> Path:
    return subject_cleaned_folder(subject_id) / "preprocessing.csv"


def enabled_feature_families() -> list[str]:
    if isinstance(FEATURE_FAMILIES, dict):
        return [name for name in FEATURE_FAMILY_ORDER if bool(FEATURE_FAMILIES.get(name, False))]
    return list(FEATURE_FAMILIES)


def get_n_workers() -> int:
    total = os.cpu_count() or 1
    requested = MAX_WORKERS
    by_percent = max(1, int(total * int(CPU_USAGE_PERCENT) / 100))
    if requested is None:
        return by_percent
    return max(int(MIN_WORKERS), min(int(requested), by_percent))


def make_directories() -> None:
    for path in [CLEANED_DIR, REPORTS_DIR, RUNNER_LOG_DIR]:
        if str(path):
            path.mkdir(parents=True, exist_ok=True)


def settings_hash_from_payload(payload: Dict[str, Any], stage: int) -> str:
    hash_payload = {
        "general": payload["general"],
        "effective_settings": effective_settings_for_stage(payload, stage),
    }

    return stable_hash(
        hash_payload
    )


def settings_hash_for(stage: int) -> str:
    return settings_hash_from_payload(apply_persisted_settings(), stage)


def load_run_settings_snapshot(run_dir: Path, expected_stage: Optional[int] = None) -> Dict[str, Any]:
    run_dir = Path(run_dir)
    path = run_dir / "settings_snapshot.json"
    snapshot = load_json(path, {}) or {}
    if not isinstance(snapshot, dict) or not isinstance(snapshot.get("stages"), dict):
        raise ValueError(f"The selected run has no valid settings snapshot: {path}")
    if expected_stage is not None:
        recorded = snapshot.get("runtime", {}).get("stage")
        if recorded is not None and int(recorded) != int(expected_stage):
            raise ValueError(
                f"The selected run's settings snapshot records Stage {recorded}, not Stage {expected_stage}."
            )
    return snapshot


def payload_with_source_settings(
    payload: Dict[str, Any],
    stage: int,
    source_run: Optional[Path],
) -> Dict[str, Any]:
    result = deepcopy(payload)
    required = SOURCE_SNAPSHOT_STAGES.get(stage, ())
    if not required:
        return result
    if source_run is None:
        raise ValueError(f"Stage {stage} requires an upstream run folder.")
    source_stage = STAGE_INPUT_SOURCE[stage]
    snapshot = load_run_settings_snapshot(source_run, source_stage)
    for settings_stage in required:
        key = str(settings_stage)
        values = snapshot["stages"].get(key)
        if not isinstance(values, dict):
            raise ValueError(
                f"The selected Stage {source_stage} snapshot does not contain Stage {settings_stage} settings."
            )
        result["stages"][key] = deepcopy(values)

    if stage in {7, 8}:
        manifest = load_json(Path(source_run) / "manifest.json", {}) or {}
        retained = manifest.get("top_k_configurations_retained_per_cell")
        if retained is not None:
            result["stages"]["6"]["TOP_K_CONFIGS"] = int(retained)
    return result

STAGE_INPUT_SOURCE = {2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 6}
STAGE_REQUIRED_OUTPUTS = {
    1: [("basic_preprocessing_summary.csv",)],
    2: [("epoch_preprocessing_summary.csv",)],
    3: [("analysis_cohort_V3.csv", "preprocessing.csv")],
    4: [("feature_cache_preflight_V3.csv",)],
    5: [("fixed_outer_folds.csv",)],
    6: [("nested_v3_topk_candidates.csv",)],
}
STAGE_REQUIRED_COLUMNS = {
    1: {"subject_id", "passed"},
    2: {"subject_id", "duration_sec", "passed"},
    3: {"subject_id"},
    4: {"subject_id", "complete_feature_cache"},
    5: {"subject_id", "outer_fold"},
    6: {"outer_fold", "task", "scale", "config_id", "inner_rank"},
}


def _stage_scheme(stage: int, payload: Dict[str, Any]) -> str:
    settings_stage = "8" if stage == 8 else "6"
    return str(payload["stages"][settings_stage].get("ORDINAL_SCHEME", "4class_original"))


def _required_output(run_dir: Path, source_stage: int) -> Path:
    for alternatives in STAGE_REQUIRED_OUTPUTS[source_stage]:
        for name in alternatives:
            path = run_dir / name
            if path.exists():
                return path
    choices = " or ".join(STAGE_REQUIRED_OUTPUTS[source_stage][0])
    raise ValueError(f"Selected Stage {source_stage} folder is missing {choices}: {run_dir}")


def _csv_ordinal_scheme(path: Path) -> str:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            row = next(csv.DictReader(handle), {})
        return str(row.get("ordinal_scheme", ""))
    except Exception:
        return ""


def _csv_columns(path: Path) -> set[str]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return set(next(csv.reader(handle), []))
    except Exception:
        return set()


def _fold_assignment_hash(path: Path) -> str:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [
            {
                "subject_id": str(row["subject_id"]).strip(),
                "outer_fold": int(float(row["outer_fold"])),
            }
            for row in csv.DictReader(handle)
        ]
    rows.sort(key=lambda row: row["subject_id"])
    return stable_hash(rows)


def validate_stage_input(stage: int, run_dir: Path, payload: Optional[Dict[str, Any]] = None) -> Path:
    if stage == 1:
        raise ValueError("Stage 1 reads the project EEG and MCMI locations directly.")
    payload = payload or apply_persisted_settings()
    run_dir = Path(run_dir).expanduser()
    if not run_dir.is_dir():
        raise ValueError(f"Input run folder does not exist: {run_dir}")

    source_stage = STAGE_INPUT_SOURCE[stage]
    required = _required_output(run_dir, source_stage)
    if not (run_dir / "manifest.json").is_file():
        raise ValueError(f"The selected run has no manifest.json: {run_dir}")
    manifest = load_json(run_dir / "manifest.json", {}) or {}
    recorded_stage = manifest.get("stage")
    if recorded_stage is not None and int(recorded_stage) != source_stage:
        raise ValueError(f"Stage {stage} requires a Stage {source_stage} run, not Stage {recorded_stage}.")
    if not manifest_complete(run_dir):
        raise ValueError(f"The selected run is not complete: {run_dir}")

    missing_required_columns = STAGE_REQUIRED_COLUMNS[source_stage] - _csv_columns(required)
    if missing_required_columns:
        raise ValueError(
            f"The selected Stage {source_stage} output is missing columns: "
            f"{sorted(missing_required_columns)}"
        )
    if source_stage == 3:
        inclusion_column = str(payload["stages"]["4"].get("FEATURE_INCLUSION_COLUMN", "minimum_include"))
        if inclusion_column not in _csv_columns(required):
            raise ValueError(
                f"The selected Stage 3 cohort does not contain the Stage 4 inclusion column: "
                f"{inclusion_column}"
            )
    if source_stage == 5:
        recorded_fold_hash = str(manifest.get("fold_assignment_hash", "")).strip()
        if recorded_fold_hash and _fold_assignment_hash(required) != recorded_fold_hash:
            raise ValueError(
                "The selected Stage 5 fold file does not match its manifest."
            )

    if stage in SOURCE_SNAPSHOT_STAGES:
        snapshot = load_run_settings_snapshot(run_dir, source_stage)
        recorded_hash = str(manifest.get("settings_hash", ""))
        if recorded_hash and settings_hash_from_payload(snapshot, source_stage) != recorded_hash:
            raise ValueError(
                "The selected run's settings snapshot does not match its manifest settings hash."
            )
        for settings_stage in SOURCE_SNAPSHOT_STAGES[stage]:
            if not isinstance(snapshot["stages"].get(str(settings_stage)), dict):
                raise ValueError(
                    f"The selected Stage {source_stage} snapshot lacks Stage {settings_stage} settings."
                )
        if stage == 4:
            available = {
                float(value)
                for value in snapshot["stages"]["2"].get("CACHE_DURATIONS", [])
            }
            requested = {
                float(value)
                for value in payload["stages"]["4"].get("CACHE_DURATIONS", [])
            }
            missing_durations = sorted(requested - available)
            if missing_durations:
                raise ValueError(
                    "The selected preprocessing lineage did not create the Stage 4 epoch durations: "
                    f"{missing_durations}"
                )

    if source_stage == 6:
        if stage == 7 and not (run_dir / "nested_v3_all_configuration_results.csv").is_file():
            raise ValueError(
                "The selected Stage 6 run is missing nested_v3_all_configuration_results.csv."
            )
        tasks = [str(task) for task in manifest.get("tasks", [])]
        scales = [str(scale) for scale in manifest.get("scales", [])]
        if not tasks or not scales or int(manifest.get("outer_folds", 0)) < 1:
            raise ValueError(
                "The selected Stage 6 manifest must record outer_folds, tasks, and scales."
            )
        source_scheme = str(manifest.get("ordinal_scheme", "")) or _csv_ordinal_scheme(required)
        desired_scheme = _stage_scheme(stage, payload)
        if "ordinal" in tasks:
            if source_scheme != desired_scheme:
                raise ValueError(
                    f"Stage {stage} is set to {desired_scheme}, but the selected Stage 6 run uses "
                    f"{source_scheme or 'an unknown ordinal scheme'}."
                )
        retained = manifest.get("top_k_configurations_retained_per_cell")
        if retained is None or int(retained) < 1:
            raise ValueError("The selected Stage 6 manifest does not record a valid retained Top K.")
        if stage == 8 and payload["stages"]["8"].get("PERMUTATION_METHOD") == "fixed_candidate":
            requested_values = [int(k) for k in payload["stages"]["8"].get("PERMUTATION_K_VALUES", [])]
            if requested_values and max(requested_values) > int(retained):
                raise ValueError(
                    f"Fixed-candidate permutation requests Top {max(requested_values)}, but the selected "
                    f"Stage 6 run retained only {int(retained)} candidates."
                )
        fold_file = Path(str(manifest.get("fold_file", "")))
        if not fold_file.is_file():
            raise ValueError("The selected Stage 6 run does not point to an existing Stage 5 fold file.")
        missing_fold_columns = {"subject_id", "outer_fold"} - _csv_columns(fold_file)
        if missing_fold_columns:
            raise ValueError(f"The Stage 5 fold file is missing columns: {sorted(missing_fold_columns)}")
        recorded_fold_hash = str(manifest.get("fold_assignment_hash", "")).strip()
        if recorded_fold_hash and _fold_assignment_hash(fold_file) != recorded_fold_hash:
            raise ValueError(
                "The Stage 5 fold file referenced by the selected Stage 6 run no longer "
                "matches the recorded fold assignment."
            )
        recorded_mcmi_hash = str(manifest.get("mcmi_sha256", "")).strip()
        current_mcmi = Path(str(payload["general"].get("MCMI_CSV", ""))).expanduser()
        if (
            recorded_mcmi_hash
            and current_mcmi.is_file()
            and file_sha256(current_mcmi) != recorded_mcmi_hash
        ):
            raise ValueError(
                "The current GUI-selected MCMI CSV does not match the one used by "
                "the selected Stage 6 run."
            )
        if stage == 7 and payload["stages"]["7"].get("INCLUDE_FOLD_BALANCE", True):
            balance_file = fold_file.parent / "fixed_outer_fold_balance.csv"
            if not balance_file.is_file():
                raise ValueError(
                    "Stage 7 is set to include fold balance, but the selected Stage 5 run "
                    "has no fixed_outer_fold_balance.csv."
                )
    return run_dir


def resolve_stage_input(
    stage: int,
    payload: Optional[Dict[str, Any]] = None,
    manual_path: str = "",
) -> Optional[Path]:
    payload = payload or apply_persisted_settings()
    if stage == 1:
        return None
    if manual_path.strip():
        return validate_stage_input(stage, Path(manual_path.strip()), payload)
    source_stage = STAGE_INPUT_SOURCE[stage]
    for candidate in run_dirs(Path(payload["general"]["REPORTS_DIR"]), STAGE_SCRIPTS[source_stage]):
        try:
            return validate_stage_input(stage, candidate, payload)
        except ValueError:
            continue
    raise ValueError(f"No compatible completed Stage {source_stage} run was found.")


def resolved_inputs_for_stage(stage: int, source_run: Optional[Path]) -> Dict[str, str]:
    if stage == 1 or source_run is None:
        return {}
    source_run = Path(source_run)
    manifest = load_json(source_run / "manifest.json", {}) or {}
    inputs = manifest.get("inputs", {}) if isinstance(manifest.get("inputs"), dict) else {}

    if stage == 2:
        return {"source_run": str(source_run), "basic_summary": str(source_run / "basic_preprocessing_summary.csv")}
    if stage == 3:
        basic = inputs.get("basic_summary") or inputs.get("basic")
        if not basic:
            raise ValueError("The selected Stage 2 manifest does not record its Stage 1 summary.")
        basic_path = Path(str(basic))
        if not basic_path.is_file():
            raise ValueError(f"The Stage 1 summary recorded by the selected run is missing: {basic_path}")
        missing = {"subject_id", "passed"} - _csv_columns(basic_path)
        if missing:
            raise ValueError(f"The recorded Stage 1 summary is missing columns: {sorted(missing)}")
        return {
            "source_run": str(source_run),
            "basic_summary": str(basic_path),
            "epoch_summary": str(source_run / "epoch_preprocessing_summary.csv"),
        }
    if stage == 4:
        cohort = source_run / "analysis_cohort_V3.csv"
        if not cohort.exists():
            cohort = source_run / "preprocessing.csv"
        return {"source_run": str(source_run), "qc_cohort": str(cohort)}
    if stage == 5:
        cohort = inputs.get("qc_cohort") or inputs.get("preprocessing_csv")
        if not cohort:
            raise ValueError("The selected Stage 4 manifest does not record its Stage 3 cohort.")
        cohort_path = Path(str(cohort))
        if not cohort_path.is_file():
            raise ValueError(f"The Stage 3 cohort recorded by the selected run is missing: {cohort_path}")
        if "subject_id" not in _csv_columns(cohort_path):
            raise ValueError("The recorded Stage 3 cohort is missing the subject_id column.")
        return {
            "source_run": str(source_run),
            "feature_preflight": str(source_run / "feature_cache_preflight_V3.csv"),
            "qc_cohort": str(cohort_path),
        }
    if stage == 6:
        return {"source_run": str(source_run), "fold_file": str(source_run / "fixed_outer_folds.csv")}
    if stage in {7, 8}:
        fold_file = manifest.get("fold_file")
        if not fold_file:
            raise ValueError("The selected Stage 6 manifest does not record its Stage 5 fold file.")
        return {"source_run": str(source_run), "stage6_run": str(source_run), "fold_file": str(fold_file)}
    return {"source_run": str(source_run)}


def find_latest_stage_input(stage: int) -> str:
    payload = apply_persisted_settings()
    if stage == 1:
        return f"EEG: {EEG_DIR} | MCMI: {MCMI_CSV}"
    try:
        return str(resolve_stage_input(stage, payload))
    except ValueError as error:
        return str(error)


def validate_general_paths() -> list[str]:
    payload = load_json(settings_path(), {}) or {}
    general = payload.get("general", {}) if isinstance(payload, dict) else {}
    problems = []
    required = {
        "EEG_DIR": "EEG directory",
        "CLEANED_DIR": "Cleaned-data directory",
        "REPORTS_DIR": "Reports directory",
        "MCMI_CSV": "MCMI CSV",
    }
    for key, label in required.items():
        if not str(general.get(key, "")).strip():
            problems.append(f"{label} is not configured.")
    eeg = Path(str(general.get("EEG_DIR", ""))).expanduser()
    mcmi = Path(str(general.get("MCMI_CSV", ""))).expanduser()
    if str(general.get("EEG_DIR", "")).strip() and not eeg.is_dir():
        problems.append(f"EEG directory does not exist: {eeg}")
    if str(general.get("MCMI_CSV", "")).strip() and not mcmi.is_file():
        problems.append(f"MCMI CSV does not exist: {mcmi}")
    for key, label in (("CLEANED_DIR", "Cleaned-data directory"), ("REPORTS_DIR", "Reports directory")):
        raw = str(general.get(key, "")).strip()
        if not raw:
            continue
        path = Path(raw).expanduser()
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            problems.append(f"{label} cannot be created: {error}")
    return problems


def check_basic_paths() -> bool:
    problems = validate_general_paths()
    if str(MCMI_CSV) and not MCMI_CSV.exists():
        problems.append(f"MCMI CSV does not exist: {MCMI_CSV}")
    if str(EEG_DIR) and not EEG_DIR.exists():
        problems.append(f"EEG directory does not exist: {EEG_DIR}")
    for problem in problems:
        print(f"[CONFIG] {problem}")
    return not problems


def validate_config() -> None:
    problems = validate_general_paths()
    if int(OUTER_N_FOLDS) != 5:
        problems.append("V3 primary validation requires five outer folds.")
    if int(INNER_N_FOLDS) != 5:
        problems.append("V3 primary validation requires five inner folds.")
    if int(INNER_N_REPEATS) < 1:
        problems.append("INNER_N_REPEATS must be at least 1.")
    if int(TOP_K_CONFIGS) < 1:
        problems.append(
            "TOP_K_CONFIGS must be at least 1."
        )

    try:
        cutoffs = ordinal_cutoffs()

        if any(
            left >= right
            for left, right in zip(
                cutoffs,
                cutoffs[1:],
            )
        ):
            problems.append(
                "The active ordinal cutoffs "
                "must be strictly increasing."
            )

    except ValueError as error:
        problems.append(
            str(error)
        )

    if problems:
        raise ValueError(
            "\n".join(problems)
        )


def print_config_summary() -> None:
    apply_persisted_settings()
    print("\n" + "=" * 70)
    print("MAPPER configuration")
    print("=" * 70)
    print(f"EEG directory: {EEG_DIR}")
    print(f"Cleaned directory: {CLEANED_DIR}")
    print(f"Reports directory: {REPORTS_DIR}")
    print(f"MCMI CSV: {MCMI_CSV}")
    print(f"Outer folds: {OUTER_N_FOLDS}")
    print(f"Inner CV: {INNER_N_FOLDS}-fold x {INNER_N_REPEATS} repeats")
    print(f"Top configurations retained per task/scale: {TOP_K_CONFIGS}")
    print("=" * 70 + "\n")


# GUI presentation metadata

STAGE_DESCRIPTIONS = {
    1: "Clean raw EEG recordings and save one continuous FIF file per subject.",
    2: "Create and clean fixed-duration epochs from the basic-cleaned EEG.",
    3: "Combine preprocessing reports, apply quality rules, and freeze the analysis cohort.",
    4: "Extract the eight predefined EEG feature families and cache them per subject.",
    5: "Create or reuse one locked five-fold subject assignment. Keep this assignment unchanged when switching between the original four-class and post-hoc three-class ordinal analyses.",
    6: "Screen every EEG feature configuration inside each outer-training sample. The ordinal task uses the original four-class or post-hoc three-class grouping selected below, with separate resumable output branches.",
    7: "Read the matching Stage-6 ordinal branch, summarize the complete configuration search, and create Top-K stability plots, including descriptive Top-50 curves.",
    8: "Choose a compatible three- or four-class Stage-6 run, evaluate its retained configurations on untouched outer folds, and run bootstrap or permutation inference.",
}

STAGE_SECTIONS = {
    1: [
        ("Recording requirements", ["MIN_RECORDING_SECONDS", "RESAMPLE_SFREQ"]),
        ("Filtering and reference", ["DO_BANDPASS", "BANDPASS_LOW", "BANDPASS_HIGH", "DO_NOTCH", "NOTCH_FREQ", "DO_AVERAGE_REFERENCE"]),
        ("Bad-channel handling", ["BAD_CHANNEL_VAR_LOW", "BAD_CHANNEL_VAR_HIGH_MULT", "MAX_BAD_CHANNELS", "DO_INTERPOLATE_BAD_CHANNELS"]),
        ("ICA artifact removal", ["DO_ICA", "DO_ECG_REMOVAL", "ICA_EOG_THRESHOLD"]),
        ("Compute", ["CPU_USAGE_PERCENT"]),
    ],
    2: [
        ("Epoch design", ["CACHE_DURATIONS", "REFERENCE_DURATION", "MIN_RECORDING_SECONDS"]),
        ("Artifact rejection", ["ARTIFACT_REJECTION", "AUTOREJECT_TIMEOUT_SECONDS", "EPOCH_FLAT_THRESHOLD", "VARIANCE_REJECT_Z"]),
        ("Compute", ["CPU_USAGE_PERCENT"]),
    ],
    3: [
        ("Cohort rule", ["FEATURE_INCLUSION_COLUMN"]),
        ("Quality-warning thresholds", ["HIGH_EPOCH_REMOVAL_PERCENT", "VERY_HIGH_EPOCH_REMOVAL_PERCENT", "LOW_CLEAN_SECONDS", "NEAR_MINIMUM_CLEAN_SECONDS"]),
    ],
    4: [
        ("Subjects and feature families", ["FEATURE_INCLUSION_COLUMN", "FEATURE_FAMILIES"]),
        ("Epoch durations", ["CACHE_DURATIONS", "REFERENCE_DURATION"]),
        ("Frequency bands", ["BANDS", "MAX_FREQ"]),
        ("Connectivity", ["CONNECTIVITY_DENSITIES", "CONNECTIVITY_USE_ABSOLUTE_WEIGHTS", "USE_GRAPH_METRICS"]),
        ("Entropy and complexity", ["ENTROPY_MAX_SAMPLES", "SAMPLE_ENTROPY_MAX_SAMPLES", "HIGUCHI_KMAX"]),
        ("Spectral parameterization", ["USE_ADAPTIVE_WELCH", "SPECPARAM_FREQ_RANGE", "SPECPARAM_PEAK_WIDTHS", "SPECPARAM_MAX_PEAKS", "SPECPARAM_MIN_PEAK_HEIGHT", "SPECPARAM_PEAK_THRESHOLD", "SPECPARAM_APERIODIC_MODE"]),
        ("Microstates", ["MICROSTATE_N_STATES", "MICROSTATE_MAX_PEAKS"]),
        ("Compute", ["MAX_FEATURE_WORKERS"]),
    ],
    5: [
        ("Outer-fold design", ["OUTER_N_FOLDS", "OUTER_FOLD_SEED", "OUTER_FOLD_CANDIDATE_ASSIGNMENTS", "LOCK_OUTER_FOLDS"]),
        ("Balancing targets", ["BALANCE_BR60", "BALANCE_BR75", "BALANCE_ORDINAL", "BALANCE_CONTINUOUS_MOMENTS", "CONTINUOUS_BALANCE_WEIGHT"]),
    ],
    6: [
        ("Nested validation", ["OUTER_N_FOLDS", "INNER_N_FOLDS", "INNER_N_REPEATS", "INNER_BASE_SEED", "REGRESSION_STRATIFICATION_BINS"]),
        ("Tasks and scales", ["RUN_BR60", "RUN_BR75", "RUN_ORDINAL", "RUN_CONTINUOUS", "MCMI_SCALE_ORDER"]),
        ("Configuration search", ["CONFIG_SELECTION_RULE", "TOP_K_CONFIGS", "SELECTION_TIE_TOLERANCE", "RUN_ALL_255_FAMILY_COMBINATIONS", "EPOCH_DURATION_MODE"]),
        ("Models", ["BINARY_MODEL_NAME", "LOGISTIC_C", "RIDGE_ALPHA", "CLASS_WEIGHT", "MAX_ITER"]),
        ("Ordinal outcome",["ORDINAL_SCHEME","BRACKET_CUTOFF_1","BRACKET_CUTOFF_2","BRACKET_CUTOFF_3",],),
        ("Compute", ["MAX_CLASSIFIER_WORKERS", "NUMERICAL_THREADS_PER_WORKER", "CLASSIFIER_PARALLEL_BACKEND", "CPU_USAGE_PERCENT"]),
    ],
    7: [
        ("Report contents", ["MAKE_PNG", "MAKE_PDF", "REPORT_TOP_CONFIGS", "TOP_K_PLOT_VALUES", "INCLUDE_FOLD_BALANCE", "INCLUDE_SELECTION_STABILITY"]),
    ],
    8: [
        ("Ordinal source", ["ORDINAL_SCHEME"]),
        ("Figures", ["MAKE_PNG", "MAKE_PDF"]),
        ("Bootstrap", ["RUN_BOOTSTRAP", "N_BOOTSTRAPS"]),
        ("Permutation", ["PERMUTATION_METHOD", "N_PERMUTATIONS", "PERMUTATION_K_VALUES", "PERMUTE_MCMI_ROWS_TOGETHER"]),
        ("Multiplicity", ["MULTIPLICITY_METHOD"]),
    ],
}

FIELD_LABELS = {
    "MIN_RECORDING_SECONDS": "Minimum usable recording (seconds)",
    "RESAMPLE_SFREQ": "Resample frequency (Hz; blank = keep original)",
    "DO_BANDPASS": "Apply band-pass filter",
    "BANDPASS_LOW": "Band-pass lower edge (Hz)",
    "BANDPASS_HIGH": "Band-pass upper edge (Hz)",
    "DO_NOTCH": "Apply mains notch filter",
    "NOTCH_FREQ": "Notch frequency (Hz)",
    "DO_AVERAGE_REFERENCE": "Apply average reference",
    "BAD_CHANNEL_VAR_LOW": "Flat-channel variance threshold",
    "BAD_CHANNEL_VAR_HIGH_MULT": "Noisy-channel variance multiplier",
    "MAX_BAD_CHANNELS": "Maximum bad channels allowed",
    "DO_INTERPOLATE_BAD_CHANNELS": "Interpolate accepted bad channels",
    "DO_ICA": "Run ICA artifact removal",
    "DO_ECG_REMOVAL": "Attempt ECG component removal",
    "ICA_EOG_THRESHOLD": "ICA blink-detection threshold",
    "CPU_USAGE_PERCENT": "Maximum CPU use (%)",
    "CACHE_DURATIONS": "Epoch durations (seconds)",
    "REFERENCE_DURATION": "Reference epoch duration (seconds)",
    "ARTIFACT_REJECTION": "Artifact-rejection method",
    "AUTOREJECT_TIMEOUT_SECONDS": "AutoReject timeout (seconds)",
    "EPOCH_FLAT_THRESHOLD": "Flat-epoch threshold (volts)",
    "VARIANCE_REJECT_Z": "Variance fallback cutoff (MAD units)",
    "FEATURE_INCLUSION_COLUMN": "Subject inclusion rule",
    "HIGH_EPOCH_REMOVAL_PERCENT": "High epoch-removal warning (%)",
    "VERY_HIGH_EPOCH_REMOVAL_PERCENT": "Very-high epoch-removal warning (%)",
    "LOW_CLEAN_SECONDS": "Low clean-duration warning (seconds)",
    "NEAR_MINIMUM_CLEAN_SECONDS": "Near-minimum clean-duration warning (seconds)",
    "FEATURE_FAMILIES": "Feature families to extract",
    "MAX_FEATURE_WORKERS": "Maximum feature-extraction workers",
    "CONNECTIVITY_DENSITIES": "Graph densities",
    "CONNECTIVITY_USE_ABSOLUTE_WEIGHTS": "Use absolute connectivity weights",
    "USE_GRAPH_METRICS": "Calculate graph metrics",
    "ENTROPY_MAX_SAMPLES": "Maximum samples for entropy",
    "SAMPLE_ENTROPY_MAX_SAMPLES": "Maximum samples for sample entropy",
    "HIGUCHI_KMAX": "Higuchi fractal dimension k-max",
    "USE_ADAPTIVE_WELCH": "Use adaptive Welch windows",
    "SPECPARAM_FREQ_RANGE": "Specparam frequency range (Hz)",
    "SPECPARAM_PEAK_WIDTHS": "Specparam peak-width range (Hz)",
    "SPECPARAM_MAX_PEAKS": "Maximum spectral peaks",
    "SPECPARAM_MIN_PEAK_HEIGHT": "Minimum peak height",
    "SPECPARAM_PEAK_THRESHOLD": "Peak-detection threshold",
    "SPECPARAM_APERIODIC_MODE": "Aperiodic model",
    "MICROSTATE_N_STATES": "Number of microstate classes",
    "MICROSTATE_MAX_PEAKS": "Maximum GFP peaks for microstates",
    "BANDS": "Frequency-band definitions",
    "MAX_FREQ": "Maximum analyzed frequency (Hz)",
    "OUTER_N_FOLDS": "Number of fixed outer folds",
    "OUTER_FOLD_SEED": "Outer-fold random seed",
    "OUTER_FOLD_CANDIDATE_ASSIGNMENTS": "Candidate fold assignments to compare",
    "LOCK_OUTER_FOLDS": "Lock the chosen fold assignment",
    "BALANCE_BR60": "Balance BR60 labels",
    "BALANCE_BR75": "Balance BR75 labels",
    "BALANCE_ORDINAL": "Balance four ordinal brackets",
    "BALANCE_CONTINUOUS_MOMENTS": "Balance continuous score distributions",
    "CONTINUOUS_BALANCE_WEIGHT": "Weight of continuous-score balance",
    "INNER_N_FOLDS": "Inner CV folds",
    "INNER_N_REPEATS": "Inner 5-fold repetitions",
    "INNER_BASE_SEED": "Inner-CV base random seed",
    "REGRESSION_STRATIFICATION_BINS": "Continuous-task stratification bins",
    "CONFIG_SELECTION_RULE": "Configuration-selection rule",
    "SELECTION_TIE_TOLERANCE": "Tie tolerance for configuration ranking",
    "TOP_K_CONFIGS": "Configurations retained per task and scale",
    "RUN_BR60": "Run binary BR60 task",
    "RUN_BR75": "Run binary BR75 task",
    "RUN_ORDINAL": "Run ordinal task",
    "RUN_CONTINUOUS": "Run continuous BR task",
    "MCMI_SCALE_ORDER": "MCMI scales to analyze",
    "RUN_ALL_255_FAMILY_COMBINATIONS": "Run all non-empty family combinations",
    "EPOCH_DURATION_MODE": "Epoch-duration search mode",
    "BINARY_MODEL_NAME": "Binary classifier",
    "LOGISTIC_C": "Binary-model C value",
    "RIDGE_ALPHA": "Ridge alpha",
    "CLASS_WEIGHT": "Binary class weighting",
    "MAX_ITER": "Maximum model iterations",
    "MAX_CLASSIFIER_WORKERS": "Maximum classifier workers",
    "NUMERICAL_THREADS_PER_WORKER": "Numerical threads per worker",
    "CLASSIFIER_PARALLEL_BACKEND": "Parallel backend",
    "BRACKET_CUTOFF_1": "Ordinal cutoff 1",
    "BRACKET_CUTOFF_2": "Ordinal cutoff 2",
    "BRACKET_CUTOFF_3": "Ordinal cutoff 3",
    "ORDINAL_SCHEME": "Ordinal grouping",
    "MAKE_PNG": "Create PNG figures",
    "MAKE_PDF": "Create PDF figures",
    "REPORT_TOP_CONFIGS": "Top configurations listed in report tables",
    "TOP_K_PLOT_VALUES": "Top-k values shown in stability plots",
    "INCLUDE_FOLD_BALANCE": "Include fold-balance report",
    "INCLUDE_SELECTION_STABILITY": "Include selection-stability report",
    "N_BOOTSTRAPS": "Bootstrap samples",
    "RUN_BOOTSTRAP": "Run subject bootstrap",
    "PERMUTATION_METHOD": "Permutation method",
    "N_PERMUTATIONS": "Permutation count",
    "PERMUTATION_K_VALUES": "Permutation Top-k values",
    "MULTIPLICITY_METHOD": "Multiple-testing correction",
    "PERMUTE_MCMI_ROWS_TOGETHER": "Permute each subject's MCMI profile together",
}

FIELD_HELP = {
    "MIN_RECORDING_SECONDS": "Subjects or subject-duration files with less usable EEG than this are rejected. The replication default is 30 seconds.",
    "RESAMPLE_SFREQ": "Leave blank to preserve the original sampling rate. Entering a value resamples every recording and changes temporal resolution, file size, and processing time.",
    "DO_BANDPASS": "Keeps only frequencies between the lower and upper edges. Turn this off only for a planned sensitivity analysis.",
    "BANDPASS_LOW": "Frequencies below this value are attenuated. The replication default is 0.5 Hz.",
    "BANDPASS_HIGH": "Frequencies above this value are attenuated. The replication default is 30 Hz, so gamma is excluded.",
    "DO_NOTCH": "Removes narrow-band electrical mains interference around the selected notch frequency.",
    "NOTCH_FREQ": "Use the local mains frequency. The replication setting is 50 Hz.",
    "DO_AVERAGE_REFERENCE": "Re-references every channel to the average of the EEG channels. This is part of the replication pipeline.",
    "BAD_CHANNEL_VAR_LOW": "Channels below this variance are treated as nearly flat. Raising it marks more channels as flat.",
    "BAD_CHANNEL_VAR_HIGH_MULT": "Channels above this multiple of the median channel variance are treated as noisy. Lower values are stricter.",
    "MAX_BAD_CHANNELS": "A subject fails basic preprocessing when more channels than this are marked bad.",
    "DO_INTERPOLATE_BAD_CHANNELS": "Reconstructs accepted bad channels from neighboring electrodes. It does not rescue subjects exceeding the maximum bad-channel count.",
    "DO_ICA": "Fits ICA to identify eye-blink and other artifact components. Turning it off changes the preprocessing method.",
    "DO_ECG_REMOVAL": "Attempts to identify and remove cardiac components when ICA is enabled.",
    "ICA_EOG_THRESHOLD": "Controls how strongly a component must resemble the frontal blink proxy. Lower values can remove more components.",
    "CPU_USAGE_PERCENT": "Caps worker selection as a percentage of available logical CPU cores. Lower values reduce load but increase runtime.",
    "CACHE_DURATIONS": "Comma-separated fixed epoch lengths. The replication analysis uses 1, 2, 3, 4, and 5 seconds.",
    "REFERENCE_DURATION": "The duration used for the stitched clean signal and reference-duration operations. It should be one of the cached durations.",
    "ARTIFACT_REJECTION": "Hybrid first attempts AutoReject and then uses the robust variance fallback if AutoReject fails or times out. This is the replication setting.",
    "AUTOREJECT_TIMEOUT_SECONDS": "Maximum time allowed for AutoReject for one subject-duration before the worker is terminated and fallback cleaning is used.",
    "EPOCH_FLAT_THRESHOLD": "Epochs with EEG amplitudes below this flat threshold are removed. The value is stored in volts.",
    "VARIANCE_REJECT_Z": "Fallback cleaning keeps epochs within this many robust MAD units of the median log-variance. Lower values reject more epochs.",
    "FEATURE_INCLUSION_COLUMN": "Chooses which QC inclusion flag defines the cohort. Minimum inclusion reproduces the main analysis; strict options are sensitivity analyses.",
    "HIGH_EPOCH_REMOVAL_PERCENT": "Creates a quality warning when at least this percentage of epochs was removed.",
    "VERY_HIGH_EPOCH_REMOVAL_PERCENT": "Creates a stronger quality warning at this removal percentage.",
    "LOW_CLEAN_SECONDS": "Flags subject-duration files with less clean data than this, while still above the hard minimum.",
    "NEAR_MINIMUM_CLEAN_SECONDS": "Marks recordings whose surviving clean duration is close to the minimum requirement.",
    "FEATURE_FAMILIES": "Select the feature families that will be cached. All eight must be enabled to reproduce the full 1,215-configuration search.",
    "MAX_FEATURE_WORKERS": "Maximum number of subjects processed in parallel during feature extraction. Reduce this if memory use is excessive.",
    "CONNECTIVITY_DENSITIES": "Comma-separated proportional graph densities used when thresholding connectivity networks.",
    "CONNECTIVITY_USE_ABSOLUTE_WEIGHTS": "Uses connection magnitude regardless of sign when creating connectivity summaries and graphs.",
    "USE_GRAPH_METRICS": "Adds graph-theory summaries to the connectivity family. Turning it off changes the feature space.",
    "ENTROPY_MAX_SAMPLES": "Maximum continuous samples used by the less expensive entropy measures. Larger values use more data and memory.",
    "SAMPLE_ENTROPY_MAX_SAMPLES": "Separate lower cap for sample entropy, which is computationally expensive.",
    "HIGUCHI_KMAX": "Maximum scale parameter for Higuchi fractal dimension. Higher values increase computation and change the estimate.",
    "USE_ADAPTIVE_WELCH": "Keeps the two-second Welch maximum but shortens the window to about one quarter of the available samples, preserving multiple segments. The replication setting is off.",
    "SPECPARAM_FREQ_RANGE": "Two numeric values defining the frequency range fitted by specparam.",
    "SPECPARAM_PEAK_WIDTHS": "Two numeric values defining the minimum and maximum oscillatory peak widths accepted by specparam.",
    "SPECPARAM_MAX_PEAKS": "Maximum number of oscillatory peaks specparam may fit per spectrum.",
    "SPECPARAM_MIN_PEAK_HEIGHT": "Minimum peak height required for an oscillatory peak.",
    "SPECPARAM_PEAK_THRESHOLD": "Relative threshold used to identify spectral peaks. Higher values are more conservative.",
    "SPECPARAM_APERIODIC_MODE": "Fixed fits a straight aperiodic component in log-log space; knee allows a bend. Fixed reproduces the project analysis.",
    "MICROSTATE_N_STATES": "Number of microstate topographies clustered. Four states reproduce the project analysis.",
    "MICROSTATE_MAX_PEAKS": "Maximum GFP peaks used for microstate clustering. Lower values reduce runtime.",
    "BANDS": "Edit the low and high edge of each named band. Changing these boundaries changes band-power, connectivity, and asymmetry features.",
    "MAX_FREQ": "Upper frequency limit used by applicable feature calculations. It should be consistent with the band-pass upper edge.",
    "OUTER_N_FOLDS": "The primary V3 design is fixed at five outer folds. Each fold is hidden once for unbiased outer evaluation.",
    "OUTER_FOLD_SEED": "Makes the candidate fold assignments reproducible. Changing the seed creates a different set of candidate assignments.",
    "OUTER_FOLD_CANDIDATE_ASSIGNMENTS": "Number of candidate five-fold assignments scored for balance before the best one is locked. More candidates can improve balance but take slightly longer.",
    "LOCK_OUTER_FOLDS": "Prevents accidental replacement of an existing fold assignment for the same cohort.",
    "BALANCE_BR60": "Includes all 15 BR60 binary labels in the fold-balance objective.",
    "BALANCE_BR75": "Includes all 15 BR75 binary labels in the fold-balance objective.",
    "BALANCE_ORDINAL": "Includes the four-category label distribution for all 15 scales in the fold-balance objective.",
    "BALANCE_CONTINUOUS_MOMENTS": "Also penalizes differences in continuous BR means and standard deviations between folds.",
    "CONTINUOUS_BALANCE_WEIGHT": "Relative importance of continuous-score balance compared with categorical label balance. Higher values emphasize means and spreads more strongly.",
    "INNER_N_FOLDS": "Number of folds used inside each outer-training sample. Five is the approved design.",
    "INNER_N_REPEATS": "Number of repeated inner five-fold partitions. Twenty repetitions produce 100 inner validation splits per outer fold.",
    "INNER_BASE_SEED": "Base seed from which reproducible inner-CV repetitions are generated.",
    "REGRESSION_STRATIFICATION_BINS": "Continuous scores are binned only to distribute them across inner folds. The model still predicts the original continuous scores.",
    "CONFIG_SELECTION_RULE": "Ranks every configuration using inner-CV performance only and retains the prespecified Top K for Stage 8. Outer-test performance is never used to reorder candidates.",
    "SELECTION_TIE_TOLERANCE": "Scores within this tiny distance are treated as tied so deterministic secondary rules can break the tie.",
    "TOP_K_CONFIGS": "Number of inner-ranked configurations retained separately for every outer fold, task, and MCMI scale. Stage 8 evaluates all of them on the untouched outer fold. The replication default is 20.",
    "RUN_BR60": "Runs the broad-elevation binary task BR >= 60.",
    "RUN_BR75": "Runs the clinically elevated binary task BR >= 75.",
    "RUN_ORDINAL": (
        "Runs the ordinal task using "
        "the grouping selected below."
    ),
    "RUN_CONTINUOUS": "Runs ridge regression on the original numerical BR scores.",
    "MCMI_SCALE_ORDER": "Use all scales to reproduce the project. Deselect scales only for an explicitly prespecified subset analysis.",
    "RUN_ALL_255_FAMILY_COMBINATIONS": "Builds every non-empty combination of the eight feature families. Together with duration variants this produces 1,215 configurations.",
    "EPOCH_DURATION_MODE": "All evaluates epoch-based configurations at every cached duration; reference evaluates them only at the reference duration.",
    "BINARY_MODEL_NAME": "Linear SVM reproduces the primary BR60/BR75 analysis. L2 logistic regression is the supported probability-model alternative.",
    "LOGISTIC_C": "Regularization parameter used by the selected binary linear model. Larger values mean weaker regularization.",
    "RIDGE_ALPHA": "Regularization strength for continuous ridge regression. Larger values shrink coefficients more strongly.",
    "CLASS_WEIGHT": "Balanced weights each binary class inversely to its frequency. This is important for rare BR75-positive groups.",
    "MAX_ITER": "Maximum optimization iterations before a model is treated as non-converged.",
    "MAX_CLASSIFIER_WORKERS": "Maximum parallel workers during nested modeling. Reduce it when memory pressure or disk contention occurs.",
    "NUMERICAL_THREADS_PER_WORKER": "Threads allowed inside each worker. One avoids CPU oversubscription when several workers run in parallel.",
    "CLASSIFIER_PARALLEL_BACKEND": "Loky uses separate processes and is the safer replication choice. Threading shares memory but may behave differently with numerical libraries.",
    "BRACKET_CUTOFF_1": "Boundary between ordinal brackets 0 and 1. The replication cutoff is 60.",
    "BRACKET_CUTOFF_2": "Boundary between ordinal brackets 1 and 2 and the BR75 binary threshold. The replication cutoff is 75.",
    "BRACKET_CUTOFF_3": (
        "Boundary between ordinal brackets 2 and 3 "
        "in the original four-class analysis. "
        "It is ignored by the post-hoc "
        "three-class analysis."
    ),
    "ORDINAL_SCHEME": (
        "The four-class option preserves the original analysis. "
        "The three-class option is the explicitly post-hoc "
        "sensitivity analysis that merges 75-84 and 85+ "
        "into one 75+ class."
    ),
    "MAKE_PNG": "Saves raster figures suitable for quick viewing and presentations.",
    "MAKE_PDF": "Saves vector PDF versions of figures for manuscripts and high-quality printing.",
    "REPORT_TOP_CONFIGS": "Number of highest-ranked configurations included in detailed descriptive tables. This does not change the Top K sent to Stage 8.",
    "TOP_K_PLOT_VALUES": "Comma-separated k values used in plots like Top 5, Top 10, Top 20, and Top 50. These curves are calculated from the complete inner-CV screening results and show whether performance extends beyond one lucky configuration.",
    "INCLUDE_FOLD_BALANCE": "Adds tables showing the subject and outcome distributions in each fixed outer fold.",
    "INCLUDE_SELECTION_STABILITY": "Reports how often configurations, feature families, and durations were selected across outer folds.",
    "N_BOOTSTRAPS": "Number of subject-level resamples used for confidence intervals. One thousand is the replication default.",
    "RUN_BOOTSTRAP": "Calculates confidence intervals from saved outer-test predictions without refitting the nested models.",
    "PERMUTATION_METHOD": "Fixed-candidate permutation refits the configurations and ranks selected in the observed Stage 6 run. Full nested permutation also repeats Stage 6 selection under every shuffle. Choose None to skip permutation inference.",
    "N_PERMUTATIONS": "Exact number of label permutations to run. No hidden cap is applied.",
    "PERMUTATION_K_VALUES": "Top-k widths evaluated under fixed-candidate permutation. Every value must be no larger than the candidates retained by Stage 6.",
    "MULTIPLICITY_METHOD": "Benjamini-Hochberg FDR controls the expected proportion of false discoveries across the tested scales within each task.",
    "PERMUTE_MCMI_ROWS_TOGETHER": "Moves each subject's complete 15-scale profile as one unit, preserving correlations among MCMI scales under permutation.",
}

CHOICE_OPTIONS = {
    "ARTIFACT_REJECTION": [
        ("Hybrid: AutoReject + variance fallback (replication)", "hybrid"),
        ("Variance fallback only", "variance"),
        ("No artifact rejection", "none"),
    ],
    "FEATURE_INCLUSION_COLUMN": [
        ("Minimum inclusion (primary analysis)", "minimum_include"),
        ("Strict quality inclusion", "strict_include_quality"),
        ("Strict method inclusion", "strict_include_method"),
    ],
    "SPECPARAM_APERIODIC_MODE": [
        ("Fixed (replication)", "fixed"),
        ("Knee", "knee"),
    ],
    "CONFIG_SELECTION_RULE": [
        ("Rank all configurations and retain Top K (replication)", "top_k"),
    ],
    "EPOCH_DURATION_MODE": [
        ("All cached durations (replication)", "all"),
        ("Reference duration only", "reference"),
    ],
    "BINARY_MODEL_NAME": [
        ("Linear SVM / SVC (replication)", "linear_svm_svc"),
        ("L2 logistic regression", "l2_logistic"),
    ],
    "CLASS_WEIGHT": [
        ("Balanced (replication)", "balanced"),
        ("No class weighting", None),
    ],
    "CLASSIFIER_PARALLEL_BACKEND": [
        ("Loky processes (recommended)", "loky"),
        ("Threads", "threading"),
    ],
    "ORDINAL_SCHEME": [
        (
            "4 classes — original analysis",
            "4class_original",
        ),
        (
            "3 classes — post-hoc merged 75+",
            "3class_posthoc",
        ),
    ],
    "PERMUTATION_METHOD": [
        ("None", "none"),
        ("Fixed candidate (manuscript method)", "fixed_candidate"),
        ("Full nested selection-aware", "full_nested"),
    ],
    "MULTIPLICITY_METHOD": [
        ("Benjamini-Hochberg FDR", "fdr_bh"),
    ],
}

FEATURE_HELP = {
    "BAND_POWER": "Absolute and relative delta, theta, alpha, and beta power from cleaned epochs.",
    "MICROSTATES": "Four-state topographic microstate dynamics from continuous EEG.",
    "CONNECTIVITY": "Band-limited connectivity summaries and graph metrics.",
    "SPECTRAL_PARAMETERIZATION": "Aperiodic and oscillatory spectral parameters from specparam/FOOOF.",
    "HJORTH": "Activity, mobility, and complexity parameters from cleaned epochs.",
    "ENTROPY": "Entropy and complexity measures, including Higuchi fractal dimension.",
    "ASYMMETRY": "Left-right power differences across homologous electrode pairs.",
    "KURTOSIS": "Excess temporal kurtosis describing non-Gaussian signal shape.",
}

OPTIONAL_INTEGER_FIELDS: set[str] = set()
OPTIONAL_FLOAT_FIELDS = {"RESAMPLE_SFREQ"}
NUMERIC_LIST_FIELDS = {
    "CACHE_DURATIONS", "CONNECTIVITY_DENSITIES", "SPECPARAM_FREQ_RANGE",
    "SPECPARAM_PEAK_WIDTHS",
    "TOP_K_PLOT_VALUES", "PERMUTATION_K_VALUES",
}

# GUI

def launch_gui() -> None:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
    from tkinter.scrolledtext import ScrolledText

    BG = "#f2f4f7"
    PANEL = "#ffffff"
    BORDER = "#d7dce2"
    ACCENT = "#2f5f8f"
    ACCENT_DARK = "#244a70"
    TEXT = "#20262d"
    MUTED = "#5f6b76"
    SUCCESS = "#2d6a4f"
    WARNING = "#9a6700"

    def friendly_name(key: str) -> str:
        if key in FIELD_LABELS:
            return FIELD_LABELS[key]
        text = key.replace("_", " ").title()
        for old, new in {
            "Mcmi": "MCMI", "Ica": "ICA", "Eeg": "EEG", "Cv": "CV",
            "Br60": "BR60", "Br75": "BR75", "Cpu": "CPU", "Pdf": "PDF",
            "Png": "PNG", "Fdr": "FDR", "Mad": "MAD", "Gfp": "GFP",
        }.items():
            text = text.replace(old, new)
        return text

    def format_default(value: Any) -> str:
        if isinstance(value, bool):
            return "On" if value else "Off"
        if value is None:
            return "blank"
        if isinstance(value, dict) and all(isinstance(v, bool) for v in value.values()):
            return f"{sum(bool(v) for v in value.values())}/{len(value)} selected"
        if isinstance(value, dict):
            return ", ".join(f"{k}: {v[0]}-{v[1]}" for k, v in value.items())
        if isinstance(value, (list, tuple, set)):
            return ", ".join(str(v) for v in value)
        return str(value)

    def field_help(stage: int, key: str, default: Any) -> str:
        base = FIELD_HELP.get(key)
        if base is None:
            if isinstance(default, bool):
                base = "Turn this option on or off. Changing it changes the behavior of this stage."
            elif isinstance(default, (int, float)) or default is None:
                base = "Numeric setting used by this stage. Change it only when the analysis plan requires a different value."
            else:
                base = "Setting used by this stage. Select a supported option rather than typing an internal code."
        return f"{base}\n\nReplication default: {format_default(default)}"

    class ToolTip:
        def __init__(self, widget, text: str, delay: int = 450):
            self.widget = widget
            self.text = text
            self.delay = delay
            self.after_id = None
            self.window = None
            widget.bind("<Enter>", self._schedule, add="+")
            widget.bind("<Leave>", self._hide, add="+")
            widget.bind("<ButtonPress>", self._hide, add="+")

        def _schedule(self, _event=None):
            self._cancel()
            self.after_id = self.widget.after(self.delay, self._show)

        def _cancel(self):
            if self.after_id is not None:
                self.widget.after_cancel(self.after_id)
                self.after_id = None

        def _show(self):
            if self.window is not None or not self.text:
                return
            self.window = win = tk.Toplevel(self.widget)
            win.wm_overrideredirect(True)
            frame = tk.Frame(win, background="#fffbea", borderwidth=1, relief="solid")
            frame.pack()
            tk.Label(
                frame, text=self.text, justify="left", background="#fffbea",
                foreground=TEXT, wraplength=440, padx=10, pady=8,
                font=("Segoe UI", 9),
            ).pack()
            win.update_idletasks()
            pointer_x, pointer_y = self.widget.winfo_pointerxy()
            width = win.winfo_reqwidth()
            height = win.winfo_reqheight()
            x = max(4, pointer_x - width - 12)
            y = min(pointer_y + 18, self.widget.winfo_screenheight() - height - 4)
            win.wm_geometry(f"+{x}+{max(4, y)}")

        def _hide(self, _event=None):
            self._cancel()
            if self.window is not None:
                self.window.destroy()
                self.window = None

    class ScrollableFrame(ttk.Frame):
        def __init__(self, master):
            super().__init__(master)
            self.canvas = tk.Canvas(self, highlightthickness=0, background=BG)
            bar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
            self.inner = ttk.Frame(self.canvas, style="Page.TFrame")
            self.window_id = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
            self.inner.bind("<Configure>", lambda _e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
            self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfigure(self.window_id, width=e.width))
            self.canvas.configure(yscrollcommand=bar.set)
            self.canvas.pack(side="left", fill="both", expand=True)
            bar.pack(side="right", fill="y")
            self.canvas.bind("<Enter>", self._bind_wheel)
            self.canvas.bind("<Leave>", self._unbind_wheel)

        def _bind_wheel(self, _event=None):
            self.canvas.bind_all("<MouseWheel>", self._wheel)

        def _unbind_wheel(self, _event=None):
            self.canvas.unbind_all("<MouseWheel>")

        def _wheel(self, event):
            self.canvas.yview_scroll(int(-event.delta / 120), "units")

    class FieldBinding:
        def __init__(self, key: str, default: Any, kind: str, **kwargs):
            self.key = key
            self.default = deepcopy(default)
            self.kind = kind
            self.__dict__.update(kwargs)

        def get(self) -> Any:
            label = friendly_name(self.key)
            try:
                if self.kind == "bool":
                    return bool(self.var.get())
                if self.kind == "choice":
                    display = self.var.get()
                    if display not in self.display_to_value:
                        raise ValueError("Select one of the available options.")
                    return deepcopy(self.display_to_value[display])
                if self.kind == "number":
                    text = self.var.get().strip()
                    if self.key in OPTIONAL_INTEGER_FIELDS:
                        return None if text == "" else int(text)
                    if self.key in OPTIONAL_FLOAT_FIELDS:
                        return None if text == "" else float(text)
                    if isinstance(self.default, int) and not isinstance(self.default, bool):
                        return int(text)
                    return float(text)
                if self.kind == "number_list":
                    text = self.var.get().strip()
                    if not text:
                        raise ValueError("Enter at least one numeric value.")
                    if text.startswith("["):
                        values = json.loads(text)
                    else:
                        values = [part.strip() for part in text.replace(";", ",").split(",") if part.strip()]
                    if not isinstance(values, list):
                        raise ValueError("Enter comma-separated numbers.")
                    use_int = bool(self.default) and all(isinstance(v, int) and not isinstance(v, bool) for v in self.default)
                    return [int(v) if use_int else float(v) for v in values]
                if self.kind == "feature_dict":
                    return {name: bool(var.get()) for name, var in self.vars.items()}
                if self.kind == "string_checklist":
                    selected = [name for name in self.order if self.vars[name].get()]
                    if not selected:
                        raise ValueError("Select at least one item.")
                    return selected
                if self.kind == "bands":
                    result = {}
                    for name, (low_var, high_var) in self.vars.items():
                        low = float(low_var.get())
                        high = float(high_var.get())
                        if low >= high:
                            raise ValueError(f"{name}: lower edge must be below upper edge.")
                        result[name] = (low, high)
                    return result
            except Exception as error:
                raise ValueError(f"{label}: {error}") from error
            raise ValueError(f"Unsupported control type for {label}.")

        def set(self, value: Any) -> None:
            if self.kind == "bool":
                self.var.set(bool(value))
            elif self.kind == "choice":
                display = self.value_to_display.get(self._hashable(value))
                if display is None and self.value_to_display:
                    display = next(iter(self.value_to_display.values()))
                self.var.set(display or "")
            elif self.kind == "number":
                self.var.set("" if value is None else str(value))
            elif self.kind == "number_list":
                self.var.set(", ".join(str(v) for v in value))
            elif self.kind == "feature_dict":
                for name, var in self.vars.items():
                    var.set(bool(value.get(name, False)))
            elif self.kind == "string_checklist":
                selected = set(value or [])
                for name, var in self.vars.items():
                    var.set(name in selected)
            elif self.kind == "bands":
                for name, (low_var, high_var) in self.vars.items():
                    low, high = value[name]
                    low_var.set(str(low))
                    high_var.set(str(high))

        @staticmethod
        def _hashable(value: Any):
            if isinstance(value, list):
                return tuple(value)
            if isinstance(value, dict):
                return tuple(sorted(value.items()))
            return value

    class InitialSetupDialog(tk.Toplevel):
        def __init__(self, master, payload):
            super().__init__(master)
            self.title("MAPPER — Project locations")
            self.transient(master)
            self.grab_set()
            self.result = None
            self.vars = {}
            self.configure(background=BG)
            self.geometry("980x500")
            self.minsize(820, 440)

            header = tk.Frame(self, background=ACCENT, padx=18, pady=14)
            header.pack(fill="x")
            tk.Label(header, text="Project locations", background=ACCENT, foreground="white", font=("Segoe UI Semibold", 17)).pack(anchor="w")
            tk.Label(header, text="These locations are shared by all eight stages and can be changed later.", background=ACCENT, foreground="#e9f1f8", font=("Segoe UI", 9)).pack(anchor="w", pady=(3, 0))

            body = ttk.Frame(self, padding=18, style="Page.TFrame")
            body.pack(fill="both", expand=True)
            body.columnconfigure(1, weight=1)
            fields = [
                ("BASE_DIR", "Project base directory", "dir", "Optional project root used to suggest the other paths."),
                ("EEG_DIR", "Raw EEG directory", "dir", "Folder containing one raw EEG file per subject."),
                ("CLEANED_DIR", "Cleaned subject directory", "dir", "Folder where subject-level cleaned EEG, epochs, and features are stored."),
                ("REPORTS_DIR", "Reports directory", "dir", "Folder containing summaries, timestamped runs, logs, and checkpoints."),
                ("MCMI_CSV", "MCMI analysis CSV", "file", "Anonymized table containing subject_id and the 15 BR-scale columns."),
                ("SUBJECT_ID_COLUMN", "Subject ID column", "text", "Column name shared by the MCMI table and EEG filenames. Replication default: subject_id."),
                ("EEG_FILE_EXTENSION", "EEG file extension", "choice", "Raw file extension expected by Stage 1. The current reader is designed for EDF files."),
            ]
            general = payload["general"]
            for row, (key, label_text, kind, help_text) in enumerate(fields):
                label = ttk.Label(body, text=label_text, style="Field.TLabel")
                label.grid(row=row, column=0, sticky="w", padx=(0, 10), pady=7)
                info = ttk.Label(body, text="ⓘ", style="Info.TLabel")
                info.grid(row=row, column=2, sticky="w", padx=(6, 4))
                ToolTip(label, help_text)
                ToolTip(info, help_text)
                var = tk.StringVar(value=str(general.get(key, "")))
                self.vars[key] = var
                if kind == "choice":
                    widget = ttk.Combobox(body, textvariable=var, values=[".edf"], state="readonly", width=18)
                else:
                    widget = ttk.Entry(body, textvariable=var)
                widget.grid(row=row, column=1, sticky="ew", pady=7)
                if kind in {"dir", "file"}:
                    def browse(k=key, typ=kind):
                        value = filedialog.askdirectory(parent=self) if typ == "dir" else filedialog.askopenfilename(parent=self, filetypes=[("CSV", "*.csv"), ("All files", "*.*")])
                        if value:
                            self.vars[k].set(value)
                            if k == "BASE_DIR":
                                base = Path(value)
                                suggestions = {
                                    "EEG_DIR": base / "Subject info" / "EEG",
                                    "CLEANED_DIR": base / "Subject info" / "EEG" / "Cleaned",
                                    "REPORTS_DIR": base / "Reports",
                                    "MCMI_CSV": base / "Subject info" / "mcmi_anonymized_for_analysis.csv",
                                }
                                for sk, sv in suggestions.items():
                                    if not self.vars[sk].get().strip():
                                        self.vars[sk].set(str(sv))
                    ttk.Button(body, text="Browse…", command=browse).grid(row=row, column=3, padx=(4, 0))

            buttons = ttk.Frame(body, style="Page.TFrame")
            buttons.grid(row=len(fields), column=0, columnspan=4, sticky="e", pady=(18, 0))
            ttk.Button(buttons, text="Cancel", command=self.destroy).pack(side="right", padx=4)
            ttk.Button(buttons, text="Save project locations", style="Accent.TButton", command=self._save).pack(side="right", padx=4)
            self.protocol("WM_DELETE_WINDOW", self.destroy)

        def _save(self):
            values = {k: v.get().strip() for k, v in self.vars.items()}
            required = ["EEG_DIR", "CLEANED_DIR", "REPORTS_DIR", "MCMI_CSV", "SUBJECT_ID_COLUMN", "EEG_FILE_EXTENSION"]
            missing = [key for key in required if not values.get(key)]
            if missing:
                messagebox.showerror("Missing settings", "Please complete: " + ", ".join(missing), parent=self)
                return
            self.result = values
            self.destroy()

    class App(tk.Tk):
        def __init__(self):
            super().__init__()
            self.title("MAPPER")
            self.configure(background=BG)
            self.payload = default_settings_payload()
            stored = load_json(settings_path(), {}) or {}
            from V3_runtime import deep_update
            deep_update(self.payload, stored)
            normalize_settings_payload(self.payload)
            geometry = self.payload.get("gui", {}).get("window_geometry", "1380x900")
            self.geometry(geometry)
            self.minsize(1080, 720)

            self.process: Optional[subprocess.Popen] = None
            self.process_thread: Optional[threading.Thread] = None
            self.output_queue: queue.Queue = queue.Queue()
            self.current_stage: Optional[int] = None
            self.current_control: Optional[RunControl] = None
            self.current_run_dir: Optional[Path] = None
            self.selected_vars: Dict[int, tk.BooleanVar] = {}
            self.resume_vars: Dict[int, tk.BooleanVar] = {}
            self.force_vars: Dict[int, tk.BooleanVar] = {}
            self.setting_bindings: Dict[int, Dict[str, FieldBinding]] = {}
            self.detected_labels: Dict[int, tk.StringVar] = {}
            self.input_mode_vars: Dict[int, tk.StringVar] = {}
            self.input_path_vars: Dict[int, tk.StringVar] = {}
            self.pending_stages: list[int] = []

            self._configure_styles()
            self._ensure_initial_setup()
            self._build_ui()
            self.after(100, self._drain_output)
            self.protocol("WM_DELETE_WINDOW", self._on_close)

        def _configure_styles(self):
            style = ttk.Style(self)
            try:
                style.theme_use("clam")
            except tk.TclError:
                pass
            style.configure(".", font=("Segoe UI", 9), background=BG, foreground=TEXT)
            style.configure("Page.TFrame", background=BG)
            style.configure("Panel.TFrame", background=PANEL)
            style.configure("TNotebook", background=BG, borderwidth=0)
            style.configure("TNotebook.Tab", padding=(15, 9), font=("Segoe UI Semibold", 9))
            style.map("TNotebook.Tab", background=[("selected", PANEL)], foreground=[("selected", ACCENT_DARK)])
            style.configure("Stage.TLabelframe", background=PANEL, bordercolor=BORDER, relief="solid")
            style.configure("Stage.TLabelframe.Label", background=PANEL, foreground=ACCENT_DARK, font=("Segoe UI Semibold", 10))
            style.configure("Field.TLabel", background=PANEL, foreground=TEXT)
            style.configure("Muted.TLabel", background=PANEL, foreground=MUTED)
            style.configure("Info.TLabel", background=PANEL, foreground=ACCENT, font=("Segoe UI Semibold", 10))
            style.configure("SectionNote.TLabel", background=PANEL, foreground=MUTED, font=("Segoe UI", 8))
            style.configure("Accent.TButton", background=ACCENT, foreground="white", padding=(12, 7), font=("Segoe UI Semibold", 9))
            style.map("Accent.TButton", background=[("active", ACCENT_DARK), ("disabled", "#a9b6c3")])
            style.configure("Quiet.TButton", padding=(10, 6))
            style.configure("Run.TCheckbutton", background=PANEL, font=("Segoe UI Semibold", 9))
            style.configure("Option.TCheckbutton", background=PANEL)

        def _ensure_initial_setup(self):
            general = self.payload.get("general", {})
            required = ["EEG_DIR", "CLEANED_DIR", "REPORTS_DIR", "MCMI_CSV"]
            if any(not str(general.get(key, "")).strip() for key in required):
                dialog = InitialSetupDialog(self, self.payload)
                self.wait_window(dialog)
                if dialog.result:
                    self.payload["general"].update(dialog.result)
                    save_json_atomic(settings_path(), self.payload)
                    apply_persisted_settings()

        def _build_ui(self):
            header = tk.Frame(self, background=ACCENT, padx=18, pady=12)
            header.pack(fill="x")
            left = tk.Frame(header, background=ACCENT)
            left.pack(side="left", fill="x", expand=True)
            tk.Label(left, text="MAPPER", background=ACCENT, foreground="white", font=("Segoe UI Semibold", 19)).pack(anchor="w")
            tk.Label(left, text="Reproducible nested MCMI–EEG analysis", background=ACCENT, foreground="#e8f0f7", font=("Segoe UI", 9)).pack(anchor="w")
            self.status_var = tk.StringVar(value="Idle")
            tk.Label(header, textvariable=self.status_var, background=ACCENT, foreground="white", font=("Segoe UI Semibold", 9)).pack(side="right", padx=6)

            toolbar = ttk.Frame(self, padding=(10, 8), style="Page.TFrame")
            toolbar.pack(fill="x")
            ttk.Button(toolbar, text="Project locations", command=self._edit_general).pack(side="left", padx=3)
            ttk.Button(toolbar, text="Refresh inputs", command=self._refresh_detected).pack(side="left", padx=3)
            ttk.Button(toolbar, text="Save settings", command=self._save_from_ui).pack(side="left", padx=3)
            ttk.Button(toolbar, text="Restore replication defaults", command=self._restore_all_defaults).pack(side="left", padx=3)
            ttk.Separator(toolbar, orient="vertical").pack(side="left", fill="y", padx=9)
            ttk.Button(toolbar, text="Run selected stages", style="Accent.TButton", command=self._run_selected).pack(side="left", padx=(10, 3))
            self.pause_button = ttk.Button(toolbar, text="Pause", command=self._toggle_pause, state="disabled")
            self.pause_button.pack(side="left", padx=3)
            ttk.Button(toolbar, text="Stop safely", command=self._request_stop).pack(side="left", padx=3)
            ttk.Button(toolbar, text="Force stop", command=self._force_stop).pack(side="left", padx=3)

            self.notebook = ttk.Notebook(self)
            self.notebook.pack(fill="both", expand=True, padx=10, pady=(0, 6))
            selected_default = set(self.payload.get("gui", {}).get("selected_stages", list(STAGE_IDS)))
            for stage in STAGE_IDS:
                scroll = ScrollableFrame(self.notebook)
                self.notebook.add(scroll, text=f"{stage}. {STAGE_TITLES[stage]}")
                self._build_stage_tab(stage, scroll.inner, stage in selected_default)

            bottom = ttk.Labelframe(self, text="Run log", padding=6, style="Stage.TLabelframe")
            bottom.pack(fill="both", expand=False, padx=10, pady=(0, 10))
            self.log = ScrolledText(bottom, height=10, wrap="word", state="disabled", font=("Consolas", 9), background="#fbfcfd", foreground=TEXT)
            self.log.pack(fill="both", expand=True)

        def _build_stage_tab(self, stage: int, parent, selected: bool):
            parent.configure(style="Page.TFrame")
            parent.columnconfigure(0, weight=1)

            stage_header = ttk.Frame(parent, padding=(14, 14), style="Panel.TFrame")
            stage_header.grid(row=0, column=0, sticky="ew", padx=12, pady=(12, 6))
            stage_header.columnconfigure(0, weight=1)
            ttk.Label(stage_header, text=f"Stage {stage}: {STAGE_TITLES[stage]}", style="Field.TLabel", font=("Segoe UI Semibold", 15)).grid(row=0, column=0, sticky="w")
            ttk.Label(stage_header, text=STAGE_DESCRIPTIONS[stage], style="Muted.TLabel", wraplength=980).grid(row=1, column=0, sticky="w", pady=(4, 0))

            execution = ttk.Labelframe(parent, text="Execution", padding=12, style="Stage.TLabelframe")
            execution.grid(row=1, column=0, sticky="ew", padx=12, pady=6)
            execution.columnconfigure(4, weight=1)
            selected_var = tk.BooleanVar(value=selected)
            resume_var = tk.BooleanVar(value=bool(self.payload["stages"][str(stage)].get("resume", True)))
            self.selected_vars[stage] = selected_var
            self.resume_vars[stage] = resume_var
            ttk.Checkbutton(execution, text="Include in pipeline run", variable=selected_var, style="Run.TCheckbutton").grid(row=0, column=0, sticky="w", padx=(0, 16))
            ttk.Checkbutton(execution, text="Resume existing progress", variable=resume_var, style="Option.TCheckbutton").grid(row=0, column=1, sticky="w", padx=(0, 16))
            if stage in {1, 2, 4}:
                force_var = tk.BooleanVar(value=bool(self.payload["stages"][str(stage)].get("force_rerun", False)))
                self.force_vars[stage] = force_var
                ttk.Checkbutton(execution, text="Force rerun / overwrite", variable=force_var, style="Option.TCheckbutton").grid(row=0, column=2, sticky="w", padx=(0, 16))
            ttk.Button(execution, text="Run this stage", style="Accent.TButton", command=lambda s=stage: self._start_sequence([s])).grid(row=0, column=5, padx=4)
            ttk.Button(execution, text="Open latest output", command=lambda s=stage: self._open_latest(s)).grid(row=0, column=6, padx=4)
            ttk.Button(execution, text="Restore stage defaults", command=lambda s=stage: self._restore_stage_defaults(s)).grid(row=0, column=7, padx=4)

            detected_box = ttk.Labelframe(parent, text="Input", padding=12, style="Stage.TLabelframe")
            detected_box.grid(row=2, column=0, sticky="ew", padx=12, pady=6)
            detected_box.columnconfigure(1, weight=1)
            detected = tk.StringVar(value="")
            self.detected_labels[stage] = detected
            ttk.Label(detected_box, textvariable=detected, style="Muted.TLabel", wraplength=1120).grid(row=0, column=0, columnspan=4, sticky="w")
            if stage > 1:
                input_pref = self.payload["inputs"][str(stage)]
                mode = tk.StringVar(value=str(input_pref.get("mode", "auto")))
                path = tk.StringVar(value=str(input_pref.get("path", "")))
                self.input_mode_vars[stage] = mode
                self.input_path_vars[stage] = path
                ttk.Radiobutton(detected_box, text="Latest compatible run", variable=mode, value="auto", command=lambda s=stage: self._refresh_detected(s)).grid(row=1, column=0, sticky="w", pady=(8, 0))
                ttk.Radiobutton(detected_box, text="Choose another run", variable=mode, value="manual").grid(row=2, column=0, sticky="w", pady=(5, 0))
                ttk.Entry(detected_box, textvariable=path).grid(row=2, column=1, sticky="ew", padx=(8, 6), pady=(5, 0))

                def browse_input(s=stage):
                    value = filedialog.askdirectory(parent=self)
                    if value:
                        self.input_mode_vars[s].set("manual")
                        self.input_path_vars[s].set(value)
                        self._refresh_detected(s)

                ttk.Button(detected_box, text="Browse…", command=browse_input).grid(row=2, column=2, pady=(5, 0))
            ttk.Button(detected_box, text="Refresh", command=lambda s=stage: self._refresh_detected(s)).grid(row=0, column=4, padx=(8, 0))

            self.setting_bindings[stage] = {}
            stage_settings = self.payload["stages"][str(stage)]
            row_index = 3
            seen = set()
            for section_name, keys in STAGE_SECTIONS[stage]:
                section = ttk.Labelframe(parent, text=section_name, padding=12, style="Stage.TLabelframe")
                section.grid(row=row_index, column=0, sticky="ew", padx=12, pady=6)
                section.columnconfigure(1, weight=1)
                control_row = 0
                for key in keys:
                    if key not in stage_settings or key in seen:
                        continue
                    seen.add(key)
                    binding = self._add_setting_control(stage, section, control_row, key, stage_settings[key])
                    self.setting_bindings[stage][key] = binding
                    control_row += 1
                row_index += 1

            # Any new setting added later is exposed.
            remaining = [key for key in stage_settings if key not in seen and key not in {"resume", "force_rerun"}]
            if remaining:
                section = ttk.Labelframe(parent, text="Additional settings", padding=12, style="Stage.TLabelframe")
                section.grid(row=row_index, column=0, sticky="ew", padx=12, pady=6)
                section.columnconfigure(1, weight=1)
                for row, key in enumerate(remaining):
                    binding = self._add_setting_control(stage, section, row, key, stage_settings[key])
                    self.setting_bindings[stage][key] = binding

            self._refresh_detected(stage)

        def _label_and_help(self, parent, row: int, stage: int, key: str, default: Any):
            text = friendly_name(key)
            help_text = field_help(stage, key, default)
            label = ttk.Label(parent, text=text, style="Field.TLabel")
            label.grid(row=row, column=0, sticky="nw", padx=(0, 8), pady=6)
            info = ttk.Label(parent, text="ⓘ", style="Info.TLabel")
            info.grid(row=row, column=2, sticky="nw", padx=(7, 3), pady=6)
            default_label = ttk.Label(parent, text=f"Default: {format_default(default)}", style="SectionNote.TLabel")
            default_label.grid(row=row, column=3, sticky="nw", padx=(8, 0), pady=7)
            ToolTip(label, help_text)
            ToolTip(info, help_text)
            ToolTip(default_label, help_text)

        def _add_setting_control(self, stage: int, parent, row: int, key: str, value: Any) -> FieldBinding:
            default = STAGE_DEFAULTS[str(stage)][key]
            self._label_and_help(parent, row, stage, key, default)

            if key == "FEATURE_FAMILIES":
                frame = ttk.Frame(parent, style="Panel.TFrame")
                frame.grid(row=row, column=1, sticky="ew", pady=4)
                vars_ = {}
                for index, name in enumerate(FEATURE_FAMILY_ORDER):
                    var = tk.BooleanVar(value=bool(value.get(name, False)))
                    vars_[name] = var
                    button = ttk.Checkbutton(frame, text=name.replace("_", " ").title(), variable=var, style="Option.TCheckbutton")
                    button.grid(row=index // 4, column=index % 4, sticky="w", padx=(0, 18), pady=3)
                    ToolTip(button, f"{FEATURE_HELP.get(name, name)}\n\nEnable all eight to reproduce the full feature space.")
                return FieldBinding(key, default, "feature_dict", vars=vars_)

            if key == "MCMI_SCALE_ORDER":
                frame = ttk.Frame(parent, style="Panel.TFrame")
                frame.grid(row=row, column=1, sticky="ew", pady=4)
                vars_ = {}
                selected = set(value or [])
                for index, name in enumerate(MCMI_SCALE_ORDER):
                    var = tk.BooleanVar(value=name in selected)
                    vars_[name] = var
                    button = ttk.Checkbutton(frame, text=name, variable=var, style="Option.TCheckbutton")
                    button.grid(row=index // 5, column=index % 5, sticky="w", padx=(0, 22), pady=3)
                    ToolTip(button, f"Include {name} in every enabled task. All 15 scales are required for exact replication.")
                return FieldBinding(key, default, "string_checklist", vars=vars_, order=list(MCMI_SCALE_ORDER))

            if key == "BANDS":
                frame = ttk.Frame(parent, style="Panel.TFrame")
                frame.grid(row=row, column=1, sticky="w", pady=4)
                ttk.Label(frame, text="Band", style="SectionNote.TLabel").grid(row=0, column=0, sticky="w", padx=(0, 8))
                ttk.Label(frame, text="Low Hz", style="SectionNote.TLabel").grid(row=0, column=1, sticky="w")
                ttk.Label(frame, text="High Hz", style="SectionNote.TLabel").grid(row=0, column=2, sticky="w")
                vars_ = {}
                for band_row, name in enumerate(BANDS, start=1):
                    low, high = value.get(name, BANDS[name])
                    low_var = tk.StringVar(value=str(low))
                    high_var = tk.StringVar(value=str(high))
                    vars_[name] = (low_var, high_var)
                    ttk.Label(frame, text=name.title(), style="Field.TLabel").grid(row=band_row, column=0, sticky="w", padx=(0, 8), pady=2)
                    ttk.Entry(frame, textvariable=low_var, width=10).grid(row=band_row, column=1, padx=(0, 6), pady=2)
                    ttk.Entry(frame, textvariable=high_var, width=10).grid(row=band_row, column=2, pady=2)
                return FieldBinding(key, default, "bands", vars=vars_)

            if isinstance(value, bool):
                var = tk.BooleanVar(value=value)
                widget = ttk.Checkbutton(parent, text="Enabled", variable=var, style="Option.TCheckbutton")
                widget.grid(row=row, column=1, sticky="w", pady=5)
                ToolTip(widget, field_help(stage, key, default))
                return FieldBinding(key, default, "bool", var=var)

            if key in CHOICE_OPTIONS:
                options = CHOICE_OPTIONS[key]
                display_to_value = {display: internal for display, internal in options}
                value_to_display = {FieldBinding._hashable(internal): display for display, internal in options}
                display = value_to_display.get(FieldBinding._hashable(value), options[0][0])
                var = tk.StringVar(value=display)
                widget = ttk.Combobox(parent, textvariable=var, values=[display for display, _ in options], state="readonly", width=52)
                widget.grid(row=row, column=1, sticky="w", pady=4)
                ToolTip(widget, field_help(stage, key, default))
                return FieldBinding(key, default, "choice", var=var, display_to_value=display_to_value, value_to_display=value_to_display)

            if key in NUMERIC_LIST_FIELDS or isinstance(value, (list, tuple)):
                var = tk.StringVar(value=", ".join(str(v) for v in value))
                widget = ttk.Entry(parent, textvariable=var, width=54)
                widget.grid(row=row, column=1, sticky="ew", pady=4)
                ToolTip(widget, field_help(stage, key, default) + "\n\nEnter numbers separated by commas.")
                return FieldBinding(key, default, "number_list", var=var)

            if isinstance(value, (int, float)) or value is None:
                var = tk.StringVar(value="" if value is None else str(value))
                widget = ttk.Entry(parent, textvariable=var, width=22)
                widget.grid(row=row, column=1, sticky="w", pady=4)
                ToolTip(widget, field_help(stage, key, default))
                return FieldBinding(key, default, "number", var=var)

            # Unsupported arbitrary strings are shown through a read-only option
            var = tk.StringVar(value=str(value))
            widget = ttk.Combobox(parent, textvariable=var, values=[str(value)], state="readonly", width=52)
            widget.grid(row=row, column=1, sticky="w", pady=4)
            display_to_value = {str(value): value}
            value_to_display = {FieldBinding._hashable(value): str(value)}
            return FieldBinding(key, default, "choice", var=var, display_to_value=display_to_value, value_to_display=value_to_display)

        def _edit_general(self):
            dialog = InitialSetupDialog(self, self.payload)
            self.wait_window(dialog)
            if dialog.result:
                self.payload["general"].update(dialog.result)
                self._save_from_ui()
                self._refresh_detected()

        def _refresh_detected(self, only_stage: Optional[int] = None):
            stages = [only_stage] if only_stage else STAGE_IDS
            for stage in stages:
                if stage in self.detected_labels:
                    try:
                        if stage == 1:
                            self.detected_labels[stage].set(
                                f"EEG: {self.payload['general'].get('EEG_DIR', '')} | "
                                f"MCMI: {self.payload['general'].get('MCMI_CSV', '')}"
                            )
                            continue
                        payload = deepcopy(self.payload)
                        live_stages = {stage}
                        if stage in {7, 8}:
                            live_stages.add(6)
                        for live_stage in live_stages:
                            for key, binding in self.setting_bindings.get(live_stage, {}).items():
                                payload["stages"][str(live_stage)][key] = binding.get()
                        mode = self.input_mode_vars[stage].get()
                        manual = self.input_path_vars[stage].get() if mode == "manual" else ""
                        selected = resolve_stage_input(stage, payload, manual)
                        prefix = "Selected" if mode == "manual" else "Latest compatible"
                        self.detected_labels[stage].set(f"{prefix}: {selected}")
                    except Exception as error:
                        self.detected_labels[stage].set(str(error))

        def _validate_stage_values(self, stage: int, values: Dict[str, Any]) -> None:
            if stage == 1 and values["DO_BANDPASS"] and values["BANDPASS_LOW"] >= values["BANDPASS_HIGH"]:
                raise ValueError("Stage 1: the band-pass lower edge must be below the upper edge.")
            if stage in {2, 4} and values["REFERENCE_DURATION"] not in values["CACHE_DURATIONS"]:
                raise ValueError(f"Stage {stage}: the reference duration must be included in the cached durations.")
            if stage == 4 and not any(values["FEATURE_FAMILIES"].values()):
                raise ValueError("Stage 4: select at least one feature family.")
            if stage == 5 and values["OUTER_N_FOLDS"] != 5:
                raise ValueError("Stage 5: the approved primary design requires exactly five outer folds.")
            if stage == 5 and values["OUTER_FOLD_CANDIDATE_ASSIGNMENTS"] < 1:
                raise ValueError("Stage 5: test at least one candidate fold assignment.")
            if stage == 5 and values["CONTINUOUS_BALANCE_WEIGHT"] < 0:
                raise ValueError("Stage 5: the continuous balance weight cannot be negative.")
            if stage == 6:
                if values["OUTER_N_FOLDS"] != 5 or values["INNER_N_FOLDS"] != 5:
                    raise ValueError("Stage 6: the approved design requires five outer and five inner folds.")
                if values["INNER_N_REPEATS"] < 1:
                    raise ValueError("Stage 6: inner repetitions must be at least one.")
                if values["TOP_K_CONFIGS"] < 1:
                    raise ValueError("Stage 6: retain at least one configuration.")
                if values["SELECTION_TIE_TOLERANCE"] < 0:
                    raise ValueError("Stage 6: tie tolerance cannot be negative.")
                if not any(values[key] for key in ["RUN_BR60", "RUN_BR75", "RUN_ORDINAL", "RUN_CONTINUOUS"]):
                    raise ValueError("Stage 6: enable at least one prediction task.")
                if not values["MCMI_SCALE_ORDER"]:
                    raise ValueError("Stage 6: select at least one MCMI scale.")
                scheme = str(
                    values["ORDINAL_SCHEME"]
                )

                if scheme not in VALID_ORDINAL_SCHEMES:
                    raise ValueError(
                        "Stage 6: select a valid "
                        "ordinal grouping."
                    )

                if not (
                    values["BRACKET_CUTOFF_1"]
                    < values["BRACKET_CUTOFF_2"]
                ):
                    raise ValueError(
                        "Stage 6: ordinal cutoffs 1 and 2 "
                        "must be strictly increasing."
                    )

                if (
                    scheme == "4class_original"
                    and not (
                        values["BRACKET_CUTOFF_2"]
                        < values["BRACKET_CUTOFF_3"]
                    )
                ):
                    raise ValueError(
                        "Stage 6: cutoff 3 must exceed cutoff 2 "
                        "for the four-class analysis."
                    )
            if stage == 7:
                if any(int(k) < 1 for k in values["TOP_K_PLOT_VALUES"]):
                    raise ValueError("Stage 7: every Top-k plot value must be at least 1.")
                if values["REPORT_TOP_CONFIGS"] < 1:
                    raise ValueError("Stage 7: report at least one top configuration.")
            if stage == 8:
                if values["ORDINAL_SCHEME"] not in VALID_ORDINAL_SCHEMES:
                    raise ValueError("Stage 8: select a valid ordinal grouping.")
                if values["N_BOOTSTRAPS"] < 1 and values["RUN_BOOTSTRAP"]:
                    raise ValueError("Stage 8: bootstrap samples must be at least 1.")
                if values["PERMUTATION_METHOD"] not in {"none", "fixed_candidate", "full_nested"}:
                    raise ValueError("Stage 8: select a valid permutation method.")
                if values["PERMUTATION_METHOD"] != "none" and values["N_PERMUTATIONS"] < 1:
                    raise ValueError("Stage 8: permutation count must be at least 1.")
                if any(int(k) < 1 for k in values["PERMUTATION_K_VALUES"]):
                    raise ValueError("Stage 8: every permutation Top-k value must be at least 1.")
                if values["PERMUTATION_METHOD"] == "fixed_candidate" and not values["PERMUTATION_K_VALUES"]:
                    raise ValueError("Stage 8: select at least one Top-k value for fixed-candidate permutation.")
                if values["MULTIPLICITY_METHOD"] != "fdr_bh":
                    raise ValueError("Stage 8: the supported multiplicity method is Benjamini-Hochberg FDR.")

        def _save_from_ui(self):
            try:
                clean_stages = {}
                for stage in STAGE_IDS:
                    current = deepcopy(self.payload["stages"][str(stage)])
                    current["resume"] = bool(self.resume_vars[stage].get())
                    if stage in self.force_vars:
                        current["force_rerun"] = bool(self.force_vars[stage].get())
                    for key, binding in self.setting_bindings[stage].items():
                        current[key] = binding.get()
                    self._validate_stage_values(stage, current)
                    clean_stages[str(stage)] = current
                self.payload["stages"] = clean_stages
                self.payload["inputs"] = {
                    str(stage): {
                        "mode": self.input_mode_vars[stage].get(),
                        "path": self.input_path_vars[stage].get().strip(),
                    }
                    for stage in STAGE_IDS if stage > 1
                }
                self.payload.setdefault("gui", {})["selected_stages"] = [stage for stage in STAGE_IDS if self.selected_vars[stage].get()]
                self.payload["gui"]["window_geometry"] = self.geometry()
                save_json_atomic(settings_path(), self.payload)
                apply_persisted_settings()
                self.status_var.set("Settings saved")
                return True
            except Exception as error:
                messagebox.showerror("Invalid setting", str(error), parent=self)
                return False

        def _restore_stage_defaults(self, stage: int):
            if not messagebox.askyesno("Restore defaults", f"Restore the replication defaults for Stage {stage}?", parent=self):
                return
            defaults = STAGE_DEFAULTS[str(stage)]
            self.resume_vars[stage].set(bool(defaults.get("resume", True)))
            if stage in self.force_vars:
                self.force_vars[stage].set(bool(defaults.get("force_rerun", False)))
            for key, binding in self.setting_bindings[stage].items():
                binding.set(defaults[key])
            self.status_var.set(f"Stage {stage} defaults restored; save to keep them")

        def _restore_all_defaults(self):
            if not messagebox.askyesno("Restore replication defaults", "Restore all Stage 1-8 scientific settings? Project locations will not change.", parent=self):
                return
            for stage in STAGE_IDS:
                defaults = STAGE_DEFAULTS[str(stage)]
                self.resume_vars[stage].set(bool(defaults.get("resume", True)))
                if stage in self.force_vars:
                    self.force_vars[stage].set(bool(defaults.get("force_rerun", False)))
                for key, binding in self.setting_bindings[stage].items():
                    binding.set(defaults[key])
            self.status_var.set("Replication defaults restored; save to keep them")

        def _run_selected(self):
            stages = [stage for stage in STAGE_IDS if self.selected_vars[stage].get()]
            if not stages:
                messagebox.showinfo("No stages selected", "Select at least one stage.", parent=self)
                return
            self._start_sequence(stages)

        def _start_sequence(self, stages: list[int]):
            if self.process is not None:
                messagebox.showwarning("Pipeline running", "Another stage is already running.", parent=self)
                return
            if not self._save_from_ui():
                return
            problems = validate_general_paths()
            if problems:
                messagebox.showerror("Project locations", "\n".join(problems), parent=self)
                return
            self.pending_stages = list(stages)
            self._launch_next()

        def _launch_next(self):
            if not self.pending_stages:
                self.status_var.set("Selected stages finished")
                self.pause_button.configure(state="disabled", text="Pause")
                self._refresh_detected()
                return
            stage = self.pending_stages.pop(0)
            script = SCRIPT_DIR / STAGE_SCRIPTS[stage]
            if not script.exists():
                messagebox.showerror("Missing script", str(script), parent=self)
                self.pending_stages.clear()
                return

            resume = bool(self.resume_vars[stage].get())
            try:
                source_run = None
                if stage > 1:
                    input_pref = self.payload["inputs"][str(stage)]
                    manual = input_pref.get("path", "") if input_pref.get("mode") == "manual" else ""
                    source_run = resolve_stage_input(stage, self.payload, str(manual))
                resolved_inputs = resolved_inputs_for_stage(stage, source_run)
            except Exception as error:
                messagebox.showerror("Invalid input run", str(error), parent=self)
                self.pending_stages.clear()
                return

            run_payload = payload_with_source_settings(
                self.payload,
                stage,
                source_run,
            )
            run_hash = settings_hash_from_payload(run_payload, stage)
            run_dir = None
            resuming_run = False
            if resume:
                for candidate in run_dirs(REPORTS_DIR, STAGE_SCRIPTS[stage]):
                    manifest = load_json(candidate / "manifest.json", {}) or {}
                    if manifest.get("status") == "complete":
                        continue
                    if manifest.get("settings_hash") != run_hash:
                        continue
                    if manifest.get("resolved_inputs", {}) != resolved_inputs:
                        continue
                    try:
                        saved_payload = load_run_settings_snapshot(candidate, stage)
                    except (OSError, TypeError, ValueError):
                        continue
                    if settings_hash_from_payload(saved_payload, stage) != run_hash:
                        continue
                    run_dir = candidate
                    resuming_run = True
                    break
            if run_dir is None:
                run_dir = make_stage_run_dir(REPORTS_DIR, STAGE_SCRIPTS[stage], resume=False)

            run_payload["runtime"] = {
                "stage": stage,
                "run_dir": str(run_dir),
                "resolved_inputs": resolved_inputs,
            }
            snapshot_path = run_dir / "settings_snapshot.json"
            if snapshot_path.exists():
                run_payload = load_run_settings_snapshot(run_dir, stage)
                if settings_hash_from_payload(run_payload, stage) != run_hash:
                    raise ValueError(
                        f"The saved settings snapshot does not match the run manifest: {run_dir}"
                    )
            else:
                save_json_atomic(snapshot_path, run_payload)
            manifest_update = {
                "stage": stage,
                "script": STAGE_SCRIPTS[stage],
                "run_dir": str(run_dir),
                "settings_snapshot": str(snapshot_path),
                "settings_hash": run_hash,
                "requested_settings": deepcopy(run_payload["stages"][str(stage)]),
                "effective_settings": effective_settings_for_stage(run_payload, stage),
                "configured_paths": deepcopy(run_payload["general"]),
                "resolved_inputs": resolved_inputs,
            }
            if resuming_run:
                prior_manifest = load_json(run_dir / "manifest.json", {}) or {}
                manifest_update["resumed_utc"] = utc_iso()
                manifest_update["resume_count"] = int(prior_manifest.get("resume_count", 0)) + 1
            else:
                manifest_update["started_utc"] = utc_iso()
            write_manifest(run_dir, manifest_update, status="running")

            control_dir = run_dir / "_control"
            control = RunControl(control_dir)
            control.clear()
            self.current_control = control
            self.current_stage = stage
            self.current_run_dir = run_dir

            env = os.environ.copy()
            env["NEUROPSYCHE_SETTINGS_FILE"] = str(snapshot_path)
            env["NEUROPSYCHE_CONTROL_DIR"] = str(control_dir)
            env["NEUROPSYCHE_RESUME"] = "1" if resume else "0"
            env["NEUROPSYCHE_STAGE_RUN_DIR"] = str(run_dir)
            env["NEUROPSYCHE_STAGE_NUMBER"] = str(stage)
            if stage in self.force_vars:
                env["NEUROPSYCHE_FORCE_RERUN"] = "1" if self.force_vars[stage].get() else "0"

            self._append_log(f"\nStarting Stage {stage}: {STAGE_TITLES[stage]}\nRun: {run_dir}\nScript: {script}\n")
            self.status_var.set(f"Running Stage {stage}: {STAGE_TITLES[stage]}")
            self.pause_button.configure(state="normal", text="Pause")
            self.process = subprocess.Popen(
                [sys.executable, "-u", str(script)],
                cwd=str(SCRIPT_DIR), env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, encoding="utf-8", errors="replace",
            )
            self.process_thread = threading.Thread(target=self._reader_thread, daemon=True)
            self.process_thread.start()

        def _reader_thread(self):
            proc = self.process
            if proc is None or proc.stdout is None:
                return
            for line in proc.stdout:
                self.output_queue.put(("line", line))
            code = proc.wait()
            self.output_queue.put(("done", code))

        def _drain_output(self):
            try:
                while True:
                    kind, value = self.output_queue.get_nowait()
                    if kind == "line":
                        self._append_log(value)
                    elif kind == "done":
                        stage = self.current_stage
                        run_dir = self.current_run_dir
                        self._append_log(f"Stage {stage} exited with code {value}.\n")
                        manifest = load_json(run_dir / "manifest.json", {}) if run_dir else {}
                        completed = bool(manifest and manifest.get("status") == "complete")
                        if run_dir is not None:
                            if value == 0 and completed:
                                update_manifest(run_dir, exit_code=0, finished_utc=utc_iso())
                            else:
                                update_manifest(
                                    run_dir,
                                    status="stopped" if value == 0 else "failed",
                                    exit_code=int(value),
                                    finished_utc=utc_iso(),
                                )
                        self.process = None
                        self.current_stage = None
                        self.current_control = None
                        self.current_run_dir = None
                        self.pause_button.configure(state="disabled", text="Pause")
                        if value == 0 and completed:
                            self.after(100, self._launch_next)
                        else:
                            self.pending_stages.clear()
                            self.status_var.set(f"Stage {stage} did not complete")
            except queue.Empty:
                pass
            self.after(100, self._drain_output)

        def _append_log(self, text: str):
            self.log.configure(state="normal")
            self.log.insert("end", text)
            self.log.see("end")
            self.log.configure(state="disabled")
            if self.current_run_dir is not None:
                with (self.current_run_dir / "run.log").open("a", encoding="utf-8", newline="") as handle:
                    handle.write(text)

        def _toggle_pause(self):
            if self.current_control is None:
                return
            if self.current_control.pause_file.exists():
                try:
                    self.current_control.pause_file.unlink()
                except FileNotFoundError:
                    pass
                self.pause_button.configure(text="Pause")
                self.status_var.set(f"Running Stage {self.current_stage}")
            else:
                self.current_control.pause_file.parent.mkdir(parents=True, exist_ok=True)
                self.current_control.pause_file.write_text("pause", encoding="utf-8")
                self.pause_button.configure(text="Resume")
                self.status_var.set("Pause requested; waiting for a safe checkpoint")

        def _request_stop(self):
            if self.current_control is not None:
                self.current_control.stop_file.parent.mkdir(parents=True, exist_ok=True)
                self.current_control.stop_file.write_text("stop", encoding="utf-8")
                self.pending_stages.clear()
                self.status_var.set("Safe stop requested; waiting for a checkpoint")

        def _force_stop(self):
            self.pending_stages.clear()
            if self.process is not None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                self.status_var.set("Process force-stopped")

        def _open_latest(self, stage: int):
            apply_persisted_settings()
            target = latest_run_dir(REPORTS_DIR, STAGE_SCRIPTS[stage], complete_only=True)
            if target is None or not target.exists():
                messagebox.showinfo("No output", "No output folder was found.", parent=self)
                return
            if os.name == "nt":
                os.startfile(str(target))
            else:
                subprocess.Popen(["xdg-open", str(target)])

        def _on_close(self):
            if self.process is not None:
                if not messagebox.askyesno("Pipeline running", "A stage is running. Force-stop it and close?", parent=self):
                    return
                self._force_stop()
            self._save_from_ui()
            self.destroy()

    app = App()
    app.mainloop()


if __name__ == "__main__":
    launch_gui()
