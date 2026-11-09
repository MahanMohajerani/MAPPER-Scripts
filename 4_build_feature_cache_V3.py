"""Build the subject-level EEG feature cache used by MAPPER."""

from __future__ import annotations

import math
import traceback
from pathlib import Path
from typing import Any, Dict, List, Tuple

import mne
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import signal, stats
from sklearn.cluster import KMeans

try:
    from specparam import SpectralModel
    SPECPARAM_AVAILABLE = True
    SPECPARAM_BACKEND = "specparam"
except Exception:
    try:
        from fooof import FOOOF as SpectralModel
        SPECPARAM_AVAILABLE = True
        SPECPARAM_BACKEND = "fooof"
    except Exception:
        SpectralModel = None
        SPECPARAM_AVAILABLE = False
        SPECPARAM_BACKEND = "not_available"

from V3_config_loader import cfg
from V3_runtime import CheckpointStore, GracefulStop, RunControl, env_bool, make_stage_run_dir, stable_hash, utc_iso, write_manifest


RESUME = env_bool("NEUROPSYCHE_RESUME", True)
OUT_DIR = make_stage_run_dir(cfg.REPORTS_DIR, Path(__file__).name, resume=RESUME)
PREPROCESSING_CSV = Path(cfg.runtime_input("qc_cohort", cfg.REPORTS_DIR / "preprocessing.csv"))
FEATURE_SUMMARY_CSV = OUT_DIR / "feature_cache_summary.csv"
FEATURE_PREFLIGHT_CSV = OUT_DIR / "feature_cache_preflight_V3.csv"
FEATURE_DEFINITION_HASH = stable_hash({
    "durations": list(cfg.CACHE_DURATIONS),
    "bands": cfg.BANDS,
    "max_freq": cfg.MAX_FREQ,
    "reference_duration": cfg.REFERENCE_DURATION,
    "connectivity_densities": list(cfg.CONNECTIVITY_DENSITIES),
    "connectivity_absolute_weights": cfg.CONNECTIVITY_USE_ABSOLUTE_WEIGHTS,
    "graph_metrics": cfg.USE_GRAPH_METRICS,
    "entropy_max_samples": cfg.ENTROPY_MAX_SAMPLES,
    "sample_entropy_max_samples": cfg.SAMPLE_ENTROPY_MAX_SAMPLES,
    "higuchi_kmax": cfg.HIGUCHI_KMAX,
    "adaptive_welch": cfg.USE_ADAPTIVE_WELCH,
    "specparam_freq_range": list(cfg.SPECPARAM_FREQ_RANGE),
    "specparam_peak_widths": list(cfg.SPECPARAM_PEAK_WIDTHS),
    "specparam_max_peaks": cfg.SPECPARAM_MAX_PEAKS,
    "specparam_min_peak_height": cfg.SPECPARAM_MIN_PEAK_HEIGHT,
    "specparam_peak_threshold": cfg.SPECPARAM_PEAK_THRESHOLD,
    "specparam_aperiodic_mode": cfg.SPECPARAM_APERIODIC_MODE,
    "microstate_n_states": cfg.MICROSTATE_N_STATES,
    "microstate_max_peaks": cfg.MICROSTATE_MAX_PEAKS,
    "continuous_source": "BasicCleanedRaw/clean_raw.fif",
})


CONTINUOUS_FAMILIES = [
    "MICROSTATES",
    "CONNECTIVITY",
    "ENTROPY",
    "ASYMMETRY",
]

EPOCH_FAMILIES = [
    "BAND_POWER",
    "SPECTRAL_PARAMETERIZATION",
    "HJORTH",
    "KURTOSIS",
]


DEFAULT_MAX_FEATURE_WORKERS = 8

EPS = 1e-20


def normalize_bool(value: Any) -> bool:
    if pd.isna(value):
        return False

    if isinstance(value, bool):
        return value

    value_str = str(value).strip().lower()

    return value_str in ["true", "1", "yes", "y"]


def family_enabled(family: str) -> bool:
    if not hasattr(cfg, "FEATURE_FAMILIES"):
        return True

    return bool(cfg.FEATURE_FAMILIES.get(family, False))


def safe_family_filename(family: str) -> str:
    return family.lower() + ".npz"


def get_feature_workers() -> int:
    base_workers = cfg.get_n_workers()

    max_feature_workers = getattr(
        cfg,
        "MAX_FEATURE_WORKERS",
        DEFAULT_MAX_FEATURE_WORKERS,
    )

    return max(1, min(int(base_workers), int(max_feature_workers)))


def get_inclusion_column() -> str:
    return getattr(cfg, "FEATURE_INCLUSION_COLUMN", "minimum_include")


def load_subjects_for_feature_extraction() -> List[str]:
    if not PREPROCESSING_CSV.exists():
        raise FileNotFoundError(
            f"Could not find preprocessing QC file:\n{PREPROCESSING_CSV}\n"
            "Run preprocessing_qc_report.py first."
        )

    df = pd.read_csv(PREPROCESSING_CSV)

    inclusion_col = get_inclusion_column()

    if "subject_id" not in df.columns:
        raise ValueError("preprocessing.csv must contain subject_id column.")

    if inclusion_col not in df.columns:
        raise ValueError(
            f"preprocessing.csv does not contain inclusion column: {inclusion_col}"
        )

    df["subject_id"] = df["subject_id"].astype(str)
    df[inclusion_col] = df[inclusion_col].apply(normalize_bool)

    included = df[df[inclusion_col] == True].copy()

    return included["subject_id"].tolist()


def continuous_feature_folder(subject_id: str) -> Path:
    return cfg.subject_features_folder(subject_id) / "continuous"


def epoch_feature_folder(subject_id: str, duration: float) -> Path:
    return (
        cfg.subject_features_folder(subject_id)
        / "epochs"
        / cfg.duration_label(duration)
    )


def continuous_feature_path(subject_id: str, family: str) -> Path:
    return continuous_feature_folder(subject_id) / safe_family_filename(family)


def epoch_feature_path(subject_id: str, duration: float, family: str) -> Path:
    return epoch_feature_folder(subject_id, duration) / safe_family_filename(family)


def validate_cached_feature(path: Path) -> tuple[bool, str, dict]:
    if not path.exists():
        return False, "missing", {}
    try:
        with np.load(path, allow_pickle=True) as z:
            required = {"X_full", "X_summary", "feature_names", "summary_feature_names"}
            missing = required.difference(z.files)
            if missing:
                return False, f"missing arrays: {sorted(missing)}", {}
            summary = np.asarray(z["X_summary"], dtype=float).ravel()
            if summary.size == 0 or not np.all(np.isfinite(summary)):
                return False, "invalid X_summary", {}
            meta = {
                "n_full_rows": int(np.asarray(z["X_full"]).shape[0]) if np.asarray(z["X_full"]).ndim else 1,
                "n_full_features": int(np.asarray(z["X_full"]).shape[-1]) if np.asarray(z["X_full"]).ndim else int(np.asarray(z["X_full"]).size),
                "n_summary_features": int(summary.size),
                "cache_version": str(np.asarray(z["cache_version"]).item()) if "cache_version" in z.files else "legacy_v2",
                "feature_definition_hash": str(np.asarray(z["feature_definition_hash"]).item()) if "feature_definition_hash" in z.files else "legacy_v2",
                "source_size": int(np.asarray(z["source_size"]).item()) if "source_size" in z.files else -1,
                "source_mtime_ns": int(np.asarray(z["source_mtime_ns"]).item()) if "source_mtime_ns" in z.files else -1,
            }
        return True, "ok", meta
    except Exception as error:
        return False, str(error)[:200], {}


def should_skip_existing(path: Path, source_file: Path) -> bool:
    resume = env_bool("NEUROPSYCHE_RESUME", bool(cfg.RESUME_CACHE))
    force = env_bool("NEUROPSYCHE_FORCE_RERUN", bool(cfg.OVERWRITE_FEATURE_CACHE))
    valid, _, meta = validate_cached_feature(path)
    if not valid or meta.get("feature_definition_hash") != FEATURE_DEFINITION_HASH:
        return False
    if not source_file.is_file():
        return False
    source_matches = (
        meta.get("source_size") == source_file.stat().st_size
        and meta.get("source_mtime_ns") == source_file.stat().st_mtime_ns
    )
    if not source_matches:
        return False
    return resume and not force


def populate_cached_row(row: dict, path: Path) -> dict:
    valid, reason, meta = validate_cached_feature(path)
    if not valid:
        row["status"] = "failed"
        row["details"] = f"Existing feature cache failed validation: {reason}"
        return row
    row.update(meta)
    row["status"] = "cached"
    row["details"] = "Reused verified feature cache file."
    return row


def save_npz_atomic(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    temp_path = path.with_name(path.name + ".tmp")

    with open(temp_path, "wb") as f:
        np.savez_compressed(f, **arrays)

    temp_path.replace(path)


def make_summary_from_epoch_features(
    X_epoch: np.ndarray,
    feature_names: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    if X_epoch.ndim != 2:
        raise ValueError("X_epoch must be 2D: n_epochs × n_features.")

    mean_values = np.nanmean(X_epoch, axis=0)
    std_values = np.nanstd(X_epoch, axis=0)
    median_values = np.nanmedian(X_epoch, axis=0)
    q75 = np.nanpercentile(X_epoch, 75, axis=0)
    q25 = np.nanpercentile(X_epoch, 25, axis=0)
    iqr_values = q75 - q25

    X_summary = np.concatenate(
        [
            mean_values,
            std_values,
            median_values,
            iqr_values,
        ],
        axis=0,
    )

    summary_names = []

    for suffix in ["mean", "std", "median", "iqr"]:
        for name in feature_names:
            summary_names.append(f"{name}_{suffix}")

    return X_summary.astype(np.float32), np.array(summary_names, dtype=str)


def save_feature_file(
    output_path: Path,
    subject_id: str,
    family: str,
    source_file: Path,
    X_full: np.ndarray,
    X_summary: np.ndarray,
    feature_names: List[str] | np.ndarray,
    summary_feature_names: List[str] | np.ndarray,
    duration_sec: float | None,
    channel_names: List[str],
    extraction_method: str,
) -> None:
    duration_value = np.nan if duration_sec is None else float(duration_sec)

    save_npz_atomic(
        output_path,
        X_full=np.asarray(X_full, dtype=np.float32),
        X_summary=np.asarray(X_summary, dtype=np.float32),
        feature_names=np.asarray(feature_names, dtype=str),
        summary_feature_names=np.asarray(summary_feature_names, dtype=str),
        subject_id=np.asarray(subject_id, dtype=str),
        feature_family=np.asarray(family, dtype=str),
        duration_sec=np.asarray(duration_value, dtype=np.float32),
        channel_names=np.asarray(channel_names, dtype=str),
        source_file=np.asarray(str(source_file), dtype=str),
        extraction_method=np.asarray(extraction_method, dtype=str),
        cache_version=np.asarray("feature_cache_v3", dtype=str),
        feature_definition_hash=np.asarray(FEATURE_DEFINITION_HASH, dtype=str),
        created_utc=np.asarray(utc_iso(), dtype=str),
        source_size=np.asarray(source_file.stat().st_size if source_file.exists() else -1, dtype=np.int64),
        source_mtime_ns=np.asarray(source_file.stat().st_mtime_ns if source_file.exists() else -1, dtype=np.int64),
    )


def make_status_row(
    subject_id: str,
    family: str,
    feature_kind: str,
    duration_sec: float | None,
    output_path: Path,
    source_file: Path,
) -> dict:
    return {
        "subject_id": subject_id,
        "feature_kind": feature_kind,
        "feature_family": family,
        "duration_sec": np.nan if duration_sec is None else duration_sec,
        "duration_label": "" if duration_sec is None else cfg.duration_label(duration_sec),
        "status": "not_started",
        "output_path": str(output_path),
        "source_file": str(source_file),
        "n_full_rows": 0,
        "n_full_features": 0,
        "n_summary_features": 0,
        "details": "",
    }

def allow_all_zero_summary_for_family(family: str) -> bool:
    return family in {
        "KURTOSIS",
    }


def validate_feature_output(
    family: str,
    X_summary: np.ndarray,
    summary_feature_names: List[str] | np.ndarray,
    allow_all_zero: bool = False,
) -> None:
    X_summary = np.asarray(X_summary, dtype=float)
    summary_feature_names = np.asarray(summary_feature_names, dtype=str)

    if X_summary.size == 0:
        raise RuntimeError(f"{family} produced an empty summary vector.")

    if len(summary_feature_names) != X_summary.size:
        raise RuntimeError(
            f"{family} feature-name mismatch: "
            f"{len(summary_feature_names)} names for {X_summary.size} values."
        )

    if not np.any(np.isfinite(X_summary)):
        raise RuntimeError(f"{family} produced no finite summary features.")

    clean = np.nan_to_num(
        X_summary,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    if not allow_all_zero and np.all(clean == 0):
        raise RuntimeError(f"{family} produced an all-zero summary vector.")

def compute_welch_psd(
    data: np.ndarray,
    sfreq: float,
    max_freq: float | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    n_times = data.shape[-1]

    target = max(2, int(round(sfreq * 2.0)))
    if bool(cfg.USE_ADAPTIVE_WELCH):
        target = min(target, max(8, n_times // 4))
    nperseg = max(2, min(n_times, target))

    noverlap = nperseg // 2

    freqs, psd = signal.welch(
        data,
        fs=sfreq,
        nperseg=nperseg,
        noverlap=noverlap,
        axis=-1,
    )

    if max_freq is not None:
        keep = freqs <= float(max_freq)
        freqs = freqs[keep]
        psd = psd[..., keep]

    return freqs, psd


def band_mean_power(
    freqs: np.ndarray,
    psd: np.ndarray,
    low: float,
    high: float,
) -> np.ndarray:
    mask = (freqs >= low) & (freqs < high)

    if not np.any(mask):
        return np.full(psd.shape[:-1], np.nan)

    return np.nanmean(psd[..., mask], axis=-1)


def all_band_powers(
    freqs: np.ndarray,
    psd: np.ndarray,
) -> Dict[str, np.ndarray]:
    outputs = {}

    for band_name, (low, high) in cfg.BANDS.items():
        outputs[band_name] = band_mean_power(freqs, psd, low, high)

    return outputs


# EPOCH FEATURE: BAND POWER

def extract_band_power_from_epochs(
    epochs: mne.Epochs,
) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray, str]:
    data = epochs.get_data(copy=True)
    sfreq = float(epochs.info["sfreq"])
    channel_names = epochs.ch_names

    freqs, psd = compute_welch_psd(
        data=data,
        sfreq=sfreq,
        max_freq=cfg.MAX_FREQ,
    )

    band_powers = all_band_powers(freqs, psd)

    total_power = band_mean_power(
        freqs,
        psd,
        min(low for low, high in cfg.BANDS.values()),
        max(high for low, high in cfg.BANDS.values()),
    )

    feature_blocks = []
    feature_names = []

    for ch_idx, ch_name in enumerate(channel_names):
        for band_name in cfg.BANDS.keys():
            values = band_powers[band_name][:, ch_idx]
            feature_blocks.append(np.log10(values + EPS))
            feature_names.append(f"{ch_name}_{band_name}_log_power")

        for band_name in cfg.BANDS.keys():
            values = band_powers[band_name][:, ch_idx]
            rel_values = values / (total_power[:, ch_idx] + EPS)
            feature_blocks.append(rel_values)
            feature_names.append(f"{ch_name}_{band_name}_relative_power")

    X_full = np.column_stack(feature_blocks).astype(np.float32)

    X_summary, summary_feature_names = make_summary_from_epoch_features(
        X_full,
        feature_names,
    )

    return (
        X_full,
        X_summary,
        feature_names,
        summary_feature_names,
        "welch_log_absolute_and_relative_band_power",
    )


# EPOCH FEATURE: HJORTH

def extract_hjorth_from_epochs(
    epochs: mne.Epochs,
) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray, str]:
    data = epochs.get_data(copy=True)
    channel_names = epochs.ch_names

    activity = np.var(data, axis=-1)

    diff1 = np.diff(data, axis=-1)
    diff2 = np.diff(diff1, axis=-1)

    var_diff1 = np.var(diff1, axis=-1)
    var_diff2 = np.var(diff2, axis=-1)

    mobility = np.sqrt(var_diff1 / (activity + EPS))
    mobility_diff = np.sqrt(var_diff2 / (var_diff1 + EPS))
    complexity = mobility_diff / (mobility + EPS)

    feature_blocks = []
    feature_names = []

    for ch_idx, ch_name in enumerate(channel_names):
        feature_blocks.append(np.log10(activity[:, ch_idx] + EPS))
        feature_names.append(f"{ch_name}_hjorth_log_activity")

        feature_blocks.append(mobility[:, ch_idx])
        feature_names.append(f"{ch_name}_hjorth_mobility")

        feature_blocks.append(complexity[:, ch_idx])
        feature_names.append(f"{ch_name}_hjorth_complexity")

    X_full = np.column_stack(feature_blocks).astype(np.float32)

    X_summary, summary_feature_names = make_summary_from_epoch_features(
        X_full,
        feature_names,
    )

    return (
        X_full,
        X_summary,
        feature_names,
        summary_feature_names,
        "hjorth_activity_mobility_complexity",
    )


# EPOCH FEATURE: KURTOSIS

def extract_kurtosis_from_epochs(
    epochs: mne.Epochs,
) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray, str]:
    data = epochs.get_data(copy=True)
    channel_names = epochs.ch_names

    kurt = stats.kurtosis(
        data,
        axis=-1,
        fisher=True,
        bias=False,
        nan_policy="omit",
    )

    feature_blocks = []
    feature_names = []

    for ch_idx, ch_name in enumerate(channel_names):
        feature_blocks.append(kurt[:, ch_idx])
        feature_names.append(f"{ch_name}_kurtosis")

    X_full = np.column_stack(feature_blocks).astype(np.float32)

    X_summary, summary_feature_names = make_summary_from_epoch_features(
        X_full,
        feature_names,
    )

    return (
        X_full,
        X_summary,
        feature_names,
        summary_feature_names,
        "epoch_channel_kurtosis",
    )


# EPOCH FEATURE: SPECTRAL PARAMETERIZATION

def get_specparam_value(model, names, default=None):
    # 1. Direct attributes, mostly FOOOF-style
    for name in names:
        if hasattr(model, name):
            try:
                value = getattr(model, name)
                if value is not None:
                    return value
            except Exception:
                pass

    # 2. Results object, mostly FOOOF-style
    if hasattr(model, "get_results"):
        try:
            results = model.get_results()

            for name in names:
                clean_name = name[:-1] if name.endswith("_") else name

                if hasattr(results, name):
                    value = getattr(results, name)
                    if value is not None:
                        return value

                if hasattr(results, clean_name):
                    value = getattr(results, clean_name)
                    if value is not None:
                        return value
        except Exception:
            pass

    # 3. get_params API
    if hasattr(model, "get_params"):
        for name in names:
            clean_name = name[:-1] if name.endswith("_") else name

            # Direct call, e.g. get_params("aperiodic_params")
            try:
                value = model.get_params(clean_name)
                if value is not None:
                    return value
            except Exception:
                pass

            # specparam 2.x aliases
            if clean_name in {"aperiodic_params", "aperiodic_params_"}:
                try:
                    value = model.get_params("aperiodic")
                    if value is not None:
                        return value
                except Exception:
                    pass

            if clean_name in {"peak_params", "peak_params_"}:
                try:
                    value = model.get_params("peak")
                    if value is not None:
                        return value
                except Exception:
                    pass

            if clean_name in {"r_squared", "r_squared_"}:
                try:
                    value = model.get_params("r_squared")
                    if value is not None:
                        return value
                except Exception:
                    pass

            if clean_name in {"error", "error_"}:
                try:
                    value = model.get_params("error")
                    if value is not None:
                        return value
                except Exception:
                    pass

    return default

def extract_fast_spectral_parameterization_from_epochs(
    epochs: mne.Epochs,
) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray, str]:
    if not SPECPARAM_AVAILABLE or SpectralModel is None:
        raise ImportError(
            "specparam/FOOOF is not installed. Install it with:\n"
            "  pip install specparam\n"
            "or:\n"
            "  pip install fooof"
        )

    data = epochs.get_data(copy=True)
    sfreq = float(epochs.info["sfreq"])
    channel_names = epochs.ch_names

    freqs, psd = compute_welch_psd(
        data=data,
        sfreq=sfreq,
        max_freq=cfg.MAX_FREQ,
    )

    # Average across epochs first, then fit one model per channel.
    mean_psd = np.nanmean(psd, axis=0)

    freq_range = list(cfg.SPECPARAM_FREQ_RANGE)
    peak_width_limits = list(cfg.SPECPARAM_PEAK_WIDTHS)

    positive_freqs = freqs[freqs > 0]

    if len(positive_freqs) > 1:
        freq_resolution = float(np.median(np.diff(positive_freqs)))
    else:
        freq_resolution = 1.0

    # Protect specparam if peak width is narrower than PSD resolution.
    if peak_width_limits[0] < freq_resolution:
        peak_width_limits[0] = freq_resolution

    if peak_width_limits[0] >= peak_width_limits[1]:
        peak_width_limits[0] = max(0.5, peak_width_limits[1] / 2.0)

    channel_feature_names = [
        "specparam_offset",
        "specparam_exponent",
        "alpha_peak_frequency",
        "alpha_peak_power",
        "alpha_peak_bandwidth",
        "alpha_peak_present",
        "specparam_r2",
        "specparam_error",
        "specparam_fit_success",
    ]

    rows = []
    fit_successes = 0
    fit_errors = []

    for ch_idx, ch_name in enumerate(channel_names):
        spectrum = mean_psd[ch_idx].astype(float)

        spectrum = np.nan_to_num(
            spectrum,
            nan=EPS,
            posinf=EPS,
            neginf=EPS,
        )
        spectrum = np.maximum(spectrum, EPS)

        try:
            fm = SpectralModel(
                peak_width_limits=peak_width_limits,
                max_n_peaks=cfg.SPECPARAM_MAX_PEAKS,
                min_peak_height=cfg.SPECPARAM_MIN_PEAK_HEIGHT,
                peak_threshold=cfg.SPECPARAM_PEAK_THRESHOLD,
                aperiodic_mode=cfg.SPECPARAM_APERIODIC_MODE,
                verbose=False,
            )

            fm.fit(freqs, spectrum, freq_range)

            # Aperiodic parameters
            aperiodic = get_specparam_value(
                fm,
                ["aperiodic_params_", "aperiodic_params"],
                default=None,
            )

            if aperiodic is None:
                raise RuntimeError("Could not read aperiodic parameters.")

            aperiodic = np.asarray(aperiodic, dtype=float).ravel()

            if len(aperiodic) < 2:
                raise RuntimeError(
                    f"Aperiodic parameters too short: {aperiodic}"
                )

            offset = float(aperiodic[0])
            exponent = float(aperiodic[1])

            # Peak parameters
            peak_params = get_specparam_value(
                fm,
                ["peak_params_", "peak_params"],
                default=np.empty((0, 3)),
            )

            peak_params = np.asarray(peak_params, dtype=float)

            if peak_params.ndim == 1 and peak_params.size == 0:
                peak_params = np.empty((0, 3))

            if peak_params.ndim == 1 and peak_params.size >= 3:
                peak_params = peak_params.reshape(1, -1)

            alpha_peak_frequency = 0.0
            alpha_peak_power = 0.0
            alpha_peak_bandwidth = 0.0
            alpha_peak_present = 0.0

            if peak_params.size > 0 and peak_params.shape[1] >= 3:
                alpha_mask = (
                    (peak_params[:, 0] >= 8.0)
                    & (peak_params[:, 0] <= 13.0)
                )

                alpha_peaks = peak_params[alpha_mask]

                if len(alpha_peaks) > 0:
                    best_idx = int(np.argmax(alpha_peaks[:, 1]))
                    best_peak = alpha_peaks[best_idx]

                    alpha_peak_frequency = float(best_peak[0])
                    alpha_peak_power = float(best_peak[1])
                    alpha_peak_bandwidth = float(best_peak[2])
                    alpha_peak_present = 1.0

            # Fit quality
            # specparam 2.0.0rc6 may not expose r_squared/error in the same
            # way as FOOOF. If unavailable, keep them as zero.
            r2 = get_specparam_value(
                fm,
                ["r_squared_", "r_squared"],
                default=0.0,
            )

            error = get_specparam_value(
                fm,
                ["error_", "error"],
                default=0.0,
            )

            try:
                r2 = float(np.asarray(r2).ravel()[0])
            except Exception:
                r2 = 0.0

            try:
                error = float(np.asarray(error).ravel()[0])
            except Exception:
                error = 0.0

            row = [
                offset,
                exponent,
                alpha_peak_frequency,
                alpha_peak_power,
                alpha_peak_bandwidth,
                alpha_peak_present,
                r2,
                error,
                1.0,
            ]

            fit_successes += 1

        except Exception as error:
            fit_errors.append(f"{ch_name}: {repr(error)}")

            row = [
                0.0,  # offset
                0.0,  # exponent
                0.0,  # alpha peak frequency
                0.0,  # alpha peak power
                0.0,  # alpha peak bandwidth
                0.0,  # alpha peak present
                0.0,  # r2
                0.0,  # error
                0.0,  # fit success
            ]

        rows.append(row)

    if fit_successes == 0:
        first_errors = " | ".join(fit_errors[:5])
        raise RuntimeError(
            "All specparam channel fits failed. "
            f"First errors: {first_errors}"
        )

    X_full = np.asarray(rows, dtype=np.float32)

    offset_values = X_full[:, 0]
    exponent_values = X_full[:, 1]
    alpha_frequency_values = X_full[:, 2]
    alpha_power_values = X_full[:, 3]
    alpha_bandwidth_values = X_full[:, 4]
    alpha_present_values = X_full[:, 5]
    r2_values = X_full[:, 6]
    error_values = X_full[:, 7]
    fit_success_values = X_full[:, 8]

    fit_success_mask = fit_success_values > 0
    alpha_present_mask = alpha_present_values > 0

    if not np.any(fit_success_mask):
        raise RuntimeError("No successful specparam channel fits.")

    if np.any(alpha_present_mask):
        iaf_mean = float(np.mean(alpha_frequency_values[alpha_present_mask]))
        iaf_std = float(np.std(alpha_frequency_values[alpha_present_mask]))

        alpha_power_mean = float(np.mean(alpha_power_values[alpha_present_mask]))
        alpha_power_std = float(np.std(alpha_power_values[alpha_present_mask]))

        alpha_bw_mean = float(np.mean(alpha_bandwidth_values[alpha_present_mask]))
        alpha_bw_std = float(np.std(alpha_bandwidth_values[alpha_present_mask]))
    else:
        iaf_mean = 0.0
        iaf_std = 0.0
        alpha_power_mean = 0.0
        alpha_power_std = 0.0
        alpha_bw_mean = 0.0
        alpha_bw_std = 0.0

    alpha_peak_rate = float(np.mean(alpha_present_values[fit_success_mask]))

    X_summary = np.asarray(
        [
            float(np.mean(offset_values[fit_success_mask])),
            float(np.std(offset_values[fit_success_mask])),
            float(np.mean(exponent_values[fit_success_mask])),
            float(np.std(exponent_values[fit_success_mask])),
            iaf_mean,
            iaf_std,
            alpha_power_mean,
            alpha_power_std,
            alpha_bw_mean,
            alpha_bw_std,
            alpha_peak_rate,
            float(np.mean(r2_values[fit_success_mask])),
            float(np.std(r2_values[fit_success_mask])),
            float(np.mean(error_values[fit_success_mask])),
            float(np.std(error_values[fit_success_mask])),
            float(np.mean(fit_success_values)),
        ],
        dtype=np.float32,
    )

    summary_feature_names = np.asarray(
        [
            "specparam_offset_mean",
            "specparam_offset_std",
            "specparam_exponent_mean",
            "specparam_exponent_std",
            "iaf_mean",
            "iaf_std",
            "alpha_peak_power_mean",
            "alpha_peak_power_std",
            "alpha_peak_bw_mean",
            "alpha_peak_bw_std",
            "alpha_peak_rate",
            "specparam_r2_mean",
            "specparam_r2_std",
            "specparam_error_mean",
            "specparam_error_std",
            "specparam_fit_success_rate",
        ],
        dtype=str,
    )

    method = (
        f"full_{SPECPARAM_BACKEND}_mean_psd_per_channel_"
        f"freq_range_{freq_range[0]}_{freq_range[1]}_"
        f"peak_widths_{peak_width_limits[0]}_{peak_width_limits[1]}_"
        f"fit_success_{fit_successes}_of_{len(channel_names)}"
    )

    return (
        X_full,
        X_summary,
        channel_feature_names,
        summary_feature_names,
        method,
    )


# CONTINUOUS FEATURE: EXPANDED ENTROPY / COMPLEXITY

def get_entropy_max_samples() -> int:
    return int(getattr(cfg, "ENTROPY_MAX_SAMPLES", 20000))


def get_sample_entropy_max_samples() -> int:
    return int(getattr(cfg, "SAMPLE_ENTROPY_MAX_SAMPLES", 8000))


def get_higuchi_kmax() -> int:
    return int(getattr(cfg, "HIGUCHI_KMAX", 8))


def downsample_vector_evenly(x: np.ndarray, max_samples: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)

    if x.size <= max_samples:
        return x

    idx = np.linspace(0, x.size - 1, max_samples).astype(int)
    return x[idx]


def histogram_entropy_1d(x: np.ndarray, bins: int = 64) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    if x.size < 10:
        return np.nan

    hist, _ = np.histogram(x, bins=bins, density=False)

    if hist.sum() <= 0:
        return np.nan

    probs = hist.astype(float) / (hist.sum() + EPS)
    probs = probs[probs > 0]

    entropy = -np.sum(probs * np.log2(probs + EPS))
    max_entropy = np.log2(bins)

    return float(entropy / (max_entropy + EPS))


def permutation_entropy_1d(
    x: np.ndarray,
    order: int = 3,
    delay: int = 1,
    max_samples: int | None = None,
) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    if max_samples is None:
        max_samples = get_entropy_max_samples()

    x = downsample_vector_evenly(x, max_samples=max_samples)

    n_patterns = x.size - delay * (order - 1)

    if n_patterns <= order:
        return np.nan

    # Build delayed embedding:
    # shape = n_patterns × order
    embedded = np.column_stack(
        [
            x[i * delay : i * delay + n_patterns]
            for i in range(order)
        ]
    )

    patterns = np.argsort(embedded, axis=1)

    _, counts = np.unique(patterns, axis=0, return_counts=True)

    probs = counts.astype(float) / (np.sum(counts) + EPS)
    entropy = -np.sum(probs * np.log2(probs + EPS))

    max_entropy = np.log2(math.factorial(order))

    return float(entropy / (max_entropy + EPS))


def sample_entropy_1d(
    x: np.ndarray,
    m: int = 2,
    r_factor: float = 0.2,
    max_samples: int | None = None,
) -> float:
    from scipy.spatial import cKDTree

    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    if max_samples is None:
        max_samples = get_sample_entropy_max_samples()

    x = downsample_vector_evenly(x, max_samples=max_samples)

    if x.size < m + 10:
        return np.nan

    sd = np.std(x)

    if not np.isfinite(sd) or sd <= 0:
        return np.nan

    r = r_factor * sd

    n_m = x.size - m + 1
    n_m1 = x.size - (m + 1) + 1

    if n_m1 <= 1:
        return np.nan

    emb_m = np.column_stack(
        [
            x[i : i + n_m]
            for i in range(m)
        ]
    )

    emb_m1 = np.column_stack(
        [
            x[i : i + n_m1]
            for i in range(m + 1)
        ]
    )

    try:
        tree_m = cKDTree(emb_m)
        tree_m1 = cKDTree(emb_m1)

        # query_pairs counts unique i<j pairs within radius r.
        B = len(tree_m.query_pairs(r=r, p=np.inf))
        A = len(tree_m1.query_pairs(r=r, p=np.inf))

        if B <= 0 or A <= 0:
            return np.nan

        return float(-np.log((A + EPS) / (B + EPS)))

    except Exception:
        return np.nan


def spectral_entropy_from_psd(psd_values: np.ndarray) -> float:
    p = np.asarray(psd_values, dtype=float)
    p = p[np.isfinite(p)]

    if p.size <= 1:
        return np.nan

    p = np.maximum(p, 0.0)

    if np.sum(p) <= 0:
        return np.nan

    p = p / (np.sum(p) + EPS)
    p = p[p > 0]

    entropy = -np.sum(p * np.log2(p + EPS))
    max_entropy = np.log2(p.size)

    return float(entropy / (max_entropy + EPS))


def band_spectral_entropy(
    freqs: np.ndarray,
    psd_values: np.ndarray,
    low: float,
    high: float,
) -> float:
    mask = (freqs >= low) & (freqs < high)

    if not np.any(mask):
        return np.nan

    return spectral_entropy_from_psd(psd_values[mask])


def higuchi_fd_1d(
    x: np.ndarray,
    kmax: int | None = None,
    max_samples: int | None = None,
) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    if max_samples is None:
        max_samples = get_entropy_max_samples()

    if kmax is None:
        kmax = get_higuchi_kmax()

    x = downsample_vector_evenly(x, max_samples=max_samples)

    n = x.size

    if n < (kmax * 3):
        return np.nan

    lk_values = []
    k_values = []

    for k in range(1, kmax + 1):
        lm_values = []

        for m in range(k):
            idx = np.arange(m, n, k)

            if idx.size < 2:
                continue

            diffs = np.abs(np.diff(x[idx]))
            length = np.sum(diffs)

            # Higuchi normalization.
            norm = (n - 1) / (((idx.size - 1) * k) + EPS)
            lm = (length * norm) / k

            if np.isfinite(lm) and lm > 0:
                lm_values.append(lm)

        if len(lm_values) > 0:
            lk_values.append(np.mean(lm_values))
            k_values.append(k)

    if len(lk_values) < 2:
        return np.nan

    log_lk = np.log(np.asarray(lk_values) + EPS)
    log_inv_k = np.log(1.0 / (np.asarray(k_values) + EPS))

    try:
        slope, _ = np.polyfit(log_inv_k, log_lk, 1)
        return float(slope)
    except Exception:
        return np.nan


def extract_entropy_from_raw(
    raw: mne.io.BaseRaw,
) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray, str]:
    data = raw.get_data()
    sfreq = float(raw.info["sfreq"])
    channel_names = raw.ch_names

    freqs, psd = compute_welch_psd(
        data=data,
        sfreq=sfreq,
        max_freq=cfg.MAX_FREQ,
    )

    entropy_feature_names = [
        "histogram_entropy",
        "permutation_entropy_order3_delay1",
        "permutation_entropy_order4_delay1",
        "permutation_entropy_order5_delay1",
        "permutation_entropy_order3_delay2",
        "sample_entropy_m2_r0p2",
        "spectral_entropy_total",
    ]

    for band_name in cfg.BANDS.keys():
        entropy_feature_names.append(f"spectral_entropy_{band_name}")

    entropy_feature_names.append("higuchi_fractal_dimension")

    X_full_rows = []

    for ch_idx, _ in enumerate(channel_names):
        x = data[ch_idx]
        p = psd[ch_idx]

        row = [
            histogram_entropy_1d(x),
            permutation_entropy_1d(x, order=3, delay=1),
            permutation_entropy_1d(x, order=4, delay=1),
            permutation_entropy_1d(x, order=5, delay=1),
            permutation_entropy_1d(x, order=3, delay=2),
            sample_entropy_1d(x, m=2, r_factor=0.2),
            spectral_entropy_from_psd(p),
        ]

        for band_name, (low, high) in cfg.BANDS.items():
            row.append(
                band_spectral_entropy(
                    freqs=freqs,
                    psd_values=p,
                    low=low,
                    high=high,
                )
            )

        row.append(higuchi_fd_1d(x))

        X_full_rows.append(row)

    X_full = np.asarray(X_full_rows, dtype=np.float32)

    summary_values = []
    summary_names = []

    for ch_idx, ch_name in enumerate(channel_names):
        for feat_idx, feat_name in enumerate(entropy_feature_names):
            summary_values.append(X_full[ch_idx, feat_idx])
            summary_names.append(f"{ch_name}_{feat_name}")

    X_summary = np.asarray(summary_values, dtype=np.float32)

    method = (
        "expanded_entropy_histogram_permutation_orders_3_4_5_"
        "sample_entropy_spectral_entropy_band_entropy_higuchi_fd"
    )

    return (
        X_full,
        X_summary,
        entropy_feature_names,
        np.asarray(summary_names, dtype=str),
        method,
    )

# CONTINUOUS FEATURE: ASYMMETRY

def extract_asymmetry_from_raw(
    raw: mne.io.BaseRaw,
) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray, str]:
    data = raw.get_data()
    sfreq = float(raw.info["sfreq"])
    channel_names = raw.ch_names
    channel_index = {name: idx for idx, name in enumerate(channel_names)}

    pairs = [
        ("Fp1", "Fp2"),
        ("F7", "F8"),
        ("F3", "F4"),
        ("C3", "C4"),
        ("T3", "T4"),
        ("T7", "T8"),
        ("P3", "P4"),
        ("O1", "O2"),
    ]

    usable_pairs = [
        (left, right)
        for left, right in pairs
        if left in channel_index and right in channel_index
    ]

    if len(usable_pairs) == 0:
        raise ValueError("No usable left-right channel pairs found for asymmetry.")

    freqs, psd = compute_welch_psd(
        data=data,
        sfreq=sfreq,
        max_freq=cfg.MAX_FREQ,
    )

    band_powers = all_band_powers(freqs, psd)

    rows = []
    full_feature_names = []

    for left, right in usable_pairs:
        left_idx = channel_index[left]
        right_idx = channel_index[right]

        row = []

        for band_name in cfg.BANDS.keys():
            left_power = band_powers[band_name][left_idx]
            right_power = band_powers[band_name][right_idx]

            log_diff = np.log10(right_power + EPS) - np.log10(left_power + EPS)
            norm_diff = (right_power - left_power) / (right_power + left_power + EPS)

            row.append(log_diff)
            row.append(norm_diff)

            full_feature_names.append(f"{right}_minus_{left}_{band_name}_log_power")
            full_feature_names.append(f"{right}_minus_{left}_{band_name}_normalized_power")

        rows.append(row)

    X_full = np.asarray(rows, dtype=np.float32)

    summary_values = []
    summary_names = []

    for pair_idx, (left, right) in enumerate(usable_pairs):
        col_idx = 0

        for band_name in cfg.BANDS.keys():
            summary_values.append(X_full[pair_idx, col_idx])
            summary_names.append(f"{right}_minus_{left}_{band_name}_log_power")
            col_idx += 1

            summary_values.append(X_full[pair_idx, col_idx])
            summary_names.append(f"{right}_minus_{left}_{band_name}_normalized_power")
            col_idx += 1

    X_summary = np.asarray(summary_values, dtype=np.float32)

    return (
        X_full,
        X_summary,
        full_feature_names,
        np.asarray(summary_names, dtype=str),
        "left_right_band_power_asymmetry",
    )


# CONTINUOUS FEATURE: CONNECTIVITY + GRAPH METRICS

def bandpass_filter_data(
    data: np.ndarray,
    sfreq: float,
    low: float,
    high: float,
) -> np.ndarray:
    nyq = sfreq / 2.0

    low = max(float(low), 0.01)
    high = min(float(high), nyq - 0.5)

    if high <= low:
        raise ValueError(f"Invalid bandpass range: {low}-{high} Hz.")

    sos = signal.butter(
        N=4,
        Wn=[low, high],
        btype="bandpass",
        fs=sfreq,
        output="sos",
    )

    return signal.sosfiltfilt(sos, data, axis=-1)


def get_connectivity_densities() -> Tuple[float, ...]:
    densities = getattr(
        cfg,
        "CONNECTIVITY_DENSITIES",
        (0.10, 0.20, 0.30),
    )

    densities = tuple(float(d) for d in densities)

    for d in densities:
        if d <= 0 or d > 1:
            raise ValueError(
                f"CONNECTIVITY_DENSITIES must be between 0 and 1. Got {d}."
            )

    return densities


def upper_triangle_values(matrix: np.ndarray) -> np.ndarray:
    n = matrix.shape[0]
    i, j = np.triu_indices(n, k=1)
    return matrix[i, j]


def threshold_matrix_by_density(
    weight_matrix: np.ndarray,
    density: float,
) -> Tuple[np.ndarray, np.ndarray]:
    W = np.asarray(weight_matrix, dtype=float).copy()
    np.fill_diagonal(W, 0.0)

    n = W.shape[0]
    upper_i, upper_j = np.triu_indices(n, k=1)
    edge_values = W[upper_i, upper_j]

    n_possible = len(edge_values)

    if n_possible == 0:
        return np.zeros_like(W), np.zeros_like(W)

    n_keep = int(round(float(density) * n_possible))
    n_keep = max(1, min(n_keep, n_possible))

    # Sort strongest edges descending.
    order = np.argsort(edge_values)[::-1]
    keep_idx = order[:n_keep]

    A = np.zeros_like(W)
    W_thr = np.zeros_like(W)

    selected_i = upper_i[keep_idx]
    selected_j = upper_j[keep_idx]

    A[selected_i, selected_j] = 1.0
    A[selected_j, selected_i] = 1.0

    W_thr[selected_i, selected_j] = W[selected_i, selected_j]
    W_thr[selected_j, selected_i] = W[selected_j, selected_i]

    return W_thr, A


def shortest_path_distances_from_weights(W: np.ndarray) -> np.ndarray:
    from scipy.sparse.csgraph import shortest_path

    W = np.asarray(W, dtype=float)

    length = np.full_like(W, np.inf, dtype=float)

    edge_mask = W > 0
    length[edge_mask] = 1.0 / (W[edge_mask] + EPS)

    np.fill_diagonal(length, 0.0)

    distances = shortest_path(
        length,
        directed=False,
        unweighted=False,
    )

    return distances


def global_efficiency_from_distances(distances: np.ndarray) -> float:
    D = np.asarray(distances, dtype=float)
    n = D.shape[0]

    if n <= 1:
        return np.nan

    mask = ~np.eye(n, dtype=bool)

    inv_d = np.zeros_like(D, dtype=float)
    finite_mask = np.isfinite(D) & (D > 0)
    inv_d[finite_mask] = 1.0 / D[finite_mask]

    return float(np.sum(inv_d[mask]) / (n * (n - 1)))


def characteristic_path_length(distances: np.ndarray) -> float:
    D = np.asarray(distances, dtype=float)
    n = D.shape[0]

    mask = ~np.eye(n, dtype=bool)
    valid = mask & np.isfinite(D) & (D > 0)

    if not np.any(valid):
        return np.nan

    return float(np.mean(D[valid]))


def binary_clustering_coefficients(A: np.ndarray) -> np.ndarray:
    A = np.asarray(A, dtype=float)
    np.fill_diagonal(A, 0.0)

    degree = np.sum(A > 0, axis=1).astype(float)

    A3 = np.linalg.matrix_power((A > 0).astype(float), 3)
    triangles_twice = np.diag(A3)

    clustering = np.zeros(A.shape[0], dtype=float)

    valid = degree >= 2
    clustering[valid] = triangles_twice[valid] / (
        degree[valid] * (degree[valid] - 1.0) + EPS
    )

    return clustering


def weighted_clustering_coefficients(W: np.ndarray) -> np.ndarray:
    W = np.asarray(W, dtype=float).copy()
    np.fill_diagonal(W, 0.0)

    max_w = np.max(W)

    if max_w <= 0:
        return np.zeros(W.shape[0], dtype=float)

    Wn = W / (max_w + EPS)

    A = (Wn > 0).astype(float)
    degree = np.sum(A, axis=1)

    W13 = Wn ** (1.0 / 3.0)
    cyc = np.diag(W13 @ W13 @ W13)

    clustering = np.zeros(W.shape[0], dtype=float)

    valid = degree >= 2
    clustering[valid] = cyc[valid] / (
        degree[valid] * (degree[valid] - 1.0) + EPS
    )

    return clustering


def local_efficiency_weighted(W: np.ndarray, A: np.ndarray) -> float:
    W = np.asarray(W, dtype=float)
    A = np.asarray(A, dtype=float)

    n = W.shape[0]
    local_values = []

    for node in range(n):
        neighbors = np.where(A[node] > 0)[0]

        if len(neighbors) < 2:
            local_values.append(0.0)
            continue

        sub_W = W[np.ix_(neighbors, neighbors)]

        try:
            sub_D = shortest_path_distances_from_weights(sub_W)
            local_values.append(global_efficiency_from_distances(sub_D))
        except Exception:
            local_values.append(0.0)

    return float(np.mean(local_values))


def binary_transitivity(A: np.ndarray) -> float:
    A = (np.asarray(A, dtype=float) > 0).astype(float)
    np.fill_diagonal(A, 0.0)

    degree = np.sum(A, axis=1)

    triangles = np.trace(np.linalg.matrix_power(A, 3)) / 6.0
    triples = np.sum(degree * (degree - 1.0)) / 2.0

    if triples <= 0:
        return 0.0

    return float((3.0 * triangles) / (triples + EPS))


def eigenvector_centrality_weighted(W: np.ndarray) -> np.ndarray:
    W = np.asarray(W, dtype=float)
    np.fill_diagonal(W, 0.0)

    if np.max(W) <= 0:
        return np.zeros(W.shape[0], dtype=float)

    try:
        eigenvalues, eigenvectors = np.linalg.eigh(W)
        idx = int(np.argmax(eigenvalues))
        centrality = np.abs(eigenvectors[:, idx])
        centrality = centrality / (np.sum(centrality) + EPS)
        return centrality.astype(float)
    except Exception:
        return np.zeros(W.shape[0], dtype=float)


def spectral_modularity_two_way(W: np.ndarray) -> float:
    W = np.asarray(W, dtype=float)
    np.fill_diagonal(W, 0.0)

    strength = np.sum(W, axis=1)
    m2 = np.sum(strength)

    if m2 <= 0:
        return 0.0

    expected = np.outer(strength, strength) / (m2 + EPS)
    B = W - expected

    try:
        eigenvalues, eigenvectors = np.linalg.eigh(B)
        leading = eigenvectors[:, int(np.argmax(eigenvalues))]
        communities = leading >= 0

        # If all nodes fall into one community, modularity is not meaningful.
        if np.all(communities == communities[0]):
            return 0.0

        same = communities[:, None] == communities[None, :]

        Q = np.sum(B[same]) / (m2 + EPS)

        return float(Q)

    except Exception:
        return 0.0


def summarize_vector(
    values: np.ndarray,
    prefix: str,
    summary_values: List[float],
    summary_names: List[str],
) -> None:
    values = np.asarray(values, dtype=float)

    summary_values.append(float(np.nanmean(values)))
    summary_names.append(f"{prefix}_mean")

    summary_values.append(float(np.nanstd(values)))
    summary_names.append(f"{prefix}_std")

    summary_values.append(float(np.nanmin(values)))
    summary_names.append(f"{prefix}_min")

    summary_values.append(float(np.nanmax(values)))
    summary_names.append(f"{prefix}_max")


def extract_connectivity_from_raw(
    raw: mne.io.BaseRaw,
) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray, str]:
    data = raw.get_data()
    sfreq = float(raw.info["sfreq"])
    channel_names = raw.ch_names
    n_channels = len(channel_names)

    densities = get_connectivity_densities()

    use_abs = getattr(
        cfg,
        "CONNECTIVITY_USE_ABSOLUTE_WEIGHTS",
        True,
    )

    matrix_layers = []
    matrix_layer_names = []

    summary_values = []
    summary_names = []

    upper_i, upper_j = np.triu_indices(n_channels, k=1)

    for band_name, (low, high) in cfg.BANDS.items():
        # Band-limited signal
        filtered = bandpass_filter_data(
            data=data,
            sfreq=sfreq,
            low=low,
            high=high,
        )

        # Correlation matrix
        corr = np.corrcoef(filtered)

        corr = np.nan_to_num(
            corr,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        corr = np.clip(corr, -1.0, 1.0)

        signed_corr = corr.copy()
        np.fill_diagonal(signed_corr, 0.0)

        abs_corr = np.abs(signed_corr)
        np.fill_diagonal(abs_corr, 0.0)

        matrix_layers.append(signed_corr)
        matrix_layer_names.append(f"{band_name}_signed_correlation_matrix")

        matrix_layers.append(abs_corr)
        matrix_layer_names.append(f"{band_name}_absolute_correlation_matrix")

        signed_edges = signed_corr[upper_i, upper_j]
        abs_edges = abs_corr[upper_i, upper_j]

        summarize_vector(
            signed_edges,
            f"{band_name}_signed_connectivity",
            summary_values,
            summary_names,
        )

        summarize_vector(
            abs_edges,
            f"{band_name}_absolute_connectivity",
            summary_values,
            summary_names,
        )

        base_weights = abs_corr if use_abs else np.maximum(signed_corr, 0.0)

        graph_densities = densities if bool(cfg.USE_GRAPH_METRICS) else []
        for density in graph_densities:
            density_label = str(density).replace(".", "p")

            W_thr, A = threshold_matrix_by_density(
                weight_matrix=base_weights,
                density=density,
            )

            matrix_layers.append(W_thr)
            matrix_layer_names.append(
                f"{band_name}_density_{density_label}_weighted_adjacency"
            )

            degree = np.sum(A > 0, axis=1).astype(float)
            strength = np.sum(W_thr, axis=1).astype(float)

            binary_clustering = binary_clustering_coefficients(A)
            weighted_clustering = weighted_clustering_coefficients(W_thr)
            eigen_centrality = eigenvector_centrality_weighted(W_thr)

            distances = shortest_path_distances_from_weights(W_thr)

            global_eff = global_efficiency_from_distances(distances)
            char_path = characteristic_path_length(distances)
            local_eff = local_efficiency_weighted(W_thr, A)
            transitivity = binary_transitivity(A)
            modularity = spectral_modularity_two_way(W_thr)

            actual_density = np.sum(A[upper_i, upper_j] > 0) / len(upper_i)

            graph_prefix = f"{band_name}_density_{density_label}"

            # Global graph metrics
            summary_values.extend(
                [
                    float(actual_density),
                    float(global_eff),
                    float(char_path),
                    float(local_eff),
                    float(transitivity),
                    float(modularity),
                ]
            )

            summary_names.extend(
                [
                    f"{graph_prefix}_actual_density",
                    f"{graph_prefix}_global_efficiency",
                    f"{graph_prefix}_characteristic_path_length",
                    f"{graph_prefix}_local_efficiency",
                    f"{graph_prefix}_binary_transitivity",
                    f"{graph_prefix}_spectral_modularity",
                ]
            )

            # Nodal metric summaries
            summarize_vector(
                degree,
                f"{graph_prefix}_degree",
                summary_values,
                summary_names,
            )

            summarize_vector(
                strength,
                f"{graph_prefix}_strength",
                summary_values,
                summary_names,
            )

            summarize_vector(
                binary_clustering,
                f"{graph_prefix}_binary_clustering",
                summary_values,
                summary_names,
            )

            summarize_vector(
                weighted_clustering,
                f"{graph_prefix}_weighted_clustering",
                summary_values,
                summary_names,
            )

            summarize_vector(
                eigen_centrality,
                f"{graph_prefix}_eigenvector_centrality",
                summary_values,
                summary_names,
            )

            # Channel-level nodal metrics
            for ch_idx, ch_name in enumerate(channel_names):
                summary_values.append(float(degree[ch_idx]))
                summary_names.append(f"{graph_prefix}_{ch_name}_degree")

                summary_values.append(float(strength[ch_idx]))
                summary_names.append(f"{graph_prefix}_{ch_name}_strength")

                summary_values.append(float(binary_clustering[ch_idx]))
                summary_names.append(f"{graph_prefix}_{ch_name}_binary_clustering")

                summary_values.append(float(weighted_clustering[ch_idx]))
                summary_names.append(f"{graph_prefix}_{ch_name}_weighted_clustering")

                summary_values.append(float(eigen_centrality[ch_idx]))
                summary_names.append(f"{graph_prefix}_{ch_name}_eigenvector_centrality")

    X_full = np.asarray(matrix_layers, dtype=np.float32)
    X_summary = np.asarray(summary_values, dtype=np.float32)

    method = "band_limited_pearson_connectivity"
    if bool(cfg.USE_GRAPH_METRICS):
        method += "_with_density_thresholded_weighted_graph_metrics"

    if not np.any(np.isfinite(X_summary)):
        raise RuntimeError("Connectivity produced no finite summary features.")

    if np.all(np.nan_to_num(X_summary, nan=0.0, posinf=0.0, neginf=0.0) == 0):
        raise RuntimeError("Connectivity produced an all-zero summary vector.")

    if len(summary_names) != X_summary.size:
        raise RuntimeError(
            f"Connectivity feature-name mismatch: "
            f"{len(summary_names)} names for {X_summary.size} values."
        )

    return (
        X_full,
        X_summary,
        matrix_layer_names,
        np.asarray(summary_names, dtype=str),
        method,
    )

# CONTINUOUS FEATURE: MICROSTATES

def normalize_maps(maps: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(maps, axis=1, keepdims=True) + EPS
    return maps / norms


def extract_microstates_from_raw(
    raw: mne.io.BaseRaw,
) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray, str]:
    data = raw.get_data()
    sfreq = float(raw.info["sfreq"])
    channel_names = raw.ch_names

    n_states = int(cfg.MICROSTATE_N_STATES)
    max_peaks = int(cfg.MICROSTATE_MAX_PEAKS)

    if data.shape[0] < 2:
        raise ValueError("Need at least 2 channels for microstate extraction.")

    # Topographic centering and GFP
    # Remove average across channels at each time point.
    # This makes topographies more comparable.
    data_centered = data - np.mean(data, axis=0, keepdims=True)

    gfp = np.std(data_centered, axis=0)

    if np.all(gfp <= 0):
        raise ValueError("GFP is zero everywhere; cannot extract microstates.")

    # GFP peak selection
    min_distance = max(1, int(round(0.02 * sfreq)))

    peaks, _ = signal.find_peaks(
        gfp,
        distance=min_distance,
    )

    if len(peaks) < n_states:
        raise ValueError(
            f"Too few GFP peaks for microstates: {len(peaks)} peaks, "
            f"need at least {n_states}."
        )

    # Keep strongest GFP peaks if there are too many.
    if len(peaks) > max_peaks:
        strongest_idx = np.argsort(gfp[peaks])[-max_peaks:]
        peaks = np.sort(peaks[strongest_idx])

    peak_maps = data_centered[:, peaks].T
    peak_maps = normalize_maps(peak_maps)

    # Cluster GFP peak maps
    kmeans = KMeans(
        n_clusters=n_states,
        random_state=cfg.RANDOM_SEED,
        n_init=50,
    )

    peak_labels = kmeans.fit_predict(peak_maps)

    maps = normalize_maps(kmeans.cluster_centers_)

    # Backfit maps to every time point
    # Normalize every time-point topography.
    time_maps = data_centered.T
    time_maps = normalize_maps(time_maps)

    # Correlation of every time point with every map.
    # Use absolute correlation because microstate polarity is usually ignored.
    corr = time_maps @ maps.T
    abs_corr = np.abs(corr)

    labels = np.argmax(abs_corr, axis=1)
    max_corr = abs_corr[np.arange(abs_corr.shape[0]), labels]

    # Sort states by GEV, highest first
    total_gfp_power = np.sum(gfp ** 2) + EPS

    gev_unsorted = []

    for state_idx in range(n_states):
        state_mask = labels == state_idx

        if not np.any(state_mask):
            gev_unsorted.append(0.0)
            continue

        gev = np.sum(
            (gfp[state_mask] ** 2) * (max_corr[state_mask] ** 2)
        ) / total_gfp_power

        gev_unsorted.append(float(gev))

    order = np.argsort(gev_unsorted)[::-1]

    old_to_new = {old: new for new, old in enumerate(order)}

    labels_sorted = np.array(
        [old_to_new[label] for label in labels],
        dtype=int,
    )

    maps_sorted = maps[order]

    # Recompute max correlations after sorting maps.
    corr_sorted = time_maps @ maps_sorted.T
    abs_corr_sorted = np.abs(corr_sorted)
    max_corr_sorted = abs_corr_sorted[
        np.arange(abs_corr_sorted.shape[0]),
        labels_sorted,
    ]

    # Compute microstate statistics
    gev_values = []
    coverage_values = []
    occurrence_values = []
    mean_duration_ms_values = []

    recording_seconds = raw.n_times / sfreq

    # Compress labels into segments.
    segment_labels = []
    segment_lengths = []

    current_label = labels_sorted[0]
    current_length = 1

    for label in labels_sorted[1:]:
        if label == current_label:
            current_length += 1
        else:
            segment_labels.append(current_label)
            segment_lengths.append(current_length)
            current_label = label
            current_length = 1

    segment_labels.append(current_label)
    segment_lengths.append(current_length)

    segment_labels = np.asarray(segment_labels, dtype=int)
    segment_lengths = np.asarray(segment_lengths, dtype=float)

    for state_idx in range(n_states):
        state_mask_time = labels_sorted == state_idx

        if not np.any(state_mask_time):
            gev_values.append(0.0)
            coverage_values.append(0.0)
            occurrence_values.append(0.0)
            mean_duration_ms_values.append(0.0)
            continue

        gev = np.sum(
            (gfp[state_mask_time] ** 2)
            * (max_corr_sorted[state_mask_time] ** 2)
        ) / total_gfp_power

        coverage = np.mean(state_mask_time)

        state_segment_mask = segment_labels == state_idx
        n_segments = np.sum(state_segment_mask)

        occurrence = n_segments / (recording_seconds + EPS)

        mean_duration_ms = (
            np.mean(segment_lengths[state_segment_mask]) / sfreq * 1000.0
            if n_segments > 0
            else 0.0
        )

        gev_values.append(float(gev))
        coverage_values.append(float(coverage))
        occurrence_values.append(float(occurrence))
        mean_duration_ms_values.append(float(mean_duration_ms))

    # Transition matrix using segment-to-segment transitions
    transition = np.zeros((n_states, n_states), dtype=float)

    if len(segment_labels) > 1:
        for a, b in zip(segment_labels[:-1], segment_labels[1:]):
            if a != b:
                transition[a, b] += 1.0

        row_sums = transition.sum(axis=1, keepdims=True)
        transition = transition / (row_sums + EPS)

    # Build summary vector
    summary_values = []
    summary_names = []

    for state_idx in range(n_states):
        summary_values.append(gev_values[state_idx])
        summary_names.append(f"microstate_{state_idx + 1}_gev")

        summary_values.append(coverage_values[state_idx])
        summary_names.append(f"microstate_{state_idx + 1}_coverage")

        summary_values.append(occurrence_values[state_idx])
        summary_names.append(f"microstate_{state_idx + 1}_occurrence_per_sec")

        summary_values.append(mean_duration_ms_values[state_idx])
        summary_names.append(f"microstate_{state_idx + 1}_mean_duration_ms")

    for state_i in range(n_states):
        for state_j in range(n_states):
            summary_values.append(transition[state_i, state_j])
            summary_names.append(
                f"microstate_transition_{state_i + 1}_to_{state_j + 1}"
            )

    for state_idx in range(n_states):
        for ch_idx, ch_name in enumerate(channel_names):
            summary_values.append(maps_sorted[state_idx, ch_idx])
            summary_names.append(f"microstate_{state_idx + 1}_map_{ch_name}")

    X_full = maps_sorted.astype(np.float32)
    X_summary = np.asarray(summary_values, dtype=np.float32)

    full_feature_names = [
        f"microstate_{state_idx + 1}_map"
        for state_idx in range(n_states)
    ]

    return (
        X_full,
        X_summary,
        full_feature_names,
        np.asarray(summary_names, dtype=str),
        "continuous_gfp_peak_kmeans_microstate_backfitting",
    )


# FEATURE DISPATCHERS

def extract_continuous_family(
    raw: mne.io.BaseRaw,
    family: str,
) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray, str]:
    if family == "MICROSTATES":
        return extract_microstates_from_raw(raw)

    if family == "CONNECTIVITY":
        return extract_connectivity_from_raw(raw)

    if family == "ENTROPY":
        return extract_entropy_from_raw(raw)

    if family == "ASYMMETRY":
        return extract_asymmetry_from_raw(raw)

    raise ValueError(f"Unknown continuous feature family: {family}")


def extract_epoch_family(
    epochs: mne.Epochs,
    family: str,
) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray, str]:
    if family == "BAND_POWER":
        return extract_band_power_from_epochs(epochs)

    if family == "SPECTRAL_PARAMETERIZATION":
        return extract_fast_spectral_parameterization_from_epochs(epochs)

    if family == "HJORTH":
        return extract_hjorth_from_epochs(epochs)

    if family == "KURTOSIS":
        return extract_kurtosis_from_epochs(epochs)

    raise ValueError(f"Unknown epoch feature family: {family}")


# PROCESS ONE SUBJECT

def process_continuous_features(subject_id: str) -> List[dict]:
    rows = []

    source_file = cfg.subject_clean_raw_path(subject_id)

    enabled_families = [
        family
        for family in CONTINUOUS_FAMILIES
        if family_enabled(family)
    ]

    if len(enabled_families) == 0:
        return rows

    # Make status rows first.
    output_paths = {
        family: continuous_feature_path(subject_id, family)
        for family in enabled_families
    }

    families_to_run = []

    for family in enabled_families:
        output_path = output_paths[family]
        row = make_status_row(
            subject_id=subject_id,
            family=family,
            feature_kind="continuous",
            duration_sec=None,
            output_path=output_path,
            source_file=source_file,
        )

        if should_skip_existing(output_path, source_file):
            rows.append(populate_cached_row(row, output_path))
        else:
            families_to_run.append((family, row))

    if len(families_to_run) == 0:
        return rows

    if not source_file.exists():
        for family, row in families_to_run:
            row["status"] = "failed"
            row["details"] = "Basic cleaned raw file not found."
            rows.append(row)
        return rows

    try:
        raw = mne.io.read_raw_fif(
            source_file,
            preload=True,
            verbose="ERROR",
        )
    except Exception as error:
        for family, row in families_to_run:
            row["status"] = "failed"
            row["details"] = f"Could not read basic cleaned raw: {str(error)[:200]}"
            rows.append(row)
        return rows

    channel_names = raw.ch_names

    for family, row in families_to_run:
        output_path = output_paths[family]

        try:
            (
                X_full,
                X_summary,
                feature_names,
                summary_feature_names,
                method,
            ) = extract_continuous_family(raw, family)

            validate_feature_output(
                family=family,
                X_summary=X_summary,
                summary_feature_names=summary_feature_names,
                allow_all_zero=allow_all_zero_summary_for_family(family),
            )

            save_feature_file(
                output_path=output_path,
                subject_id=subject_id,
                family=family,
                source_file=source_file,
                X_full=X_full,
                X_summary=X_summary,
                feature_names=feature_names,
                summary_feature_names=summary_feature_names,
                duration_sec=None,
                channel_names=channel_names,
                extraction_method=method,
            )

            row["status"] = "passed"
            row["n_full_rows"] = int(X_full.shape[0]) if X_full.ndim >= 1 else 1
            row["n_full_features"] = int(X_full.shape[-1]) if X_full.ndim >= 2 else int(X_full.size)
            row["n_summary_features"] = int(X_summary.size)
            row["details"] = f"Feature extraction passed using {method}."

        except Exception as error:
            row["status"] = "failed"
            row["details"] = (
                f"Feature extraction failed: {str(error)[:200]} | "
                f"{traceback.format_exc()[:500]}"
            )

        rows.append(row)

    return rows


def process_epoch_features(subject_id: str) -> List[dict]:
    rows = []

    enabled_families = [
        family
        for family in EPOCH_FAMILIES
        if family_enabled(family)
    ]

    if len(enabled_families) == 0:
        return rows

    for duration in cfg.CACHE_DURATIONS:
        source_file = cfg.subject_epoch_path(subject_id, float(duration))

        output_paths = {
            family: epoch_feature_path(subject_id, float(duration), family)
            for family in enabled_families
        }

        families_to_run = []

        for family in enabled_families:
            output_path = output_paths[family]

            row = make_status_row(
                subject_id=subject_id,
                family=family,
                feature_kind="epoch",
                duration_sec=float(duration),
                output_path=output_path,
                source_file=source_file,
            )

            if should_skip_existing(output_path, source_file):
                rows.append(populate_cached_row(row, output_path))
            else:
                families_to_run.append((family, row))

        if len(families_to_run) == 0:
            continue

        if not source_file.exists():
            for family, row in families_to_run:
                row["status"] = "failed"
                row["details"] = "Cleaned epoch file not found."
                rows.append(row)
            continue

        try:
            epochs = mne.read_epochs(
                source_file,
                preload=True,
                verbose="ERROR",
            )
        except Exception as error:
            for family, row in families_to_run:
                row["status"] = "failed"
                row["details"] = f"Could not read epoch file: {str(error)[:200]}"
                rows.append(row)
            continue

        channel_names = epochs.ch_names

        for family, row in families_to_run:
            output_path = output_paths[family]

            try:
                (
                    X_full,
                    X_summary,
                    feature_names,
                    summary_feature_names,
                    method,
                ) = extract_epoch_family(epochs, family)

                validate_feature_output(
                    family=family,
                    X_summary=X_summary,
                    summary_feature_names=summary_feature_names,
                    allow_all_zero=allow_all_zero_summary_for_family(family),
                )

                save_feature_file(
                    output_path=output_path,
                    subject_id=subject_id,
                    family=family,
                    source_file=source_file,
                    X_full=X_full,
                    X_summary=X_summary,
                    feature_names=feature_names,
                    summary_feature_names=summary_feature_names,
                    duration_sec=float(duration),
                    channel_names=channel_names,
                    extraction_method=method,
                )

                row["status"] = "passed"
                row["n_full_rows"] = int(X_full.shape[0]) if X_full.ndim >= 1 else 1
                row["n_full_features"] = int(X_full.shape[-1]) if X_full.ndim >= 2 else int(X_full.size)
                row["n_summary_features"] = int(X_summary.size)
                row["details"] = f"Feature extraction passed using {method}."

            except Exception as error:
                row["status"] = "failed"
                row["details"] = (
                    f"Feature extraction failed: {str(error)[:200]} | "
                    f"{traceback.format_exc()[:500]}"
                )

            rows.append(row)

        # Help Python release memory between durations.
        del epochs

    return rows


def process_one_subject(subject_id: str) -> List[dict]:
    rows = []

    rows.extend(process_continuous_features(subject_id))
    rows.extend(process_epoch_features(subject_id))

    return rows


# MAIN

def _write_feature_summary(rows_by_key: dict[str, dict]) -> None:
    rows = list(rows_by_key.values())
    rows.sort(key=lambda r: (str(r.get("subject_id", "")), str(r.get("feature_kind", "")), str(r.get("feature_family", "")), str(r.get("duration_label", ""))))
    df = pd.DataFrame(rows)
    tmp = FEATURE_SUMMARY_CSV.with_suffix(".csv.tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(FEATURE_SUMMARY_CSV)


def _make_preflight(subject_ids: list[str], summary_df: pd.DataFrame) -> pd.DataFrame:
    expected_cont = [f for f in CONTINUOUS_FAMILIES if family_enabled(f)]
    expected_epoch = [f for f in EPOCH_FAMILIES if family_enabled(f)]
    expected_jobs = len(expected_cont) + len(expected_epoch) * len(cfg.CACHE_DURATIONS)
    rows = []
    for sid in subject_ids:
        sub = summary_df[summary_df["subject_id"].astype(str) == str(sid)] if not summary_df.empty else pd.DataFrame()
        ok = sub["status"].astype(str).isin(["passed", "cached"]).sum() if not sub.empty else 0
        rows.append({
            "subject_id": str(sid),
            "expected_feature_files": expected_jobs,
            "valid_feature_files": int(ok),
            "complete_feature_cache": bool(ok == expected_jobs),
            "feature_definition_hash": FEATURE_DEFINITION_HASH,
        })
    return pd.DataFrame(rows)


def main() -> None:
    mne.set_log_level("ERROR")
    cfg.apply_persisted_settings()
    cfg.make_directories()
    control = RunControl.from_environment(OUT_DIR / "_control")
    checkpoint = CheckpointStore(OUT_DIR / "_checkpoints" / "completed_feature_jobs.txt")
    resume = env_bool("NEUROPSYCHE_RESUME", bool(cfg.RESUME_CACHE))
    force = env_bool("NEUROPSYCHE_FORCE_RERUN", bool(cfg.OVERWRITE_FEATURE_CACHE))

    print("\n" + "=" * 70)
    print("Starting V3 feature cache builder")
    print(f"QC input: {PREPROCESSING_CSV}")
    print(f"Resume: {resume} | forced rerun: {force}")
    subject_ids = load_subjects_for_feature_extraction()
    n_workers = get_feature_workers()

    rows_by_key: dict[str, dict] = {}
    if FEATURE_SUMMARY_CSV.exists() and resume and not force:
        try:
            prior = pd.read_csv(FEATURE_SUMMARY_CSV, dtype={"subject_id": str})
            for _, row in prior.iterrows():
                key = f"{row['subject_id']}|{row['feature_kind']}|{row['feature_family']}|{row.get('duration_label','')}"
                rows_by_key[key] = row.to_dict()
        except Exception as error:
            print(f"Could not load prior feature summary: {error}")

    batch_size = max(1, n_workers)
    try:
        for start in range(0, len(subject_ids), batch_size):
            control.checkpoint(f"before feature subject batch {start // batch_size + 1}")
            batch = subject_ids[start:start + batch_size]
            batch_results = Parallel(n_jobs=n_workers, backend="loky")(
                delayed(process_one_subject)(subject_id) for subject_id in batch
            )
            for subject_rows in batch_results:
                for row in subject_rows:
                    key = f"{row['subject_id']}|{row['feature_kind']}|{row['feature_family']}|{row.get('duration_label','')}"
                    rows_by_key[key] = row
                    checkpoint.add(key)
            _write_feature_summary(rows_by_key)
            print(f"Checkpoint: {min(start + batch_size, len(subject_ids))}/{len(subject_ids)} subjects", flush=True)
    except GracefulStop as stop:
        _write_feature_summary(rows_by_key)
        print(f"Graceful stop: {stop}")
        return

    summary_df = pd.DataFrame(rows_by_key.values())
    preflight = _make_preflight(subject_ids, summary_df)
    preflight.to_csv(FEATURE_PREFLIGHT_CSV, index=False)
    print("\n" + "=" * 70)
    print("Feature cache builder finished")
    print(f"Jobs: {len(summary_df)} | complete subjects: {int(preflight['complete_feature_cache'].sum())}/{len(preflight)}")
    print(f"Summary: {FEATURE_SUMMARY_CSV}")
    print(f"Preflight: {FEATURE_PREFLIGHT_CSV}")
    print("=" * 70)
    write_manifest(OUT_DIR, {
        "script": Path(__file__).name,
        "stage": 4,
        "inputs": {"qc_cohort": str(PREPROCESSING_CSV)},
        "outputs": {"feature_summary": str(FEATURE_SUMMARY_CSV), "feature_preflight": str(FEATURE_PREFLIGHT_CSV)},
        "n_subjects": int(len(preflight)),
        "n_complete_subjects": int(preflight["complete_feature_cache"].sum()),
        "feature_definition_hash": FEATURE_DEFINITION_HASH,
        "settings_hash": cfg.settings_hash_for(4),
    })
    (OUT_DIR / "FEATURE_CACHE_COMPLETE.flag").write_text("complete", encoding="utf-8")


if __name__ == "__main__":
    main()
