"""Epoch-level EEG preprocessing for the MAPPER pipeline."""

from __future__ import annotations

import traceback
from pathlib import Path
from typing import List, Tuple
import subprocess
import sys
import tempfile

import mne
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from V3_config_loader import cfg
from V3_runtime import CheckpointStore, GracefulStop, RunControl, env_bool, make_stage_run_dir, load_json, save_json_atomic, stable_hash, utc_iso, write_manifest


# TRY TO IMPORT AUTOREJECT

try:
    from autoreject import AutoReject
    AUTOREJECT_AVAILABLE = True
except Exception:
    AutoReject = None
    AUTOREJECT_AVAILABLE = False


# PATHS

RESUME = env_bool("NEUROPSYCHE_RESUME", True)
OUT_DIR = make_stage_run_dir(cfg.REPORTS_DIR, Path(__file__).name, resume=RESUME)
BASIC_SUMMARY_CSV = Path(cfg.runtime_input("basic_summary", cfg.REPORTS_DIR / "basic_preprocessing_summary.csv"))
EPOCH_SUMMARY_CSV = OUT_DIR / "epoch_preprocessing_summary.csv"


# HELPER FUNCTIONS

def duration_label(duration: float) -> str:
    return cfg.duration_label(duration)


def subject_input_raw_path(subject_id: str) -> Path:
    return cfg.subject_clean_raw_path(subject_id)


def subject_output_folder(subject_id: str) -> Path:
    return cfg.subject_cleaned_epochs_folder(subject_id)


def subject_epoch_path(subject_id: str, duration: float) -> Path:
    return cfg.subject_epoch_path(subject_id, duration)


def subject_stitched_path(subject_id: str) -> Path:
    return cfg.subject_stitched_clean_raw_path(subject_id)


def subject_metadata_path(subject_id: str) -> Path:
    return cfg.subject_epoch_metadata_path(subject_id)


def get_config_value(name: str, default_value):
    return getattr(cfg, name, default_value)


def make_empty_row(subject_id: str, duration: float) -> dict:
    return {
        "subject_id": subject_id,
        "duration_sec": duration,
        "duration_label": duration_label(duration),
        "status": "not_started",
        "passed": False,
        "input_raw_path": str(subject_input_raw_path(subject_id)),
        "output_epoch_path": str(subject_epoch_path(subject_id, duration)),
        "output_stitched_path": str(subject_stitched_path(subject_id)),
        "is_reference_duration": duration == cfg.REFERENCE_DURATION,
        "stitched_saved": False,
        "raw_duration_sec": np.nan,
        "sfreq": np.nan,
        "n_channels": np.nan,
        "n_epochs_initial": 0,
        "n_epochs_after_flat": 0,
        "n_epochs_final": 0,
        "n_epochs_removed_total": 0,
        "percent_epochs_removed": np.nan,
        "clean_seconds_final": 0.0,
        "minimum_clean_seconds_required": cfg.MIN_RECORDING_SECONDS,
        "artifact_rejection_setting": cfg.ARTIFACT_REJECTION,
        "artifact_method_used": "",
        "autoreject_available": AUTOREJECT_AVAILABLE,
        "autoreject_success": False,
        "autoreject_failed_reason": "",
        "variance_fallback_used": False,
        "details": "",
    }


def load_passed_subject_ids() -> List[str]:
    if not BASIC_SUMMARY_CSV.exists():
        raise FileNotFoundError(
            f"Could not find basic preprocessing summary:\n{BASIC_SUMMARY_CSV}\n"
            "Run preprocess_basic.py first."
        )

    df = pd.read_csv(BASIC_SUMMARY_CSV)

    if "subject_id" not in df.columns:
        raise ValueError("basic_preprocessing_summary.csv must contain subject_id column.")

    if "passed" not in df.columns:
        raise ValueError("basic_preprocessing_summary.csv must contain passed column.")

    passed_df = df[df["passed"] == True].copy()
    subject_ids = passed_df["subject_id"].astype(str).tolist()

    return subject_ids


def create_fixed_length_epochs(raw: mne.io.BaseRaw, duration: float) -> mne.Epochs:
    sfreq = raw.info["sfreq"]

    events = mne.make_fixed_length_events(
        raw,
        id=1,
        start=0,
        duration=duration,
        overlap=0,
    )

    tmax = duration - (1.0 / sfreq)

    epochs = mne.Epochs(
        raw,
        events,
        event_id=1,
        tmin=0,
        tmax=tmax,
        baseline=None,
        preload=True,
        reject_by_annotation=True,
        verbose="ERROR",
    )

    return epochs


def drop_flat_epochs(epochs: mne.Epochs) -> mne.Epochs:
    flat_threshold = get_config_value("EPOCH_FLAT_THRESHOLD", 1e-6)

    epochs_clean = epochs.copy()

    try:
        epochs_clean.drop_bad(
            flat={"eeg": flat_threshold},
            verbose="ERROR",
        )
    except Exception:
        # If MNE drop_bad fails for any reason, continue without flat dropping.
        pass

    return epochs_clean


def reject_epochs_by_variance(epochs: mne.Epochs) -> Tuple[mne.Epochs, dict]:
    variance_z = get_config_value("VARIANCE_REJECT_Z", 3.5)

    data = epochs.get_data(copy=True)

    if data.shape[0] == 0:
        info = {
            "kept": 0,
            "removed": 0,
            "low_threshold": np.nan,
            "high_threshold": np.nan,
            "reason": "No epochs available.",
        }
        return epochs.copy(), info

    epoch_variances = np.var(data, axis=(1, 2))
    log_variances = np.log10(epoch_variances + 1e-20)

    median_value = np.median(log_variances)
    mad_value = np.median(np.abs(log_variances - median_value))

    if mad_value == 0:
        mad_value = np.std(log_variances)

    if mad_value == 0:
        keep_mask = np.ones(len(log_variances), dtype=bool)
        low_threshold = float(median_value)
        high_threshold = float(median_value)
    else:
        low_threshold = median_value - variance_z * mad_value
        high_threshold = median_value + variance_z * mad_value
        keep_mask = (log_variances >= low_threshold) & (log_variances <= high_threshold)

    kept_indices = np.where(keep_mask)[0]

    epochs_clean = epochs.copy()[kept_indices]

    info = {
        "kept": int(len(kept_indices)),
        "removed": int(len(epochs) - len(kept_indices)),
        "low_threshold": float(low_threshold),
        "high_threshold": float(high_threshold),
        "reason": f"Variance rejection using median ± {variance_z} × MAD.",
    }

    return epochs_clean, info


def _run_autoreject_directly(epochs: mne.Epochs) -> mne.Epochs:
    ar = AutoReject(n_jobs=1, random_state=cfg.RANDOM_SEED, verbose=False)
    return ar.fit_transform(epochs)


def reject_epochs_with_autoreject(epochs: mne.Epochs) -> Tuple[mne.Epochs | None, str]:
    if not AUTOREJECT_AVAILABLE:
        return None, "AutoReject is not installed."
    timeout_seconds = float(get_config_value("AUTOREJECT_TIMEOUT_SECONDS", 600))
    worker = Path(__file__).resolve().parent / "V3_autoreject_worker.py"
    with tempfile.TemporaryDirectory(prefix="mapper_ar_") as temp_dir:
        temp_dir = Path(temp_dir)
        input_path = temp_dir / "input-epo.fif"
        output_path = temp_dir / "output-epo.fif"
        epochs.save(input_path, overwrite=True, verbose="ERROR")
        proc = subprocess.Popen(
            [sys.executable, str(worker), str(input_path), str(output_path), str(int(cfg.RANDOM_SEED))],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            stdout, _ = proc.communicate(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
            return None, f"AutoReject timed out after {timeout_seconds:g} seconds."
        if proc.returncode != 0 or not output_path.exists():
            return None, (stdout or "AutoReject subprocess failed.")[-300:]
        try:
            return mne.read_epochs(output_path, preload=True, verbose="ERROR"), ""
        except Exception as error:
            return None, f"Could not read AutoReject output: {error}"


def apply_hybrid_artifact_rejection(epochs: mne.Epochs, duration: float) -> Tuple[mne.Epochs, dict]:
    method = str(cfg.ARTIFACT_REJECTION).lower().strip()

    info = {
        "method_used": "",
        "autoreject_success": False,
        "autoreject_failed_reason": "",
        "variance_fallback_used": False,
    }

    if method == "none":
        info["method_used"] = "none"
        return epochs.copy(), info

    if method == "variance":
        epochs_var, variance_info = reject_epochs_by_variance(epochs)
        info["method_used"] = "variance"
        info["variance_fallback_used"] = True
        info["variance_info"] = variance_info
        return epochs_var, info

    if method != "hybrid":
        epochs_var, variance_info = reject_epochs_by_variance(epochs)
        info["method_used"] = "variance"
        info["variance_fallback_used"] = True
        info["autoreject_failed_reason"] = (
            f"Unknown ARTIFACT_REJECTION setting: {cfg.ARTIFACT_REJECTION}. "
            "Used variance fallback."
        )
        info["variance_info"] = variance_info
        return epochs_var, info

    # Hybrid mode: try AutoReject first
    epochs_ar, ar_error = reject_epochs_with_autoreject(epochs)

    if epochs_ar is not None:
        ar_clean_seconds = len(epochs_ar) * duration

        if ar_clean_seconds >= cfg.MIN_RECORDING_SECONDS:
            info["method_used"] = "autoreject"
            info["autoreject_success"] = True
            return epochs_ar, info

        info["autoreject_failed_reason"] = (
            f"AutoReject worked but left only {ar_clean_seconds:.2f}s clean data. "
            f"Minimum required is {cfg.MIN_RECORDING_SECONDS:.2f}s. "
            "Using variance fallback."
        )

    else:
        info["autoreject_failed_reason"] = ar_error

    # Fallback: variance rejection
    epochs_var, variance_info = reject_epochs_by_variance(epochs)
    info["method_used"] = "variance"
    info["variance_fallback_used"] = True
    info["variance_info"] = variance_info

    return epochs_var, info


def stitch_epochs_to_raw(epochs: mne.Epochs) -> mne.io.RawArray:
    data = epochs.get_data(copy=True)

    if data.shape[0] == 0:
        raise ValueError("Cannot stitch epochs because there are zero epochs.")

    # Epochs data shape:
    #   n_epochs × n_channels × n_times
    #
    # RawArray needs:
    #   n_channels × total_times
    stitched_data = np.concatenate(data, axis=1)

    info = epochs.info.copy()

    stitched_raw = mne.io.RawArray(
        stitched_data,
        info,
        verbose="ERROR",
    )

    return stitched_raw


SETTINGS_HASH = stable_hash({
    "durations": list(cfg.CACHE_DURATIONS),
    "reference_duration": cfg.REFERENCE_DURATION,
    "artifact_rejection": cfg.ARTIFACT_REJECTION,
    "timeout": cfg.AUTOREJECT_TIMEOUT_SECONDS,
    "flat_threshold": cfg.EPOCH_FLAT_THRESHOLD,
    "variance_z": cfg.VARIANCE_REJECT_Z,
    "minimum_seconds": cfg.MIN_RECORDING_SECONDS,
})


def cached_epoch_matches(subject_id: str, output_epoch_path: Path, input_raw_path: Path) -> bool:
    metadata_path = subject_metadata_path(subject_id)
    if not output_epoch_path.is_file() or not input_raw_path.is_file() or not metadata_path.is_file():
        return False
    metadata = load_json(metadata_path, {}) or {}
    return (
        metadata.get("settings_hash") == SETTINGS_HASH
        and metadata.get("source_size") == input_raw_path.stat().st_size
        and metadata.get("source_mtime_ns") == input_raw_path.stat().st_mtime_ns
    )


def _cached_epoch_row(subject_id: str, duration: float, output_epoch_path: Path) -> dict:
    row = make_empty_row(subject_id, duration)
    if EPOCH_SUMMARY_CSV.exists():
        try:
            prior = pd.read_csv(EPOCH_SUMMARY_CSV, dtype={"subject_id": str})
            matches = prior[(prior["subject_id"].astype(str) == str(subject_id)) & (pd.to_numeric(prior["duration_sec"], errors="coerce") == float(duration))]
            if not matches.empty:
                row.update(matches.iloc[-1].to_dict())
        except Exception:
            pass
    metadata = load_json(subject_metadata_path(subject_id), {}) or {}
    for saved in metadata.get("status_rows", []):
        try:
            if float(saved.get("duration_sec")) == float(duration):
                row.update(saved)
                break
        except Exception:
            pass
    row["subject_id"] = str(subject_id)
    row["duration_sec"] = float(duration)
    row["duration_label"] = duration_label(duration)
    row["output_epoch_path"] = str(output_epoch_path)
    row["status"] = "cached"
    row["passed"] = True
    row["details"] = (str(row.get("details", "")).strip() + " | Reused verified epoch file.").strip(" |")
    return row


def process_one_subject_duration(subject_id: str, duration: float) -> Tuple[dict, mne.Epochs | None]:
    row = make_empty_row(subject_id, duration)

    input_raw_path = subject_input_raw_path(subject_id)
    output_epoch_path = subject_epoch_path(subject_id, duration)

    if not input_raw_path.exists():
        row["status"] = "failed"
        row["details"] = "Input cleaned raw file not found."
        return row, None

    resume = env_bool("NEUROPSYCHE_RESUME", bool(get_config_value("RESUME_EPOCH_PREPROCESSING", True)))
    force_rerun = env_bool("NEUROPSYCHE_FORCE_RERUN", bool(get_config_value("FORCE_REBUILD_EPOCHS", False)))
    if cached_epoch_matches(subject_id, output_epoch_path, input_raw_path) and resume and not force_rerun:
        row = _cached_epoch_row(subject_id, duration, output_epoch_path)
        if duration == cfg.REFERENCE_DURATION:
            try:
                cached_epochs = mne.read_epochs(output_epoch_path, preload=True, verbose="ERROR")
                row["n_epochs_final"] = len(cached_epochs)
                row["clean_seconds_final"] = round(float(len(cached_epochs) * duration), 3)
                return row, cached_epochs
            except Exception as error:
                row["passed"] = False
                row["status"] = "failed"
                row["details"] += f" Could not read cached reference epochs: {str(error)[:150]}"
        return row, None

    try:
        raw = mne.io.read_raw_fif(
            input_raw_path,
            preload=True,
            verbose="ERROR",
        )

        raw_duration = raw.n_times / raw.info["sfreq"]

        row["raw_duration_sec"] = round(float(raw_duration), 3)
        row["sfreq"] = round(float(raw.info["sfreq"]), 3)
        row["n_channels"] = len(raw.ch_names)

        epochs = create_fixed_length_epochs(raw, duration)
        row["n_epochs_initial"] = len(epochs)

        if len(epochs) == 0:
            row["status"] = "failed"
            row["details"] = "No epochs were created."
            return row, None

        epochs_after_flat = drop_flat_epochs(epochs)
        row["n_epochs_after_flat"] = len(epochs_after_flat)

        if len(epochs_after_flat) == 0:
            row["status"] = "failed"
            row["details"] = "All epochs were removed by flat-epoch rejection."
            return row, None

        epochs_final, rejection_info = apply_hybrid_artifact_rejection(
            epochs_after_flat,
            duration=duration,
        )

        row["artifact_method_used"] = rejection_info.get("method_used", "")
        row["autoreject_success"] = rejection_info.get("autoreject_success", False)
        row["autoreject_failed_reason"] = rejection_info.get("autoreject_failed_reason", "")
        row["variance_fallback_used"] = rejection_info.get("variance_fallback_used", False)

        row["n_epochs_final"] = len(epochs_final)
        row["n_epochs_removed_total"] = row["n_epochs_initial"] - row["n_epochs_final"]

        if row["n_epochs_initial"] > 0:
            row["percent_epochs_removed"] = round(
                100.0 * row["n_epochs_removed_total"] / row["n_epochs_initial"],
                3,
            )

        clean_seconds = len(epochs_final) * duration
        row["clean_seconds_final"] = round(float(clean_seconds), 3)

        if clean_seconds < cfg.MIN_RECORDING_SECONDS:
            row["status"] = "failed"
            row["details"] = (
                f"Too little clean data after artifact rejection: "
                f"{clean_seconds:.2f}s. "
                f"Minimum required: {cfg.MIN_RECORDING_SECONDS:.2f}s."
            )
            return row, None

        subject_output_folder(subject_id).mkdir(parents=True, exist_ok=True)

        epochs_final.save(
            output_epoch_path,
            overwrite=True,
            verbose="ERROR",
        )

        row["status"] = "passed"
        row["passed"] = True
        row["details"] = "Epoch preprocessing passed."

        return row, epochs_final

    except Exception as error:
        row["status"] = "failed"
        row["details"] = (
            f"Unexpected epoch preprocessing error: {str(error)[:150]} | "
            f"{traceback.format_exc()[:500]}"
        )
        return row, None


def process_one_subject(subject_id: str) -> List[dict]:
    rows = []
    reference_epochs = None

    for duration in cfg.CACHE_DURATIONS:
        row, epochs_final = process_one_subject_duration(subject_id, duration)

        if duration == cfg.REFERENCE_DURATION and epochs_final is not None:
            reference_epochs = epochs_final

        rows.append(row)

    # Save stitched clean raw from reference-duration epochs
    stitched_path = subject_stitched_path(subject_id)

    should_make_stitched = reference_epochs is not None

    reference_row = next((row for row in rows if row["is_reference_duration"]), None)
    if (
        stitched_path.exists()
        and reference_row is not None
        and reference_row.get("status") == "cached"
        and env_bool("NEUROPSYCHE_RESUME", bool(get_config_value("RESUME_EPOCH_PREPROCESSING", True)))
        and not env_bool("NEUROPSYCHE_FORCE_RERUN", bool(get_config_value("FORCE_REBUILD_EPOCHS", False)))
    ):
        should_make_stitched = False

        for row in rows:
            if row["is_reference_duration"]:
                row["stitched_saved"] = True
                row["details"] = row["details"] + " Stitched file already exists."

    if should_make_stitched:
        try:
            stitched_raw = stitch_epochs_to_raw(reference_epochs)
            stitched_raw.save(
                stitched_path,
                overwrite=True,
                verbose="ERROR",
            )

            for row in rows:
                if row["is_reference_duration"]:
                    row["stitched_saved"] = True
                    row["details"] = row["details"] + " Stitched clean raw saved."

        except Exception as error:
            for row in rows:
                if row["is_reference_duration"]:
                    row["stitched_saved"] = False
                    row["details"] = (
                        row["details"]
                        + f" Stitched clean raw failed: {str(error)[:150]}"
                    )

    # Save subject metadata
    try:
        subject_output_folder(subject_id).mkdir(parents=True, exist_ok=True)

        metadata = {
            "subject_id": subject_id,
            "input_raw_path": str(subject_input_raw_path(subject_id)),
            "output_folder": str(subject_output_folder(subject_id)),
            "cache_durations": list(cfg.CACHE_DURATIONS),
            "reference_duration": cfg.REFERENCE_DURATION,
            "artifact_rejection": cfg.ARTIFACT_REJECTION,
            "minimum_clean_seconds_required": cfg.MIN_RECORDING_SECONDS,
            "autoreject_available": AUTOREJECT_AVAILABLE,
            "settings_hash": SETTINGS_HASH,
            "source_size": subject_input_raw_path(subject_id).stat().st_size if subject_input_raw_path(subject_id).exists() else None,
            "source_mtime_ns": subject_input_raw_path(subject_id).stat().st_mtime_ns if subject_input_raw_path(subject_id).exists() else None,
            "created_utc": utc_iso(),
            "status_rows": rows,
            "note": (
                "stitched_clean_raw.fif concatenates clean reference-duration epochs; "
                "the locked V3 feature definitions still use BasicCleanedRaw for continuous families."
            ),
        }
        save_json_atomic(subject_metadata_path(subject_id), metadata)

    except Exception:
        pass

    return rows


# MAIN

def _write_epoch_summary(rows_by_key: dict[str, dict]) -> None:
    rows = list(rows_by_key.values())
    rows.sort(key=lambda r: (str(r.get("subject_id", "")), float(r.get("duration_sec", 0))))
    df = pd.DataFrame(rows)
    tmp = EPOCH_SUMMARY_CSV.with_suffix(".csv.tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(EPOCH_SUMMARY_CSV)


def main() -> None:
    mne.set_log_level("ERROR")
    cfg.apply_persisted_settings()
    cfg.make_directories()
    cfg.CLEANED_DIR.mkdir(parents=True, exist_ok=True)
    cfg.print_config_summary()
    if not cfg.check_basic_paths():
        raise SystemExit(2)

    subject_ids = load_passed_subject_ids()
    n_workers = cfg.get_n_workers()
    control = RunControl.from_environment(OUT_DIR / "_control")
    checkpoint = CheckpointStore(OUT_DIR / "_checkpoints" / "completed_subject_durations.txt")
    resume = env_bool("NEUROPSYCHE_RESUME", bool(cfg.RESUME_EPOCH_PREPROCESSING))
    force = env_bool("NEUROPSYCHE_FORCE_RERUN", bool(cfg.FORCE_REBUILD_EPOCHS))

    rows_by_key: dict[str, dict] = {}
    if EPOCH_SUMMARY_CSV.exists() and resume and not force:
        try:
            prior = pd.read_csv(EPOCH_SUMMARY_CSV, dtype={"subject_id": str})
            for _, row in prior.iterrows():
                key = f"{row['subject_id']}|{float(row['duration_sec']):g}"
                rows_by_key[key] = row.to_dict()
        except Exception as error:
            print(f"Could not load prior epoch summary: {error}")

    print("\n" + "=" * 70)
    print("Starting V3 epoch preprocessing")
    print(f"Subjects: {len(subject_ids)} | durations: {cfg.CACHE_DURATIONS} | workers: {n_workers}")
    batch_size = max(1, n_workers)
    try:
        for start in range(0, len(subject_ids), batch_size):
            control.checkpoint(f"before epoch subject batch {start // batch_size + 1}")
            batch = subject_ids[start:start + batch_size]
            batch_results = Parallel(n_jobs=n_workers, backend="loky")(
                delayed(process_one_subject)(subject_id) for subject_id in batch
            )
            for subject_rows in batch_results:
                for row in subject_rows:
                    key = f"{row['subject_id']}|{float(row['duration_sec']):g}"
                    rows_by_key[key] = row
                    checkpoint.add(key)
            _write_epoch_summary(rows_by_key)
            print(f"Checkpoint: {min(start + batch_size, len(subject_ids))}/{len(subject_ids)} subjects", flush=True)
    except GracefulStop as stop:
        _write_epoch_summary(rows_by_key)
        print(f"Graceful stop: {stop}")
        return

    summary_df = pd.DataFrame(rows_by_key.values())
    if summary_df.empty:
        summary_df = pd.DataFrame(columns=list(make_empty_row("", float(cfg.CACHE_DURATIONS[0])).keys()))
        summary_df.to_csv(EPOCH_SUMMARY_CSV, index=False)
    passed_count = int(summary_df["passed"].fillna(False).astype(bool).sum()) if "passed" in summary_df.columns else 0
    print("\n" + "=" * 70)
    print("Epoch preprocessing finished")
    print(f"Rows: {len(summary_df)} | passed: {passed_count}")
    print(f"Summary: {EPOCH_SUMMARY_CSV}")
    print("=" * 70)
    write_manifest(OUT_DIR, {
        "script": Path(__file__).name,
        "stage": 2,
        "inputs": {"basic_summary": str(BASIC_SUMMARY_CSV)},
        "outputs": {"epoch_summary": str(EPOCH_SUMMARY_CSV), "cleaned_dir": str(cfg.CLEANED_DIR)},
        "n_rows": int(len(summary_df)),
        "n_passed": passed_count,
        "settings_hash": cfg.settings_hash_for(2),
    })
    (OUT_DIR / "EPOCH_PREPROCESSING_COMPLETE.flag").write_text("complete", encoding="utf-8")


if __name__ == "__main__":
    main()
