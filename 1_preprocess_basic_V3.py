"""Basic EEG preprocessing for the MAPPER pipeline."""

from __future__ import annotations

import traceback
from pathlib import Path
from typing import List, Tuple

import mne
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from V3_config_loader import cfg
from V3_runtime import CheckpointStore, GracefulStop, RunControl, env_bool, file_sha256, make_stage_run_dir, load_json, save_json_atomic, stable_hash, utc_iso, write_manifest


# OUTPUT PATHS

RESUME = env_bool("NEUROPSYCHE_RESUME", True)
OUT_DIR = make_stage_run_dir(cfg.REPORTS_DIR, Path(__file__).name, resume=RESUME)
SUMMARY_CSV = OUT_DIR / "basic_preprocessing_summary.csv"


# SMALL HELPER FUNCTIONS

def subject_raw_path(subject_id: str) -> Path:
    return cfg.EEG_DIR / f"{subject_id}{cfg.EEG_FILE_EXTENSION}"


def subject_cleaned_path(subject_id: str) -> Path:
    return cfg.subject_clean_raw_path(subject_id)


def subject_metadata_path(subject_id: str) -> Path:
    return cfg.subject_clean_raw_metadata_path(subject_id)


def preprocessing_settings_dict() -> dict:
    return {
        "min_recording_seconds": cfg.MIN_RECORDING_SECONDS,
        "resample_sfreq": cfg.RESAMPLE_SFREQ,
        "do_bandpass": cfg.DO_BANDPASS,
        "bandpass_low": cfg.BANDPASS_LOW,
        "bandpass_high": cfg.BANDPASS_HIGH,
        "do_notch": cfg.DO_NOTCH,
        "notch_freq": cfg.NOTCH_FREQ,
        "do_average_reference": cfg.DO_AVERAGE_REFERENCE,
        "bad_channel_var_low": cfg.BAD_CHANNEL_VAR_LOW,
        "bad_channel_var_high_mult": cfg.BAD_CHANNEL_VAR_HIGH_MULT,
        "max_bad_channels": cfg.MAX_BAD_CHANNELS,
        "do_interpolate_bad_channels": cfg.DO_INTERPOLATE_BAD_CHANNELS,
        "do_ica": cfg.DO_ICA,
        "do_ecg_removal": cfg.DO_ECG_REMOVAL,
        "ica_eog_threshold": cfg.ICA_EOG_THRESHOLD,
    }


def make_status_row(subject_id: str) -> dict:
    return {
        "subject_id": subject_id,
        "raw_path": str(subject_raw_path(subject_id)),
        "cleaned_path": str(subject_cleaned_path(subject_id)),
        "status": "not_started",
        "passed": False,
        "file_found": False,
        "raw_duration_sec": np.nan,
        "final_duration_sec": np.nan,
        "sfreq": np.nan,
        "n_channels_raw": np.nan,
        "n_channels_final": np.nan,
        "bad_channels": "",
        "n_bad_channels": 0,
        "bad_channel_details": "",
        "interpolation_applied": False,
        "ica_applied": False,
        "ica_skipped": False,
        "eog_proxy_channel": "",
        "ica_blink_components_removed": 0,
        "ica_ecg_components_removed": 0,
        "details": "",
    }


def clean_channel_names(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    rename_map = {}

    for ch in raw.ch_names:
        new_name = ch.strip()
        new_name = new_name.replace("-LE", "")
        new_name = new_name.replace("EEG ", "")
        new_name = new_name.strip()

        if new_name != ch:
            rename_map[ch] = new_name

    if rename_map:
        raw.rename_channels(rename_map)

    return raw


def drop_non_eeg_channels(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    drop_names = []

    for ch in raw.ch_names:
        ch_lower = ch.lower().strip()

        if ch_lower in ["trig", "trigger", "status"]:
            drop_names.append(ch)

    if drop_names:
        raw.drop_channels(drop_names)

    return raw


def ensure_eeg_channel_types(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    eeg_picks = mne.pick_types(raw.info, eeg=True, exclude=[])

    if len(eeg_picks) == 0:
        channel_type_map = {ch: "eeg" for ch in raw.ch_names}
        raw.set_channel_types(channel_type_map)

    return raw


def detect_bad_channels(raw: mne.io.BaseRaw) -> Tuple[List[str], List[str]]:
    data = raw.get_data()
    ch_vars = np.var(data, axis=1)

    median_var = float(np.median(ch_vars))
    high_threshold = median_var * cfg.BAD_CHANNEL_VAR_HIGH_MULT

    bads = []
    bad_details = []

    for idx, var_value in enumerate(ch_vars):
        ch_name = raw.ch_names[idx]

        if var_value < cfg.BAD_CHANNEL_VAR_LOW:
            bads.append(ch_name)
            bad_details.append(f"{ch_name}: flat variance={var_value:.3e}")

        elif var_value > high_threshold:
            bads.append(ch_name)
            bad_details.append(
                f"{ch_name}: noisy variance={var_value:.3e}, "
                f"median={median_var:.3e}, threshold={high_threshold:.3e}"
            )

    return bads, bad_details


def find_frontal_eog_proxy(
    raw: mne.io.BaseRaw,
    bad_channels_before_interpolation: List[str],
) -> str | None:
    bad_set = set(bad_channels_before_interpolation)

    for name in ["Fp1", "Fp2", "Fpz"]:
        if name in raw.ch_names and name not in bad_set:
            return name

    return None


def run_ica_artifact_removal(
    raw: mne.io.BaseRaw,
    bad_channels_before_interpolation: List[str],
) -> Tuple[mne.io.BaseRaw, dict, List[str]]:
    messages = []

    ica_info = {
        "ica_applied": False,
        "ica_skipped": False,
        "eog_proxy_channel": "",
        "blink_components_removed": 0,
        "ecg_components_removed": 0,
    }

    if not cfg.DO_ICA:
        ica_info["ica_skipped"] = True
        messages.append("ICA skipped by configuration.")
        return raw, ica_info, messages

    eog_ch = find_frontal_eog_proxy(
        raw,
        bad_channels_before_interpolation=bad_channels_before_interpolation,
    )

    if eog_ch is None:
        ica_info["ica_skipped"] = True
        messages.append(
            "ICA skipped: no usable frontal blink-proxy channel. "
            "Fp1/Fp2/Fpz were missing or marked bad before interpolation."
        )
        return raw, ica_info, messages

    ica_info["eog_proxy_channel"] = eog_ch

    n_channels = len(raw.ch_names)
    max_components = min(20, n_channels)
    min_components = min(15, max_components)

    ica = None
    fitted = False
    used_components = max_components

    for n_components in range(max_components, min_components - 1, -1):
        try:
            this_ica = mne.preprocessing.ICA(
                n_components=n_components,
                random_state=cfg.RANDOM_SEED,
                max_iter="auto",
                verbose="ERROR",
            )
            this_ica.fit(raw, verbose="ERROR")

            ica = this_ica
            fitted = True
            used_components = n_components
            break

        except Exception:
            continue

    if not fitted or ica is None:
        ica_info["ica_skipped"] = True
        messages.append("ICA failed: could not fit ICA model.")
        return raw, ica_info, messages

    blink_components = []
    ecg_components = []

    try:
        blink_components, _ = ica.find_bads_eog(
            raw,
            ch_name=eog_ch,
            threshold=cfg.ICA_EOG_THRESHOLD,
            verbose="ERROR",
        )
    except Exception as error:
        messages.append(f"Blink ICA detection failed: {str(error)[:100]}")

    if cfg.DO_ECG_REMOVAL:
        try:
            ecg_epochs = mne.preprocessing.create_ecg_epochs(
                raw,
                tmin=-0.5,
                tmax=0.5,
                baseline=(None, 0),
                verbose="ERROR",
            )
            ecg_components, _ = ica.find_bads_ecg(
                ecg_epochs,
                method="ctps",
                verbose="ERROR",
            )
        except Exception:
            messages.append("ECG ICA removal skipped: no usable ECG/heartbeat signal detected.")

    components_to_remove = sorted(set(blink_components + ecg_components))

    if components_to_remove:
        ica.exclude = components_to_remove
        ica.apply(raw, verbose="ERROR")

    ica_info["ica_applied"] = True
    ica_info["ica_skipped"] = False
    ica_info["blink_components_removed"] = len(blink_components)
    ica_info["ecg_components_removed"] = len(ecg_components)

    messages.append(
        f"ICA applied using {eog_ch}; "
        f"n_components={used_components}; "
        f"blink_removed={len(blink_components)}; "
        f"ecg_removed={len(ecg_components)}."
    )

    return raw, ica_info, messages


# V3 RESUME / CONTROL HELPERS

SETTINGS_HASH = stable_hash(preprocessing_settings_dict())
_PRIOR_ROWS: dict[str, dict] = {}


def cached_cleaned_file_matches(raw_path: Path, cleaned_path: Path, metadata_path: Path) -> bool:
    if not raw_path.is_file() or not cleaned_path.is_file() or not metadata_path.is_file():
        return False
    metadata = load_json(metadata_path, {}) or {}
    return (
        metadata.get("settings_hash") == SETTINGS_HASH
        and metadata.get("source_size") == raw_path.stat().st_size
        and metadata.get("source_mtime_ns") == raw_path.stat().st_mtime_ns
    )

def _load_cached_status(subject_id: str, raw_path: Path, cleaned_path: Path, metadata_path: Path) -> dict:
    row = dict(_PRIOR_ROWS.get(str(subject_id), {}))
    if not row and SUMMARY_CSV.exists():
        try:
            prior = pd.read_csv(SUMMARY_CSV, dtype={"subject_id": str})
            match = prior[prior["subject_id"].astype(str) == str(subject_id)]
            if not match.empty:
                row = match.iloc[-1].to_dict()
        except Exception:
            pass
    metadata = load_json(metadata_path, {}) or {}
    if not row and isinstance(metadata.get("status_row"), dict):
        row = dict(metadata["status_row"])
    if not row:
        row = make_status_row(subject_id)
    row["subject_id"] = str(subject_id)
    row["raw_path"] = str(raw_path)
    row["cleaned_path"] = str(cleaned_path)
    row["status"] = "cached"
    row["passed"] = True
    row["file_found"] = raw_path.exists()
    details = str(row.get("details", "")).strip()
    row["details"] = (details + " | " if details else "") + "Reused verified cleaned file."
    return row

# MAIN PREPROCESSING FUNCTION FOR ONE SUBJECT

def preprocess_one_subject(subject_id: str) -> dict:
    status = make_status_row(subject_id)
    messages = []

    raw_path = subject_raw_path(subject_id)
    cleaned_path = subject_cleaned_path(subject_id)
    metadata_path = subject_metadata_path(subject_id)

    # Resume behavior
    force_rerun = env_bool("NEUROPSYCHE_FORCE_RERUN", bool(cfg.FORCE_REBUILD_CLEANED_EEG))
    resume = env_bool("NEUROPSYCHE_RESUME", bool(cfg.RESUME_PREPROCESSING))
    if cached_cleaned_file_matches(raw_path, cleaned_path, metadata_path) and resume and not force_rerun:
        return _load_cached_status(subject_id, raw_path, cleaned_path, metadata_path)

    # Check file exists
    if not raw_path.exists():
        status["status"] = "failed"
        status["details"] = "Raw EDF file not found."
        return status

    status["file_found"] = True

    # Load EDF
    try:
        raw = mne.io.read_raw_edf(
            raw_path,
            preload=True,
            verbose="ERROR",
        )
    except Exception as error:
        status["status"] = "failed"
        status["details"] = f"Could not read EDF: {str(error)[:150]}"
        return status

    if raw.n_times == 0:
        status["status"] = "failed"
        status["details"] = "EDF file has zero samples."
        return status

    raw_duration = raw.n_times / raw.info["sfreq"]

    status["raw_duration_sec"] = round(float(raw_duration), 3)
    status["sfreq"] = round(float(raw.info["sfreq"]), 3)
    status["n_channels_raw"] = len(raw.ch_names)

    # Reject short recordings before the expensive preprocessing steps.
    if raw_duration < cfg.MIN_RECORDING_SECONDS:
        status["status"] = "failed"
        status["details"] = (
            f"Raw recording too short: {raw_duration:.2f}s. "
            f"Minimum required: {cfg.MIN_RECORDING_SECONDS:.2f}s."
        )
        return status

    try:
        # Channel cleanup
        raw = clean_channel_names(raw)
        raw = drop_non_eeg_channels(raw)
        raw = ensure_eeg_channel_types(raw)

        # Resample, if requested
        if cfg.RESAMPLE_SFREQ is not None:
            raw.resample(cfg.RESAMPLE_SFREQ, npad="auto")
            messages.append(f"Resampled to {cfg.RESAMPLE_SFREQ} Hz.")

        # Montage
        try:
            raw.set_montage(
                "standard_1020",
                match_case=False,
                on_missing="warn",
            )
        except Exception:
            messages.append("Could not fully set montage; continuing anyway.")

        # Bandpass filter
        if cfg.DO_BANDPASS:
            raw.filter(
                cfg.BANDPASS_LOW,
                cfg.BANDPASS_HIGH,
                fir_design="firwin",
                verbose="ERROR",
            )
            messages.append(f"Bandpass filtered: {cfg.BANDPASS_LOW}-{cfg.BANDPASS_HIGH} Hz.")

        # Notch filter
        if cfg.DO_NOTCH:
            raw.notch_filter(
                cfg.NOTCH_FREQ,
                fir_design="firwin",
                verbose="ERROR",
            )
            messages.append(f"Notch filtered at {cfg.NOTCH_FREQ} Hz.")

        # Average reference
        if cfg.DO_AVERAGE_REFERENCE:
            raw.set_eeg_reference(
                "average",
                projection=False,
                verbose="ERROR",
            )
            messages.append("Average reference applied.")

        # Bad-channel detection
        bads, bad_details = detect_bad_channels(raw)

        status["bad_channels"] = ", ".join(bads)
        status["n_bad_channels"] = len(bads)
        status["bad_channel_details"] = " | ".join(bad_details)

        if bads:
            raw.info["bads"] = bads
            messages.append("Bad channels detected: " + ", ".join(bads))

            if len(bads) > cfg.MAX_BAD_CHANNELS:
                status["status"] = "failed"
                status["details"] = (
                    f"Too many bad channels: {len(bads)}. "
                    f"Maximum allowed: {cfg.MAX_BAD_CHANNELS}. "
                    + " ".join(messages)
                )
                return status

            if cfg.DO_INTERPOLATE_BAD_CHANNELS:
                try:
                    raw.interpolate_bads(
                        reset_bads=True,
                        verbose="ERROR",
                    )
                    status["interpolation_applied"] = True
                    messages.append("Bad channels interpolated.")
                except Exception as error:
                    messages.append(f"Bad-channel interpolation failed: {str(error)[:100]}")

        # ICA artifact removal
        raw, ica_info, ica_messages = run_ica_artifact_removal(
            raw,
            bad_channels_before_interpolation=bads,
        )
        messages.extend(ica_messages)

        status["ica_applied"] = ica_info["ica_applied"]
        status["ica_skipped"] = ica_info["ica_skipped"]
        status["eog_proxy_channel"] = ica_info["eog_proxy_channel"]
        status["ica_blink_components_removed"] = ica_info["blink_components_removed"]
        status["ica_ecg_components_removed"] = ica_info["ecg_components_removed"]

        # Final duration check BEFORE epoching
        final_duration = raw.n_times / raw.info["sfreq"]

        status["final_duration_sec"] = round(float(final_duration), 3)
        status["sfreq"] = round(float(raw.info["sfreq"]), 3)
        status["n_channels_final"] = len(raw.ch_names)

        if final_duration < cfg.MIN_RECORDING_SECONDS:
            status["status"] = "failed"
            status["details"] = (
                f"Final preprocessed recording too short: {final_duration:.2f}s. "
                f"Minimum required: {cfg.MIN_RECORDING_SECONDS:.2f}s. "
                + " ".join(messages)
            )
            return status

        # Save cleaned continuous raw EEG
        cfg.subject_basic_cleaned_folder(subject_id).mkdir(parents=True, exist_ok=True)

        raw.save(
            cleaned_path,
            overwrite=True,
            verbose="ERROR",
        )

        status["status"] = "passed"
        status["passed"] = True
        status["details"] = " ".join(messages) if messages else "Basic preprocessing passed."

        metadata = {
            "subject_id": subject_id,
            "raw_path": str(raw_path),
            "cleaned_path": str(cleaned_path),
            "preprocessing_settings": preprocessing_settings_dict(),
            "settings_hash": SETTINGS_HASH,
            "created_utc": utc_iso(),
            "source_size": raw_path.stat().st_size if raw_path.exists() else None,
            "source_mtime_ns": raw_path.stat().st_mtime_ns if raw_path.exists() else None,
            "messages": messages,
            "status_row": status,
        }
        save_json_atomic(metadata_path, metadata)
        return status

    except Exception as error:
        status["status"] = "failed"
        status["details"] = (
            f"Unexpected preprocessing error: {str(error)[:150]} | "
            f"{traceback.format_exc()[:500]}"
        )
        return status


# LOAD SUBJECT IDS

def load_subject_ids() -> List[str]:
    df = pd.read_csv(cfg.MCMI_CSV)

    if cfg.SUBJECT_ID_COLUMN not in df.columns:
        raise ValueError(
            f"MCMI CSV must contain a '{cfg.SUBJECT_ID_COLUMN}' column."
        )

    subject_ids = df[cfg.SUBJECT_ID_COLUMN].astype(str).tolist()
    return subject_ids


# MAIN

def _write_summary(rows_by_subject: dict[str, dict]) -> None:
    ordered = [rows_by_subject[k] for k in sorted(rows_by_subject, key=str)]
    df = pd.DataFrame(ordered)
    tmp = SUMMARY_CSV.with_suffix(".csv.tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(SUMMARY_CSV)


def main() -> None:
    global _PRIOR_ROWS
    mne.set_log_level("ERROR")
    cfg.apply_persisted_settings()
    cfg.make_directories()
    cfg.CLEANED_DIR.mkdir(parents=True, exist_ok=True)
    cfg.print_config_summary()
    if not cfg.check_basic_paths():
        raise SystemExit(2)

    subject_ids = load_subject_ids()
    n_workers = cfg.get_n_workers()
    control = RunControl.from_environment(OUT_DIR / "_control")
    checkpoint = CheckpointStore(OUT_DIR / "_checkpoints" / "completed_subjects.txt")
    force_rerun = env_bool("NEUROPSYCHE_FORCE_RERUN", bool(cfg.FORCE_REBUILD_CLEANED_EEG))
    resume = env_bool("NEUROPSYCHE_RESUME", bool(cfg.RESUME_PREPROCESSING))

    rows_by_subject: dict[str, dict] = {}
    if SUMMARY_CSV.exists() and resume and not force_rerun:
        try:
            prior_df = pd.read_csv(SUMMARY_CSV, dtype={"subject_id": str})
            rows_by_subject = {str(r["subject_id"]): r.to_dict() for _, r in prior_df.iterrows()}
        except Exception as error:
            print(f"Could not read prior summary; rebuilding it from outputs: {error}")
    _PRIOR_ROWS = dict(rows_by_subject)

    print("\n" + "=" * 70)
    print("Starting V3 basic preprocessing")
    print("=" * 70)
    print(f"Subjects: {len(subject_ids)} | workers: {n_workers} | resume: {resume} | force: {force_rerun}")

    batch_size = max(1, n_workers * 2)
    try:
        for start in range(0, len(subject_ids), batch_size):
            control.checkpoint(f"before subject batch {start // batch_size + 1}")
            batch = subject_ids[start:start + batch_size]
            results = Parallel(n_jobs=n_workers, backend="loky")(
                delayed(preprocess_one_subject)(subject_id)
                for subject_id in batch
            )
            for row in results:
                sid = str(row["subject_id"])
                rows_by_subject[sid] = row
                checkpoint.add(sid)
            _write_summary(rows_by_subject)
            print(f"Checkpoint: {min(start + batch_size, len(subject_ids))}/{len(subject_ids)} subjects", flush=True)
    except GracefulStop as stop:
        _write_summary(rows_by_subject)
        print(f"Graceful stop: {stop}")
        return

    summary_df = pd.DataFrame([rows_by_subject[str(s)] for s in subject_ids if str(s) in rows_by_subject])
    n_passed = int(summary_df.get("passed", pd.Series(dtype=bool)).fillna(False).astype(bool).sum())
    n_failed = int((summary_df.get("status", pd.Series(dtype=str)) == "failed").sum())
    n_cached = int((summary_df.get("status", pd.Series(dtype=str)) == "cached").sum())
    print("\n" + "=" * 70)
    print("Basic preprocessing finished")
    print(f"Passed: {n_passed} | Failed: {n_failed} | Cached: {n_cached}")
    print(f"Summary: {SUMMARY_CSV}")
    print("=" * 70)
    write_manifest(OUT_DIR, {
        "script": Path(__file__).name,
        "stage": 1,
        "inputs": {"eeg_dir": str(cfg.EEG_DIR), "mcmi_csv": str(cfg.MCMI_CSV)},
        "outputs": {"basic_summary": str(SUMMARY_CSV), "cleaned_dir": str(cfg.CLEANED_DIR)},
        "n_subjects": int(len(summary_df)),
        "n_passed": n_passed,
        "n_failed": n_failed,
        "n_cached": n_cached,
        "mcmi_sha256": file_sha256(cfg.MCMI_CSV),
        "settings_hash": cfg.settings_hash_for(1),
    })
    (OUT_DIR / "PREPROCESSING_COMPLETE.flag").write_text("complete", encoding="utf-8")


if __name__ == "__main__":
    main()
