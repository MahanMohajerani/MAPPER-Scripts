"""Build preprocessing QC tables and the frozen analysis cohort."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from V3_config_loader import cfg
from V3_runtime import GracefulStop, RunControl, env_bool, file_sha256, make_stage_run_dir, stable_hash, write_manifest


# INPUT FILES

BASIC_SUMMARY_CSV = Path(cfg.runtime_input("basic_summary", cfg.REPORTS_DIR / "basic_preprocessing_summary.csv"))
EPOCH_SUMMARY_CSV = Path(cfg.runtime_input("epoch_summary", cfg.REPORTS_DIR / "epoch_preprocessing_summary.csv"))


# OUTPUT FILES

RESUME = env_bool("NEUROPSYCHE_RESUME", True)
OUT_DIR = make_stage_run_dir(cfg.REPORTS_DIR, Path(__file__).name, resume=RESUME)
GENERAL_PREPROCESSING_CSV = OUT_DIR / "preprocessing.csv"
SUBJECT_DURATION_CSV = OUT_DIR / "preprocessing_subject_duration.csv"
ANALYSIS_COHORT_CSV = OUT_DIR / "analysis_cohort_V3.csv"
SUBJECT_PREPROCESSING_FILENAME = "preprocessing.csv"


# QC THRESHOLDS
#
# These are intentionally simple and transparent.

HIGH_EPOCH_REMOVAL_PERCENT = float(getattr(cfg, "HIGH_EPOCH_REMOVAL_PERCENT", 50.0))
VERY_HIGH_EPOCH_REMOVAL_PERCENT = float(getattr(cfg, "VERY_HIGH_EPOCH_REMOVAL_PERCENT", 75.0))

LOW_CLEAN_SECONDS = float(getattr(cfg, "LOW_CLEAN_SECONDS", 60.0))
NEAR_MINIMUM_CLEAN_SECONDS = float(getattr(cfg, "NEAR_MINIMUM_CLEAN_SECONDS", 120.0))


# SMALL HELPERS

def normalize_bool(value: Any) -> bool:
    if pd.isna(value):
        return False

    if isinstance(value, bool):
        return value

    value_str = str(value).strip().lower()

    return value_str in ["true", "1", "yes", "y"]


def safe_float(value: Any, default: float = np.nan) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if pd.isna(value):
            return default
        return int(float(value))
    except Exception:
        return default


def safe_str(value: Any, default: str = "") -> str:
    if pd.isna(value):
        return default

    return str(value)


def get_value(row: pd.Series | None, column: str, default: Any = None) -> Any:
    if row is None:
        return default

    if column not in row.index:
        return default

    return row[column]


def duration_key(duration: float) -> float:
    return round(float(duration), 6)


def ordered_unique(values: list[str]) -> list[str]:
    seen = set()
    output = []

    for value in values:
        if value not in seen:
            seen.add(value)
            output.append(value)

    return output


def join_reasons(reasons: list[str]) -> str:
    clean_reasons = [str(reason) for reason in reasons if str(reason).strip()]

    return "; ".join(clean_reasons)


def count_true(series: pd.Series) -> int:
    return int(series.apply(normalize_bool).sum())


def file_exists(path_value: Any) -> bool:
    try:
        return Path(str(path_value)).exists()
    except Exception:
        return False


# LOAD INPUTS

def load_mcmi_subject_ids() -> list[str]:
    if not cfg.MCMI_CSV.exists():
        return []

    df = pd.read_csv(cfg.MCMI_CSV)

    if cfg.SUBJECT_ID_COLUMN not in df.columns:
        return []

    return df[cfg.SUBJECT_ID_COLUMN].astype(str).tolist()


def load_basic_summary() -> pd.DataFrame:
    if not BASIC_SUMMARY_CSV.exists():
        raise FileNotFoundError(
            f"Could not find basic preprocessing summary:\n{BASIC_SUMMARY_CSV}"
        )

    df = pd.read_csv(BASIC_SUMMARY_CSV)

    if "subject_id" not in df.columns:
        raise ValueError("basic_preprocessing_summary.csv must contain subject_id.")

    df["subject_id"] = df["subject_id"].astype(str)

    if "passed" in df.columns:
        df["passed"] = df["passed"].apply(normalize_bool)

    return df


def load_epoch_summary() -> pd.DataFrame:
    if not EPOCH_SUMMARY_CSV.exists():
        raise FileNotFoundError(
            f"Could not find epoch preprocessing summary:\n{EPOCH_SUMMARY_CSV}"
        )

    df = pd.read_csv(EPOCH_SUMMARY_CSV)

    required_columns = ["subject_id", "duration_sec", "passed"]

    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"epoch_preprocessing_summary.csv must contain {col}.")

    df["subject_id"] = df["subject_id"].astype(str)
    df["duration_sec"] = df["duration_sec"].apply(safe_float)
    df["duration_key"] = df["duration_sec"].apply(duration_key)
    df["passed"] = df["passed"].apply(normalize_bool)

    for col in [
        "stitched_saved",
        "autoreject_available",
        "autoreject_success",
        "variance_fallback_used",
        "is_reference_duration",
    ]:
        if col in df.columns:
            df[col] = df[col].apply(normalize_bool)

    for col in [
        "raw_duration_sec",
        "sfreq",
        "n_channels",
        "n_epochs_initial",
        "n_epochs_after_flat",
        "n_epochs_final",
        "n_epochs_removed_total",
        "percent_epochs_removed",
        "clean_seconds_final",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    duplicate_mask = df.duplicated(["subject_id", "duration_key"], keep=False)

    if duplicate_mask.any():
        duplicates = df.loc[
            duplicate_mask,
            ["subject_id", "duration_sec"],
        ].sort_values(["subject_id", "duration_sec"])

        print("\nDuplicate subject-duration rows found:")
        print(duplicates.head(30))

        raise ValueError(
            "epoch_preprocessing_summary.csv has duplicate subject-duration rows."
        )

    return df


# SUBJECT-DURATION QC

def make_subject_duration_qc(
    subject_ids: list[str],
    basic_df: pd.DataFrame,
    epoch_df: pd.DataFrame,
) -> pd.DataFrame:
    basic_map = {}

    for _, row in basic_df.iterrows():
        subject_id = str(row["subject_id"])
        basic_map[subject_id] = row.copy()

    epoch_map = {}

    for _, row in epoch_df.iterrows():
        subject_id = str(row["subject_id"])
        dur_key = duration_key(float(row["duration_sec"]))
        epoch_map[(subject_id, dur_key)] = row.copy()

    rows = []

    for subject_id in subject_ids:
        basic_row = basic_map.get(subject_id)
        passed_basic = normalize_bool(get_value(basic_row, "passed", False))

        for duration in cfg.CACHE_DURATIONS:
            dur = float(duration)
            dur_key = duration_key(dur)
            epoch_row = epoch_map.get((subject_id, dur_key))

            epoch_row_exists = epoch_row is not None

            passed_epoch = normalize_bool(get_value(epoch_row, "passed", False))
            epoch_status = safe_str(get_value(epoch_row, "status", "missing_epoch_row"))

            percent_removed = safe_float(
                get_value(epoch_row, "percent_epochs_removed", np.nan)
            )
            clean_seconds = safe_float(
                get_value(epoch_row, "clean_seconds_final", np.nan)
            )

            artifact_method = safe_str(
                get_value(epoch_row, "artifact_method_used", "")
            )

            autoreject_reason = safe_str(
                get_value(epoch_row, "autoreject_failed_reason", "")
            )

            variance_fallback = normalize_bool(
                get_value(epoch_row, "variance_fallback_used", False)
            )

            autoreject_timeout = "timed out" in autoreject_reason.lower()

            high_epoch_removal = (
                passed_epoch
                and not np.isnan(percent_removed)
                and percent_removed >= HIGH_EPOCH_REMOVAL_PERCENT
            )

            very_high_epoch_removal = (
                passed_epoch
                and not np.isnan(percent_removed)
                and percent_removed >= VERY_HIGH_EPOCH_REMOVAL_PERCENT
            )

            low_clean_seconds = (
                passed_epoch
                and not np.isnan(clean_seconds)
                and clean_seconds <= LOW_CLEAN_SECONDS
            )

            near_minimum_clean_seconds = (
                passed_epoch
                and not np.isnan(clean_seconds)
                and clean_seconds > LOW_CLEAN_SECONDS
                and clean_seconds <= NEAR_MINIMUM_CLEAN_SECONDS
            )

            failed_epoch_preprocessing = not passed_epoch

            major_reasons = []
            minor_reasons = []

            if not passed_basic:
                major_reasons.append("failed_basic_preprocessing")

            if not epoch_row_exists and passed_basic:
                major_reasons.append("missing_epoch_summary_row")

            if failed_epoch_preprocessing and passed_basic:
                major_reasons.append("failed_epoch_preprocessing")

            if high_epoch_removal:
                major_reasons.append(
                    f"high_epoch_removal_{percent_removed:.1f}%"
                )

            if very_high_epoch_removal:
                major_reasons.append(
                    f"very_high_epoch_removal_{percent_removed:.1f}%"
                )

            if low_clean_seconds:
                major_reasons.append(
                    f"low_clean_seconds_{clean_seconds:.1f}s"
                )

            if autoreject_timeout:
                minor_reasons.append("autoreject_timeout")

            if variance_fallback:
                minor_reasons.append("variance_fallback_used")

            if near_minimum_clean_seconds:
                minor_reasons.append(
                    f"near_minimum_clean_seconds_{clean_seconds:.1f}s"
                )

            major_qc_warning_duration = len(major_reasons) > 0
            minor_qc_warning_duration = len(minor_reasons) > 0
            review_needed_duration = (
                major_qc_warning_duration or minor_qc_warning_duration
            )

            current_input_raw_path = cfg.subject_clean_raw_path(subject_id)
            current_epoch_path = cfg.subject_epoch_path(subject_id, dur)
            current_stitched_path = cfg.subject_stitched_clean_raw_path(subject_id)

            row_out = {
                "subject_id": subject_id,
                "duration_sec": dur,
                "duration_label": cfg.duration_label(dur),

                # Basic preprocessing
                "passed_basic": passed_basic,
                "basic_status": safe_str(get_value(basic_row, "status", "")),
                "basic_details": safe_str(get_value(basic_row, "details", "")),
                "basic_raw_duration_sec": safe_float(
                    get_value(basic_row, "raw_duration_sec", np.nan)
                ),
                "basic_final_duration_sec": safe_float(
                    get_value(basic_row, "final_duration_sec", np.nan)
                ),
                "basic_n_bad_channels": safe_int(
                    get_value(basic_row, "n_bad_channels", 0)
                ),
                "basic_bad_channels": safe_str(
                    get_value(basic_row, "bad_channels", "")
                ),
                "basic_ica_applied": normalize_bool(
                    get_value(basic_row, "ica_applied", False)
                ),
                "basic_ica_skipped": normalize_bool(
                    get_value(basic_row, "ica_skipped", False)
                ),
                "basic_eog_proxy_channel": safe_str(
                    get_value(basic_row, "eog_proxy_channel", "")
                ),
                "basic_ica_blink_components_removed": safe_int(
                    get_value(basic_row, "ica_blink_components_removed", 0)
                ),
                "basic_ica_ecg_components_removed": safe_int(
                    get_value(basic_row, "ica_ecg_components_removed", 0)
                ),

                # Epoch preprocessing
                "epoch_row_exists": epoch_row_exists,
                "epoch_status": epoch_status,
                "passed_epoch_preprocessing": passed_epoch,
                "is_reference_duration": dur == float(cfg.REFERENCE_DURATION),
                "stitched_saved": normalize_bool(
                    get_value(epoch_row, "stitched_saved", False)
                ),
                "epoch_raw_duration_sec": safe_float(
                    get_value(epoch_row, "raw_duration_sec", np.nan)
                ),
                "sfreq": safe_float(get_value(epoch_row, "sfreq", np.nan)),
                "n_channels": safe_int(get_value(epoch_row, "n_channels", 0)),
                "n_epochs_initial": safe_int(
                    get_value(epoch_row, "n_epochs_initial", 0)
                ),
                "n_epochs_after_flat": safe_int(
                    get_value(epoch_row, "n_epochs_after_flat", 0)
                ),
                "n_epochs_final": safe_int(
                    get_value(epoch_row, "n_epochs_final", 0)
                ),
                "n_epochs_removed_total": safe_int(
                    get_value(epoch_row, "n_epochs_removed_total", 0)
                ),
                "percent_epochs_removed": percent_removed,
                "clean_seconds_final": clean_seconds,
                "artifact_method_used": artifact_method,
                "autoreject_success": normalize_bool(
                    get_value(epoch_row, "autoreject_success", False)
                ),
                "autoreject_failed_reason": autoreject_reason,
                "variance_fallback_used": variance_fallback,

                # QC flags
                "failed_epoch_preprocessing": failed_epoch_preprocessing,
                "autoreject_timeout": autoreject_timeout,
                "high_epoch_removal": high_epoch_removal,
                "very_high_epoch_removal": very_high_epoch_removal,
                "low_clean_seconds": low_clean_seconds,
                "near_minimum_clean_seconds": near_minimum_clean_seconds,
                "major_qc_warning_duration": major_qc_warning_duration,
                "minor_qc_warning_duration": minor_qc_warning_duration,
                "review_needed_duration": review_needed_duration,
                "major_qc_reasons_duration": join_reasons(major_reasons),
                "minor_qc_reasons_duration": join_reasons(minor_reasons),

                # Current expected paths
                "clean_raw_path": str(current_input_raw_path),
                "epoch_path": str(current_epoch_path),
                "stitched_clean_raw_path": str(current_stitched_path),
                "clean_raw_exists": current_input_raw_path.exists(),
                "epoch_file_exists": current_epoch_path.exists(),
                "stitched_clean_raw_exists": current_stitched_path.exists(),

                # Original details
                "epoch_details": safe_str(get_value(epoch_row, "details", "")),
            }

            rows.append(row_out)

    qc_df = pd.DataFrame(rows)

    return qc_df


# SUBJECT-LEVEL QC

def make_subject_qc(subject_duration_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    expected_n_durations = len(cfg.CACHE_DURATIONS)

    for subject_id, sub_df in subject_duration_df.groupby("subject_id", sort=False):
        passed_basic = normalize_bool(sub_df["passed_basic"].iloc[0])

        n_durations_passed = count_true(sub_df["passed_epoch_preprocessing"])
        n_epoch_rows_found = count_true(sub_df["epoch_row_exists"])

        passed_all_durations = (
            passed_basic
            and n_durations_passed == expected_n_durations
        )

        reference_mask = sub_df["is_reference_duration"].apply(normalize_bool)
        reference_df = sub_df[reference_mask]

        if len(reference_df) > 0:
            passed_reference_duration = normalize_bool(
                reference_df["passed_epoch_preprocessing"].iloc[0]
            )
            stitched_saved = normalize_bool(reference_df["stitched_saved"].iloc[0])
            stitched_exists = normalize_bool(
                reference_df["stitched_clean_raw_exists"].iloc[0]
            )
        else:
            passed_reference_duration = False
            stitched_saved = False
            stitched_exists = False

        any_timeout = count_true(sub_df["autoreject_timeout"]) > 0
        any_variance = count_true(sub_df["variance_fallback_used"]) > 0
        any_high_epoch_removal = count_true(sub_df["high_epoch_removal"]) > 0
        any_very_high_epoch_removal = count_true(sub_df["very_high_epoch_removal"]) > 0
        any_low_clean_seconds = count_true(sub_df["low_clean_seconds"]) > 0
        any_near_minimum_clean_seconds = (
            count_true(sub_df["near_minimum_clean_seconds"]) > 0
        )

        n_timeout_rows = count_true(sub_df["autoreject_timeout"])
        n_variance_rows = count_true(sub_df["variance_fallback_used"])
        n_high_epoch_removal_rows = count_true(sub_df["high_epoch_removal"])
        n_very_high_epoch_removal_rows = count_true(sub_df["very_high_epoch_removal"])
        n_low_clean_seconds_rows = count_true(sub_df["low_clean_seconds"])
        n_near_minimum_clean_seconds_rows = count_true(
            sub_df["near_minimum_clean_seconds"]
        )

        n_autoreject_rows = int(
            (
                sub_df["artifact_method_used"]
                .fillna("")
                .astype(str)
                .str.lower()
                .str.strip()
                == "autoreject"
            ).sum()
        )

        n_epoch_files_existing = count_true(sub_df["epoch_file_exists"])

        passed_rows_df = sub_df[sub_df["passed_epoch_preprocessing"].apply(normalize_bool)]

        if len(passed_rows_df) > 0:
            min_clean_seconds = float(
                pd.to_numeric(
                    passed_rows_df["clean_seconds_final"],
                    errors="coerce",
                ).min()
            )
            max_percent_epochs_removed = float(
                pd.to_numeric(
                    passed_rows_df["percent_epochs_removed"],
                    errors="coerce",
                ).max()
            )
        else:
            min_clean_seconds = np.nan
            max_percent_epochs_removed = np.nan

        major_qc_warning_subject = (
            not passed_basic
            or not passed_all_durations
            or any_high_epoch_removal
            or any_very_high_epoch_removal
            or any_low_clean_seconds
        )

        minor_qc_warning_subject = (
            any_timeout
            or any_variance
            or any_near_minimum_clean_seconds
        )

        minimum_include = passed_basic and passed_all_durations

        strict_include_quality = (
            minimum_include
            and not any_high_epoch_removal
            and not any_very_high_epoch_removal
            and not any_low_clean_seconds
        )

        strict_include_method = (
            strict_include_quality
            and not any_timeout
            and not any_variance
        )

        review_needed = (
            minimum_include
            and (
                any_high_epoch_removal
                or any_very_high_epoch_removal
                or any_low_clean_seconds
                or any_timeout
                or any_variance
                or any_near_minimum_clean_seconds
            )
        )

        subject_reasons = []

        if not passed_basic:
            subject_reasons.append("failed_basic_preprocessing")

        if passed_basic and not passed_all_durations:
            subject_reasons.append(
                f"passed_{n_durations_passed}_of_{expected_n_durations}_durations"
            )

        if any_high_epoch_removal:
            subject_reasons.append(
                f"high_epoch_removal_rows={n_high_epoch_removal_rows}"
            )

        if any_very_high_epoch_removal:
            subject_reasons.append(
                f"very_high_epoch_removal_rows={n_very_high_epoch_removal_rows}"
            )

        if any_low_clean_seconds:
            subject_reasons.append(
                f"low_clean_seconds_rows={n_low_clean_seconds_rows}"
            )

        if any_timeout:
            subject_reasons.append(f"autoreject_timeout_rows={n_timeout_rows}")

        if any_variance:
            subject_reasons.append(f"variance_fallback_rows={n_variance_rows}")

        if any_near_minimum_clean_seconds:
            subject_reasons.append(
                f"near_minimum_clean_seconds_rows={n_near_minimum_clean_seconds_rows}"
            )

        if not subject_reasons:
            subject_reasons.append("no_qc_warnings")

        if not passed_basic:
            qc_group = "exclude_failed_basic"

        elif not passed_all_durations:
            qc_group = "exclude_failed_epoch"

        elif strict_include_method:
            qc_group = "include_clean"

        elif strict_include_quality:
            qc_group = "include_minor_method_warning"

        else:
            qc_group = "include_major_quality_warning"

        subject_folder = cfg.subject_cleaned_folder(subject_id)

        row_out = {
            "subject_id": subject_id,
            "qc_group": qc_group,
            "passed_basic": passed_basic,
            "passed_all_durations": passed_all_durations,
            "passed_reference_duration": passed_reference_duration,
            "n_expected_durations": expected_n_durations,
            "n_epoch_rows_found": n_epoch_rows_found,
            "n_durations_passed": n_durations_passed,

            # Inclusion labels
            "minimum_include": minimum_include,
            "strict_include_quality": strict_include_quality,
            "strict_include_method": strict_include_method,
            "review_needed": review_needed,
            "major_qc_warning_subject": major_qc_warning_subject,
            "minor_qc_warning_subject": minor_qc_warning_subject,
            "qc_reasons": join_reasons(subject_reasons),

            # Counts
            "n_autoreject_rows": n_autoreject_rows,
            "n_variance_fallback_rows": n_variance_rows,
            "n_autoreject_timeout_rows": n_timeout_rows,
            "n_high_epoch_removal_rows": n_high_epoch_removal_rows,
            "n_very_high_epoch_removal_rows": n_very_high_epoch_removal_rows,
            "n_low_clean_seconds_rows": n_low_clean_seconds_rows,
            "n_near_minimum_clean_seconds_rows": n_near_minimum_clean_seconds_rows,

            # Summary metrics
            "min_clean_seconds_across_durations": min_clean_seconds,
            "max_percent_epochs_removed_across_durations": max_percent_epochs_removed,

            # File checks
            "n_epoch_files_existing": n_epoch_files_existing,
            "clean_raw_exists": normalize_bool(sub_df["clean_raw_exists"].iloc[0]),
            "stitched_saved": stitched_saved,
            "stitched_clean_raw_exists": stitched_exists,

            # Paths
            "subject_folder": str(subject_folder),
            "subject_preprocessing_csv": str(
                subject_folder / SUBJECT_PREPROCESSING_FILENAME
            ),
            "clean_raw_path": str(cfg.subject_clean_raw_path(subject_id)),
            "stitched_clean_raw_path": str(
                cfg.subject_stitched_clean_raw_path(subject_id)
            ),
        }

        rows.append(row_out)

    subject_qc_df = pd.DataFrame(rows)

    return subject_qc_df


# SAVE PER-SUBJECT CSV FILES

def save_subject_preprocessing_csvs(
    subject_duration_df: pd.DataFrame,
    subject_qc_df: pd.DataFrame,
) -> None:
    subject_qc_map = {}

    for _, row in subject_qc_df.iterrows():
        subject_qc_map[str(row["subject_id"])] = row.copy()

    for subject_id, sub_duration_df in subject_duration_df.groupby(
        "subject_id",
        sort=False,
    ):
        subject_summary = subject_qc_map[str(subject_id)]

        # Add the most useful subject-level labels to every duration row.
        per_subject_df = sub_duration_df.copy()

        subject_columns_to_repeat = [
            "qc_group",
            "minimum_include",
            "strict_include_quality",
            "strict_include_method",
            "review_needed",
            "major_qc_warning_subject",
            "minor_qc_warning_subject",
            "qc_reasons",
            "n_durations_passed",
            "n_autoreject_rows",
            "n_variance_fallback_rows",
            "n_autoreject_timeout_rows",
            "min_clean_seconds_across_durations",
            "max_percent_epochs_removed_across_durations",
        ]

        for col in subject_columns_to_repeat:
            per_subject_df[col] = subject_summary[col]

        subject_folder = cfg.subject_cleaned_folder(str(subject_id))
        subject_folder.mkdir(parents=True, exist_ok=True)

        output_path = subject_folder / SUBJECT_PREPROCESSING_FILENAME
        per_subject_df.to_csv(output_path, index=False)


# PRINT SUMMARY

def print_summary(subject_qc_df: pd.DataFrame, subject_duration_df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("Preprocessing QC report finished")
    print("=" * 70)

    print(f"Subjects in QC report: {len(subject_qc_df)}")
    print(f"Subject-duration rows: {len(subject_duration_df)}")

    print("\nSubject QC groups:")
    print(subject_qc_df["qc_group"].value_counts(dropna=False))

    print("\nInclusion counts:")
    print(f"  minimum_include: {int(subject_qc_df['minimum_include'].sum())}")
    print(
        f"  strict_include_quality: "
        f"{int(subject_qc_df['strict_include_quality'].sum())}"
    )
    print(
        f"  strict_include_method: "
        f"{int(subject_qc_df['strict_include_method'].sum())}"
    )

    print("\nSubject-level warning counts:")
    print(f"  review_needed: {int(subject_qc_df['review_needed'].sum())}")
    print(
        f"  major_qc_warning_subject: "
        f"{int(subject_qc_df['major_qc_warning_subject'].sum())}"
    )
    print(
        f"  minor_qc_warning_subject: "
        f"{int(subject_qc_df['minor_qc_warning_subject'].sum())}"
    )

    print("\nDuration-level warning counts:")
    print(
        f"  high_epoch_removal rows: "
        f"{int(subject_duration_df['high_epoch_removal'].sum())}"
    )
    print(
        f"  very_high_epoch_removal rows: "
        f"{int(subject_duration_df['very_high_epoch_removal'].sum())}"
    )
    print(
        f"  low_clean_seconds rows: "
        f"{int(subject_duration_df['low_clean_seconds'].sum())}"
    )
    print(
        f"  near_minimum_clean_seconds rows: "
        f"{int(subject_duration_df['near_minimum_clean_seconds'].sum())}"
    )
    print(
        f"  autoreject_timeout rows: "
        f"{int(subject_duration_df['autoreject_timeout'].sum())}"
    )
    print(
        f"  variance_fallback rows: "
        f"{int(subject_duration_df['variance_fallback_used'].sum())}"
    )

    print("\nSaved files:")
    print(f"  General subject-level report: {GENERAL_PREPROCESSING_CSV}")
    print(f"  Detailed duration-level report: {SUBJECT_DURATION_CSV}")
    print(f"  Per-subject files: Cleaned/<subject_id>/preprocessing.csv")

    print("=" * 70 + "\n")


# MAIN

def main() -> None:
    cfg.apply_persisted_settings()
    cfg.make_directories()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    control = RunControl.from_environment(OUT_DIR / "_control")

    if RESUME and ANALYSIS_COHORT_CSV.exists() and GENERAL_PREPROCESSING_CSV.exists() and SUBJECT_DURATION_CSV.exists():
        cohort = pd.read_csv(ANALYSIS_COHORT_CSV, dtype={"subject_id": str})
        write_manifest(OUT_DIR, {
            "script": Path(__file__).name,
            "stage": 3,
            "inputs": {"basic": str(BASIC_SUMMARY_CSV), "epochs": str(EPOCH_SUMMARY_CSV), "mcmi": str(cfg.MCMI_CSV)},
            "outputs": {"subject_qc": str(GENERAL_PREPROCESSING_CSV), "duration_qc": str(SUBJECT_DURATION_CSV), "cohort": str(ANALYSIS_COHORT_CSV)},
            "n_subjects": int(len(cohort)),
            "n_included": int(cohort.get("final_include", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()),
            "mcmi_sha256": file_sha256(cfg.MCMI_CSV),
            "settings_hash": cfg.settings_hash_for(3),
        })
        (OUT_DIR / "QC_COMPLETE.flag").write_text("complete", encoding="utf-8")
        print(f"Resuming: completed QC outputs already exist in {OUT_DIR}")
        return

    try:
        print("\n" + "=" * 70)
        print("Starting V3 preprocessing QC report")
        print(f"Run folder: {OUT_DIR}")
        print(f"Basic summary: {BASIC_SUMMARY_CSV}")
        print(f"Epoch summary: {EPOCH_SUMMARY_CSV}")
        control.checkpoint("before loading preprocessing summaries")

        mcmi_subject_ids = load_mcmi_subject_ids()
        basic_df = load_basic_summary()
        epoch_df = load_epoch_summary()
        all_subject_ids = ordered_unique(
            mcmi_subject_ids
            + basic_df["subject_id"].astype(str).tolist()
            + epoch_df["subject_id"].astype(str).tolist()
        )

        control.checkpoint("before duration-level QC")
        subject_duration_df = make_subject_duration_qc(
            subject_ids=all_subject_ids,
            basic_df=basic_df,
            epoch_df=epoch_df,
        )
        control.checkpoint("before subject-level QC")
        subject_qc_df = make_subject_qc(subject_duration_df)

        subject_qc_df.to_csv(GENERAL_PREPROCESSING_CSV, index=False)
        subject_duration_df.to_csv(SUBJECT_DURATION_CSV, index=False)

        inclusion_col = str(getattr(cfg, "FEATURE_INCLUSION_COLUMN", "minimum_include"))
        cohort = subject_qc_df.copy()
        if inclusion_col not in cohort.columns:
            raise ValueError(f"QC output does not contain inclusion column: {inclusion_col}")
        cohort["analysis_include"] = cohort[inclusion_col].apply(normalize_bool)
        cohort["analysis_inclusion_column"] = inclusion_col
        cohort["mcmi_row_present"] = cohort["subject_id"].astype(str).isin(set(mcmi_subject_ids))
        cohort["final_include"] = cohort["analysis_include"] & cohort["mcmi_row_present"]
        cohort["final_exclusion_reason"] = ""
        cohort.loc[~cohort["analysis_include"], "final_exclusion_reason"] = "failed_preprocessing_inclusion_rule"
        cohort.loc[cohort["analysis_include"] & ~cohort["mcmi_row_present"], "final_exclusion_reason"] = "missing_mcmi_row"
        cohort.to_csv(ANALYSIS_COHORT_CSV, index=False)

        save_subject_preprocessing_csvs(subject_duration_df, subject_qc_df)
        print_summary(subject_qc_df, subject_duration_df)
        write_manifest(OUT_DIR, {
            "script": Path(__file__).name,
            "stage": 3,
            "inputs": {"basic": str(BASIC_SUMMARY_CSV), "epochs": str(EPOCH_SUMMARY_CSV), "mcmi": str(cfg.MCMI_CSV)},
            "outputs": {"subject_qc": str(GENERAL_PREPROCESSING_CSV), "duration_qc": str(SUBJECT_DURATION_CSV), "cohort": str(ANALYSIS_COHORT_CSV)},
            "n_subjects": int(len(cohort)),
            "n_included": int(cohort["final_include"].sum()),
            "cohort_hash": stable_hash(cohort[["subject_id", "final_include"]].astype(str).to_dict("records")),
            "mcmi_sha256": file_sha256(cfg.MCMI_CSV),
            "settings_hash": cfg.settings_hash_for(3),
        })
        (OUT_DIR / "QC_COMPLETE.flag").write_text("complete", encoding="utf-8")
        print(f"Frozen analysis cohort: {ANALYSIS_COHORT_CSV}")
    except GracefulStop as stop:
        print(f"Graceful stop: {stop}")


if __name__ == "__main__":
    main()
