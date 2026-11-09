"""Create descriptive reports from the Stage 6 screening results."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from V3_config_loader import cfg
from V3_runtime import (
    GracefulStop,
    RunControl,
    env_bool,
    make_stage_run_dir,
    write_manifest,
)


RESUME = env_bool("NEUROPSYCHE_RESUME", True)
RUN_SCRIPT_NAME = cfg.ordinal_stage_script_name(Path(__file__).name)
OUT_DIR = make_stage_run_dir(cfg.REPORTS_DIR, RUN_SCRIPT_NAME, resume=RESUME)

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


def nested_input_dir() -> Path:
    raw = cfg.runtime_input("stage6_run")
    run = Path(raw) if raw else None
    if run is None or not run.exists():
        raise FileNotFoundError(
            "No Stage 6 nested-analysis run was found for the selected "
            "ordinal scheme."
        )
    return run


def fold_balance_file() -> Path | None:
    raw = cfg.runtime_input("fold_file")
    if not raw:
        return None
    path = Path(raw).parent / "fixed_outer_fold_balance.csv"
    return path if path.exists() else None


def fold_balance_for_selected_scheme(path: Path) -> pd.DataFrame:
    """Present the locked Stage-5 balance under the active ordinal labels."""
    balance = pd.read_csv(path)

    if cfg.ordinal_scheme_name() == "3class_posthoc":
        required = {"ordinal_2", "ordinal_3"}
        if not required.issubset(balance.columns):
            raise ValueError(
                "The Stage 5 balance report lacks ordinal_2/ordinal_3 columns "
                "needed to construct the post-hoc 75+ class."
            )
        balance["ordinal_2"] = (
            pd.to_numeric(balance["ordinal_2"], errors="coerce").fillna(0)
            + pd.to_numeric(balance["ordinal_3"], errors="coerce").fillna(0)
        ).astype(int)
        balance = balance.drop(columns=["ordinal_3"])

    balance["ordinal_scheme"] = cfg.ordinal_scheme_name()
    balance["ordinal_class_labels"] = " | ".join(cfg.ordinal_labels())
    return balance


def load_first_existing(run: Path, names: list[str], required: bool = True) -> pd.DataFrame:
    for name in names:
        path = run / name
        if path.exists():
            return pd.read_csv(path)
    if required:
        raise FileNotFoundError(f"None of these files exists in {run}: {names}")
    return pd.DataFrame()


def positive_k_values() -> list[int]:
    raw = getattr(cfg, "TOP_K_PLOT_VALUES", [5, 10, 20, 50])
    values = sorted({int(v) for v in raw if int(v) >= 1})
    if not values:
        values = [5, 10, 20, 50]
    return values


def rank_all_configuration_rows(all_configs: pd.DataFrame) -> pd.DataFrame:
    data = all_configs.copy()
    if "status" in data.columns:
        data = data[data["status"].astype(str).str.lower().eq("ok")].copy()
    data["inner_metric_mean"] = pd.to_numeric(data["inner_metric_mean"], errors="coerce")
    data["config_id"] = pd.to_numeric(data["config_id"], errors="coerce")
    data["n_features"] = pd.to_numeric(data.get("n_features"), errors="coerce")
    data = data[np.isfinite(data["inner_metric_mean"])].copy()
    if data.empty:
        return data

    if "inner_rank" in data.columns:
        data["inner_rank"] = pd.to_numeric(data["inner_rank"], errors="coerce")
        rank_is_complete = data["inner_rank"].notna().all()
        rank_is_unique = all(
            group["inner_rank"].nunique() == len(group)
            for _, group in data.groupby(["outer_fold", "task", "scale"], sort=False)
        )
        if rank_is_complete and rank_is_unique:
            data["inner_rank"] = data["inner_rank"].astype(int)
            return data.sort_values(
                ["outer_fold", "task", "scale", "inner_rank"],
                kind="mergesort",
            ).reset_index(drop=True)

    ranked_parts = []
    for _, group in data.groupby(["outer_fold", "task", "scale"], sort=False):
        group = group.sort_values(
            ["inner_metric_mean", "n_features", "config_id"],
            ascending=[False, True, True],
            kind="mergesort",
        ).copy()
        group["inner_rank"] = np.arange(1, len(group) + 1, dtype=int)
        ranked_parts.append(group)
    return pd.concat(ranked_parts, ignore_index=True)


def create_topk_screening_summary(
    ranked: pd.DataFrame,
    k_values: Iterable[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fold_rows = []
    for (outer_fold, task, scale), group in ranked.groupby(
        ["outer_fold", "task", "scale"], sort=False
    ):
        group = group.sort_values("inner_rank")
        values = pd.to_numeric(group["inner_metric_mean"], errors="coerce").dropna().to_numpy()
        for k in k_values:
            actual_k = min(int(k), len(values))
            if actual_k < 1:
                continue
            fold_rows.append({
                "outer_fold": int(outer_fold),
                "task": str(task),
                "scale": str(scale),
                "requested_k": int(k),
                "actual_k": int(actual_k),
                "topk_inner_metric_mean": float(np.mean(values[:actual_k])),
                "topk_inner_metric_sd_within_cell": float(np.std(values[:actual_k], ddof=1)) if actual_k > 1 else 0.0,
                "best_inner_metric": float(values[0]),
                "kth_inner_metric": float(values[actual_k - 1]),
                "n_valid_configurations": int(len(values)),
            })
    fold_summary = pd.DataFrame(fold_rows)
    if fold_summary.empty:
        return fold_summary, pd.DataFrame()

    aggregate = (
        fold_summary.groupby(["task", "scale", "requested_k"], as_index=False)
        .agg(
            topk_mean_across_outer_folds=("topk_inner_metric_mean", "mean"),
            topk_sd_across_outer_folds=("topk_inner_metric_mean", "std"),
            topk_min_outer_fold=("topk_inner_metric_mean", "min"),
            topk_max_outer_fold=("topk_inner_metric_mean", "max"),
            mean_actual_k=("actual_k", "mean"),
            n_outer_folds=("outer_fold", "nunique"),
        )
    )
    return fold_summary, aggregate


def create_topk_plot(summary: pd.DataFrame, output: Path, k_values: list[int]) -> None:
    if summary.empty:
        return
    scale_order = [s for s in cfg.MCMI_SCALE_ORDER if s in set(summary["scale"].astype(str))]
    remaining = [s for s in summary["scale"].astype(str).unique() if s not in scale_order]
    scale_order.extend(remaining)

    fig, axes = plt.subplots(2, 2, figsize=(16, 9), constrained_layout=False)
    axes = axes.ravel()
    handles = []
    labels = []

    for ax, task in zip(axes, TASK_ORDER):
        sub = summary[summary["task"].astype(str) == task].copy()
        ax.set_title(TASK_TITLES[task], fontweight="bold")
        ax.set_ylabel(TASK_YLABELS[task])
        ax.set_xticks(np.arange(len(scale_order)))
        ax.set_xticklabels(scale_order, rotation=45, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.25)

        for k in k_values:
            line = sub[sub["requested_k"] == int(k)].set_index("scale")
            y = [
                float(line.loc[scale, "topk_mean_across_outer_folds"])
                if scale in line.index else np.nan
                for scale in scale_order
            ]
            plotted, = ax.plot(
                np.arange(len(scale_order)),
                y,
                marker="o",
                linewidth=1.6,
                markersize=4,
                label=f"Top {k}",
            )
            if ax is axes[0]:
                handles.append(plotted)
                labels.append(f"Top {k}")

        chance = TASK_CHANCE[task]
        chance_line = ax.axhline(chance, linestyle="--", linewidth=1.0, label="Chance")
        if ax is axes[0]:
            handles.append(chance_line)
            labels.append("Chance")

    fig.suptitle(
        "Top-k mean performance across EEG feature configurations",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )
    fig.text(
        0.5,
        0.035,
        "For each scale and task, configurations were ranked using inner-CV performance only; "
        "each line shows the mean of the top k configurations, averaged across the five outer-training folds.",
        ha="center",
        fontsize=9,
    )
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False, bbox_to_anchor=(0.5, 0.005))
    fig.tight_layout(rect=(0.02, 0.08, 1, 0.95))

    if bool(getattr(cfg, "MAKE_PNG", True)):
        fig.savefig(output / "topk_mean_performance_across_EEG_configurations.png", dpi=200)
    if bool(getattr(cfg, "MAKE_PDF", True)):
        fig.savefig(output / "topk_mean_performance_across_EEG_configurations.pdf")
    plt.close(fig)


def selected_family_recurrence(candidates: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in candidates.iterrows():
        families = [x for x in str(row.get("feature_set", "")).split("+") if x]
        for family in families:
            rows.append({
                "task": row["task"],
                "scale": row["scale"],
                "outer_fold": row["outer_fold"],
                "inner_rank": row.get("inner_rank"),
                "feature_family": family,
            })
    if not rows:
        return pd.DataFrame(columns=["task", "scale", "feature_family", "candidate_occurrences"])
    return (
        pd.DataFrame(rows)
        .groupby(["task", "scale", "feature_family"], as_index=False)
        .size()
        .rename(columns={"size": "candidate_occurrences"})
        .sort_values(["task", "scale", "candidate_occurrences"], ascending=[True, True, False])
    )


def create_rank1_metric_figure(metrics: pd.DataFrame, output: Path) -> None:
    if metrics.empty:
        return
    for task in metrics["task"].dropna().unique().tolist():
        sub = metrics[metrics["task"] == task].copy()
        if sub.empty:
            continue
        primary_col = "roc_auc" if str(task).startswith("binary_") else ("qwk" if task == "ordinal" else "pearson_r")
        if primary_col not in sub.columns:
            primary_col = "primary"
        sub = sub.sort_values(primary_col, ascending=True)
        fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(sub))))
        ax.barh(sub["scale"].astype(str), pd.to_numeric(sub[primary_col], errors="coerce"))
        ax.set_xlabel(primary_col)
        ax.set_title(f"Outer-test performance of inner rank 1 — {task}")
        fig.tight_layout()
        if bool(getattr(cfg, "MAKE_PNG", True)):
            fig.savefig(output / f"rank1_outer_performance_{task}.png", dpi=180)
        if bool(getattr(cfg, "MAKE_PDF", True)):
            fig.savefig(output / f"rank1_outer_performance_{task}.pdf")
        plt.close(fig)


def main() -> None:
    cfg.apply_persisted_settings()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    control = RunControl.from_environment(OUT_DIR / "_control")
    complete_marker = OUT_DIR / "REPORT_COMPLETE.flag"
    if RESUME and complete_marker.exists():
        print(f"Resuming: report is already complete in {OUT_DIR}")
        return

    source = nested_input_dir()
    try:
        control.checkpoint("before loading complete configuration results")
        all_configs = load_first_existing(
            source,
            ["nested_v3_all_configuration_results.csv", "nested_v3_inner_config_summary.csv.gz"],
        )
        topk_candidates = load_first_existing(
            source,
            ["nested_v3_topk_candidates.csv"],
            required=False,
        )
        rank1_metrics = load_first_existing(
            source,
            ["nested_v3_final_metrics_rank1.csv", "nested_v3_final_metrics.csv"],
            required=False,
        )
        rank1_predictions = load_first_existing(
            source,
            [
                "nested_v3_outer_subject_predictions_rank1.csv.gz",
                "nested_v3_outer_subject_predictions.csv.gz",
                "nested_v3_outer_subject_predictions.csv",
            ],
            required=False,
        )

        ranked = rank_all_configuration_rows(all_configs)
        if ranked.empty:
            raise RuntimeError("Stage 6 contains no valid configuration screening results.")

        k_values = positive_k_values()
        fold_topk, aggregate_topk = create_topk_screening_summary(ranked, k_values)
        fold_topk.to_csv(OUT_DIR / "topk_inner_screening_by_outer_fold.csv", index=False)
        aggregate_topk.to_csv(OUT_DIR / "topk_inner_screening_summary.csv", index=False)

        descriptive_k = int(getattr(cfg, "REPORT_TOP_CONFIGS", 20))
        if descriptive_k < 1:
            raise ValueError("REPORT_TOP_CONFIGS must be at least 1.")
        ranked[ranked["inner_rank"] <= descriptive_k].to_csv(
            OUT_DIR / f"top_{descriptive_k}_configurations_per_outer_fold_task_scale.csv",
            index=False,
        )

        if not topk_candidates.empty:
            topk_candidates.to_csv(OUT_DIR / "retained_topk_candidates_for_stage8.csv", index=False)
            if bool(getattr(cfg, "INCLUDE_SELECTION_STABILITY", True)):
                selected_family_recurrence(topk_candidates).to_csv(
                    OUT_DIR / "topk_feature_family_recurrence.csv", index=False
                )
                duration_stability = (
                    topk_candidates.groupby(["task", "scale", "duration_label"], dropna=False)
                    .size()
                    .reset_index(name="candidate_occurrences")
                    .sort_values(["task", "scale", "candidate_occurrences"], ascending=[True, True, False])
                )
                duration_stability.to_csv(OUT_DIR / "topk_epoch_duration_recurrence.csv", index=False)

        if not rank1_metrics.empty:
            rank1_metrics.to_csv(OUT_DIR / "rank1_outer_test_metrics.csv", index=False)
        if not rank1_predictions.empty:
            fold_rows = []
            for (task, scale, fold), group in rank1_predictions.groupby(["task", "scale", "outer_fold"]):
                y = group["y_true"].to_numpy()
                score = group["prediction_score"].to_numpy(dtype=float)
                pred = group["prediction_label"].to_numpy()
                row = {"task": task, "scale": scale, "outer_fold": fold, "n": len(group)}
                try:
                    if str(task).startswith("binary_"):
                        from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score
                        row["roc_auc"] = roc_auc_score(y, score) if len(np.unique(y)) == 2 else np.nan
                        row["pr_auc"] = average_precision_score(y, score) if len(np.unique(y)) == 2 else np.nan
                        row["balanced_accuracy"] = balanced_accuracy_score(y, pred)
                    elif task == "ordinal":
                        from sklearn.metrics import cohen_kappa_score, accuracy_score
                        row["qwk"] = cohen_kappa_score(y.astype(int), pred.astype(int), weights="quadratic")
                        row["accuracy"] = accuracy_score(y.astype(int), pred.astype(int))
                    else:
                        row["pearson_r"] = np.corrcoef(y, score)[0, 1] if np.std(y) > 0 and np.std(score) > 0 else np.nan
                except Exception:
                    pass
                fold_rows.append(row)
            pd.DataFrame(fold_rows).to_csv(OUT_DIR / "rank1_performance_by_outer_fold.csv", index=False)

        balance_path = fold_balance_file()
        if balance_path is not None and bool(getattr(cfg, "INCLUDE_FOLD_BALANCE", True)):
            fold_balance_for_selected_scheme(balance_path).to_csv(
                OUT_DIR / "fixed_outer_fold_balance.csv",
                index=False,
            )

        control.checkpoint("before figures")
        create_topk_plot(aggregate_topk, OUT_DIR, k_values)
        create_rank1_metric_figure(rank1_metrics, OUT_DIR)

        lines = [
            "MAPPER complete configuration-screening report",
            f"Source Stage 6 run: {source}",
            f"Successful configuration results: {len(ranked):,}",
            f"Top-k plot values: {', '.join(map(str, k_values))}",
            f"Ordinal scheme: {cfg.ordinal_scheme_name()}",
            f"Ordinal classes: {' | '.join(cfg.ordinal_labels())}",
            "",
            "Every configuration was ranked separately within each outer-training fold, task, and scale.",
            "The Top-k screening plots use inner-CV results only and therefore do not select candidates using outer-test performance.",
            "Stage 8 evaluates every retained Top-K candidate on the untouched outer-test folds and produces the final Top-K metrics.",
        ]
        (OUT_DIR / "README_results.txt").write_text("\n".join(lines), encoding="utf-8")

        write_manifest(OUT_DIR, {
            "script": Path(__file__).name,
            "stage": 7,
            "source_nested_run": str(source),
            "n_successful_configuration_rows": int(len(ranked)),
            "n_retained_topk_candidate_rows": int(len(topk_candidates)),
            "top_k_plot_values": k_values,
            "report_top_configurations": descriptive_k,
            "selection_stability_included": bool(getattr(cfg, "INCLUDE_SELECTION_STABILITY", True)),
            "ordinal_scheme": cfg.ordinal_scheme_name(),
            "ordinal_cutoffs": list(cfg.ordinal_cutoffs()),
            "ordinal_labels": cfg.ordinal_labels(),
            "ordinal_n_classes": cfg.ordinal_n_classes(),
            "topk_plot_uses_inner_cv_only": True,
            "outer_test_used_for_ranking": False,
            "settings_hash": cfg.settings_hash_for(7),
        })
        complete_marker.write_text("complete", encoding="utf-8")
        print(f"Nested screening report saved to: {OUT_DIR}")
    except GracefulStop as stop:
        print(f"Graceful stop: {stop}")


if __name__ == "__main__":
    main()
