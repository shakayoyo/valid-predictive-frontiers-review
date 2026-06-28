from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm


RUN_LABELS = {
    "minimax_anes_schema_budgetgrid": "ANES96 vote",
    "minimax_fair_schema_budgetgrid": "Fair affairs",
    "minimax_randhie_schema_budgetgrid": "RAND HIE visits",
    "minimax_modechoice_schema_budgetgrid": "Modechoice car",
    "minimax_star98_schema_budgetgrid": "Star98 education",
    "minimax_fertility_schema_budgetgrid": "Fertility demography",
    "minimax_adult_schema_budgetgrid": "Adult income",
    "minimax_acs_schema_budgetgrid": "ACS income",
}

TASK_ORDER = list(RUN_LABELS.values())
GAP_TASK_ORDER = [
    "ANES96 vote",
    "RAND HIE visits",
    "Adult income",
    "Fair affairs",
    "Modechoice car",
    "Star98 education",
    "Fertility demography",
    "ACS income",
]
FAMILY_LABELS = {
    "llm_prompt_policy_frontier": "LLM prompt-policy",
    "reference": "Reference",
}
FAMILY_COLORS = {
    "llm_prompt_policy_frontier": "#0072B2",
    "reference": "#4D4D4D",
}
COVERAGE_STYLES = {
    0.8: "-",
    0.9: "--",
}
COVERAGE_COLORS = {
    0.8: "#0072B2",
    0.9: "#D55E00",
}
COVERAGE_MARKERS = {
    0.8: "o",
    0.9: "s",
}


def filesystem_path(path: Path) -> str:
    resolved = str(path.resolve())
    if sys.platform != "win32" or resolved.startswith("\\\\?\\"):
        return resolved
    if resolved.startswith("\\\\"):
        return "\\\\?\\UNC\\" + resolved[2:]
    return "\\\\?\\" + resolved


def ensure_dir(path: Path) -> None:
    os.makedirs(filesystem_path(path), exist_ok=True)


def load_frontier(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(filesystem_path(path))
    required = {
        "run_label",
        "seed",
        "budget",
        "coverage_target",
        "method_family",
        "frontier_status",
        "frontier_set_size",
        "coverage",
        "point_risk",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns in {path}: {sorted(missing)}")
    return frame.copy()


def real_frontier_rows(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.loc[frame["run_label"].isin(RUN_LABELS)].copy()
    out["dataset_name"] = out["run_label"].map(RUN_LABELS)
    out["coverage_target"] = out["coverage_target"].astype(float)
    out["budget"] = out["budget"].astype(int)
    return out


def policy_rows(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[frame["method_family"].isin(FAMILY_LABELS)].copy()


def _mean_or_nan(series: pd.Series) -> float:
    clean = series.dropna()
    if clean.empty:
        return np.nan
    return float(clean.mean())


def _sem_or_nan(series: pd.Series) -> float:
    clean = series.dropna()
    if len(clean) <= 1:
        return np.nan
    return float(clean.std(ddof=1) / np.sqrt(len(clean)))


def summarize_frontier(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["dataset_name", "run_label", "method_family", "coverage_target", "budget"]
    for keys, group in frame.groupby(group_cols, dropna=False):
        attained = group.loc[group["frontier_status"] == "attained"]
        mean_size = _mean_or_nan(attained["frontier_set_size"])
        sem_size = _sem_or_nan(attained["frontier_set_size"])
        rows.append(
            {
                "dataset_name": keys[0],
                "run_label": keys[1],
                "method_family": keys[2],
                "coverage_target": keys[3],
                "budget": keys[4],
                "n_cells": int(len(group)),
                "n_attained": int(len(attained)),
                "attainment_rate": float(len(attained) / len(group)) if len(group) else np.nan,
                "mean_frontier_set_size": mean_size,
                "se_frontier_set_size": sem_size,
                "ci95_low_frontier_set_size": mean_size - 1.96 * sem_size if np.isfinite(sem_size) else mean_size,
                "ci95_high_frontier_set_size": mean_size + 1.96 * sem_size if np.isfinite(sem_size) else mean_size,
                "mean_coverage": _mean_or_nan(attained["coverage"]),
                "mean_point_risk": _mean_or_nan(attained["point_risk"]),
            }
        )
    return pd.DataFrame(rows)


def paired_gap_summary(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["dataset_name", "run_label", "seed", "budget", "coverage_target"]
    attained = frame.loc[
        (frame["frontier_status"] == "attained") & frame["method_family"].isin(FAMILY_LABELS)
    ].copy()
    left = attained.loc[attained["method_family"] == "llm_prompt_policy_frontier", keys + ["frontier_set_size", "coverage", "point_risk"]]
    right = attained.loc[attained["method_family"] == "reference", keys + ["frontier_set_size", "coverage", "point_risk"]]
    paired = left.merge(right, on=keys, suffixes=("_llm", "_reference"))
    paired["set_size_gap_llm_minus_reference"] = paired["frontier_set_size_llm"] - paired["frontier_set_size_reference"]
    paired["coverage_gap_llm_minus_reference"] = paired["coverage_llm"] - paired["coverage_reference"]
    paired["risk_gap_llm_minus_reference"] = paired["point_risk_llm"] - paired["point_risk_reference"]

    rows = []
    group_cols = ["dataset_name", "run_label", "coverage_target", "budget"]
    for keys2, group in paired.groupby(group_cols, dropna=False):
        gap = group["set_size_gap_llm_minus_reference"]
        rows.append(
            {
                "dataset_name": keys2[0],
                "run_label": keys2[1],
                "coverage_target": keys2[2],
                "budget": keys2[3],
                "n_paired_cells": int(len(group)),
                "mean_set_size_gap_llm_minus_reference": _mean_or_nan(gap),
                "se_set_size_gap_llm_minus_reference": _sem_or_nan(gap),
                "mean_coverage_gap_llm_minus_reference": _mean_or_nan(group["coverage_gap_llm_minus_reference"]),
                "mean_risk_gap_llm_minus_reference": _mean_or_nan(group["risk_gap_llm_minus_reference"]),
            }
        )
    return paired, pd.DataFrame(rows)


def summarize_task_gaps(paired: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["dataset_name", "run_label", "coverage_target"]
    for keys, group in paired.groupby(group_cols, dropna=False):
        gap = group["set_size_gap_llm_minus_reference"]
        risk_gap = group["risk_gap_llm_minus_reference"]
        mean_gap = _mean_or_nan(gap)
        se_gap = _sem_or_nan(gap)
        rows.append(
            {
                "dataset_name": keys[0],
                "run_label": keys[1],
                "coverage_target": keys[2],
                "n_paired_cells": int(len(group)),
                "mean_set_size_gap_llm_minus_reference": mean_gap,
                "se_set_size_gap_llm_minus_reference": se_gap,
                "ci95_low_set_size_gap_llm_minus_reference": mean_gap - 1.96 * se_gap
                if np.isfinite(se_gap)
                else mean_gap,
                "ci95_high_set_size_gap_llm_minus_reference": mean_gap + 1.96 * se_gap
                if np.isfinite(se_gap)
                else mean_gap,
                "mean_risk_gap_llm_minus_reference": _mean_or_nan(risk_gap),
                "mean_coverage_gap_llm_minus_reference": _mean_or_nan(
                    group["coverage_gap_llm_minus_reference"]
                ),
            }
        )
    return pd.DataFrame(rows)


def task_coverage_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["dataset_name", "run_label", "method_family", "coverage_target"]
    for keys, group in frame.groupby(group_cols, dropna=False):
        attained = group.loc[group["frontier_status"] == "attained"]
        coverage_excess = attained["coverage"] - float(keys[3])
        rows.append(
            {
                "dataset_name": keys[0],
                "run_label": keys[1],
                "method_family": keys[2],
                "coverage_target": keys[3],
                "n_cells": int(len(group)),
                "n_attained": int(len(attained)),
                "attainment_rate": float(len(attained) / len(group)) if len(group) else np.nan,
                "mean_coverage": _mean_or_nan(attained["coverage"]),
                "mean_coverage_excess": _mean_or_nan(coverage_excess),
                "se_coverage_excess": _sem_or_nan(coverage_excess),
                "mean_frontier_set_size": _mean_or_nan(attained["frontier_set_size"]),
            }
        )
    return pd.DataFrame(rows)


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    frame.to_csv(filesystem_path(path), index=False)


def present_task_order(*frames: pd.DataFrame) -> list[str]:
    present: set[str] = set()
    for frame in frames:
        if not frame.empty and "dataset_name" in frame.columns:
            present.update(frame["dataset_name"].dropna().astype(str).unique().tolist())
    return [task for task in TASK_ORDER if task in present]


def add_frontier_curves(ax: plt.Axes, summary: pd.DataFrame, dataset_name: str) -> None:
    sub = summary.loc[summary["dataset_name"] == dataset_name]
    for method_family in FAMILY_LABELS:
        for coverage_target in sorted(sub["coverage_target"].unique()):
            curve = sub.loc[
                (sub["method_family"] == method_family)
                & (np.isclose(sub["coverage_target"], coverage_target))
            ].sort_values("budget")
            if curve.empty:
                continue
            label = f"{FAMILY_LABELS[method_family]}, {coverage_target:.1f}"
            ax.plot(
                curve["budget"],
                curve["mean_frontier_set_size"],
                color=FAMILY_COLORS[method_family],
                linestyle=COVERAGE_STYLES.get(float(coverage_target), "-"),
                marker="o",
                markersize=3.4,
                linewidth=1.5,
                label=label,
            )
            low = curve["ci95_low_frontier_set_size"].to_numpy(dtype=float)
            high = curve["ci95_high_frontier_set_size"].to_numpy(dtype=float)
            if np.isfinite(low).any() and np.isfinite(high).any():
                ax.fill_between(
                    curve["budget"].to_numpy(dtype=float),
                    low,
                    high,
                    color=FAMILY_COLORS[method_family],
                    alpha=0.10,
                    linewidth=0,
                )
    ax.set_title(dataset_name, fontsize=10.5, pad=5)
    ax.set_ylim(0.75, 2.06)
    ax.grid(axis="y", color="0.88", linewidth=0.6)
    ax.tick_params(axis="both", labelsize=8.5)


def add_gap_heatmap(ax: plt.Axes, gaps: pd.DataFrame, task_order: list[str], panel_label: str) -> None:
    budgets = sorted(gaps["budget"].unique())
    row_keys = []
    row_labels = []
    for task in task_order:
        for coverage in sorted(gaps.loc[gaps["dataset_name"] == task, "coverage_target"].unique()):
            row_keys.append((task, coverage))
            row_labels.append(f"{task}, {coverage:.1f}")
    matrix = np.full((len(row_keys), len(budgets)), np.nan)
    for i, (task, coverage) in enumerate(row_keys):
        for j, budget in enumerate(budgets):
            hit = gaps.loc[
                (gaps["dataset_name"] == task)
                & (np.isclose(gaps["coverage_target"], coverage))
                & (gaps["budget"] == budget)
            ]
            if not hit.empty:
                matrix[i, j] = float(hit["mean_set_size_gap_llm_minus_reference"].iloc[0])

    finite = matrix[np.isfinite(matrix)]
    limit = max(0.10, float(np.nanmax(np.abs(finite)))) if finite.size else 0.10
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    image = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", norm=norm)
    ax.set_title(f"{panel_label}  LLM minus reference frontier set size")
    ax.set_xlabel("Labelled-data budget")
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_xticks(np.arange(len(budgets)))
    ax.set_xticklabels([str(int(budget)) for budget in budgets])
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if np.isfinite(value):
                color = "white" if abs(value) > 0.55 * limit else "black"
                ax.text(j, i, f"{value:+.2f}", ha="center", va="center", fontsize=7, color=color)
            else:
                ax.text(j, i, "NA", ha="center", va="center", fontsize=7, color="0.4")
    return image


def add_coverage_panel(ax: plt.Axes, task_summary: pd.DataFrame, task_order: list[str], panel_label: str) -> None:
    offsets = {
        ("llm_prompt_policy_frontier", 0.8): -0.21,
        ("reference", 0.8): -0.07,
        ("llm_prompt_policy_frontier", 0.9): 0.07,
        ("reference", 0.9): 0.21,
    }
    markers = {0.8: "o", 0.9: "s"}
    x_base = np.arange(len(task_order))
    for (method_family, coverage_target), sub in task_summary.groupby(["method_family", "coverage_target"]):
        sub = sub.set_index("dataset_name").reindex(task_order).reset_index()
        x = x_base + offsets.get((method_family, float(coverage_target)), 0.0)
        y = sub["mean_coverage_excess"].to_numpy(dtype=float)
        se = sub["se_coverage_excess"].to_numpy(dtype=float)
        yerr = np.where(np.isfinite(se), 1.96 * se, 0.0)
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            fmt=markers.get(float(coverage_target), "o"),
            color=FAMILY_COLORS.get(method_family, "0.25"),
            markerfacecolor=FAMILY_COLORS.get(method_family, "0.25"),
            markeredgecolor="white",
            markeredgewidth=0.5,
            markersize=5,
            linewidth=1.0,
            capsize=2.5,
            label=f"{FAMILY_LABELS[method_family]}, {coverage_target:.1f}",
        )
    ax.axhline(0.0, color="0.25", linewidth=0.9, linestyle=":")
    ax.set_xticks(x_base)
    ax.set_xticklabels(task_order, rotation=32, ha="right", fontsize=8)
    ax.set_ylabel("Coverage minus target")
    ax.set_title(f"{panel_label}  Empirical coverage relative to target")
    ax.grid(axis="y", color="0.88", linewidth=0.6)
    ax.legend(fontsize=7, frameon=False, ncol=2, loc="upper left")


def make_task_frontier_figure(summary: pd.DataFrame, output_pdf: Path, output_png: Path) -> None:
    ensure_dir(output_pdf.parent)
    task_order = present_task_order(summary)
    if not task_order:
        raise ValueError("No tasks available for frontier figure.")
    n_cols = 4 if len(task_order) >= 4 else len(task_order)
    n_rows = int(np.ceil(len(task_order) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12.2, 3.05 * n_rows), sharey=True)
    axes_array = np.atleast_1d(axes).ravel()
    panel_letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    for ax, letter, task in zip(axes_array, panel_letters, task_order):
        add_frontier_curves(ax, summary, task)
        ax.text(-0.13, 1.05, letter, transform=ax.transAxes, fontsize=12, fontweight="bold", va="top")
    for ax in axes_array[len(task_order) :]:
        ax.axis("off")
    for row_idx in range(n_rows):
        axes_array[row_idx * n_cols].set_ylabel("Mean valid set size")
    handles, labels = axes_array[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=8, frameon=False, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.01))
    fig.supxlabel("Labelled-data budget", fontsize=10, y=0.02)
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    fig.savefig(filesystem_path(output_pdf), bbox_inches="tight")
    fig.savefig(filesystem_path(output_png), dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_task_gap_figure(gap_summary: pd.DataFrame, output_pdf: Path, output_png: Path) -> None:
    ensure_dir(output_pdf.parent)
    task_order = [task for task in GAP_TASK_ORDER if task in set(gap_summary["dataset_name"])]
    if not task_order:
        task_order = present_task_order(gap_summary)
    if not task_order:
        raise ValueError("No tasks available for task-gap figure.")

    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    y_base = np.arange(len(task_order))
    ax.axvspan(-0.35, 0.0, color="#F1F7FD", zorder=0)
    ax.axvspan(0.0, 0.90, color="#F7F7F7", zorder=0)
    for y in y_base:
        ax.axhline(y, color="white", linewidth=1.0, zorder=0)
    ax.axvline(0.0, color="0.15", linewidth=1.0)

    offsets = {0.8: -0.16, 0.9: 0.16}
    for coverage_target in sorted(gap_summary["coverage_target"].unique()):
        sub = gap_summary.loc[np.isclose(gap_summary["coverage_target"], coverage_target)].set_index("dataset_name")
        ordered = sub.reindex(task_order)
        x = ordered["mean_set_size_gap_llm_minus_reference"].to_numpy(dtype=float)
        low = ordered["ci95_low_set_size_gap_llm_minus_reference"].to_numpy(dtype=float)
        high = ordered["ci95_high_set_size_gap_llm_minus_reference"].to_numpy(dtype=float)
        y = y_base + offsets.get(float(coverage_target), 0.0)
        xerr = np.vstack([x - low, high - x])
        ax.errorbar(
            x,
            y,
            xerr=xerr,
            fmt=COVERAGE_MARKERS.get(float(coverage_target), "o"),
            color=COVERAGE_COLORS.get(float(coverage_target), "0.25"),
            markerfacecolor=COVERAGE_COLORS.get(float(coverage_target), "0.25"),
            markeredgecolor="white",
            markeredgewidth=0.7,
            markersize=6.4,
            linewidth=1.2,
            capsize=3,
            label=f"Target coverage {coverage_target:.1f}",
            zorder=3,
        )

    finite_low = gap_summary["ci95_low_set_size_gap_llm_minus_reference"].to_numpy(dtype=float)
    finite_high = gap_summary["ci95_high_set_size_gap_llm_minus_reference"].to_numpy(dtype=float)
    xmin = min(-0.28, float(np.nanmin(finite_low)) - 0.05)
    xmax = max(0.82, float(np.nanmax(finite_high)) + 0.05)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(-0.65, len(task_order) - 0.35)
    ax.set_yticks(y_base)
    ax.set_yticklabels(task_order)
    ax.invert_yaxis()
    ax.set_xlabel("LLM minus reference valid-set size (labels)")
    ax.set_title("Task-level valid-set gap", loc="left", fontsize=12, fontweight="bold", pad=8)
    ax.text(
        xmin + 0.02,
        -0.46,
        "LLM shorter",
        ha="left",
        va="center",
        fontsize=9,
        color="#005B8E",
    )
    ax.text(
        xmax - 0.02,
        -0.46,
        "Reference shorter",
        ha="right",
        va="center",
        fontsize=9,
        color="0.25",
    )
    ax.grid(axis="x", color="0.86", linewidth=0.7)
    ax.tick_params(axis="both", labelsize=9.5)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.11), fontsize=9, ncol=2)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(filesystem_path(output_pdf), bbox_inches="tight")
    fig.savefig(filesystem_path(output_png), dpi=240, bbox_inches="tight")
    plt.close(fig)


def make_gap_coverage_figure(gaps: pd.DataFrame, coverage_summary: pd.DataFrame, output_pdf: Path, output_png: Path) -> None:
    ensure_dir(output_pdf.parent)
    task_order = present_task_order(gaps, coverage_summary)
    if not task_order:
        raise ValueError("No tasks available for gap/coverage figure.")
    fig = plt.figure(figsize=(13.0, 6.8))
    grid = fig.add_gridspec(1, 2, width_ratios=[1.2, 0.8], wspace=0.38)
    ax_gap = fig.add_subplot(grid[0, 0])
    image = add_gap_heatmap(ax_gap, gaps, task_order, "A")
    cbar = fig.colorbar(image, ax=ax_gap, orientation="horizontal", fraction=0.075, pad=0.18)
    cbar.set_label("Set-size gap (LLM - reference)")
    ax_cov = fig.add_subplot(grid[0, 1])
    add_coverage_panel(ax_cov, coverage_summary, task_order, "B")

    fig.suptitle("Supplementary frontier gap and coverage summaries", fontsize=13)
    fig.savefig(filesystem_path(output_pdf), bbox_inches="tight")
    fig.savefig(filesystem_path(output_png), dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_figure(
    summary: pd.DataFrame,
    paired: pd.DataFrame,
    gaps: pd.DataFrame,
    coverage_summary: pd.DataFrame,
    output_pdf: Path,
    output_png: Path,
) -> None:
    task_gap_summary = summarize_task_gaps(paired)
    write_csv(task_gap_summary, output_pdf.parent.parent / "frontier_diagnostics_task_gap_summary.csv")
    make_task_gap_figure(task_gap_summary, output_pdf, output_png)
    make_task_frontier_figure(
        summary,
        output_pdf.with_name("F02_valid_frontier_curves_supplementary.pdf"),
        output_png.with_name("F02_valid_frontier_curves_supplementary.png"),
    )
    make_gap_coverage_figure(
        gaps,
        coverage_summary,
        output_pdf.with_name("F02_frontier_gap_coverage_diagnostics.pdf"),
        output_png.with_name("F02_frontier_gap_coverage_diagnostics.png"),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frontier-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output = Path(args.output_dir).resolve()
    ensure_dir(output)
    real = policy_rows(real_frontier_rows(load_frontier(Path(args.frontier_csv).resolve())))
    if real.empty:
        raise ValueError("No real schema budget-grid frontier rows found.")

    summary = summarize_frontier(real)
    paired, gaps = paired_gap_summary(real)
    coverage_summary = task_coverage_summary(real)
    write_csv(summary, output / "frontier_diagnostics_by_budget.csv")
    write_csv(paired, output / "frontier_diagnostics_paired_seed_gaps.csv")
    write_csv(gaps, output / "frontier_diagnostics_paired_gaps_by_budget.csv")
    write_csv(coverage_summary, output / "frontier_diagnostics_by_task_coverage.csv")
    make_figure(
        summary,
        paired,
        gaps,
        coverage_summary,
        output / "figures" / "F02_valid_frontier_diagnostics.pdf",
        output / "figures" / "F02_valid_frontier_diagnostics.png",
    )
    print(
        {
            "status": "ok",
            "output_dir": str(output),
            "n_rows": int(len(real)),
            "n_paired_seed_cells": int(len(paired)),
        }
    )


if __name__ == "__main__":
    main()
