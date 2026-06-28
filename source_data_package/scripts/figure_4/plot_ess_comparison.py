from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


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

TASK_COLORS = {
    "ANES96 vote": "#1f77b4",
    "Fair affairs": "#d62728",
    "RAND HIE visits": "#2ca02c",
    "Modechoice car": "#6b6b6b",
    "Star98 education": "#CC79A7",
    "Fertility demography": "#8c564b",
    "Adult income": "#009E73",
    "ACS income": "#0072B2",
    "Combined": "#4d4d4d",
}

TASK_SHORT_LABELS = {
    "ANES96 vote": "ANES96",
    "Fair affairs": "Fair",
    "RAND HIE visits": "RAND HIE",
    "Modechoice car": "Modechoice",
    "Star98 education": "Star98",
    "Fertility demography": "Fertility",
    "Adult income": "Adult",
    "ACS income": "ACS",
    "Combined": "Combined",
}


def load_comparison(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        "run_label",
        "coverage_target",
        "target_budget",
        "fess_ess_bracket",
        "ress_ess_bracket",
        "fess_ess_point_on_grid",
        "ress_ess_point_on_grid",
        "ess_brackets_disagree",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns in {path}: {sorted(missing)}")
    return frame.copy()


def real_budgetgrid_rows(frame: pd.DataFrame) -> pd.DataFrame:
    keep = frame["run_label"].isin(RUN_LABELS)
    out = frame.loc[keep].copy()
    out["dataset_name"] = out["run_label"].map(RUN_LABELS)
    return out


def disagreement_summary(frame: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["dataset_name", "coverage_target", "target_budget"]
    summary = (
        frame.groupby(group_cols, dropna=False)
        .agg(
            n_cells=("ess_brackets_disagree", "size"),
            n_disagree=("ess_brackets_disagree", "sum"),
        )
        .reset_index()
    )
    summary["disagreement_rate"] = summary["n_disagree"] / summary["n_cells"]
    return summary


def task_summary(frame: pd.DataFrame) -> pd.DataFrame:
    summary = (
        frame.groupby(["dataset_name"], dropna=False)
        .agg(
            n_cells=("ess_brackets_disagree", "size"),
            n_disagree=("ess_brackets_disagree", "sum"),
        )
        .reset_index()
    )
    combined = pd.DataFrame(
        [
            {
                "dataset_name": "Combined",
                "n_cells": int(len(frame)),
                "n_disagree": int(frame["ess_brackets_disagree"].sum()),
            }
        ]
    )
    out = pd.concat([summary, combined], ignore_index=True)
    out["disagreement_rate"] = out["n_disagree"] / out["n_cells"]
    intervals = out.apply(lambda row: wilson_interval(int(row.n_disagree), int(row.n_cells)), axis=1)
    out["wilson95_low"] = [low for low, _ in intervals]
    out["wilson95_high"] = [high for _, high in intervals]
    return out


def wilson_interval(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n <= 0:
        return (math.nan, math.nan)
    phat = k / n
    denom = 1.0 + z * z / n
    center = (phat + z * z / (2.0 * n)) / denom
    half = z * math.sqrt((phat * (1.0 - phat) + z * z / (4.0 * n)) / n) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        frame.to_csv(handle, index=False)


def bracket_numeric_order(bracket: str) -> float:
    value = str(bracket)
    if value.startswith("<="):
        return float(value[2:])
    if value.startswith(">"):
        return float(value[1:]) + 20.0
    if "-" in value:
        left, right = value.split("-", maxsplit=1)
        return (float(left) + float(right)) / 2.0
    return np.nan


def sample_size_display_coordinate(point_on_grid: object, bracket: object) -> float:
    point = pd.to_numeric(pd.Series([point_on_grid]), errors="coerce").iloc[0]
    if np.isfinite(point):
        return float(point)
    return bracket_numeric_order(str(bracket))


def sample_size_coordinate_summary(frame: pd.DataFrame) -> pd.DataFrame:
    plot = frame.copy()
    plot["fess_numeric"] = [
        sample_size_display_coordinate(point, bracket)
        for point, bracket in zip(plot["fess_ess_point_on_grid"], plot["fess_ess_bracket"])
    ]
    plot["ress_numeric"] = [
        sample_size_display_coordinate(point, bracket)
        for point, bracket in zip(plot["ress_ess_point_on_grid"], plot["ress_ess_bracket"])
    ]
    plot["coordinate_relation"] = np.select(
        [
            plot["fess_numeric"] > plot["ress_numeric"],
            plot["fess_numeric"] < plot["ress_numeric"],
        ],
        ["FESS larger", "R-ESS larger"],
        default="same coordinate",
    )
    group_cols = ["ress_numeric", "fess_numeric", "coordinate_relation", "coverage_target"]
    summary = (
        plot.groupby(group_cols, dropna=False)
        .agg(n_cells=("run_label", "size"))
        .reset_index()
    )
    return summary


def add_disagreement_heatmap(ax: plt.Axes, summary: pd.DataFrame) -> None:
    row_keys = []
    row_labels = []
    for dataset_name in RUN_LABELS.values():
        for coverage in sorted(summary.loc[summary["dataset_name"] == dataset_name, "coverage_target"].unique()):
            row_keys.append((dataset_name, coverage))
            row_labels.append(f"{dataset_name}\ncoverage {coverage:.1f}")
    budgets = sorted(summary["target_budget"].unique())
    matrix = np.full((len(row_keys), len(budgets)), np.nan)
    for i, (dataset_name, coverage) in enumerate(row_keys):
        for j, budget in enumerate(budgets):
            hit = summary[
                (summary["dataset_name"] == dataset_name)
                & (summary["coverage_target"] == coverage)
                & (summary["target_budget"] == budget)
            ]
            if not hit.empty:
                matrix[i, j] = float(hit["disagreement_rate"].iloc[0])
    cmap = plt.cm.YlOrRd.copy()
    cmap.set_bad("#eeeeee")
    image = ax.imshow(matrix, aspect="auto", vmin=0.0, vmax=1.0, cmap=cmap)
    ax.set_title("A  R-ESS/FESS disagreement rate")
    ax.set_xlabel("LLM labelled-data budget")
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_xticks(np.arange(len(budgets)))
    ax.set_xticklabels([str(int(b)) for b in budgets])
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if np.isfinite(matrix[i, j]):
                color = "white" if matrix[i, j] > 0.7 else "black"
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=7, color=color)
            else:
                ax.text(j, i, "NA", ha="center", va="center", fontsize=7, color="0.35")
    return image


def add_sample_size_scatter(ax: plt.Axes, frame: pd.DataFrame) -> None:
    plot = sample_size_coordinate_summary(frame)
    markers = {0.8: "o", 0.9: "s"}
    relation_colors = {
        "FESS larger": "#2a7fb8",
        "R-ESS larger": "#d95f02",
        "same coordinate": "#8c8c8c",
    }
    max_count = int(plot["n_cells"].max()) if not plot.empty else 1
    marker_sizes = 38 + 250 * np.sqrt(plot["n_cells"].to_numpy(dtype=float) / max_count)
    plot = plot.assign(marker_size=marker_sizes)
    for (relation, coverage), sub in plot.groupby(["coordinate_relation", "coverage_target"]):
        ax.scatter(
            sub["ress_numeric"],
            sub["fess_numeric"],
            s=sub["marker_size"],
            alpha=0.62 if relation != "same coordinate" else 0.50,
            marker=markers.get(float(coverage), "o"),
            color=relation_colors.get(relation, "#333333"),
            edgecolor="white",
            linewidth=0.8,
        )
    limit = max(plot["fess_numeric"].max(), plot["ress_numeric"].max()) + 8
    ax.fill_between([0, limit], [0, limit], [limit, limit], color="#dcebf7", alpha=0.22, linewidth=0)
    ax.fill_between([0, limit], [0, 0], [0, limit], color="#f6e7dc", alpha=0.18, linewidth=0)
    ax.plot([0, limit], [0, limit], color="0.35", linestyle="--", linewidth=1.0)
    ax.set_xlim(20, limit)
    ax.set_ylim(20, limit)
    ax.set_xlabel("R-ESS reference-budget coordinate")
    ax.set_ylabel("FESS reference-budget coordinate")
    ax.set_title("B  Two sample-size coordinates")
    ax.grid(color="0.9", linewidth=0.6)
    ax.text(
        0.06,
        0.92,
        "FESS larger",
        transform=ax.transAxes,
        color=relation_colors["FESS larger"],
        fontsize=9,
        fontweight="bold",
        ha="left",
        va="top",
    )
    ax.text(
        0.69,
        0.19,
        "R-ESS larger",
        transform=ax.transAxes,
        color=relation_colors["R-ESS larger"],
        fontsize=9,
        fontweight="bold",
        ha="left",
        va="bottom",
    )
    ax.text(
        0.59,
        0.61,
        "same coordinate",
        transform=ax.transAxes,
        color="0.35",
        fontsize=8,
        rotation=38,
        ha="center",
        va="center",
    )
    coverage_handles = [
        Line2D(
            [0],
            [0],
            marker=markers[cov],
            color="0.25",
            linestyle="none",
            markersize=6,
            label=f"{cov:.1f} coverage",
        )
        for cov in (0.8, 0.9)
    ]
    legend = ax.legend(
        handles=coverage_handles,
        title="Coverage",
        title_fontsize=8,
        fontsize=8,
        frameon=True,
        ncol=1,
        loc="upper right",
        bbox_to_anchor=(0.985, 0.985),
        borderaxespad=0.2,
        handletextpad=0.35,
        columnspacing=0.7,
    )
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("none")
    legend.get_frame().set_alpha(0.86)

    ax.text(
        0.985,
        0.035,
        "Bubble area = cells",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="0.35",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.6},
    )


def add_task_bars(ax: plt.Axes, summary: pd.DataFrame) -> None:
    task_rows = summary[summary["dataset_name"] != "Combined"].sort_values(
        ["disagreement_rate", "dataset_name"], ascending=[False, True]
    )
    combined = summary[summary["dataset_name"] == "Combined"]
    sub = task_rows.reset_index(drop=True)
    colors = ["#2a7fb8" for _ in sub["dataset_name"]]
    lower = sub["disagreement_rate"].to_numpy() - sub["wilson95_low"].to_numpy()
    upper = sub["wilson95_high"].to_numpy() - sub["disagreement_rate"].to_numpy()
    xerr = np.vstack([np.maximum(lower, 0.0), np.maximum(upper, 0.0)])
    y = np.arange(len(sub))
    bars = ax.barh(
        y,
        sub["disagreement_rate"],
        color=colors,
        height=0.62,
        xerr=xerr,
        capsize=4,
        ecolor="0.2",
        linewidth=0.8,
    )
    ax.set_xlim(0, 1.04)
    ax.set_yticks(y)
    ax.set_yticklabels([TASK_SHORT_LABELS.get(name, name) for name in sub["dataset_name"]])
    ax.invert_yaxis()
    ax.set_xlabel("Disagreement rate")
    ax.set_title("A  Disagreement varies by task")
    ax.grid(axis="x", color="0.9", linewidth=0.6)
    if not combined.empty:
        combined_row = combined.iloc[0]
        combined_rate = float(combined_row.disagreement_rate)
        ax.axvline(combined_rate, color="#4d4d4d", linewidth=1.1, linestyle=":", zorder=0)
        ax.text(
            combined_rate + 0.015,
            len(sub) - 0.65,
            f"overall {int(combined_row.n_disagree)}/{int(combined_row.n_cells)}",
            ha="left",
            va="bottom",
            fontsize=8,
            color="#4d4d4d",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1.0},
            clip_on=False,
        )
    for bar, (_, row) in zip(bars, sub.iterrows()):
        ax.text(
            min(1.02, float(row.wilson95_high) + 0.035),
            bar.get_y() + bar.get_height() / 2,
            f"{int(row.n_disagree)}/{int(row.n_cells)}",
            ha="left",
            va="center",
            fontsize=8,
        )


def make_figure(frame: pd.DataFrame, output_pdf: Path, output_png: Path) -> None:
    summary = disagreement_summary(frame)
    totals = task_summary(frame)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax_bars, ax_scatter) = plt.subplots(
        1,
        2,
        figsize=(11.2, 4.7),
        gridspec_kw={"width_ratios": [0.96, 1.2]},
    )
    add_task_bars(ax_bars, totals)
    add_sample_size_scatter(ax_scatter, frame)
    fig.tight_layout(w_pad=2.2)
    fig.savefig(output_pdf)
    fig.savefig(output_png, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    frame = real_budgetgrid_rows(load_comparison(Path(args.comparison_csv).resolve()))
    if frame.empty:
        raise ValueError("No real schema budget-grid rows found in comparison CSV.")
    write_csv(disagreement_summary(frame), output / "ess_disagreement_by_budget.csv")
    write_csv(task_summary(frame), output / "ess_disagreement_by_task.csv")
    make_figure(
        frame,
        output / "figures" / "F01_equivalent_sample_size_comparison.pdf",
        output / "figures" / "F01_equivalent_sample_size_comparison.png",
    )
    print(
        {
            "status": "ok",
            "output_dir": str(output),
            "n_rows": int(len(frame)),
            "n_disagreements": int(frame["ess_brackets_disagree"].sum()),
        }
    )


if __name__ == "__main__":
    main()
