from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
DARK = "#2F2F2F"
MID = "#6B6B6B"
LIGHT = "#D9D9D9"
VERY_LIGHT = "#F6F6F6"
PALE_BLUE = "#EAF4FB"
PALE_ORANGE = "#FCEFE8"
PALE_GREEN = "#EAF7F1"

MINIMAX_RUNS = [
    "minimax_anes_schema_budgetgrid",
    "minimax_fair_schema_budgetgrid",
    "minimax_randhie_schema_budgetgrid",
    "minimax_modechoice_schema_budgetgrid",
    "minimax_star98_schema_budgetgrid",
    "minimax_fertility_schema_budgetgrid",
    "minimax_adult_schema_budgetgrid",
    "minimax_acs_schema_budgetgrid",
]

TASK_NAMES = {
    "minimax_anes_schema_budgetgrid": "ANES96",
    "minimax_fair_schema_budgetgrid": "Fair",
    "minimax_randhie_schema_budgetgrid": "RAND HIE",
    "minimax_modechoice_schema_budgetgrid": "Modechoice",
    "minimax_star98_schema_budgetgrid": "Star98",
    "minimax_fertility_schema_budgetgrid": "Fertility",
    "minimax_adult_schema_budgetgrid": "Adult",
    "minimax_acs_schema_budgetgrid": "ACS",
}

RANKING_FILES = {
    "minimax_anes_schema_budgetgrid": "remote_results_dag/minimax_anes_schema_budgetgrid_s10/results/E10B_anes_budget_grid_risk_vs_frontier_rankings.csv",
    "minimax_fair_schema_budgetgrid": "remote_results_dag/minimax_fair_schema_budgetgrid_s10/results/E11B_fair_budget_grid_risk_vs_frontier_rankings.csv",
    "minimax_randhie_schema_budgetgrid": "remote_results_dag/minimax_randhie_schema_budgetgrid_s10/results/E12B_randhie_budget_grid_risk_vs_frontier_rankings.csv",
    "minimax_modechoice_schema_budgetgrid": "remote_results_dag/minimax_modechoice_schema_budgetgrid_s10/results/E13B_modechoice_budget_grid_risk_vs_frontier_rankings.csv",
    "minimax_star98_schema_budgetgrid": "remote_results_dag/minimax_star98_schema_budgetgrid_s10/results/E14B_star98_budget_grid_risk_vs_frontier_rankings.csv",
    "minimax_fertility_schema_budgetgrid": "remote_results_dag/minimax_fertility_schema_budgetgrid_s10/results/E16B_fertility_budget_grid_risk_vs_frontier_rankings.csv",
    "minimax_adult_schema_budgetgrid": "remote_results_dag/minimax_adult_schema_budgetgrid_s10_strongref/results/E17B_adult_budget_grid_risk_vs_frontier_rankings.csv",
    "minimax_acs_schema_budgetgrid": "remote_results_dag/minimax_acs_schema_budgetgrid_s10_strongref/results/E18B_acs_budget_grid_risk_vs_frontier_rankings.csv",
}


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.4,
            "axes.titlesize": 8.2,
            "axes.labelsize": 7.1,
            "xtick.labelsize": 6.3,
            "ytick.labelsize": 6.3,
            "legend.fontsize": 6.2,
            "axes.linewidth": 0.65,
            "xtick.major.width": 0.55,
            "ytick.major.width": 0.55,
            "xtick.major.size": 2.4,
            "ytick.major.size": 2.4,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.055,
        1.045,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10.2,
        fontweight="bold",
        color=DARK,
    )


def strip_top_right(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", pad=1.3)


def rounded_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    facecolor: str,
    edgecolor: str = LIGHT,
    fontsize: float = 6.6,
    weight: str = "normal",
    color: str = DARK,
) -> None:
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.015,rounding_size=0.018",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=0.75,
    )
    ax.add_patch(box)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=color,
        fontweight=weight,
    )


def arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str = MID,
    lw: float = 0.8,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=8,
            linewidth=lw,
            color=color,
            shrinkA=1.5,
            shrinkB=1.5,
        )
    )


def load_frontier_gaps(run_dir: Path, output_dir: Path) -> pd.DataFrame:
    source = run_dir / "artifacts/frontier_versions_v18_strongref_adult10_acs_complete/frontier_versions_seed_level.csv"
    df = pd.read_csv(source)
    sub = df[
        df["run_label"].isin(MINIMAX_RUNS)
        & df["method_family"].isin(["llm_prompt_policy_frontier", "reference"])
    ].copy()
    index_cols = ["run_label", "dataset_id", "task_id", "seed", "budget", "coverage_target"]
    pivot = sub.pivot_table(
        index=index_cols,
        columns="method_family",
        values=["frontier_set_size", "point_risk"],
        aggfunc="first",
    ).dropna()
    pivot.columns = ["_".join(col) for col in pivot.columns]
    out = pivot.reset_index()
    out["task"] = out["run_label"].map(TASK_NAMES)
    out["risk_gap_llm_minus_reference"] = (
        out["point_risk_llm_prompt_policy_frontier"] - out["point_risk_reference"]
    )
    out["set_size_gap_llm_minus_reference"] = (
        out["frontier_set_size_llm_prompt_policy_frontier"] - out["frontier_set_size_reference"]
    )
    source_dir = output_dir / "source_data"
    source_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(source_dir / "figure_1_llm_reference_frontier_gaps.csv", index=False)
    return out


def load_rank_disagreement(run_dir: Path, output_dir: Path) -> pd.DataFrame:
    rows = []
    for run_label, rel_path in RANKING_FILES.items():
        df = pd.read_csv(run_dir / rel_path)
        n_cells = int(len(df))
        n_disagree = int(df["rank_disagreement_indicator"].sum())
        rows.append(
            {
                "run_label": run_label,
                "task": TASK_NAMES[run_label],
                "n_cells": n_cells,
                "n_disagree": n_disagree,
                "rate": n_disagree / n_cells,
            }
        )
    out = pd.DataFrame(rows)
    source_dir = output_dir / "source_data"
    source_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(source_dir / "figure_1_rank_disagreement_by_task.csv", index=False)
    return out


def load_ess_disagreement(run_dir: Path, output_dir: Path) -> pd.Series:
    source = run_dir / "artifacts/ess_budgetgrid_v11_strongref_adult10_acs_complete/uncertainty/ess_disagreement_uncertainty_by_task.csv"
    df = pd.read_csv(source)
    combined = df[df["dataset_name"] == "Combined"].iloc[0].copy()
    source_dir = output_dir / "source_data"
    source_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([combined]).to_csv(source_dir / "figure_1_ess_fess_disagreement_combined.csv", index=False)
    return combined


def add_panel_a(ax: plt.Axes) -> None:
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    panel_label(ax, "a")
    ax.text(
        0.02,
        1.02,
        "One case: error and uncertainty differ",
        transform=ax.transAxes,
        fontsize=7.9,
        color=DARK,
        fontweight="bold",
    )

    axis_y = 0.30
    ax.plot([0.09, 0.91], [axis_y, axis_y], color="0.78", lw=1.1)
    ax.text(0.50, 0.19, "outcome space for a future case", fontsize=6.1, color=MID, ha="center")
    for x, label in [(0.14, "low"), (0.86, "high")]:
        ax.plot([x, x], [axis_y - 0.025, axis_y + 0.025], color="0.72", lw=0.8)
        ax.text(x, axis_y - 0.075, label, fontsize=5.6, color=MID, ha="center")

    true_x = 0.63
    pred_x = 0.46
    set_low = 0.34
    set_high = 0.78

    ax.plot([true_x, true_x], [axis_y - 0.055, axis_y + 0.22], color=DARK, lw=0.9)
    ax.text(true_x + 0.075, axis_y + 0.235, "true outcome", fontsize=5.8, color=DARK, ha="left")

    ax.scatter([pred_x], [axis_y], s=34, color=ORANGE, edgecolor="white", linewidth=0.5, zorder=3)
    ax.text(pred_x - 0.085, axis_y + 0.13, "point\nprediction", fontsize=5.8, color=ORANGE, ha="center", va="center", fontweight="bold")
    ax.annotate(
        "",
        xy=(true_x, axis_y + 0.105),
        xytext=(pred_x, axis_y + 0.105),
        arrowprops={"arrowstyle": "<->", "lw": 0.8, "color": ORANGE},
    )
    ax.text((pred_x + true_x) / 2, axis_y + 0.145, "point loss", fontsize=5.6, color=ORANGE, ha="center")

    set_y = 0.72
    ax.plot([set_low, set_high], [set_y, set_y], color=GREEN, lw=4.0, solid_capstyle="round")
    ax.plot([set_low, set_low], [set_y - 0.045, set_y + 0.045], color=GREEN, lw=1.0)
    ax.plot([set_high, set_high], [set_y - 0.045, set_y + 0.045], color=GREEN, lw=1.0)
    ax.text((set_low + set_high) / 2, set_y + 0.115, "valid set or interval", fontsize=6.2, color=GREEN, ha="center", fontweight="bold")
    ax.annotate(
        "",
        xy=(set_high, set_y - 0.135),
        xytext=(set_low, set_y - 0.135),
        arrowprops={"arrowstyle": "<->", "lw": 0.8, "color": GREEN},
    )
    ax.text(set_low + 0.005, set_y - 0.205, "set size", fontsize=5.7, color=GREEN, ha="left")

    ax.text(
        0.07,
        0.02,
        "At fixed coverage, shorter sets leave less unresolved uncertainty.",
        transform=ax.transAxes,
        fontsize=6.1,
        color=MID,
        ha="left",
        va="bottom",
    )


def add_panel_b(ax: plt.Axes, gaps: pd.DataFrame) -> None:
    panel_label(ax, "b")
    ax.text(
        0.02,
        1.03,
        "Frontier: shortest valid set after coverage",
        transform=ax.transAxes,
        fontsize=7.9,
        color=DARK,
        fontweight="bold",
    )
    adult = gaps[(gaps["run_label"] == "minimax_adult_schema_budgetgrid") & (gaps["coverage_target"] == 0.8)]
    curve = (
        adult.groupby("budget")[
            [
                "frontier_set_size_llm_prompt_policy_frontier",
                "frontier_set_size_reference",
            ]
        ]
        .mean()
        .reset_index()
        .sort_values("budget")
    )
    ax.plot(
        curve["budget"],
        curve["frontier_set_size_reference"],
        color=DARK,
        lw=1.55,
        marker="o",
        ms=2.9,
        label="Reference",
    )
    ax.plot(
        curve["budget"],
        curve["frontier_set_size_llm_prompt_policy_frontier"],
        color=BLUE,
        lw=1.55,
        marker="s",
        ms=2.9,
        label="LLM policy space",
    )
    if not curve.empty:
        budget = 120 if 120 in set(curve["budget"]) else float(curve["budget"].median())
        row = curve[curve["budget"] == budget].iloc[0]
        y0 = row["frontier_set_size_reference"]
        y1 = row["frontier_set_size_llm_prompt_policy_frontier"]
        ax.plot([budget, budget], [y0, y1], color=ORANGE, lw=1.05)
        ax.annotate(
            "valid-set gap",
            xy=(budget, (y0 + y1) / 2),
            xytext=(budget - 34, (y0 + y1) / 2 + 0.03),
        arrowprops={"arrowstyle": "-", "lw": 0.7, "color": ORANGE},
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 0.2},
        fontsize=6.0,
        color=ORANGE,
        va="center",
        )
    ax.set_xlabel("Labelled-data budget")
    ax.set_ylabel("Mean valid set size")
    ax.set_xlim(curve["budget"].min() - 8, curve["budget"].max() + 12)
    ax.set_ylim(0.92, max(1.62, curve.iloc[:, 1:].max().max() + 0.08))
    ax.grid(color="0.90", lw=0.5)
    ax.legend(loc="upper right", frameon=False)
    ax.text(
        0.02,
        0.04,
        "Adult income, 80% coverage",
        transform=ax.transAxes,
        fontsize=5.9,
        color=MID,
        ha="left",
    )
    strip_top_right(ax)


def add_panel_c(ax: plt.Axes, gaps: pd.DataFrame) -> None:
    panel_label(ax, "c")
    ax.text(
        0.02,
        1.03,
        "Low point risk can leave wider valid sets",
        transform=ax.transAxes,
        fontsize=7.9,
        color=DARK,
        fontweight="bold",
    )
    ax.axvspan(-0.28, 0, ymin=0.5, ymax=1.0, color=PALE_ORANGE, zorder=0)
    ax.axhline(0, color="0.55", lw=0.65)
    ax.axvline(0, color="0.55", lw=0.65)
    ax.scatter(
        gaps["risk_gap_llm_minus_reference"],
        gaps["set_size_gap_llm_minus_reference"],
        s=7,
        color="0.72",
        alpha=0.34,
        linewidth=0,
        zorder=1,
    )
    means = (
        gaps.groupby("task")[["risk_gap_llm_minus_reference", "set_size_gap_llm_minus_reference"]]
        .mean()
        .reset_index()
    )
    palette = {
        "ANES96": BLUE,
        "RAND HIE": GREEN,
        "Adult": "#7E57C2",
        "ACS": ORANGE,
        "Fair": "#CC79A7",
        "Modechoice": "#56B4E9",
        "Star98": "#999933",
        "Fertility": "#A6761D",
    }
    label_offsets = {
        "ANES96": (0.050, -0.085),
        "RAND HIE": (-0.185, -0.075),
        "Adult": (-0.095, 0.120),
        "ACS": (0.012, 0.035),
        "Fair": (0.010, 0.035),
        "Modechoice": (0.010, 0.035),
        "Star98": (0.014, 0.035),
        "Fertility": (-0.055, 0.035),
    }
    for _, row in means.iterrows():
        color = palette.get(row["task"], DARK)
        x = row["risk_gap_llm_minus_reference"]
        y = row["set_size_gap_llm_minus_reference"]
        ax.scatter([x], [y], s=34, color=color, edgecolor="white", linewidth=0.4, zorder=3)
        dx, dy = label_offsets.get(row["task"], (0.01, 0.03))
        ax.annotate(
            row["task"],
            xy=(x, y),
            xytext=(x + dx, y + dy),
            arrowprops={"arrowstyle": "-", "lw": 0.35, "color": "0.62"},
            fontsize=5.6,
            color=DARK,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 0.15},
            zorder=4,
        )
    ax.text(
        -0.245,
        0.88,
        "LLM lower point risk,\nwider valid set",
        fontsize=6.1,
        color=ORANGE,
        ha="left",
        va="top",
    )
    ax.text(0.98, 0.04, "Reference lower risk", transform=ax.transAxes, fontsize=5.8, color=MID, ha="right")
    ax.text(0.02, 0.04, "LLM lower risk", transform=ax.transAxes, fontsize=5.8, color=MID, ha="left")
    ax.set_xlabel("Point-risk gap, LLM minus reference")
    ax.set_ylabel("Valid-set-size gap, LLM minus reference")
    ax.set_xlim(-0.28, 0.64)
    ax.set_ylim(-0.82, 1.14)
    ax.set_xticks([-0.2, 0.0, 0.2, 0.4, 0.6])
    ax.set_yticks([-0.5, 0.0, 0.5, 1.0])
    ax.grid(color="0.90", lw=0.5)
    strip_top_right(ax)


def add_panel_d(ax: plt.Axes, disagreement: pd.DataFrame, ess_combined: pd.Series) -> None:
    panel_label(ax, "d")
    ax.text(
        0.02,
        1.03,
        "Rankings and translations often disagree",
        transform=ax.transAxes,
        fontsize=7.9,
        color=DARK,
        fontweight="bold",
    )
    overall_n = int(disagreement["n_cells"].sum())
    overall_d = int(disagreement["n_disagree"].sum())
    overall_rate = overall_d / overall_n
    ess_d = int(ess_combined["n_disagree"])
    ess_n = int(ess_combined["n_cells"])
    ess_rate = ess_d / ess_n

    def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
        phat = k / n
        denom = 1 + z**2 / n
        centre = (phat + z**2 / (2 * n)) / denom
        half = z * np.sqrt((phat * (1 - phat) + z**2 / (4 * n)) / n) / denom
        return centre - half, centre + half

    rows = [
        {
            "y": 1.0,
            "rate": overall_rate,
            "count": f"{overall_d}/{overall_n}",
            "label": "Point-risk ranking differs",
            "detail": "risk frontier vs valid-set frontier",
            "color": BLUE,
            "ci": wilson_interval(overall_d, overall_n),
        },
        {
            "y": 0.0,
            "rate": ess_rate,
            "count": f"{ess_d}/{ess_n}",
            "label": "R-ESS and FESS differ",
            "detail": "sample-size translations",
            "color": ORANGE,
            "ci": (float(ess_combined["wilson95_low"]), float(ess_combined["wilson95_high"])),
        },
    ]
    for row in rows:
        lo, hi = row["ci"]
        ax.hlines(row["y"], lo, hi, color=row["color"], lw=1.7, zorder=2)
        ax.plot([lo, lo], [row["y"] - 0.065, row["y"] + 0.065], color=row["color"], lw=1.0, zorder=2)
        ax.plot([hi, hi], [row["y"] - 0.065, row["y"] + 0.065], color=row["color"], lw=1.0, zorder=2)
        ax.errorbar(
            row["rate"],
            row["y"],
            fmt="o",
            color=row["color"],
            markeredgecolor="white",
            markeredgewidth=0.45,
            markersize=5.2,
            zorder=3,
        )
        ax.text(
            0.03,
            row["y"] + 0.17,
            row["label"],
            fontsize=6.2,
            color=DARK,
            fontweight="bold",
            ha="left",
            va="center",
        )
        ax.text(
            0.03,
            row["y"] - 0.17,
            f"{row['count']} cells",
            fontsize=5.55,
            color=MID,
            ha="left",
            va="center",
        )
        ax.text(
            min(0.97, row["rate"] + 0.035),
            row["y"],
            f"{100 * row['rate']:.0f}%",
            fontsize=6.2,
            color=DARK,
            fontweight="bold",
            ha="left",
            va="center",
        )
    ax.set_yticks([])
    ax.set_xlim(0, 1.02)
    ax.set_ylim(-0.58, 1.55)
    ax.set_xlabel("Fraction of cells with disagreement")
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0", "0.25", "0.50", "0.75", "1.00"])
    ax.grid(axis="x", color="0.90", lw=0.5)
    ax.axvline(0.5, color="0.72", lw=0.65, ls=":")
    ax.text(
        0.04,
        -0.36,
        "Eight binary MiniMax tasks;\nintervals are descriptive.",
        fontsize=5.55,
        color=MID,
        ha="left",
        va="center",
    )
    strip_top_right(ax)


def make_figure(output_pdf: Path, output_png: Path, run_dir: Path, output_dir: Path) -> None:
    configure_matplotlib()
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    gaps = load_frontier_gaps(run_dir, output_dir)
    disagreement = load_rank_disagreement(run_dir, output_dir)
    ess_combined = load_ess_disagreement(run_dir, output_dir)

    fig = plt.figure(figsize=(7.35, 4.95))
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=[0.96, 1.04],
        height_ratios=[0.88, 1.05],
        wspace=0.34,
        hspace=0.36,
    )
    add_panel_a(fig.add_subplot(grid[0, 0]))
    add_panel_b(fig.add_subplot(grid[0, 1]), gaps)
    add_panel_c(fig.add_subplot(grid[1, 0]), gaps)
    add_panel_d(fig.add_subplot(grid[1, 1]), disagreement, ess_combined)
    fig.savefig(output_pdf, bbox_inches="tight")
    fig.savefig(output_png, dpi=360, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-dir", default=".")
    args = parser.parse_args()
    output = Path(args.output_dir).resolve()
    run_dir = Path(args.run_dir).resolve()
    make_figure(
        output / "figures" / "F00_valid_frontier_conceptual_schematic.pdf",
        output / "figures" / "F00_valid_frontier_conceptual_schematic.png",
        run_dir,
        output,
    )
    print({"status": "ok", "output_dir": str(output)})


if __name__ == "__main__":
    main()
