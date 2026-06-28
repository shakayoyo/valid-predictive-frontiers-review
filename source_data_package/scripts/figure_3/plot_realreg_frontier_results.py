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
from matplotlib.lines import Line2D


TASK_LABELS = {
    "fair_affairs_rate": "Fair affairs\ncount",
    "randhie_physician_visit_count": "RAND HIE\nvisits",
    "fertility_2011_total_fertility_rate": "Fertility\nrate",
}

TASK_UNITS = {
    "fair_affairs_rate": "affairs",
    "randhie_physician_visit_count": "visits",
    "fertility_2011_total_fertility_rate": "births per woman",
}

FAMILY_LABELS = {
    "reference": "Reference",
    "llm_cached": "MiniMax fixed",
    "llm_prompt_optimized": "MiniMax optimized",
}

FAMILY_COLORS = {
    "reference": "#4D4D4D",
    "llm_cached": "#0072B2",
    "llm_prompt_optimized": "#D55E00",
}

FAMILY_MARKERS = {
    "reference": "o",
    "llm_cached": "s",
    "llm_prompt_optimized": "^",
}

COVERAGE_MARKERS = {
    0.8: "o",
    0.9: "D",
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


def load_best(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(filesystem_path(path))
    required = {
        "task_id",
        "method_family",
        "coverage_target",
        "coverage",
        "mean_set_size",
        "point_risk",
        "method_id",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns in {path}: {sorted(missing)}")
    out = frame.loc[frame["task_id"].isin(TASK_LABELS)].copy()
    out["coverage_target"] = out["coverage_target"].astype(float)
    out["mean_set_size"] = out["mean_set_size"].astype(float)
    out["point_risk"] = out["point_risk"].astype(float)
    return out


def add_reference_ratios(frame: pd.DataFrame) -> pd.DataFrame:
    keys = ["task_id", "coverage_target"]
    ref = frame.loc[frame["method_family"] == "reference", keys + ["mean_set_size", "point_risk"]].rename(
        columns={
            "mean_set_size": "reference_interval_length",
            "point_risk": "reference_rmse",
        }
    )
    out = frame.merge(ref, on=keys, how="left")
    out["interval_ratio_to_reference"] = out["mean_set_size"] / out["reference_interval_length"]
    out["rmse_ratio_to_reference"] = out["point_risk"] / out["reference_rmse"]
    out["log2_interval_ratio"] = np.log2(out["interval_ratio_to_reference"])
    out["log2_rmse_ratio"] = np.log2(out["rmse_ratio_to_reference"])
    return out


def write_source_tables(frame: pd.DataFrame, out_dir: Path) -> None:
    table_dir = out_dir / "tables"
    ensure_dir(table_dir)
    frame.to_csv(filesystem_path(table_dir / "realreg_valid_best_with_ratios.csv"), index=False)
    display = frame.loc[
        frame["method_family"].isin(["reference", "llm_cached", "llm_prompt_optimized"]),
        [
            "task_id",
            "method_family",
            "coverage_target",
            "coverage",
            "mean_set_size",
            "point_risk",
            "interval_ratio_to_reference",
            "rmse_ratio_to_reference",
            "method_id",
        ],
    ].copy()
    display["task_label"] = display["task_id"].map(lambda x: TASK_LABELS[x].replace("\n", " "))
    display["method_family_label"] = display["method_family"].map(FAMILY_LABELS)
    display.to_csv(filesystem_path(table_dir / "realreg_figure_source.csv"), index=False)


def _ordered_tasks(frame: pd.DataFrame) -> list[str]:
    present = set(frame["task_id"].dropna().astype(str))
    return [task for task in TASK_LABELS if task in present]


def _ordered_coverages(frame: pd.DataFrame) -> list[float]:
    return sorted(float(x) for x in frame["coverage_target"].dropna().unique())


def plot_raw_intervals(ax: plt.Axes, frame: pd.DataFrame, task_id: str) -> None:
    sub = frame.loc[frame["task_id"] == task_id]
    coverages = _ordered_coverages(sub)
    x = np.arange(len(coverages))
    for family in ["reference", "llm_cached", "llm_prompt_optimized"]:
        curve = sub.loc[sub["method_family"] == family].sort_values("coverage_target")
        if curve.empty:
            continue
        ax.plot(
            x,
            curve["mean_set_size"],
            color=FAMILY_COLORS[family],
            marker=FAMILY_MARKERS[family],
            markersize=4.5,
            linewidth=1.6,
            label=FAMILY_LABELS[family],
        )
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(c * 100)}%" for c in coverages])
    ax.set_title(TASK_LABELS[task_id].replace("\n", " "), fontsize=8.6, pad=4)
    ax.set_xlabel("Target coverage", fontsize=8)
    ax.set_ylabel(f"Interval length\n({TASK_UNITS[task_id]})", fontsize=8)
    ax.tick_params(axis="both", labelsize=7.5, length=2.5)
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ymax = float(sub["mean_set_size"].max())
    ax.set_ylim(bottom=0, top=ymax * 1.18 if ymax > 0 else 1)


def plot_interval_ratios(ax: plt.Axes, frame: pd.DataFrame) -> None:
    sub = frame.loc[frame["method_family"].isin(["llm_cached", "llm_prompt_optimized"])].copy()
    tasks = _ordered_tasks(sub)
    coverages = _ordered_coverages(sub)
    base_y = np.arange(len(tasks))[::-1]
    offsets = {
        ("llm_cached", coverages[0]): 0.18,
        ("llm_prompt_optimized", coverages[0]): 0.06,
        ("llm_cached", coverages[-1]): -0.06,
        ("llm_prompt_optimized", coverages[-1]): -0.18,
    }
    ax.axvspan(-1.15, 0, color="#EAF4FB", alpha=0.70, zorder=0)
    ax.axvspan(0, 2.65, color="#F5F5F5", zorder=0)
    ax.axvline(0, color="#555555", linewidth=0.9, zorder=1)
    ax.text(
        0.03,
        0.96,
        "MiniMax shorter",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.2,
        color=FAMILY_COLORS["llm_cached"],
    )
    ax.text(
        0.97,
        0.96,
        "Reference shorter",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7.2,
        color="#666666",
    )
    for family in ["llm_cached", "llm_prompt_optimized"]:
        for coverage in coverages:
            pts = sub.loc[
                (sub["method_family"] == family) & np.isclose(sub["coverage_target"], coverage)
            ].set_index("task_id")
            x = [float(pts.loc[task, "log2_interval_ratio"]) if task in pts.index else np.nan for task in tasks]
            y = [base_y[i] + offsets[(family, coverage)] for i, task in enumerate(tasks)]
            ax.scatter(
                x,
                y,
                s=46,
                marker=COVERAGE_MARKERS.get(float(coverage), "o"),
                color=FAMILY_COLORS[family],
                alpha=0.92,
                edgecolor="white",
                linewidth=0.5,
                label=f"{FAMILY_LABELS[family]}, {int(coverage * 100)}%",
                zorder=3,
            )
    for task_index, task_id in enumerate(tasks):
        task_rows = sub.loc[sub["task_id"] == task_id]
        lo = float(task_rows["log2_interval_ratio"].min())
        hi = float(task_rows["log2_interval_ratio"].max())
        y0 = base_y[task_index]
        ax.plot([lo, hi], [y0, y0], color="#BBBBBB", linewidth=1.0, zorder=2)

    ax.set_title("Interval length relative to reference", loc="left", fontsize=9.2, fontweight="bold")
    ax.set_xlabel("MiniMax/reference interval length", fontsize=8.2)
    ax.set_yticks(base_y)
    ax.set_yticklabels([TASK_LABELS[t] for t in tasks], fontsize=8)
    ax.tick_params(axis="both", labelsize=7.5, length=2.5)
    ax.grid(axis="x", color="#E6E6E6", linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    xmin = min(-0.65, float(np.nanmin(sub["log2_interval_ratio"])) - 0.30)
    xmax = max(0.65, float(np.nanmax(sub["log2_interval_ratio"])) + 0.30)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(base_y.min() - 0.55, base_y.max() + 0.55)
    xticks = [-1, 0, 1, 2]
    ax.set_xticks([tick for tick in xticks if xmin <= tick <= xmax])
    ax.set_xticklabels([f"{2**tick:g}x" for tick in ax.get_xticks()])


def plot_risk_interval_scatter(ax: plt.Axes, frame: pd.DataFrame) -> None:
    sub = frame.loc[frame["method_family"].isin(["llm_cached", "llm_prompt_optimized"])].copy()
    task_edge = {
        "fair_affairs_rate": "#444444",
        "randhie_physician_visit_count": "#777777",
        "fertility_2011_total_fertility_rate": "#BBBBBB",
    }
    for _, row in sub.iterrows():
        family = str(row["method_family"])
        coverage = float(row["coverage_target"])
        ax.scatter(
            row["log2_rmse_ratio"],
            row["log2_interval_ratio"],
            s=44,
            marker=COVERAGE_MARKERS.get(coverage, "o"),
            color=FAMILY_COLORS[family],
            edgecolor=task_edge[str(row["task_id"])],
            linewidth=0.8,
            alpha=0.92,
            zorder=3,
        )
    label_offsets = {
        "fair_affairs_rate": (-0.08, -0.34),
        "randhie_physician_visit_count": (0.05, 0.18),
        "fertility_2011_total_fertility_rate": (-0.36, -0.10),
    }
    for task_id, label in {
        "fair_affairs_rate": "Fair",
        "randhie_physician_visit_count": "RAND HIE",
        "fertility_2011_total_fertility_rate": "Fertility",
    }.items():
        task_rows = sub.loc[sub["task_id"] == task_id]
        dx, dy = label_offsets[task_id]
        ax.text(
            float(task_rows["log2_rmse_ratio"].mean()) + dx,
            float(task_rows["log2_interval_ratio"].mean()) + dy,
            label,
            fontsize=6.8,
            color="#555555",
        )
    ax.axhline(0, color="#777777", linewidth=0.8)
    ax.axvline(0, color="#777777", linewidth=0.8)
    ax.set_title("Point error versus interval length", loc="left", fontsize=8.9, fontweight="bold")
    ax.set_xlabel("MiniMax/reference RMSE", fontsize=8.2)
    ax.set_ylabel("MiniMax/reference interval length", fontsize=8.2)
    ax.tick_params(axis="both", labelsize=7.5, length=2.5)
    ax.grid(color="#E6E6E6", linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    xmin = min(-0.5, float(np.nanmin(sub["log2_rmse_ratio"])) - 0.35)
    xmax = max(0.5, float(np.nanmax(sub["log2_rmse_ratio"])) + 0.35)
    ymin = min(-0.5, float(np.nanmin(sub["log2_interval_ratio"])) - 0.35)
    ymax = max(0.5, float(np.nanmax(sub["log2_interval_ratio"])) + 0.35)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    x_ticks = [-1, 0, 1, 2]
    y_ticks = [-1, 0, 1, 2]
    ax.set_xticks([tick for tick in x_ticks if xmin <= tick <= xmax])
    ax.set_yticks([tick for tick in y_ticks if ymin <= tick <= ymax])
    ax.set_xticklabels([f"{2**tick:g}x" for tick in ax.get_xticks()])
    ax.set_yticklabels([f"{2**tick:g}x" for tick in ax.get_yticks()])
    ax.text(
        0.03,
        0.78,
        "lower RMSE,\nwider interval",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.6,
        color="#555555",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1.5},
    )


def build_figure(frame: pd.DataFrame, out_dir: Path) -> None:
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 8.6,
            "axes.labelsize": 8,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
        }
    )
    fig, (ratio_ax, scatter_ax) = plt.subplots(1, 2, figsize=(7.2, 3.18), constrained_layout=False)
    plot_interval_ratios(ratio_ax, frame)
    ratio_ax.text(-0.12, 1.07, "A", transform=ratio_ax.transAxes, fontsize=10, fontweight="bold")

    plot_risk_interval_scatter(scatter_ax, frame)
    scatter_ax.text(-0.14, 1.07, "B", transform=scatter_ax.transAxes, fontsize=10, fontweight="bold")

    family_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=FAMILY_COLORS["llm_cached"],
            markeredgecolor="white",
            markersize=5.6,
            label=FAMILY_LABELS["llm_cached"],
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=FAMILY_COLORS["llm_prompt_optimized"],
            markeredgecolor="white",
            markersize=5.6,
            label=FAMILY_LABELS["llm_prompt_optimized"],
        ),
    ]
    coverage_handles = [
        Line2D(
            [0],
            [0],
            marker=COVERAGE_MARKERS[c],
            color="#666666",
            linestyle="None",
            markerfacecolor="#666666",
            markeredgecolor="#666666",
            markersize=5.1,
            label=f"{int(c * 100)}% coverage",
        )
        for c in _ordered_coverages(frame)
    ]
    fig.legend(
        family_handles + coverage_handles,
        [h.get_label() for h in family_handles + coverage_handles],
        loc="upper center",
        ncol=4,
        frameon=False,
        fontsize=7.6,
        bbox_to_anchor=(0.52, 0.995),
        handletextpad=0.35,
        columnspacing=1.2,
    )

    fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.88], w_pad=2.0)

    figure_dir = out_dir / "figures"
    ensure_dir(figure_dir)
    for suffix in ["pdf", "png"]:
        path = figure_dir / f"F03_realreg_interval_frontiers.{suffix}"
        fig.savefig(filesystem_path(path), dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot real-valued/count valid interval frontiers for MiniMax regression extensions."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("."),
        help="Research Evo run directory.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Path to E17_E19_realreg_valid_best_by_family.csv.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output artifact directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    input_path = args.input
    if input_path is None:
        input_path = (
            run_dir
            / "artifacts"
            / "realreg_minimax_full_20260613"
            / "results"
            / "E17_E19_realreg_valid_best_by_family.csv"
        )
    if not input_path.is_absolute():
        input_path = run_dir / input_path
    out_dir = args.out_dir
    if out_dir is None:
        out_dir = run_dir / "artifacts" / "realreg_frontier_results_v1"
    if not out_dir.is_absolute():
        out_dir = run_dir / out_dir

    best = add_reference_ratios(load_best(input_path))
    write_source_tables(best, out_dir)
    build_figure(best, out_dir)
    print(f"Wrote real-regression frontier figure and tables to {out_dir}")


if __name__ == "__main__":
    main()
