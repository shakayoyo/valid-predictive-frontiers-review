from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


FAMILY_ORDER = ["seed_selection", "component_grid", "oprolite_minimax"]
FAMILY_LABELS = {
    "fixed_only": "Fixed prompts",
    "seed_selection": "Seed-prompt\nselection",
    "component_grid": "Component\nsearch",
    "oprolite_minimax": "LLM proposal\nsearch",
}
FAMILY_COLORS = {
    "fixed_only": "#333333",
    "seed_selection": "#737373",
    "component_grid": "#0072B2",
    "oprolite_minimax": "#D55E00",
}
FAMILY_MARKERS = {
    "fixed_only": "D",
    "seed_selection": "o",
    "component_grid": "s",
    "oprolite_minimax": "^",
}

def jitter(n: int, width: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(-width, width, size=n)


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.10,
        1.04,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=11,
        fontweight="bold",
    )


def add_gap_panel(ax: plt.Axes, pairwise: pd.DataFrame) -> None:
    for i, family in enumerate(FAMILY_ORDER):
        sub = pairwise[pairwise["prompt_search_family"].eq(family)].copy()
        if sub.empty:
            continue
        y = sub["set_size_gap_vs_baseline"].astype(float).to_numpy()
        x = np.full(len(y), i, dtype=float) + jitter(len(y), 0.065, seed=2026 + i)
        ax.scatter(
            x,
            y,
            s=34,
            color=FAMILY_COLORS[family],
            alpha=0.72,
            edgecolor="white",
            linewidth=0.45,
            zorder=2,
        )
        mean = float(np.mean(y))
        low, high = np.percentile(y, [25, 75])
        ax.plot([i - 0.18, i + 0.18], [mean, mean], color="black", linewidth=1.5, zorder=3)
        ax.vlines(i, low, high, color="black", linewidth=1.0, zorder=3)
        ax.text(i, mean - 0.025 if mean <= 0 else mean + 0.025, f"{mean:+.3f}", ha="center", va="top" if mean <= 0 else "bottom", fontsize=8)

    ax.axhline(0, color="0.25", linewidth=0.9)
    ax.set_xticks(np.arange(len(FAMILY_ORDER)))
    ax.set_xticklabels([FAMILY_LABELS[f] for f in FAMILY_ORDER])
    ax.set_ylabel("Set-size gap vs fixed prompts", fontsize=9)
    ax.set_title("Valid-set gains", loc="left", fontsize=9.5)
    ax.grid(axis="y", color="0.90", linewidth=0.6)
    ax.set_ylim(-0.42, 0.08)
    ax.text(
        0.02,
        0.05,
        "lower is better",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        color="0.35",
    )
    ax.tick_params(axis="both", labelsize=8.5)
    add_panel_label(ax, "A")


def add_risk_frontier_panel(ax: plt.Axes, frontiers: pd.DataFrame) -> None:
    plot = frontiers[frontiers["frontier_status"].eq("attained")].copy()
    plot["point_risk"] = pd.to_numeric(plot["point_risk"], errors="coerce")
    plot["frontier_set_size"] = pd.to_numeric(plot["frontier_set_size"], errors="coerce")
    plot = plot.dropna(subset=["point_risk", "frontier_set_size"])

    direct_order = ["fixed_only", "seed_selection", "component_grid", "oprolite_minimax"]
    for i, family in enumerate(direct_order):
        sub = plot[plot["prompt_search_family"].eq(family)].copy()
        if sub.empty:
            continue
        x = sub["point_risk"].to_numpy(dtype=float) + jitter(len(sub), 0.00045, seed=4100 + i)
        y = sub["frontier_set_size"].to_numpy(dtype=float) + jitter(len(sub), 0.006, seed=5100 + i)
        ax.scatter(
            x,
            y,
            s=42 if family != "fixed_only" else 34,
            marker=FAMILY_MARKERS[family],
            color=FAMILY_COLORS[family],
            alpha=0.78 if family != "fixed_only" else 0.62,
            edgecolor="white",
            linewidth=0.45,
            label=FAMILY_LABELS[family].replace("\n", " "),
            zorder=2,
        )

    ax.text(
        0.03,
        0.04,
        "lower risk, shorter sets",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        color="0.35",
    )
    ax.set_xlim(0.055, 0.090)
    ax.set_ylim(0.82, 1.35)
    ax.set_xlabel("Point risk", fontsize=9)
    ax.set_ylabel("Frontier set size", fontsize=9)
    ax.set_title("Frontiers in the risk--set-size plane", loc="left", fontsize=9.5)
    ax.grid(color="0.90", linewidth=0.6)
    ax.tick_params(axis="both", labelsize=8.5)
    add_panel_label(ax, "B")


def make_figure(pairwise: pd.DataFrame, frontiers: pd.DataFrame, output_pdf: Path, output_png: Path) -> None:
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.25), gridspec_kw={"width_ratios": [0.95, 1.05]})
    ax_gap, ax_budget = axes
    add_gap_panel(ax_gap, pairwise)
    add_risk_frontier_panel(ax_budget, frontiers)

    handles = [
        Line2D(
            [0],
            [0],
            marker=FAMILY_MARKERS[f],
            linestyle="none",
            markerfacecolor=FAMILY_COLORS[f],
            markeredgecolor="white",
            markersize=6,
            label=FAMILY_LABELS[f].replace("\n", " "),
        )
        for f in ["fixed_only"] + FAMILY_ORDER
    ]
    fig.legend(handles=handles, frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.015), fontsize=8.0)
    fig.tight_layout(rect=[0, 0.10, 1, 1], w_pad=1.6)
    fig.savefig(output_pdf, bbox_inches="tight")
    fig.savefig(output_png, dpi=260, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot E20 prompt-search equivalent-budget diagnostics.")
    parser.add_argument("--equivalent-csv", required=True)
    parser.add_argument("--pairwise-gaps-csv", required=True)
    parser.add_argument("--frontiers-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output = Path(args.output_dir).resolve()
    pairwise = pd.read_csv(Path(args.pairwise_gaps_csv))
    frontiers = pd.read_csv(Path(args.frontiers_csv))
    make_figure(
        pairwise,
        frontiers,
        output / "figures" / "E20_prompt_search_gap_scatter.pdf",
        output / "figures" / "E20_prompt_search_gap_scatter.png",
    )
    print({"status": "ok", "output_dir": str(output), "n_rows": int(len(pairwise))})


if __name__ == "__main__":
    main()
