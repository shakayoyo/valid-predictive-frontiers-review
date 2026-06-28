from __future__ import annotations

import argparse
from pathlib import Path
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle


BLUE = "#0072B2"
GREEN = "#009E73"
ORANGE = "#D55E00"
PURPLE = "#6A51A3"
DARK = "#2F2F2F"
MID = "#666666"
LIGHT = "#D9D9D9"
VERY_LIGHT = "#F7F7F7"
BLUE_LIGHT = "#EAF4FB"
GREEN_LIGHT = "#EAF7F1"
ORANGE_LIGHT = "#FCEFE8"
PURPLE_LIGHT = "#F0ECF7"


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.4,
            "axes.linewidth": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def wrap_text(text: str, width: int) -> str:
    return "\n".join(textwrap.wrap(text, width=width, break_long_words=False))


def add_stage(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    body: str,
    color: str,
    face: str,
    width: int = 20,
) -> None:
    ax.add_patch(Rectangle((x, y), w, h, facecolor=face, edgecolor=color, linewidth=0.9))
    ax.text(x + 0.014, y + h - 0.024, title, ha="left", va="top", fontsize=7.1, fontweight="bold", color=color)
    ax.text(
        x + 0.014,
        y + h - 0.086,
        wrap_text(body, width),
        ha="left",
        va="top",
        fontsize=6.15,
        color=DARK,
        linespacing=1.14,
    )


def add_arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], color: str = MID) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=8,
            linewidth=0.8,
            color=color,
            shrinkA=1.5,
            shrinkB=1.5,
        )
    )


def add_split_bar(ax: plt.Axes) -> None:
    x0, y0, w, h = 0.18, 0.37, 0.64, 0.060
    roles = [
        ("Develop/select", 0.24, "#B8A9D6"),
        ("Fit", 0.18, "#BFD7EA"),
        ("Calibrate", 0.26, "#A8DDB5"),
        ("Evaluate", 0.32, "#F6C6A6"),
    ]
    x = x0
    for label, frac, color in roles:
        ww = w * frac
        ax.add_patch(Rectangle((x, y0), ww, h, facecolor=color, edgecolor="white", linewidth=0.8))
        ax.text(x + ww / 2, y0 + h / 2, label, ha="center", va="center", fontsize=6.0, color=DARK)
        x += ww
    ax.text(
        x0,
        y0 + h + 0.018,
        "Label roles are separated before calibration and final reporting",
        fontsize=6.2,
        color=MID,
    )


def add_record_strip(ax: plt.Axes) -> None:
    x, y, w, h = 0.04, 0.05, 0.92, 0.18
    ax.add_patch(Rectangle((x, y), w, h, facecolor=VERY_LIGHT, edgecolor=LIGHT, linewidth=0.8))
    ax.text(x + 0.018, y + h - 0.036, "Reproducible record", ha="left", va="top", fontsize=6.9, fontweight="bold", color=DARK)
    chips = [
        "parsed outputs",
        "parse status",
        "prompt/schema hashes",
        "split ids",
        "derived tables",
        "figure data",
        "source files",
    ]
    chip_x = x + 0.28
    chip_y = y + h - 0.064
    for chip in chips:
        chip_w = 0.0105 * len(chip) + 0.027
        if chip_x + chip_w > x + w - 0.015:
            chip_x = x + 0.28
            chip_y -= 0.061
        ax.add_patch(Rectangle((chip_x, chip_y - 0.030), chip_w, 0.041, facecolor="white", edgecolor=LIGHT, linewidth=0.7))
        ax.text(chip_x + chip_w / 2, chip_y - 0.010, chip, ha="center", va="center", fontsize=5.75, color=DARK)
        chip_x += chip_w + 0.010


def make_figure(output_pdf: Path, output_png: Path) -> None:
    configure_matplotlib()
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.25, 2.95))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.text(
        0.040,
        0.93,
        "Construct systems first; measure remaining uncertainty after calibration.",
        ha="left",
        va="top",
        fontsize=7.4,
        color=MID,
    )

    stages = [
        (
            0.040,
            "Task",
            "target population and label or outcome scale",
            DARK,
            "white",
            18,
        ),
        (
            0.230,
            "Systems",
            "LLM systems; reference learners",
            BLUE,
            BLUE_LIGHT,
            18,
        ),
        (
            0.420,
            "Data roles",
            "develop/select, fit, calibrate, evaluate",
            PURPLE,
            PURPLE_LIGHT,
            17,
        ),
        (
            0.610,
            "Coverage filter",
            "include if coverage >= target - 0.03",
            GREEN,
            GREEN_LIGHT,
            17,
        ),
        (
            0.800,
            "Frontier",
            "shortest set or interval; ESS translations",
            ORANGE,
            ORANGE_LIGHT,
            18,
        ),
    ]
    y, w, h = 0.58, 0.150, 0.25
    for x, title, body, color, face, wrap in stages:
        add_stage(ax, x, y, w, h, title, body, color, face, width=wrap)
    for (x, *_), (x_next, *__) in zip(stages[:-1], stages[1:]):
        add_arrow(ax, (x + w + 0.012, y + h / 2), (x_next - 0.012, y + h / 2))

    add_split_bar(ax)

    ax.plot([0.495, 0.495], [0.58, 0.450], color=PURPLE, lw=0.8)
    ax.plot([0.685, 0.685], [0.58, 0.450], color=GREEN, lw=0.8)
    add_arrow(ax, (0.495, 0.365), (0.495, 0.245), color="0.45")
    add_arrow(ax, (0.685, 0.365), (0.685, 0.245), color="0.45")
    add_record_strip(ax)

    fig.savefig(output_pdf, bbox_inches="tight")
    fig.savefig(output_png, dpi=360, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir).resolve()
    make_figure(
        output / "figures" / "F00_experimental_workflow_schematic.pdf",
        output / "figures" / "F00_experimental_workflow_schematic.png",
    )
    print({"status": "ok", "output_dir": str(output)})


if __name__ == "__main__":
    main()
