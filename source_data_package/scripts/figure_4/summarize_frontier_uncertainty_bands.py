from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

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

T_CRIT_95 = {
    1: 12.7062047364,
    2: 4.3026527299,
    3: 3.1824463053,
    4: 2.7764451052,
    5: 2.5705818366,
    6: 2.4469118488,
    7: 2.3646242510,
    8: 2.3060041352,
    9: 2.2621571627,
    10: 2.2281388519,
}


def t_interval(values: pd.Series) -> tuple[float, float, float]:
    vals = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    n = int(len(vals))
    mean = float(vals.mean()) if n else math.nan
    if n <= 1:
        return mean, math.nan, math.nan
    sd = float(vals.std(ddof=1))
    half = T_CRIT_95.get(n - 1, 1.959963984540054) * sd / math.sqrt(n)
    return mean, mean - half, mean + half


def quantile_interval(values: np.ndarray) -> tuple[float, float]:
    clean = values[np.isfinite(values)]
    if len(clean) == 0:
        return math.nan, math.nan
    return float(np.quantile(clean, 0.025)), float(np.quantile(clean, 0.975))


def bool_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def bracket_numeric_order(bracket: object) -> float:
    value = str(bracket).strip()
    numbers = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", value)]
    if not numbers:
        return math.nan
    if value.startswith("<="):
        return numbers[0]
    if value.startswith(">"):
        # Treat above-grid brackets as one displayed grid step beyond the
        # largest available budget. This is an ordering coordinate, not a
        # continuous sample-size estimate.
        return numbers[0] + 20.0
    if "-" in value and len(numbers) >= 2:
        return (numbers[0] + numbers[1]) / 2.0
    return numbers[0]


def load_frontier_pairs(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
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
        raise ValueError(f"Missing required frontier columns: {sorted(missing)}")

    keep = (
        frame["run_label"].isin(RUN_LABELS)
        & frame["method_family"].isin({"llm_prompt_policy_frontier", "reference"})
        & (frame["frontier_status"].astype(str) == "attained")
    )
    out = frame.loc[keep].copy()
    out["dataset_name"] = out["run_label"].map(RUN_LABELS)
    id_cols = ["run_label", "dataset_name", "seed", "budget", "coverage_target"]
    value_cols = ["frontier_set_size", "coverage", "point_risk"]
    pivot = out.pivot_table(
        index=id_cols,
        columns="method_family",
        values=value_cols,
        aggfunc="first",
    )
    pivot.columns = [f"{value}_{family}" for value, family in pivot.columns]
    paired = pivot.reset_index()
    need = [
        "frontier_set_size_llm_prompt_policy_frontier",
        "frontier_set_size_reference",
        "point_risk_llm_prompt_policy_frontier",
        "point_risk_reference",
    ]
    paired = paired.dropna(subset=need).copy()
    paired["set_size_gap_llm_minus_reference"] = (
        paired["frontier_set_size_llm_prompt_policy_frontier"]
        - paired["frontier_set_size_reference"]
    )
    paired["point_risk_gap_llm_minus_reference"] = (
        paired["point_risk_llm_prompt_policy_frontier"]
        - paired["point_risk_reference"]
    )
    if {
        "coverage_llm_prompt_policy_frontier",
        "coverage_reference",
    }.issubset(paired.columns):
        paired["coverage_gap_llm_minus_reference"] = (
            paired["coverage_llm_prompt_policy_frontier"]
            - paired["coverage_reference"]
        )
    return paired


def summarize_frontier_cells(pairs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["dataset_name", "coverage_target", "budget"]
    for keys, sub in pairs.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        gap = pd.to_numeric(sub["set_size_gap_llm_minus_reference"], errors="coerce")
        risk_gap = pd.to_numeric(sub["point_risk_gap_llm_minus_reference"], errors="coerce")
        mean, low, high = t_interval(gap)
        risk_mean, risk_low, risk_high = t_interval(risk_gap)
        row.update(
            {
                "n_seed_pairs": int(len(gap.dropna())),
                "mean_set_size_gap_llm_minus_reference": mean,
                "t95_low_set_size_gap": low,
                "t95_high_set_size_gap": high,
                "median_set_size_gap_llm_minus_reference": float(gap.median()),
                "share_llm_wider": float((gap > 0).mean()),
                "mean_point_risk_gap_llm_minus_reference": risk_mean,
                "t95_low_point_risk_gap": risk_low,
                "t95_high_point_risk_gap": risk_high,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


def cluster_mean(frame: pd.DataFrame, value_col: str) -> float:
    vals = pd.to_numeric(frame[value_col], errors="coerce").dropna()
    return float(vals.mean()) if len(vals) else math.nan


def bootstrap_tasks(frame: pd.DataFrame, value_col: str, rng: np.random.Generator, n_boot: int) -> np.ndarray:
    grouped = (
        frame.assign(_value=pd.to_numeric(frame[value_col], errors="coerce"))
        .dropna(subset=["_value"])
        .groupby("dataset_name", dropna=False)["_value"]
        .agg(["sum", "count"])
        .reset_index()
        .sort_values("dataset_name")
    )
    sums = grouped["sum"].to_numpy(dtype=float)
    counts = grouped["count"].to_numpy(dtype=float)
    n_tasks = len(grouped)
    out = np.empty(n_boot, dtype=float)
    for idx in range(n_boot):
        sampled = rng.integers(0, n_tasks, size=n_tasks)
        denominator = counts[sampled].sum()
        out[idx] = sums[sampled].sum() / denominator if denominator else math.nan
    return out


def bootstrap_task_seed(frame: pd.DataFrame, value_col: str, rng: np.random.Generator, n_boot: int) -> np.ndarray:
    grouped = (
        frame.assign(_value=pd.to_numeric(frame[value_col], errors="coerce"))
        .dropna(subset=["_value"])
        .groupby(["dataset_name", "seed"], dropna=False)["_value"]
        .agg(["sum", "count"])
        .reset_index()
        .sort_values(["dataset_name", "seed"])
    )
    task_arrays: list[tuple[np.ndarray, np.ndarray]] = []
    for _, sub in grouped.groupby("dataset_name", dropna=False):
        task_arrays.append((sub["sum"].to_numpy(dtype=float), sub["count"].to_numpy(dtype=float)))
    n_tasks = len(task_arrays)
    out = np.empty(n_boot, dtype=float)
    for idx in range(n_boot):
        numerator = 0.0
        denominator = 0.0
        for task_idx in rng.integers(0, n_tasks, size=n_tasks):
            sums, counts = task_arrays[int(task_idx)]
            sampled_seed_idx = rng.integers(0, len(sums), size=len(sums))
            numerator += sums[sampled_seed_idx].sum()
            denominator += counts[sampled_seed_idx].sum()
        out[idx] = numerator / denominator if denominator else math.nan
    return out


def bootstrap_seeds(frame: pd.DataFrame, value_col: str, rng: np.random.Generator, n_boot: int) -> np.ndarray:
    grouped = (
        frame.assign(_value=pd.to_numeric(frame[value_col], errors="coerce"))
        .dropna(subset=["_value"])
        .groupby("seed", dropna=False)["_value"]
        .agg(["sum", "count"])
        .reset_index()
        .sort_values("seed")
    )
    sums = grouped["sum"].to_numpy(dtype=float)
    counts = grouped["count"].to_numpy(dtype=float)
    n_seeds = len(grouped)
    out = np.empty(n_boot, dtype=float)
    for idx in range(n_boot):
        sampled = rng.integers(0, n_seeds, size=n_seeds)
        denominator = counts[sampled].sum()
        out[idx] = sums[sampled].sum() / denominator if denominator else math.nan
    return out


def bootstrap_summary(
    frame: pd.DataFrame,
    *,
    value_col: str,
    statistic: str,
    rng: np.random.Generator,
    n_boot: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    def add_row(scope: str, sub: pd.DataFrame, mode: str) -> None:
        observed = cluster_mean(sub, value_col)
        if mode == "task":
            task_low, task_high = quantile_interval(bootstrap_tasks(sub, value_col, rng, n_boot))
            task_seed_low, task_seed_high = quantile_interval(
                bootstrap_task_seed(sub, value_col, rng, n_boot)
            )
            seed_low, seed_high = math.nan, math.nan
        else:
            task_low, task_high = math.nan, math.nan
            task_seed_low, task_seed_high = math.nan, math.nan
            seed_low, seed_high = quantile_interval(bootstrap_seeds(sub, value_col, rng, n_boot))
        rows.append(
            {
                "statistic": statistic,
                "scope": scope,
                "n_tasks": int(sub["dataset_name"].nunique()),
                "n_seed_clusters": int(sub[["dataset_name", "seed"]].drop_duplicates().shape[0]),
                "n_cells": int(len(sub)),
                "observed_mean": observed,
                "task_bootstrap95_low": task_low,
                "task_bootstrap95_high": task_high,
                "task_seed_bootstrap95_low": task_seed_low,
                "task_seed_bootstrap95_high": task_seed_high,
                "seed_bootstrap95_low": seed_low,
                "seed_bootstrap95_high": seed_high,
            }
        )

    add_row("combined", frame, "task")
    for coverage, sub in frame.groupby("coverage_target", dropna=False):
        add_row(f"coverage_{float(coverage):.1f}", sub, "task")
    for dataset_name, sub in frame.groupby("dataset_name", dropna=False):
        add_row(str(dataset_name), sub, "seed")
    for (dataset_name, coverage), sub in frame.groupby(["dataset_name", "coverage_target"], dropna=False):
        add_row(f"{dataset_name}|coverage_{float(coverage):.1f}", sub, "seed")
    return pd.DataFrame(rows)


def load_ess_rows(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        "run_label",
        "seed",
        "coverage_target",
        "target_budget",
        "fess_ess_bracket",
        "ress_ess_bracket",
        "fess_ess_status",
        "ress_ess_status",
        "ess_brackets_disagree",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required ESS columns: {sorted(missing)}")
    out = frame.loc[frame["run_label"].isin(RUN_LABELS)].copy()
    out["dataset_name"] = out["run_label"].map(RUN_LABELS)
    out["fess_numeric_order"] = out["fess_ess_bracket"].map(bracket_numeric_order)
    out["ress_numeric_order"] = out["ress_ess_bracket"].map(bracket_numeric_order)
    out["fess_minus_ress_numeric_order"] = out["fess_numeric_order"] - out["ress_numeric_order"]
    out["ess_brackets_disagree"] = bool_series(out["ess_brackets_disagree"])
    out["fess_above_grid"] = out["fess_ess_status"].astype(str).eq("above_grid")
    out["ress_above_grid"] = out["ress_ess_status"].astype(str).eq("above_grid")
    return out


def summarize_ess_cells(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["dataset_name", "coverage_target", "target_budget"]
    for keys, sub in frame.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        gap = sub["fess_minus_ress_numeric_order"]
        mean, low, high = t_interval(gap)
        row.update(
            {
                "n_seed_cells": int(len(sub)),
                "mean_fess_minus_ress_order": mean,
                "t95_low_fess_minus_ress_order": low,
                "t95_high_fess_minus_ress_order": high,
                "median_fess_minus_ress_order": float(pd.to_numeric(gap, errors="coerce").median()),
                "share_brackets_disagree": float(sub["ess_brackets_disagree"].mean()),
                "share_fess_above_grid": float(sub["fess_above_grid"].mean()),
                "share_ress_above_grid": float(sub["ress_above_grid"].mean()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


def write_markdown(
    frontier_bootstrap: pd.DataFrame,
    ess_bootstrap: pd.DataFrame,
    output_path: Path,
) -> None:
    def interval(low: object, high: object) -> str:
        return f"{float(low):.3f} to {float(high):.3f}"

    combined_gap = frontier_bootstrap.loc[frontier_bootstrap["scope"] == "combined"].iloc[0]
    combined_ess = ess_bootstrap.loc[ess_bootstrap["scope"] == "combined"].iloc[0]
    lines = [
        "# Frontier and FESS Uncertainty Bands",
        "",
        "Purpose: summarize uncertainty in continuous frontier gaps and in finite-grid FESS/R-ESS bracket locations.",
        "Positive frontier gaps mean the LLM prompt-policy frontier has larger conformal prediction sets than the reference frontier.",
        "Positive FESS--R-ESS order means the frontier-equivalent bracket is larger than the risk-equivalent bracket on the displayed finite budget grid.",
        "",
        "## Combined Summaries",
        "",
        "| Quantity | Mean | Task bootstrap 95% | Task-seed bootstrap 95% |",
        "|---|---:|---:|---:|",
        "| LLM minus reference set-size gap | {mean:.3f} | {tint} | {tsint} |".format(
            mean=float(combined_gap.observed_mean),
            tint=interval(combined_gap.task_bootstrap95_low, combined_gap.task_bootstrap95_high),
            tsint=interval(combined_gap.task_seed_bootstrap95_low, combined_gap.task_seed_bootstrap95_high),
        ),
        "| FESS minus R-ESS bracket order | {mean:.3f} | {tint} | {tsint} |".format(
            mean=float(combined_ess.observed_mean),
            tint=interval(combined_ess.task_bootstrap95_low, combined_ess.task_bootstrap95_high),
            tsint=interval(combined_ess.task_seed_bootstrap95_low, combined_ess.task_seed_bootstrap95_high),
        ),
        "",
        "## Coverage-Level Frontier Gaps",
        "",
        "| Scope | Mean LLM-reference set-size gap | Task bootstrap 95% | Task-seed bootstrap 95% |",
        "|---|---:|---:|---:|",
    ]
    for row in frontier_bootstrap.loc[frontier_bootstrap["scope"].str.startswith("coverage_")].itertuples(index=False):
        lines.append(
            "| {scope} | {mean:.3f} | {tint} | {tsint} |".format(
                scope=row.scope,
                mean=float(row.observed_mean),
                tint=interval(row.task_bootstrap95_low, row.task_bootstrap95_high),
                tsint=interval(row.task_seed_bootstrap95_low, row.task_seed_bootstrap95_high),
            )
        )
    lines.extend(
        [
            "",
            "Interpretation: cell-level t intervals by task, coverage, and budget are written separately for display bands.",
            "Combined task/task-seed bootstrap intervals are intentionally wider and should carry inferential language.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frontier-csv", required=True)
    parser.add_argument("--ess-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260611)
    args = parser.parse_args()

    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    frontier_pairs = load_frontier_pairs(Path(args.frontier_csv).resolve())
    ess_rows = load_ess_rows(Path(args.ess_csv).resolve())

    frontier_cells = summarize_frontier_cells(frontier_pairs)
    ess_cells = summarize_ess_cells(ess_rows)
    frontier_bootstrap = bootstrap_summary(
        frontier_pairs,
        value_col="set_size_gap_llm_minus_reference",
        statistic="frontier_set_size_gap_llm_minus_reference",
        rng=rng,
        n_boot=args.n_bootstrap,
    )
    ess_bootstrap = bootstrap_summary(
        ess_rows,
        value_col="fess_minus_ress_numeric_order",
        statistic="fess_minus_ress_numeric_order",
        rng=rng,
        n_boot=args.n_bootstrap,
    )

    frontier_pairs.to_csv(output / "paired_frontier_gaps_seed_level.csv", index=False)
    frontier_cells.to_csv(output / "paired_frontier_gap_bands_by_cell.csv", index=False)
    frontier_bootstrap.to_csv(output / "paired_frontier_gap_bootstrap_summary.csv", index=False)
    ess_cells.to_csv(output / "ess_bracket_order_bands_by_cell.csv", index=False)
    ess_bootstrap.to_csv(output / "ess_bracket_order_bootstrap_summary.csv", index=False)
    write_markdown(frontier_bootstrap, ess_bootstrap, output / "frontier_fess_uncertainty_bands_summary.md")

    print(
        {
            "status": "ok",
            "output_dir": str(output),
            "n_frontier_pairs": int(len(frontier_pairs)),
            "n_ess_rows": int(len(ess_rows)),
            "n_bootstrap": int(args.n_bootstrap),
        }
    )


if __name__ == "__main__":
    main()
