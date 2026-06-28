from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.compute_equivalent_sample_size import equivalent_budget
from scripts.compute_equivalent_sample_size import write_csv
from src.utils.logging import write_json


KEYS = ["experiment_id", "dataset_id", "task_id", "seed", "coverage_target"]


def bracket_numeric_order(bracket: object) -> float:
    value = str(bracket)
    if value.startswith("<="):
        return float(value[2:])
    if value.startswith(">"):
        return float(value[1:]) + 20.0
    if "-" in value:
        left, right = value.split("-", maxsplit=1)
        return (float(left) + float(right)) / 2.0
    return np.nan


def compute_frontier_equivalent_budget(
    frontiers: pd.DataFrame,
    target_families: list[str],
    reference_family: str,
) -> pd.DataFrame:
    attained = frontiers[frontiers["frontier_status"].eq("attained")].copy()
    ref = attained[attained["prompt_search_family"].eq(reference_family)].copy()
    targets = attained[attained["prompt_search_family"].isin(target_families)].copy()
    rows: list[dict[str, object]] = []
    for _, target in targets.iterrows():
        key = {name: target[name] for name in KEYS}
        ref_curve = ref.copy()
        for name, value in key.items():
            ref_curve = ref_curve[ref_curve[name].eq(value)]
        eq = equivalent_budget(ref_curve, float(target["frontier_set_size"]), "frontier_set_size")
        rows.append(
            {
                **key,
                "target_prompt_search_family": target["prompt_search_family"],
                "reference_prompt_search_family": reference_family,
                "target_budget": int(target["budget"]),
                "target_metric": "frontier_set_size",
                "target_value": float(target["frontier_set_size"]),
                "target_frontier_method": target.get("frontier_method", ""),
                "target_coverage": float(target.get("coverage", np.nan)),
                "target_point_risk": float(target.get("point_risk", np.nan)),
                **{f"fess_{k}": v for k, v in eq.items()},
            }
        )
    return pd.DataFrame(rows)


def build_risk_frontiers(all_policies: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["experiment_id", "dataset_id", "task_id", "seed", "budget", "coverage_target", "prompt_search_family"]
    idx = all_policies.groupby(group_cols, dropna=False)["point_risk"].idxmin()
    out = all_policies.loc[idx].copy()
    out["risk_frontier_method"] = out["method_id"]
    out["risk_frontier_value"] = out["point_risk"].astype(float)
    return out[group_cols + ["risk_frontier_method", "risk_frontier_value"]].reset_index(drop=True)


def compute_risk_equivalent_budget(
    risk_frontiers: pd.DataFrame,
    target_families: list[str],
    reference_family: str,
) -> pd.DataFrame:
    ref = risk_frontiers[risk_frontiers["prompt_search_family"].eq(reference_family)].copy()
    targets = risk_frontiers[risk_frontiers["prompt_search_family"].isin(target_families)].copy()
    rows: list[dict[str, object]] = []
    for _, target in targets.iterrows():
        key = {name: target[name] for name in KEYS}
        ref_curve = ref.copy()
        for name, value in key.items():
            ref_curve = ref_curve[ref_curve[name].eq(value)]
        eq = equivalent_budget(ref_curve, float(target["risk_frontier_value"]), "risk_frontier_value")
        rows.append(
            {
                **key,
                "target_prompt_search_family": target["prompt_search_family"],
                "reference_prompt_search_family": reference_family,
                "target_budget": int(target["budget"]),
                "target_metric": "point_risk",
                "target_value": float(target["risk_frontier_value"]),
                "target_risk_method": target["risk_frontier_method"],
                **{f"ress_{k}": v for k, v in eq.items()},
            }
        )
    return pd.DataFrame(rows)


def comparison_table(fess: pd.DataFrame, ress: pd.DataFrame) -> pd.DataFrame:
    keys = KEYS + ["target_prompt_search_family", "reference_prompt_search_family", "target_budget"]
    left = fess[
        keys
        + [
            "target_value",
            "target_frontier_method",
            "target_coverage",
            "target_point_risk",
            "fess_ess_status",
            "fess_ess_bracket",
            "fess_ess_point_on_grid",
        ]
    ].rename(columns={"target_value": "frontier_set_size"})
    right = ress[
        keys
        + [
            "target_value",
            "target_risk_method",
            "ress_ess_status",
            "ress_ess_bracket",
            "ress_ess_point_on_grid",
        ]
    ].rename(columns={"target_value": "point_risk"})
    out = left.merge(right, on=keys, how="inner")
    out["equivalent_brackets_disagree"] = out["fess_ess_bracket"].astype(str) != out["ress_ess_bracket"].astype(str)
    out["fess_numeric_order"] = out["fess_ess_bracket"].map(bracket_numeric_order)
    out["ress_numeric_order"] = out["ress_ess_bracket"].map(bracket_numeric_order)
    out["fess_minus_ress_order"] = out["fess_numeric_order"] - out["ress_numeric_order"]
    return out


def summarize_comparison(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    rows = []
    group_cols = ["target_prompt_search_family", "coverage_target", "target_budget"]
    for key, sub in frame.groupby(group_cols, dropna=False):
        family, coverage, budget = key
        rows.append(
            {
                "target_prompt_search_family": family,
                "coverage_target": coverage,
                "target_budget": budget,
                "n_cells": int(len(sub)),
                "n_bracket_disagree": int(sub["equivalent_brackets_disagree"].sum()),
                "bracket_disagreement_rate": float(sub["equivalent_brackets_disagree"].mean()),
                "fess_above_fixed_grid": int(sub["fess_ess_status"].eq("above_grid").sum()),
                "ress_above_fixed_grid": int(sub["ress_ess_status"].eq("above_grid").sum()),
                "mean_fess_minus_ress_order": float(sub["fess_minus_ress_order"].mean()),
                "median_fess_bracket_order": float(sub["fess_numeric_order"].median()),
                "median_ress_bracket_order": float(sub["ress_numeric_order"].median()),
            }
        )
    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute E20 prompt-search equivalent-budget tables relative to fixed-only.")
    parser.add_argument("--frontiers-csv", required=True)
    parser.add_argument("--all-policies-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reference-family", default="fixed_only")
    parser.add_argument("--target-families", nargs="+", default=["seed_selection", "component_grid", "oprolite_minimax"])
    args = parser.parse_args()

    frontiers = pd.read_csv(Path(args.frontiers_csv))
    all_policies = pd.read_csv(Path(args.all_policies_csv))
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)

    risk_frontiers = build_risk_frontiers(all_policies)
    fess = compute_frontier_equivalent_budget(frontiers, args.target_families, args.reference_family)
    ress = compute_risk_equivalent_budget(risk_frontiers, args.target_families, args.reference_family)
    comparison = comparison_table(fess, ress)
    summary = summarize_comparison(comparison)

    write_csv(risk_frontiers, output / "risk_frontiers_by_prompt_search_family.csv")
    write_csv(fess, output / "frontier_equivalent_fixed_budget_seed_level.csv")
    write_csv(ress, output / "risk_equivalent_fixed_budget_seed_level.csv")
    write_csv(comparison, output / "prompt_search_equivalent_budget_comparison.csv")
    write_csv(summary, output / "prompt_search_equivalent_budget_summary.csv")
    write_json(
        output / "prompt_search_equivalent_budget_manifest.json",
        {
            "frontiers_csv": str(Path(args.frontiers_csv).resolve()),
            "all_policies_csv": str(Path(args.all_policies_csv).resolve()),
            "reference_family": args.reference_family,
            "target_families": args.target_families,
            "n_risk_frontier_rows": int(len(risk_frontiers)),
            "n_fess_rows": int(len(fess)),
            "n_ress_rows": int(len(ress)),
            "n_comparison_rows": int(len(comparison)),
        },
    )
    print(
        {
            "status": "ok",
            "output_dir": str(output),
            "n_comparison_rows": int(len(comparison)),
            "n_bracket_disagreements": int(comparison["equivalent_brackets_disagree"].sum()),
        }
    )


if __name__ == "__main__":
    main()
