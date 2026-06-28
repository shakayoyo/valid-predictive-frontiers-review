from __future__ import annotations

import argparse
import fnmatch
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.compute_frontier_versions import (
    build_first_schema_budgetgrid_rows,
    build_schema_budgetgrid_rows,
)
from src.frontier import compute_valid_frontier
from src.utils.logging import write_json


DEFAULT_RUNS = [
    ("minimax_anes_schema_budgetgrid", [("minimax_anes_schema_budgetgrid_s10", "E10B_anes_budget_grid_all_policies.csv")]),
    ("minimax_fair_schema_budgetgrid", [("minimax_fair_schema_budgetgrid_s10", "E11B_fair_budget_grid_all_policies.csv")]),
    ("minimax_randhie_schema_budgetgrid", [("minimax_randhie_schema_budgetgrid_s10", "E12B_randhie_budget_grid_all_policies.csv")]),
    ("minimax_modechoice_schema_budgetgrid", [("minimax_modechoice_schema_budgetgrid_s10", "E13B_modechoice_budget_grid_all_policies.csv")]),
    ("minimax_star98_schema_budgetgrid", [("minimax_star98_schema_budgetgrid_s10", "E14B_star98_budget_grid_all_policies.csv")]),
    ("minimax_fertility_schema_budgetgrid", [("minimax_fertility_schema_budgetgrid_s10", "E16B_fertility_budget_grid_all_policies.csv")]),
    ("minimax_adult_schema_budgetgrid", [("minimax_adult_schema_budgetgrid_s10_strongref", "E17B_adult_budget_grid_all_policies.csv")]),
    ("minimax_acs_schema_budgetgrid", [("minimax_acs_schema_budgetgrid_s10_strongref", "E18B_acs_budget_grid_all_policies.csv")]),
]


def build_rows(remote_root: Path, run_patterns: list[str]) -> pd.DataFrame:
    frames = []
    for run_label, choices in DEFAULT_RUNS:
        if run_patterns and not any(fnmatch.fnmatch(run_label, pattern) for pattern in run_patterns):
            continue
        rows = build_first_schema_budgetgrid_rows(remote_root, choices)
        if rows.empty:
            continue
        rows = rows.copy()
        rows.insert(0, "run_label", run_label)
        frames.append(rows)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def select_from_training_seeds(
    train: pd.DataFrame,
    *,
    coverage_target: float,
    tolerance: float,
) -> pd.Series | None:
    if train.empty:
        return None
    candidate_cols = ["method_id", "budget"]
    agg = (
        train.groupby(candidate_cols, as_index=False)
        .agg(
            selection_coverage=("coverage", "mean"),
            selection_mean_set_size=("mean_set_size", "mean"),
            selection_point_risk=("point_risk", "mean"),
            selection_n_seeds=("seed", "nunique"),
        )
    )
    valid = agg[agg["selection_coverage"] >= coverage_target - tolerance].copy()
    if valid.empty:
        return None
    return valid.sort_values(["selection_mean_set_size", "selection_point_risk", "method_id"]).iloc[0]


def heldout_oracle(heldout_family: pd.DataFrame, *, tolerance: float) -> pd.Series | None:
    if heldout_family.empty:
        return None
    frontier = compute_valid_frontier(heldout_family, tolerance=tolerance)
    attained = frontier[frontier["frontier_status"] == "attained"].copy()
    if attained.empty:
        return None
    return attained.iloc[0]


def leave_one_seed_out(rows: pd.DataFrame, *, tolerance: float, families: set[str]) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    required = {
        "run_label",
        "experiment_id",
        "dataset_id",
        "task_id",
        "seed",
        "budget",
        "coverage_target",
        "method_family",
        "method_id",
        "coverage",
        "mean_set_size",
        "point_risk",
    }
    missing = sorted(required.difference(rows.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    use = rows[rows["method_family"].isin(families)].copy()
    use = use.dropna(subset=["seed", "budget", "coverage_target", "coverage", "mean_set_size", "point_risk"])
    group_cols = ["run_label", "experiment_id", "dataset_id", "task_id", "coverage_target", "method_family"]
    out = []
    for key, frame in use.groupby(group_cols, dropna=False):
        key_dict = dict(zip(group_cols, key))
        seeds = sorted(frame["seed"].dropna().unique().tolist())
        if len(seeds) < 3:
            continue
        budgets = sorted(frame["budget"].dropna().unique().tolist())
        for budget in budgets:
            budget_frame = frame[frame["budget"] == budget].copy()
            for heldout_seed in seeds:
                train = budget_frame[budget_frame["seed"] != heldout_seed]
                heldout = budget_frame[budget_frame["seed"] == heldout_seed]
                if train.empty or heldout.empty:
                    continue
                selected = select_from_training_seeds(
                    train,
                    coverage_target=float(key_dict["coverage_target"]),
                    tolerance=tolerance,
                )
                oracle = heldout_oracle(heldout, tolerance=tolerance)
                base = key_dict | {"budget": budget, "heldout_seed": heldout_seed}
                if selected is None:
                    out.append(
                        base
                        | {
                            "selection_status": "no_selection_valid",
                            "selected_method": "",
                            "selection_coverage": np.nan,
                            "selection_mean_set_size": np.nan,
                            "selection_point_risk": np.nan,
                            "selection_n_seeds": int(train["seed"].nunique()),
                            "heldout_coverage": np.nan,
                            "heldout_mean_set_size": np.nan,
                            "heldout_point_risk": np.nan,
                            "heldout_empirically_admissible": False,
                            "oracle_method": "" if oracle is None else oracle["frontier_method"],
                            "oracle_mean_set_size": np.nan if oracle is None else oracle["frontier_set_size"],
                            "oracle_point_risk": np.nan if oracle is None else oracle["point_risk"],
                            "selected_minus_oracle_set_size": np.nan,
                        }
                    )
                    continue
                selected_rows = heldout[
                    (heldout["method_id"] == selected["method_id"])
                    & (heldout["budget"] == selected["budget"])
                ]
                if selected_rows.empty:
                    continue
                eval_row = selected_rows.iloc[0]
                oracle_size = np.nan if oracle is None else float(oracle["frontier_set_size"])
                heldout_size = float(eval_row["mean_set_size"])
                out.append(
                    base
                    | {
                        "selection_status": "selected",
                        "selected_method": selected["method_id"],
                        "selection_coverage": float(selected["selection_coverage"]),
                        "selection_mean_set_size": float(selected["selection_mean_set_size"]),
                        "selection_point_risk": float(selected["selection_point_risk"]),
                        "selection_n_seeds": int(selected["selection_n_seeds"]),
                        "heldout_coverage": float(eval_row["coverage"]),
                        "heldout_mean_set_size": heldout_size,
                        "heldout_point_risk": float(eval_row["point_risk"]),
                        "heldout_empirically_admissible": bool(
                            float(eval_row["coverage"]) >= float(key_dict["coverage_target"]) - tolerance
                        ),
                        "oracle_method": "" if oracle is None else oracle["frontier_method"],
                        "oracle_mean_set_size": oracle_size,
                        "oracle_point_risk": np.nan if oracle is None else float(oracle["point_risk"]),
                        "selected_minus_oracle_set_size": np.nan
                        if np.isnan(oracle_size)
                        else heldout_size - oracle_size,
                    }
                )
    return pd.DataFrame(out)


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    selected = rows[rows["selection_status"] == "selected"].copy()
    if selected.empty:
        return pd.DataFrame()
    group_cols = ["method_family", "coverage_target"]
    return (
        selected.groupby(group_cols, dropna=False)
        .agg(
            n_heldout_cells=("selection_status", "size"),
            n_tasks=("task_id", "nunique"),
            n_run_labels=("run_label", "nunique"),
            selection_admissible_rate=("heldout_empirically_admissible", "mean"),
            mean_selected_set_size=("heldout_mean_set_size", "mean"),
            mean_oracle_set_size=("oracle_mean_set_size", "mean"),
            mean_selected_minus_oracle=("selected_minus_oracle_set_size", "mean"),
            median_selected_minus_oracle=("selected_minus_oracle_set_size", "median"),
            mean_selected_risk=("heldout_point_risk", "mean"),
            mean_oracle_risk=("oracle_point_risk", "mean"),
        )
        .reset_index()
    )


def task_summary(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    selected = rows[rows["selection_status"] == "selected"].copy()
    if selected.empty:
        return pd.DataFrame()
    group_cols = ["run_label", "task_id", "method_family", "coverage_target"]
    return (
        selected.groupby(group_cols, dropna=False)
        .agg(
            n_heldout_cells=("selection_status", "size"),
            selection_admissible_rate=("heldout_empirically_admissible", "mean"),
            mean_selected_set_size=("heldout_mean_set_size", "mean"),
            mean_oracle_set_size=("oracle_mean_set_size", "mean"),
            mean_selected_minus_oracle=("selected_minus_oracle_set_size", "mean"),
            mean_selected_risk=("heldout_point_risk", "mean"),
            mean_oracle_risk=("oracle_point_risk", "mean"),
        )
        .reset_index()
    )


def write_markdown(summary_df: pd.DataFrame, task_df: pd.DataFrame, output: Path) -> None:
    lines = [
        "# Seed-Heldout Deployable-Policy Proxy",
        "",
        "Purpose: estimate how much the audit lower envelope can differ from a policy selected without the held-out seed's audit outcomes.",
        "For each task, coverage, family, budget, and held-out split seed, the diagnostic selects a fixed method using all other seeds and evaluates that method on the held-out seed.",
        "This is a seed-level proxy, not a row-level nested holdout and not a new conformal guarantee.",
        "",
        "## Combined Summary",
        "",
    ]
    if summary_df.empty:
        lines.append("No eligible held-out cells were available.")
    else:
        show = summary_df.copy()
        for col in [
            "selection_admissible_rate",
            "mean_selected_set_size",
            "mean_oracle_set_size",
            "mean_selected_minus_oracle",
            "median_selected_minus_oracle",
            "mean_selected_risk",
            "mean_oracle_risk",
        ]:
            show[col] = show[col].map(lambda x: "" if pd.isna(x) else f"{x:.3f}")
        lines.append(show.to_markdown(index=False))
    lines.extend(["", "## Task-Level Summary", ""])
    if task_df.empty:
        lines.append("No eligible task-level held-out cells were available.")
    else:
        show = task_df.copy()
        for col in [
            "selection_admissible_rate",
            "mean_selected_set_size",
            "mean_oracle_set_size",
            "mean_selected_minus_oracle",
            "mean_selected_risk",
            "mean_oracle_risk",
        ]:
            show[col] = show[col].map(lambda x: "" if pd.isna(x) else f"{x:.3f}")
        lines.append(show.to_markdown(index=False))
    lines.extend(
        [
            "",
            "Interpretation: positive selected-minus-oracle values are expected because the audit lower envelope is allowed to choose the empirically shortest admissible method on the held-out seed, while the proxy selected policy is fixed before seeing that seed.",
            "ACS-complete analyses should rerun this diagnostic after the ACS/Folktables run is merged.",
        ]
    )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--remote-results-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tolerance", type=float, default=0.03)
    parser.add_argument(
        "--family",
        action="append",
        default=["llm_prompt_policy_frontier", "reference"],
        help="Method family or policy space to include. Repeatable.",
    )
    parser.add_argument(
        "--run-pattern",
        action="append",
        default=["minimax_*_schema_budgetgrid"],
        help="fnmatch pattern for run labels. Repeatable.",
    )
    args = parser.parse_args()

    remote_root = Path(args.remote_results_dir).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)

    all_rows = build_rows(remote_root, args.run_pattern)
    heldout = leave_one_seed_out(all_rows, tolerance=args.tolerance, families=set(args.family))
    summary_df = summarize(heldout)
    task_df = task_summary(heldout)

    heldout.to_csv(output / "seed_heldout_deployable_policy_rows.csv", index=False)
    summary_df.to_csv(output / "seed_heldout_deployable_policy_summary.csv", index=False)
    task_df.to_csv(output / "seed_heldout_deployable_policy_task_summary.csv", index=False)
    write_markdown(summary_df, task_df, output / "seed_heldout_deployable_policy_summary.md")
    write_json(
        output / "seed_heldout_deployable_policy_manifest.json",
        {
            "remote_results_dir": str(remote_root),
            "tolerance": args.tolerance,
            "families": args.family,
            "run_patterns": args.run_pattern,
            "n_input_rows": int(len(all_rows)),
            "n_heldout_rows": int(len(heldout)),
            "n_summary_rows": int(len(summary_df)),
            "n_task_summary_rows": int(len(task_df)),
            "diagnostic_scope": "leave-one-seed-out proxy; not row-level nested holdout",
        },
    )
    print(
        {
            "status": "ok",
            "output_dir": str(output),
            "n_input_rows": int(len(all_rows)),
            "n_heldout_rows": int(len(heldout)),
            "n_summary_rows": int(len(summary_df)),
        }
    )


if __name__ == "__main__":
    main()
