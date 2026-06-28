from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.frontier import compute_valid_frontier
from src.utils.logging import write_json


def filesystem_path(path: Path) -> str:
    resolved = str(path.resolve())
    if sys.platform != "win32" or resolved.startswith("\\\\?\\"):
        return resolved
    if resolved.startswith("\\\\"):
        return "\\\\?\\UNC\\" + resolved[2:]
    return "\\\\?\\" + resolved


def ensure_dir(path: Path) -> None:
    os.makedirs(filesystem_path(path), exist_ok=True)


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    frame.to_csv(filesystem_path(path), index=False)


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not os.path.exists(filesystem_path(path)):
        return pd.DataFrame()
    try:
        return pd.read_csv(filesystem_path(path))
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def add_policy_space(rows: pd.DataFrame, policy_space: str, method_families: set[str] | None = None) -> pd.DataFrame:
    if rows.empty:
        return rows.copy()
    out = rows.copy()
    if method_families is not None:
        out = out[out["method_family"].isin(method_families)].copy()
    if out.empty:
        return out
    out["method_family"] = policy_space
    return out


def family_frontiers(rows: pd.DataFrame, tolerance: float) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    return compute_valid_frontier(rows, tolerance=tolerance)


def attach_run_label(df: pd.DataFrame, run_label: str) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.copy()
    out.insert(0, "run_label", run_label)
    return out


def normalize_experiment_id(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty or "source_experiment_id" not in rows.columns:
        return rows.copy()
    out = rows.copy()
    source = out["source_experiment_id"]
    mask = source.notna() & (source.astype(str) != "")
    out.loc[mask, "experiment_id"] = source[mask].astype(str)
    return out


def summarize_frontiers(frontiers: pd.DataFrame) -> pd.DataFrame:
    if frontiers.empty:
        return pd.DataFrame()
    summary = (
        frontiers.groupby(["run_label", "method_family", "coverage_target"], dropna=False)
        .agg(
            n_cells=("frontier_status", "size"),
            n_attained=("frontier_status", lambda s: int((s == "attained").sum())),
            mean_frontier_set_size=("frontier_set_size", "mean"),
            median_frontier_set_size=("frontier_set_size", "median"),
            mean_coverage=("coverage", "mean"),
            mean_point_risk=("point_risk", "mean"),
        )
        .reset_index()
    )
    summary["attainment_rate"] = summary["n_attained"] / summary["n_cells"]
    return summary


def pairwise_gaps(frontiers: pd.DataFrame) -> pd.DataFrame:
    if frontiers.empty:
        return pd.DataFrame()
    keys = ["run_label", "experiment_id", "dataset_id", "task_id", "seed", "budget", "coverage_target"]
    valid = frontiers[frontiers["frontier_status"] == "attained"].copy()
    out = []
    for run_label, frame in valid.groupby("run_label"):
        spaces = sorted(frame["method_family"].unique())
        for left in spaces:
            for right in spaces:
                if left >= right:
                    continue
                a = frame[frame["method_family"] == left]
                b = frame[frame["method_family"] == right]
                merged = a.merge(b, on=keys, suffixes=(f"_{left}", f"_{right}"))
                if merged.empty:
                    continue
                out.append(
                    pd.DataFrame(
                        {
                            "run_label": run_label,
                            "left_policy_space": left,
                            "right_policy_space": right,
                            "n_common_cells": len(merged),
                            "mean_set_size_gap_left_minus_right": merged[f"frontier_set_size_{left}"].mean()
                            - merged[f"frontier_set_size_{right}"].mean(),
                            "mean_risk_gap_left_minus_right": merged[f"point_risk_{left}"].mean()
                            - merged[f"point_risk_{right}"].mean(),
                        },
                        index=[0],
                    )
                )
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def build_minimax_real_rows(root: Path) -> pd.DataFrame:
    frames = [
        read_csv_if_exists(root / "minimax_real_l0_l1_n120" / "results" / "E04_protocol_tiers.csv"),
        read_csv_if_exists(root / "minimax_real_l0_l1_n120" / "results" / "E05_protocol_tiers.csv"),
    ]
    rows = pd.concat([f for f in frames if not f.empty], ignore_index=True) if any(not f.empty for f in frames) else pd.DataFrame()
    if rows.empty:
        return rows
    rows = normalize_experiment_id(rows)
    combined = [
        rows,
        add_policy_space(rows, "llm_fixed_prompt_frontier", {"llm_cached"}),
        add_policy_space(rows, "overall_frontier", {"reference", "llm_cached"}),
    ]
    return pd.concat([x for x in combined if not x.empty], ignore_index=True)


def build_promptopt_rows(root: Path) -> pd.DataFrame:
    rows = read_csv_if_exists(root / "minimax_promptopt_survey_n100" / "results" / "E09_prompt_optimized_frontier.csv")
    if rows.empty:
        return rows
    combined = [
        rows,
        add_policy_space(rows, "llm_prompt_policy_frontier", {"llm_cached", "llm_prompt_optimized"}),
    ]
    return pd.concat([x for x in combined if not x.empty], ignore_index=True)


def build_promptopt_common_rows(root: Path) -> pd.DataFrame:
    rows = read_csv_if_exists(root / "minimax_promptopt_survey_n100" / "results" / "E09_prompt_optimized_frontier_common_budget.csv")
    if rows.empty:
        return rows
    combined = [
        rows,
        add_policy_space(rows, "llm_prompt_policy_frontier", {"llm_cached", "llm_prompt_optimized"}),
    ]
    return pd.concat([x for x in combined if not x.empty], ignore_index=True)


def build_anes_real_rows(root: Path) -> pd.DataFrame:
    rows = read_csv_if_exists(root / "minimax_anes_promptopt_n180" / "results" / "E10_anes_all_policies.csv")
    if rows.empty:
        return rows
    combined = [
        rows,
        add_policy_space(rows, "llm_fixed_prompt_frontier", {"llm_cached"}),
        add_policy_space(rows, "llm_prompt_policy_frontier", {"llm_cached", "llm_prompt_optimized"}),
        add_policy_space(rows, "overall_frontier", {"reference", "llm_cached", "llm_prompt_optimized"}),
    ]
    return pd.concat([x for x in combined if not x.empty], ignore_index=True)


def build_anes_schema_rows(root: Path) -> pd.DataFrame:
    rows = read_csv_if_exists(root / "minimax_anes_schema_promptopt_n180" / "results" / "E10_anes_all_policies.csv")
    if rows.empty:
        return rows
    combined = [
        rows,
        add_policy_space(rows, "llm_fixed_prompt_frontier", {"llm_cached"}),
        add_policy_space(rows, "llm_prompt_policy_frontier", {"llm_cached", "llm_prompt_optimized"}),
        add_policy_space(rows, "overall_frontier", {"reference", "llm_cached", "llm_prompt_optimized"}),
    ]
    return pd.concat([x for x in combined if not x.empty], ignore_index=True)


def build_fair_schema_rows(root: Path) -> pd.DataFrame:
    rows = read_csv_if_exists(root / "minimax_fair_schema_promptopt_n180" / "results" / "E11_fair_all_policies.csv")
    if rows.empty:
        return rows
    combined = [
        rows,
        add_policy_space(rows, "llm_fixed_prompt_frontier", {"llm_cached"}),
        add_policy_space(rows, "llm_prompt_policy_frontier", {"llm_cached", "llm_prompt_optimized"}),
        add_policy_space(rows, "overall_frontier", {"reference", "llm_cached", "llm_prompt_optimized"}),
    ]
    return pd.concat([x for x in combined if not x.empty], ignore_index=True)


def build_schema_budgetgrid_rows(root: Path, dirname: str, filename: str) -> pd.DataFrame:
    rows = read_csv_if_exists(root / dirname / "results" / filename)
    if rows.empty:
        return rows
    combined = [
        rows,
        add_policy_space(rows, "llm_fixed_prompt_frontier", {"llm_cached"}),
        add_policy_space(rows, "llm_prompt_policy_frontier", {"llm_cached", "llm_prompt_optimized"}),
        add_policy_space(rows, "overall_frontier", {"reference", "llm_cached", "llm_prompt_optimized"}),
    ]
    return pd.concat([x for x in combined if not x.empty], ignore_index=True)


def build_first_schema_budgetgrid_rows(root: Path, choices: list[tuple[str, str]]) -> pd.DataFrame:
    for dirname, filename in choices:
        rows = build_schema_budgetgrid_rows(root, dirname, filename)
        if not rows.empty:
            return rows
    return pd.DataFrame()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--remote-results-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tolerance", type=float, default=0.03)
    args = parser.parse_args()

    remote_root = Path(args.remote_results_dir).resolve()
    output = Path(args.output_dir).resolve()
    ensure_dir(output)

    runs = [
        ("minimax_real_reference_vs_fixed", build_minimax_real_rows(remote_root)),
        ("minimax_promptopt_main", build_promptopt_rows(remote_root)),
        ("minimax_promptopt_common_budget", build_promptopt_common_rows(remote_root)),
        ("minimax_anes_real_promptopt", build_anes_real_rows(remote_root)),
        ("minimax_anes_schema_promptopt", build_anes_schema_rows(remote_root)),
        ("minimax_fair_schema_promptopt", build_fair_schema_rows(remote_root)),
        (
            "minimax_anes_schema_budgetgrid",
            build_first_schema_budgetgrid_rows(
                remote_root,
                [
                    ("minimax_anes_schema_budgetgrid_s10", "E10B_anes_budget_grid_all_policies.csv"),
                    ("minimax_anes_schema_budgetgrid_v1", "E10B_anes_budget_grid_all_policies.csv"),
                ],
            ),
        ),
        (
            "minimax_fair_schema_budgetgrid",
            build_first_schema_budgetgrid_rows(
                remote_root,
                [
                    ("minimax_fair_schema_budgetgrid_s10", "E11B_fair_budget_grid_all_policies.csv"),
                    ("minimax_fair_schema_budgetgrid_v1", "E11B_fair_budget_grid_all_policies.csv"),
                ],
            ),
        ),
        (
            "minimax_randhie_schema_budgetgrid",
            build_first_schema_budgetgrid_rows(
                remote_root,
                [
                    ("minimax_randhie_schema_budgetgrid_s10", "E12B_randhie_budget_grid_all_policies.csv"),
                    ("minimax_randhie_schema_budgetgrid_n180", "E12_randhie_all_policies.csv"),
                ],
            ),
        ),
        (
            "minimax_modechoice_schema_budgetgrid",
            build_first_schema_budgetgrid_rows(
                remote_root,
                [
                    ("minimax_modechoice_schema_budgetgrid_s10", "E13B_modechoice_budget_grid_all_policies.csv"),
                    ("minimax_modechoice_schema_budgetgrid_n160", "E13_modechoice_all_policies.csv"),
                ],
            ),
        ),
        (
            "minimax_star98_schema_budgetgrid",
            build_first_schema_budgetgrid_rows(
                remote_root,
                [
                    ("minimax_star98_schema_budgetgrid_s10", "E14B_star98_budget_grid_all_policies.csv"),
                    ("minimax_star98_schema_budgetgrid_n200", "E14_star98_all_policies.csv"),
                ],
            ),
        ),
        (
            "minimax_fertility_schema_budgetgrid",
            build_first_schema_budgetgrid_rows(
                remote_root,
                [
                    ("minimax_fertility_schema_budgetgrid_s10", "E16B_fertility_budget_grid_all_policies.csv"),
                    ("minimax_fertility_schema_budgetgrid_n150", "E16_fertility_all_policies.csv"),
                ],
            ),
        ),
        (
            "minimax_adult_schema_budgetgrid",
            build_first_schema_budgetgrid_rows(
                remote_root,
                [
                    ("minimax_adult_schema_budgetgrid_s10_strongref", "E17B_adult_budget_grid_all_policies.csv"),
                    ("minimax_adult_schema_budgetgrid_s10", "E17B_adult_budget_grid_all_policies.csv"),
                    ("minimax_adult_schema_promptopt_n180_v2", "E17_adult_all_policies.csv"),
                ],
            ),
        ),
        (
            "minimax_acs_schema_budgetgrid",
            build_first_schema_budgetgrid_rows(
                remote_root,
                [
                    ("minimax_acs_schema_budgetgrid_s10_strongref", "E18B_acs_budget_grid_all_policies.csv"),
                    ("minimax_acs_schema_promptopt_strongref", "E18_acs_all_policies.csv"),
                ],
            ),
        ),
        (
            "qwen25_anes_fixed_s10",
            build_first_schema_budgetgrid_rows(
                remote_root,
                [
                    ("qwen25_anes_fixed_s10", "E15_anes_local_instruct_all_policies.csv"),
                ],
            ),
        ),
        (
            "qwen25_star98_fixed_s10",
            build_first_schema_budgetgrid_rows(
                remote_root,
                [
                    ("qwen25_star98_fixed_s10", "E15_star98_local_instruct_all_policies.csv"),
                ],
            ),
        ),
        (
            "qwen25_7b_anes_fixed_s10",
            build_first_schema_budgetgrid_rows(
                remote_root,
                [
                    ("qwen25_7b_anes_fixed_s10", "E15_anes_local_instruct_all_policies.csv"),
                ],
            ),
        ),
        (
            "qwen25_7b_star98_fixed_s10",
            build_first_schema_budgetgrid_rows(
                remote_root,
                [
                    ("qwen25_7b_star98_fixed_s10", "E15_star98_local_instruct_all_policies.csv"),
                ],
            ),
        ),
        (
            "qwen25_7b_anes_promptselect_s10",
            build_first_schema_budgetgrid_rows(
                remote_root,
                [
                    ("qwen25_7b_anes_promptselect_s10", "E15_anes_local_instruct_all_policies.csv"),
                ],
            ),
        ),
        (
            "qwen25_7b_star98_promptselect_s10",
            build_first_schema_budgetgrid_rows(
                remote_root,
                [
                    ("qwen25_7b_star98_promptselect_s10", "E15_star98_local_instruct_all_policies.csv"),
                ],
            ),
        ),
        (
            "qwen25_7b_fair_promptselect_s10",
            build_first_schema_budgetgrid_rows(
                remote_root,
                [
                    ("qwen25_7b_fair_promptselect_s10", "E15_fair_local_instruct_all_policies.csv"),
                ],
            ),
        ),
        (
            "qwen25_7b_randhie_promptselect_s10",
            build_first_schema_budgetgrid_rows(
                remote_root,
                [
                    ("qwen25_7b_randhie_promptselect_s10", "E15_randhie_local_instruct_all_policies.csv"),
                ],
            ),
        ),
        (
            "qwen25_7b_modechoice_promptselect_s10",
            build_first_schema_budgetgrid_rows(
                remote_root,
                [
                    ("qwen25_7b_modechoice_promptselect_s10", "E15_modechoice_local_instruct_all_policies.csv"),
                ],
            ),
        ),
        (
            "qwen25_7b_adult_promptselect_s10",
            build_first_schema_budgetgrid_rows(
                remote_root,
                [
                    ("qwen25_7b_adult_promptselect_s10", "E15_adult_local_instruct_all_policies.csv"),
                ],
            ),
        ),
    ]
    frontier_frames = []
    for run_label, rows in runs:
        if rows.empty:
            continue
        frontier_frames.append(attach_run_label(family_frontiers(rows, args.tolerance), run_label))

    frontiers = pd.concat(frontier_frames, ignore_index=True) if frontier_frames else pd.DataFrame()
    summary = summarize_frontiers(frontiers)
    gaps = pairwise_gaps(frontiers)

    write_csv(frontiers, output / "frontier_versions_seed_level.csv")
    write_csv(summary, output / "frontier_versions_summary.csv")
    write_csv(gaps, output / "frontier_versions_pairwise_gaps.csv")
    write_json(
        output / "frontier_versions_manifest.json",
        {
            "remote_results_dir": str(remote_root),
            "tolerance": args.tolerance,
            "n_frontier_rows": int(len(frontiers)),
            "n_summary_rows": int(len(summary)),
            "n_gap_rows": int(len(gaps)),
            "policy_spaces": sorted(frontiers["method_family"].dropna().unique().tolist()) if not frontiers.empty else [],
        },
    )

    print(
        {
            "status": "ok",
            "output_dir": str(output),
            "n_frontier_rows": int(len(frontiers)),
            "n_summary_rows": int(len(summary)),
            "n_gap_rows": int(len(gaps)),
        }
    )


if __name__ == "__main__":
    main()
