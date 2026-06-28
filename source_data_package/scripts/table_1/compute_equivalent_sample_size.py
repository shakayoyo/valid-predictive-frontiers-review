from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.logging import write_json


RUN_LABELS = {
    "minimax_anes_schema_promptopt_n180": "minimax_anes_schema_promptopt",
    "minimax_fair_schema_promptopt_n180": "minimax_fair_schema_promptopt",
    "minimax_anes_promptopt_n180": "minimax_anes_real_promptopt",
    "minimax_promptopt_survey_n100": "minimax_promptopt_main",
    "minimax_anes_schema_budgetgrid_v1": "minimax_anes_schema_budgetgrid",
    "minimax_anes_schema_budgetgrid_s10": "minimax_anes_schema_budgetgrid",
    "minimax_fair_schema_budgetgrid_v1": "minimax_fair_schema_budgetgrid",
    "minimax_fair_schema_budgetgrid_s10": "minimax_fair_schema_budgetgrid",
    "minimax_randhie_schema_budgetgrid_n180": "minimax_randhie_schema_budgetgrid",
    "minimax_randhie_schema_budgetgrid_s10": "minimax_randhie_schema_budgetgrid",
    "minimax_modechoice_schema_budgetgrid_n160": "minimax_modechoice_schema_budgetgrid",
    "minimax_modechoice_schema_budgetgrid_s10": "minimax_modechoice_schema_budgetgrid",
    "minimax_star98_schema_budgetgrid_n200": "minimax_star98_schema_budgetgrid",
    "minimax_star98_schema_budgetgrid_s10": "minimax_star98_schema_budgetgrid",
    "minimax_fertility_schema_budgetgrid_n150": "minimax_fertility_schema_budgetgrid",
    "minimax_fertility_schema_budgetgrid_s10": "minimax_fertility_schema_budgetgrid",
    "minimax_adult_schema_promptopt_n180_v2": "minimax_adult_schema_budgetgrid",
    "minimax_adult_schema_budgetgrid_s10": "minimax_adult_schema_budgetgrid",
    "minimax_adult_schema_budgetgrid_s10_strongref": "minimax_adult_schema_budgetgrid",
    "minimax_acs_schema_promptopt_strongref": "minimax_acs_schema_budgetgrid",
    "minimax_acs_schema_budgetgrid_s10_strongref": "minimax_acs_schema_budgetgrid",
    "qwen25_7b_adult_promptselect_s10": "qwen25_7b_adult_promptselect_s10",
}

RUN_SOURCE_PRIORITY = {
    "minimax_anes_schema_budgetgrid_s10": 10,
    "minimax_fair_schema_budgetgrid_s10": 10,
    "minimax_randhie_schema_budgetgrid_s10": 10,
    "minimax_modechoice_schema_budgetgrid_s10": 10,
    "minimax_anes_schema_budgetgrid_v1": 1,
    "minimax_fair_schema_budgetgrid_v1": 1,
    "minimax_randhie_schema_budgetgrid_n180": 1,
    "minimax_modechoice_schema_budgetgrid_n160": 1,
    "minimax_star98_schema_budgetgrid_s10": 10,
    "minimax_star98_schema_budgetgrid_n200": 1,
    "minimax_fertility_schema_budgetgrid_s10": 10,
    "minimax_fertility_schema_budgetgrid_n150": 1,
    "minimax_adult_schema_budgetgrid_s10": 10,
    "minimax_adult_schema_budgetgrid_s10_strongref": 20,
    "minimax_adult_schema_promptopt_n180_v2": 1,
    "minimax_acs_schema_promptopt_strongref": 20,
    "minimax_acs_schema_budgetgrid_s10_strongref": 30,
    "qwen25_7b_adult_promptselect_s10": 10,
}

POLICY_SPACES = {
    "reference": {"reference"},
    "llm_fixed_prompt_frontier": {"llm_cached"},
    "llm_prompt_optimized": {"llm_prompt_optimized"},
    "llm_prompt_policy_frontier": {"llm_cached", "llm_prompt_optimized"},
    "overall_frontier": {"reference", "llm_cached", "llm_prompt_optimized"},
}

KEYS = ["run_label", "experiment_id", "dataset_id", "task_id", "seed", "coverage_target"]


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


def equivalent_budget(reference_curve: pd.DataFrame, target_value: float, metric: str) -> dict[str, object]:
    curve = reference_curve.dropna(subset=[metric]).sort_values("budget")
    if curve.empty or not np.isfinite(target_value):
        return {
            "ess_status": "not_estimable",
            "ess_bracket": "NA",
            "ess_lower_budget": np.nan,
            "ess_upper_budget": np.nan,
            "ess_point_on_grid": np.nan,
            "matched_reference_value": np.nan,
        }
    budgets = curve["budget"].astype(int).to_numpy()
    values = curve[metric].astype(float).to_numpy()
    hits = np.flatnonzero(values <= float(target_value))
    if len(hits) == 0:
        return {
            "ess_status": "above_grid",
            "ess_bracket": f">{int(budgets[-1])}",
            "ess_lower_budget": int(budgets[-1]),
            "ess_upper_budget": np.nan,
            "ess_point_on_grid": np.nan,
            "matched_reference_value": np.nan,
        }
    hit = int(hits[0])
    upper = int(budgets[hit])
    lower = 0 if hit == 0 else int(budgets[hit - 1])
    bracket = f"<={upper}" if hit == 0 else f"{lower}-{upper}"
    return {
        "ess_status": "attained",
        "ess_bracket": bracket,
        "ess_lower_budget": lower,
        "ess_upper_budget": upper,
        "ess_point_on_grid": upper,
        "matched_reference_value": float(values[hit]),
    }


def compute_frontier_ess(
    frontier_rows: pd.DataFrame,
    target_spaces: list[str],
    reference_space: str,
) -> pd.DataFrame:
    if frontier_rows.empty:
        return pd.DataFrame()
    attained = frontier_rows[frontier_rows["frontier_status"] == "attained"].copy()
    ref = attained[attained["method_family"] == reference_space].copy()
    targets = attained[attained["method_family"].isin(target_spaces)].copy()
    rows = []
    for _, target in targets.iterrows():
        key = {name: target[name] for name in KEYS}
        ref_curve = ref.copy()
        for name, value in key.items():
            ref_curve = ref_curve[ref_curve[name] == value]
        eq = equivalent_budget(ref_curve, float(target["frontier_set_size"]), "frontier_set_size")
        rows.append(
            {
                **key,
                "target_policy_space": target["method_family"],
                "reference_policy_space": reference_space,
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


def all_policy_files(remote_results_dir: Path) -> list[Path]:
    paths: list[Path] = []
    for current, _, filenames in os.walk(filesystem_path(remote_results_dir)):
        current_path = Path(current)
        if current_path.name != "results":
            continue
        for filename in filenames:
            if filename.endswith("_all_policies.csv"):
                paths.append(current_path / filename)
    return sorted(paths, key=lambda path: path.as_posix())


def load_all_policy_rows(remote_results_dir: Path) -> pd.DataFrame:
    frames = []
    paths = all_policy_files(remote_results_dir)
    best_priority: dict[str, int] = {}
    for path in paths:
        run_dir = path.parents[1].name
        label = RUN_LABELS.get(run_dir, run_dir)
        priority = RUN_SOURCE_PRIORITY.get(run_dir, 0)
        best_priority[label] = max(best_priority.get(label, priority), priority)
    for path in paths:
        frame = read_csv_if_exists(path)
        if frame.empty:
            continue
        run_dir = path.parents[1].name
        run_label = RUN_LABELS.get(run_dir, run_dir)
        priority = RUN_SOURCE_PRIORITY.get(run_dir, 0)
        if priority < best_priority.get(run_label, priority):
            continue
        frame.insert(0, "run_label", run_label)
        frame.insert(1, "source_result_dir", run_dir)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_risk_frontiers(all_rows: pd.DataFrame) -> pd.DataFrame:
    if all_rows.empty:
        return pd.DataFrame()
    frames = []
    group_cols = ["run_label", "experiment_id", "dataset_id", "task_id", "seed", "budget", "coverage_target"]
    for policy_space, families in POLICY_SPACES.items():
        sub = all_rows[all_rows["method_family"].isin(families)].copy()
        if sub.empty:
            continue
        idx = sub.groupby(group_cols)["point_risk"].idxmin()
        best = sub.loc[idx].copy()
        best["policy_space"] = policy_space
        best["risk_frontier_method"] = best["method_id"]
        best["risk_frontier_value"] = best["point_risk"].astype(float)
        frames.append(best[group_cols + ["policy_space", "risk_frontier_method", "risk_frontier_value"]])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def compute_risk_ess(
    risk_frontiers: pd.DataFrame,
    target_spaces: list[str],
    reference_space: str,
) -> pd.DataFrame:
    if risk_frontiers.empty:
        return pd.DataFrame()
    ref = risk_frontiers[risk_frontiers["policy_space"] == reference_space].copy()
    targets = risk_frontiers[risk_frontiers["policy_space"].isin(target_spaces)].copy()
    rows = []
    for _, target in targets.iterrows():
        key = {name: target[name] for name in KEYS}
        ref_curve = ref.copy()
        for name, value in key.items():
            ref_curve = ref_curve[ref_curve[name] == value]
        eq = equivalent_budget(ref_curve, float(target["risk_frontier_value"]), "risk_frontier_value")
        rows.append(
            {
                **key,
                "target_policy_space": target["policy_space"],
                "reference_policy_space": reference_space,
                "target_budget": int(target["budget"]),
                "target_metric": "point_risk",
                "target_value": float(target["risk_frontier_value"]),
                "target_risk_method": target["risk_frontier_method"],
                **{f"ress_{k}": v for k, v in eq.items()},
            }
        )
    return pd.DataFrame(rows)


def summarize_equivalents(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    bracket_col = f"{prefix}_ess_bracket"
    status_col = f"{prefix}_ess_status"
    summary = (
        frame.groupby(["run_label", "dataset_id", "task_id", "target_policy_space", "coverage_target", "target_budget", status_col, bracket_col], dropna=False)
        .size()
        .rename("n_cells")
        .reset_index()
    )
    total = (
        frame.groupby(["run_label", "dataset_id", "task_id", "target_policy_space", "coverage_target", "target_budget"], dropna=False)
        .size()
        .rename("n_total")
        .reset_index()
    )
    out = summary.merge(total, on=["run_label", "dataset_id", "task_id", "target_policy_space", "coverage_target", "target_budget"], how="left")
    out["cell_fraction"] = out["n_cells"] / out["n_total"]
    return out


def comparison_table(fess: pd.DataFrame, ress: pd.DataFrame) -> pd.DataFrame:
    if fess.empty or ress.empty:
        return pd.DataFrame()
    keys = ["run_label", "experiment_id", "dataset_id", "task_id", "seed", "coverage_target", "target_policy_space", "target_budget"]
    left = fess[keys + ["target_value", "fess_ess_status", "fess_ess_bracket", "fess_ess_point_on_grid"]].rename(columns={"target_value": "frontier_set_size"})
    right = ress[keys + ["target_value", "ress_ess_status", "ress_ess_bracket", "ress_ess_point_on_grid"]].rename(columns={"target_value": "point_risk"})
    out = left.merge(right, on=keys, how="inner")
    out["ess_brackets_disagree"] = out["fess_ess_bracket"].astype(str) != out["ress_ess_bracket"].astype(str)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frontier-seed-level", required=True)
    parser.add_argument("--remote-results-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target-spaces", nargs="+", default=["llm_prompt_policy_frontier"])
    parser.add_argument("--reference-space", default="reference")
    args = parser.parse_args()

    frontier_path = Path(args.frontier_seed_level).resolve()
    remote_root = Path(args.remote_results_dir).resolve()
    output = Path(args.output_dir).resolve()
    ensure_dir(output)

    frontier_rows = read_csv_if_exists(frontier_path)
    all_rows = load_all_policy_rows(remote_root)
    risk_frontiers = build_risk_frontiers(all_rows)
    fess = compute_frontier_ess(frontier_rows, args.target_spaces, args.reference_space)
    ress = compute_risk_ess(risk_frontiers, args.target_spaces, args.reference_space)
    comparison = comparison_table(fess, ress)
    fess_summary = summarize_equivalents(fess, "fess")
    ress_summary = summarize_equivalents(ress, "ress")

    write_csv(risk_frontiers, output / "risk_frontiers_seed_level.csv")
    write_csv(fess, output / "frontier_equivalent_sample_size_seed_level.csv")
    write_csv(ress, output / "risk_equivalent_sample_size_seed_level.csv")
    write_csv(comparison, output / "ess_comparison_seed_level.csv")
    write_csv(fess_summary, output / "frontier_equivalent_sample_size_summary.csv")
    write_csv(ress_summary, output / "risk_equivalent_sample_size_summary.csv")
    write_json(
        output / "equivalent_sample_size_manifest.json",
        {
            "frontier_seed_level": str(frontier_path),
            "remote_results_dir": str(remote_root),
            "target_spaces": args.target_spaces,
            "reference_space": args.reference_space,
            "n_all_policy_rows": int(len(all_rows)),
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
            "n_fess_rows": int(len(fess)),
            "n_ress_rows": int(len(ress)),
            "n_comparison_rows": int(len(comparison)),
        }
    )


if __name__ == "__main__":
    main()
