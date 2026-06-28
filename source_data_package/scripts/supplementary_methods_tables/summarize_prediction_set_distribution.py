from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from scripts.recompute_schema_realdata_budget_grid import build_task
from scripts.run_experiments import finite_conformal_quantile_available
from src.conformal import _classification_covered, _classification_set_sizes, conformal_quantile, make_split
from src.data.loaders import TaskData
from src.models import build_method


RUNS = {
    "minimax_anes_schema_budgetgrid": {
        "kind": "anes",
        "budget_dir": "remote_results_dag/minimax_anes_schema_budgetgrid_s10",
        "fallback_source_dir": "remote_results_dag/minimax_anes_schema_promptopt_n180",
        "prefix": "E10B_anes_budget_grid",
        "dataset_manifest": "E10_dataset_manifest.json",
        "cache_name": "minimax_anes_prompt_cache.csv",
    },
    "minimax_fair_schema_budgetgrid": {
        "kind": "fair",
        "budget_dir": "remote_results_dag/minimax_fair_schema_budgetgrid_s10",
        "fallback_source_dir": "remote_results_dag/minimax_fair_schema_promptopt_n180",
        "prefix": "E11B_fair_budget_grid",
        "dataset_manifest": "E11_dataset_manifest.json",
        "cache_name": "minimax_fair_prompt_cache.csv",
    },
    "minimax_randhie_schema_budgetgrid": {
        "kind": "randhie",
        "budget_dir": "remote_results_dag/minimax_randhie_schema_budgetgrid_s10",
        "fallback_source_dir": "remote_results_dag/minimax_randhie_schema_budgetgrid_n180",
        "prefix": "E12B_randhie_budget_grid",
        "dataset_manifest": "E12_dataset_manifest.json",
        "cache_name": "minimax_randhie_prompt_cache.csv",
    },
    "minimax_modechoice_schema_budgetgrid": {
        "kind": "modechoice",
        "budget_dir": "remote_results_dag/minimax_modechoice_schema_budgetgrid_s10",
        "fallback_source_dir": "remote_results_dag/minimax_modechoice_schema_budgetgrid_n160",
        "prefix": "E13B_modechoice_budget_grid",
        "dataset_manifest": "E13_dataset_manifest.json",
        "cache_name": "minimax_modechoice_prompt_cache.csv",
    },
    "minimax_star98_schema_budgetgrid": {
        "kind": "star98",
        "budget_dir": "remote_results_dag/minimax_star98_schema_budgetgrid_s10",
        "fallback_source_dir": "remote_results_dag/minimax_star98_schema_budgetgrid_n200",
        "prefix": "E14B_star98_budget_grid",
        "dataset_manifest": "E14_dataset_manifest.json",
        "cache_name": "minimax_star98_prompt_cache.csv",
    },
    "minimax_fertility_schema_budgetgrid": {
        "kind": "fertility",
        "budget_dir": "remote_results_dag/minimax_fertility_schema_budgetgrid_s10",
        "fallback_source_dir": "remote_results_dag/minimax_fertility_schema_budgetgrid_n150",
        "prefix": "E16B_fertility_budget_grid",
        "dataset_manifest": "E16_dataset_manifest.json",
        "cache_name": "minimax_fertility_prompt_cache.csv",
    },
    "minimax_adult_schema_budgetgrid": {
        "kind": "adult",
        "budget_dir": "remote_results_dag/minimax_adult_schema_budgetgrid_s10_strongref",
        "fallback_source_dir": "remote_results_dag/minimax_adult_schema_promptopt_n180_v2",
        "prefix": "E17B_adult_budget_grid",
        "dataset_manifest": "E17_dataset_manifest.json",
        "cache_name": "minimax_adult_prompt_cache.csv",
    },
    "minimax_acs_schema_budgetgrid": {
        "kind": "acs",
        "budget_dir": "remote_results_dag/minimax_acs_schema_promptopt_strongref",
        "fallback_source_dir": "remote_results_dag/minimax_acs_schema_promptopt_strongref",
        "prefix": "E18_acs",
        "dataset_manifest": "E18_dataset_manifest.json",
        "cache_name": "minimax_acs_prompt_cache.csv",
    },
}

DISPLAY_NAMES = {
    "minimax_anes_schema_budgetgrid": "ANES96 vote",
    "minimax_fair_schema_budgetgrid": "Fair affairs",
    "minimax_randhie_schema_budgetgrid": "RAND HIE visits",
    "minimax_modechoice_schema_budgetgrid": "Modechoice car",
    "minimax_star98_schema_budgetgrid": "Star98 education",
    "minimax_fertility_schema_budgetgrid": "Fertility demography",
    "minimax_adult_schema_budgetgrid": "Adult income",
    "minimax_acs_schema_budgetgrid": "ACS income",
}

FRONTIER_METHOD_FAMILIES = {
    "llm_prompt_policy_frontier",
    "reference",
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_source_dir(run_dir: Path, config: dict) -> Path:
    manifest_path = run_dir / config["budget_dir"] / "raw_outputs" / f"{config['prefix']}_manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        source_dir = Path(str(manifest.get("source_dir", "")))
        if source_dir.exists():
            return source_dir
    return run_dir / config["fallback_source_dir"]


def final_row_indices(n_rows: int, optimization_rows: int, final_rows: int) -> list[int]:
    return list(range(optimization_rows, min(optimization_rows + final_rows, n_rows)))


def final_task(task: TaskData, rows: list[int]) -> TaskData:
    return TaskData(
        task.dataset_id,
        task.task_id,
        task.task_type,
        task.X.iloc[rows].reset_index(drop=True),
        task.y.iloc[rows].reset_index(drop=True),
        task.groups.iloc[rows].reset_index(drop=True),
        task.metadata,
    )


def distribution_from_sizes(sizes: np.ndarray) -> dict[str, float | int]:
    sizes = np.asarray(sizes, dtype=int)
    n = int(len(sizes))
    if n == 0:
        return {
            "audit_n": 0,
            "empty_set_rate": math.nan,
            "one_label_rate": math.nan,
            "two_label_rate": math.nan,
            "mean_set_size": math.nan,
            "median_set_size": math.nan,
            "q90_set_size": math.nan,
        }
    return {
        "audit_n": n,
        "empty_set_rate": float(np.mean(sizes == 0)),
        "one_label_rate": float(np.mean(sizes == 1)),
        "two_label_rate": float(np.mean(sizes == 2)),
        "mean_set_size": float(np.mean(sizes)),
        "median_set_size": float(np.median(sizes)),
        "q90_set_size": float(np.quantile(sizes, 0.9)),
    }


def llm_distribution(
    *,
    task: TaskData,
    cache: pd.DataFrame,
    method_id: str,
    final_rows: list[int],
    budget: int,
    coverage: float,
    seed: int,
    fractions: dict[str, float],
) -> dict[str, float | int]:
    rows = cache[
        (cache["method_id"].astype(str) == method_id)
        & (cache["row_index"].astype(int).isin(final_rows))
        & (cache["parse_status"].astype(str) == "ok")
    ].copy()
    if rows.empty:
        raise ValueError(f"No parse-ok cache rows for {method_id}")
    rows = rows.sort_values("row_index")
    y = task.y.iloc[rows["row_index"].astype(int).to_numpy()].reset_index(drop=True).to_numpy().astype(int)
    proba = rows[["p0", "p1"]].to_numpy(dtype=float)
    split = make_split(len(y), budget, seed, fractions)
    y_cal = y[split.cal]
    q = conformal_quantile(1.0 - proba[split.cal, y_cal], 1.0 - coverage)
    if not np.isfinite(q):
        q = 1.0
    audit_proba = proba[split.audit]
    y_audit = y[split.audit]
    sizes = _classification_set_sizes(audit_proba, q)
    covered = _classification_covered(audit_proba, y_audit, q)
    pred = np.argmax(audit_proba, axis=1)
    out = distribution_from_sizes(sizes)
    out.update(
        {
            "coverage": float(np.mean(covered)),
            "point_risk": float(1.0 - accuracy_score(y_audit, pred)),
            "conformal_q": float(q),
            "parse_failure_rate": float(1.0 - len(rows) / max(1, len(cache[(cache["method_id"].astype(str) == method_id) & (cache["row_index"].astype(int).isin(final_rows))]))),
        }
    )
    return out


def reference_distribution(
    *,
    task: TaskData,
    method_id: str,
    budget: int,
    coverage: float,
    seed: int,
    fractions: dict[str, float],
) -> dict[str, float | int]:
    split = make_split(len(task.y), budget, seed, fractions)
    x_fit = task.X.iloc[split.fit]
    y_fit = task.y.iloc[split.fit]
    x_cal = task.X.iloc[split.cal]
    y_cal = task.y.iloc[split.cal].to_numpy().astype(int)
    x_audit = task.X.iloc[split.audit]
    y_audit = task.y.iloc[split.audit].to_numpy().astype(int)
    model = build_method(method_id, task.task_type, x_fit, seed).fit(x_fit, y_fit)
    proba_cal = model.predict_proba(x_cal)
    y_cal = np.clip(y_cal, 0, proba_cal.shape[1] - 1)
    q = conformal_quantile(1.0 - proba_cal[np.arange(len(y_cal)), y_cal], 1.0 - coverage)
    proba = model.predict_proba(x_audit)
    sizes = _classification_set_sizes(proba, q)
    covered = _classification_covered(proba, y_audit, q)
    pred = np.argmax(proba, axis=1)
    out = distribution_from_sizes(sizes)
    out.update(
        {
            "coverage": float(np.mean(covered)),
            "point_risk": float(1.0 - accuracy_score(y_audit, pred)),
            "conformal_q": float(q),
            "parse_failure_rate": 0.0,
        }
    )
    return out


def load_frontier_rows(run_dir: Path, frontier_csv: Path) -> pd.DataFrame:
    frame = pd.read_csv(run_dir / frontier_csv)
    frame = frame[
        frame["run_label"].isin(RUNS)
        & frame["method_family"].isin(FRONTIER_METHOD_FAMILIES)
        & frame["frontier_status"].astype(str).eq("attained")
    ].copy()
    frame["dataset_name"] = frame["run_label"].map(DISPLAY_NAMES)
    return frame


def summarize_task(seed_level: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["dataset_name", "run_label", "method_family", "coverage_target"]
    rows = []
    for keys, group in seed_level.groupby(group_cols, dropna=False):
        row = {col: value for col, value in zip(group_cols, keys)}
        archived = pd.to_numeric(group["archived_frontier_set_size"], errors="coerce")
        recomputed = pd.to_numeric(group["mean_set_size"], errors="coerce")
        gap = (recomputed - archived).abs()
        row.update(
            {
                "n_cells": int(len(group)),
                "total_audit_n": int(group["audit_n"].sum()),
                "mean_empty_set_rate": float(group["empty_set_rate"].mean()),
                "mean_one_label_rate": float(group["one_label_rate"].mean()),
                "mean_two_label_rate": float(group["two_label_rate"].mean()),
                "mean_coverage": float(group["coverage"].mean()),
                "mean_set_size": float(group["mean_set_size"].mean()),
                "mean_archived_frontier_set_size": float(archived.mean()),
                "mean_abs_recompute_gap": float(gap.mean()),
                "max_abs_recompute_gap": float(gap.max()),
                "max_empty_set_rate": float(group["empty_set_rate"].max()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def write_markdown(task_summary: pd.DataFrame, path: Path) -> None:
    lines = [
        "# Prediction-Set Size Distribution Diagnostics",
        "",
        "Purpose: report practical split-conformal set-size composition for frontier-selected methods.",
        "Rates are recomputed from cached LLM probabilities or refitted reference learners using the archived split protocol.",
        "The archived frontier mean is the source value used by the manuscript frontier/ESS calculations; recomputation-gap columns distinguish exact cache replay from refitted diagnostic rows.",
        "",
        "| Task | Family | Coverage | Empty | One label | Two labels | Recomputed mean | Archived mean | Mean gap | Coverage | Cells |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in task_summary.sort_values(["dataset_name", "method_family", "coverage_target"]).itertuples(index=False):
        lines.append(
            "| {task} | {family} | {coverage:.1f} | {empty:.3f} | {one:.3f} | {two:.3f} | {size:.3f} | {archived:.3f} | {gap:.3f} | {cov:.3f} | {cells} |".format(
                task=row.dataset_name,
                family=row.method_family,
                coverage=float(row.coverage_target),
                empty=float(row.mean_empty_set_rate),
                one=float(row.mean_one_label_rate),
                two=float(row.mean_two_label_rate),
                size=float(row.mean_set_size),
                archived=float(row.mean_archived_frontier_set_size),
                gap=float(row.mean_abs_recompute_gap),
                cov=float(row.mean_coverage),
                cells=int(row.n_cells),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default=".")
    parser.add_argument(
        "--frontier-csv",
        default="artifacts/frontier_versions_v16_adult_s10/frontier_versions_seed_level.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/prediction_set_distribution_20260610",
    )
    parser.add_argument("--acs-csv", default=None)
    parser.add_argument(
        "--run-labels",
        nargs="+",
        default=None,
        help="Optional run labels to include, for targeted recomputation.",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    output = (run_dir / args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    frontier = load_frontier_rows(run_dir, Path(args.frontier_csv))
    if args.run_labels:
        frontier = frontier[frontier["run_label"].isin(args.run_labels)].copy()
        if frontier.empty:
            raise ValueError(f"No frontier rows matched --run-labels {args.run_labels}")
    rows = []
    task_cache: dict[str, tuple[TaskData, list[int], pd.DataFrame]] = {}
    fractions = {"fit": 0.5, "calibration": 0.25, "selection": 0.25}

    for item in frontier.itertuples(index=False):
        config = RUNS[item.run_label]
        if item.run_label not in task_cache:
            source_dir = resolve_source_dir(run_dir, config)
            manifest = read_json(source_dir / "raw_outputs" / config["dataset_manifest"])
            n_rows = int(manifest.get("n_rows", manifest.get("optimization_rows", 0) + manifest.get("final_rows", 0)))
            optimization_rows = int(manifest["optimization_rows"])
            final_n = int(manifest["final_rows"])
            task = build_task(config["kind"], n_rows, args.acs_csv)
            rows_idx = final_row_indices(len(task.y), optimization_rows, final_n)
            cache = pd.read_csv(source_dir / "raw_outputs" / config["cache_name"])
            task_cache[item.run_label] = (task, rows_idx, cache)

        full_task, final_rows, cache = task_cache[item.run_label]
        budget = int(item.budget)
        coverage = float(item.coverage_target)
        seed = int(item.seed)
        if not finite_conformal_quantile_available(budget, coverage, fractions):
            continue
        method_id = str(item.frontier_method)
        if method_id.startswith("cached_llm"):
            metrics = llm_distribution(
                task=full_task,
                cache=cache,
                method_id=method_id,
                final_rows=final_rows,
                budget=budget,
                coverage=coverage,
                seed=seed,
                fractions=fractions,
            )
        else:
            metrics = reference_distribution(
                task=final_task(full_task, final_rows),
                method_id=method_id,
                budget=budget,
                coverage=coverage,
                seed=seed,
                fractions=fractions,
            )
        rows.append(
            {
                "dataset_name": item.dataset_name,
                "run_label": item.run_label,
                "seed": seed,
                "budget": budget,
                "coverage_target": coverage,
                "method_family": item.method_family,
                "frontier_method": method_id,
                "archived_frontier_set_size": float(item.frontier_set_size),
                **metrics,
            }
        )

    seed_level = pd.DataFrame(rows)
    task_summary = summarize_task(seed_level)
    seed_level.to_csv(output / "prediction_set_distribution_seed_level.csv", index=False)
    task_summary.to_csv(output / "prediction_set_distribution_task_summary.csv", index=False)
    write_markdown(task_summary, output / "prediction_set_distribution_summary.md")
    print(
        {
            "status": "ok",
            "output_dir": str(output),
            "n_seed_level_rows": int(len(seed_level)),
            "n_task_summary_rows": int(len(task_summary)),
        }
    )


if __name__ == "__main__":
    main()
