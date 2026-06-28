from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from scripts.recompute_schema_realdata_budget_grid import build_task
from scripts.run_experiments import finite_conformal_quantile_available
from scripts.summarize_prediction_set_distribution import (
    DISPLAY_NAMES,
    FRONTIER_METHOD_FAMILIES,
    RUNS,
    final_row_indices,
    final_task,
    load_frontier_rows,
    read_json,
    resolve_source_dir,
)
from src.conformal import _classification_covered, _classification_set_sizes, conformal_quantile, make_split
from src.data.loaders import TaskData
from src.models import build_method


def group_metrics(
    *,
    proba: np.ndarray,
    y: np.ndarray,
    groups: pd.Series,
    q: float,
) -> pd.DataFrame:
    sizes = _classification_set_sizes(proba, q)
    covered = _classification_covered(proba, y, q)
    pred = np.argmax(proba, axis=1)
    frame = pd.DataFrame(
        {
            "group": groups.astype(str).to_numpy(),
            "covered": covered.astype(float),
            "set_size": sizes.astype(int),
            "error": (pred != y).astype(float),
        }
    )
    rows: list[dict[str, object]] = []
    for group_value, sub in frame.groupby("group", dropna=False):
        n = int(len(sub))
        if n == 0:
            continue
        sizes_sub = sub["set_size"].to_numpy(dtype=int)
        rows.append(
            {
                "group": str(group_value),
                "audit_n": n,
                "coverage": float(sub["covered"].mean()),
                "mean_set_size": float(sizes_sub.mean()),
                "empty_set_rate": float(np.mean(sizes_sub == 0)),
                "one_label_rate": float(np.mean(sizes_sub == 1)),
                "two_label_rate": float(np.mean(sizes_sub == 2)),
                "point_risk": float(sub["error"].mean()),
            }
        )
    return pd.DataFrame(rows)


def llm_group_rows(
    *,
    task: TaskData,
    cache: pd.DataFrame,
    method_id: str,
    final_rows: list[int],
    budget: int,
    coverage: float,
    seed: int,
    fractions: dict[str, float],
) -> pd.DataFrame:
    rows = cache[
        (cache["method_id"].astype(str) == method_id)
        & (cache["row_index"].astype(int).isin(final_rows))
        & (cache["parse_status"].astype(str) == "ok")
    ].copy()
    if rows.empty:
        raise ValueError(f"No parse-ok cache rows for {method_id}")
    rows = rows.sort_values("row_index")
    row_index = rows["row_index"].astype(int).to_numpy()
    y = task.y.iloc[row_index].reset_index(drop=True).to_numpy().astype(int)
    groups = task.groups.iloc[row_index].reset_index(drop=True)
    proba = rows[["p0", "p1"]].to_numpy(dtype=float)
    split = make_split(len(y), budget, seed, fractions)
    y_cal = y[split.cal]
    q = conformal_quantile(1.0 - proba[split.cal, y_cal], 1.0 - coverage)
    if not np.isfinite(q):
        q = 1.0
    return group_metrics(
        proba=proba[split.audit],
        y=y[split.audit],
        groups=groups.iloc[split.audit].reset_index(drop=True),
        q=float(q),
    )


def reference_group_rows(
    *,
    task: TaskData,
    method_id: str,
    budget: int,
    coverage: float,
    seed: int,
    fractions: dict[str, float],
) -> pd.DataFrame:
    split = make_split(len(task.y), budget, seed, fractions)
    x_fit = task.X.iloc[split.fit]
    y_fit = task.y.iloc[split.fit]
    x_cal = task.X.iloc[split.cal]
    y_cal = task.y.iloc[split.cal].to_numpy().astype(int)
    x_audit = task.X.iloc[split.audit]
    y_audit = task.y.iloc[split.audit].to_numpy().astype(int)
    groups = task.groups.iloc[split.audit].reset_index(drop=True)
    model = build_method(method_id, task.task_type, x_fit, seed).fit(x_fit, y_fit)
    proba_cal = model.predict_proba(x_cal)
    y_cal = np.clip(y_cal, 0, proba_cal.shape[1] - 1)
    q = conformal_quantile(1.0 - proba_cal[np.arange(len(y_cal)), y_cal], 1.0 - coverage)
    proba = model.predict_proba(x_audit)
    return group_metrics(proba=proba, y=y_audit, groups=groups, q=float(q))


def summarize(seed_level: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["dataset_name", "run_label", "method_family", "coverage_target", "group"]
    rows: list[dict[str, object]] = []
    for keys, sub in seed_level.groupby(group_cols, dropna=False):
        row = {col: value for col, value in zip(group_cols, keys)}
        row.update(
            {
                "n_cells": int(len(sub)),
                "total_audit_n": int(pd.to_numeric(sub["audit_n"]).sum()),
                "mean_coverage": float(pd.to_numeric(sub["coverage"]).mean()),
                "mean_set_size": float(pd.to_numeric(sub["mean_set_size"]).mean()),
                "mean_empty_set_rate": float(pd.to_numeric(sub["empty_set_rate"]).mean()),
                "mean_one_label_rate": float(pd.to_numeric(sub["one_label_rate"]).mean()),
                "mean_two_label_rate": float(pd.to_numeric(sub["two_label_rate"]).mean()),
                "mean_point_risk": float(pd.to_numeric(sub["point_risk"]).mean()),
                "min_group_audit_n": int(pd.to_numeric(sub["audit_n"]).min()),
                "max_group_audit_n": int(pd.to_numeric(sub["audit_n"]).max()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


def write_markdown(summary: pd.DataFrame, path: Path) -> None:
    lines = [
        "# Group-Conditional Coverage And Set-Size Diagnostics",
        "",
        "Purpose: report descriptive group-conditional coverage, set size, and prediction-set composition for frontier-selected methods.",
        "These are audit diagnostics, not group-conditional conformal guarantees.",
        "",
        "| Task | Group | Family | Coverage | Total group audit n | Mean coverage | Mean set size | Empty | One label | Two labels | Mean point risk | Cells |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            "| {task} | {group} | {family} | {coverage:.1f} | {n} | {cov:.3f} | {size:.3f} | {empty:.3f} | {one:.3f} | {two:.3f} | {risk:.3f} | {cells} |".format(
                task=row.dataset_name,
                group=row.group,
                family=row.method_family,
                coverage=float(row.coverage_target),
                n=int(row.total_audit_n),
                cov=float(row.mean_coverage),
                size=float(row.mean_set_size),
                empty=float(row.mean_empty_set_rate),
                one=float(row.mean_one_label_rate),
                two=float(row.mean_two_label_rate),
                risk=float(row.mean_point_risk),
                cells=int(row.n_cells),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default=".")
    parser.add_argument(
        "--frontier-csv",
        default="artifacts/frontier_versions_v18_strongref_adult10_acs_complete/frontier_versions_seed_level.csv",
    )
    parser.add_argument("--output-dir", default="artifacts/group_conditional_diagnostics_v1_adult_acs_complete")
    parser.add_argument("--acs-csv", default=None)
    parser.add_argument(
        "--run-labels",
        nargs="+",
        default=["minimax_adult_schema_budgetgrid", "minimax_acs_schema_budgetgrid"],
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    output = (run_dir / args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    frontier = load_frontier_rows(run_dir, Path(args.frontier_csv))
    frontier = frontier[
        frontier["run_label"].isin(args.run_labels)
        & frontier["method_family"].isin(FRONTIER_METHOD_FAMILIES)
    ].copy()
    if frontier.empty:
        raise ValueError(f"No frontier rows matched --run-labels {args.run_labels}")

    fractions = {"fit": 0.5, "calibration": 0.25, "selection": 0.25}
    task_cache: dict[str, tuple[TaskData, list[int], pd.DataFrame]] = {}
    rows: list[dict[str, object]] = []
    for item in frontier.itertuples(index=False):
        config = RUNS[item.run_label]
        if item.run_label not in task_cache:
            source_dir = resolve_source_dir(run_dir, config)
            manifest = read_json(source_dir / "raw_outputs" / config["dataset_manifest"])
            n_rows = int(manifest.get("n_rows", manifest.get("optimization_rows", 0) + manifest.get("final_rows", 0)))
            task = build_task(config["kind"], n_rows, args.acs_csv)
            final_rows = final_row_indices(len(task.y), int(manifest["optimization_rows"]), int(manifest["final_rows"]))
            cache = pd.read_csv(source_dir / "raw_outputs" / config["cache_name"])
            task_cache[item.run_label] = (task, final_rows, cache)

        full_task, final_rows, cache = task_cache[item.run_label]
        budget = int(item.budget)
        coverage = float(item.coverage_target)
        seed = int(item.seed)
        if not finite_conformal_quantile_available(budget, coverage, fractions):
            continue
        method_id = str(item.frontier_method)
        if method_id.startswith("cached_llm"):
            group_frame = llm_group_rows(
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
            group_frame = reference_group_rows(
                task=final_task(full_task, final_rows),
                method_id=method_id,
                budget=budget,
                coverage=coverage,
                seed=seed,
                fractions=fractions,
            )
        for row in group_frame.to_dict(orient="records"):
            rows.append(
                {
                    "dataset_name": DISPLAY_NAMES[item.run_label],
                    "run_label": item.run_label,
                    "seed": seed,
                    "budget": budget,
                    "coverage_target": coverage,
                    "method_family": item.method_family,
                    "frontier_method": method_id,
                    "archived_frontier_set_size": float(item.frontier_set_size),
                    **row,
                }
            )

    seed_level = pd.DataFrame(rows)
    summary = summarize(seed_level)
    seed_level.to_csv(output / "group_conditional_diagnostics_seed_level.csv", index=False)
    summary.to_csv(output / "group_conditional_diagnostics_summary.csv", index=False)
    write_markdown(summary, output / "group_conditional_diagnostics_summary.md")
    print(
        {
            "status": "ok",
            "output_dir": str(output),
            "n_seed_level_rows": int(len(seed_level)),
            "n_summary_rows": int(len(summary)),
        }
    )


if __name__ == "__main__":
    main()
