from __future__ import annotations

import argparse
import math
from pathlib import Path

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


def load_real_rows(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    out = frame.loc[frame["run_label"].isin(RUN_LABELS)].copy()
    out["dataset_name"] = out["run_label"].map(RUN_LABELS)
    out["disagree"] = out["ess_brackets_disagree"].astype(bool).astype(int)
    if out.empty:
        raise ValueError("No real schema budget-grid rows found.")
    return out


def wilson_interval(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n <= 0:
        return (math.nan, math.nan)
    phat = k / n
    denom = 1.0 + z * z / n
    center = (phat + z * z / (2.0 * n)) / denom
    half = z * math.sqrt((phat * (1.0 - phat) + z * z / (4.0 * n)) / n) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def seed_rate_summary(frame: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rates = (
        frame.groupby(group_cols + ["seed"], dropna=False)["disagree"]
        .mean()
        .rename("seed_disagreement_rate")
        .reset_index()
    )
    rows = []
    for values, sub in rates.groupby(group_cols, dropna=False):
        if not isinstance(values, tuple):
            values = (values,)
        vals = sub["seed_disagreement_rate"].astype(float)
        n = int(len(vals))
        mean = float(vals.mean())
        sd = float(vals.std(ddof=1)) if n > 1 else 0.0
        if n > 1:
            tcrit = T_CRIT_95.get(n - 1, 1.959963984540054)
            half = tcrit * sd / math.sqrt(n)
            low = max(0.0, mean - half)
            high = min(1.0, mean + half)
        else:
            low = math.nan
            high = math.nan
        row = {col: value for col, value in zip(group_cols, values)}
        row.update(
            {
                "seed_n": n,
                "seed_rate_mean": mean,
                "seed_rate_sd": sd,
                "seed_rate_min": float(vals.min()),
                "seed_rate_max": float(vals.max()),
                "seed_t95_low": low,
                "seed_t95_high": high,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(frame: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    cell = (
        frame.groupby(group_cols, dropna=False)
        .agg(n_cells=("disagree", "size"), n_disagree=("disagree", "sum"))
        .reset_index()
    )
    cell["disagreement_rate"] = cell["n_disagree"] / cell["n_cells"]
    intervals = cell.apply(lambda row: wilson_interval(int(row.n_disagree), int(row.n_cells)), axis=1)
    cell["wilson95_low"] = [low for low, _ in intervals]
    cell["wilson95_high"] = [high for _, high in intervals]
    return cell.merge(seed_rate_summary(frame, group_cols), on=group_cols, how="left")


def add_combined(frame: pd.DataFrame, by_coverage: bool) -> pd.DataFrame:
    out = frame.copy()
    combined = frame.copy()
    combined["dataset_name"] = "Combined"
    group_cols = ["dataset_name", "coverage_target"] if by_coverage else ["dataset_name"]
    return summarize(pd.concat([out, combined], ignore_index=True), group_cols)


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    frame = load_real_rows(Path(args.comparison_csv).resolve())

    by_task = add_combined(frame, by_coverage=False)
    by_task_coverage = add_combined(frame, by_coverage=True)
    by_task_budget = summarize(frame, ["dataset_name", "coverage_target", "target_budget"])

    write_csv(by_task, output / "ess_disagreement_uncertainty_by_task.csv")
    write_csv(by_task_coverage, output / "ess_disagreement_uncertainty_by_task_coverage.csv")
    write_csv(by_task_budget, output / "ess_disagreement_uncertainty_by_task_budget.csv")
    print(
        {
            "status": "ok",
            "output_dir": str(output),
            "n_rows": int(len(frame)),
            "n_disagreements": int(frame["disagree"].sum()),
        }
    )


if __name__ == "__main__":
    main()
