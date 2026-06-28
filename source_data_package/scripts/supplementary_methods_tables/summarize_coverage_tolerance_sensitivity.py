from __future__ import annotations

import argparse
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


def parse_mapping(values: list[str]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected label=path, got {value!r}")
        label, raw_path = value.split("=", maxsplit=1)
        label = label.strip()
        if not label:
            raise ValueError(f"Empty label in {value!r}")
        out[label] = Path(raw_path).resolve()
    return out


def load_frontier_summary(label: str, directory: Path) -> pd.DataFrame:
    path = directory / "frontier_versions_seed_level.csv"
    frame = pd.read_csv(path)
    frame = frame.loc[
        frame["run_label"].isin(RUN_LABELS)
        & frame["method_family"].isin({"llm_prompt_policy_frontier", "reference"})
    ].copy()
    frame["dataset_name"] = frame["run_label"].map(RUN_LABELS)
    frame["attained"] = frame["frontier_status"].astype(str).eq("attained")
    summary = (
        frame.groupby(["dataset_name", "method_family", "coverage_target"], dropna=False)
        .agg(
            n_cells=("frontier_status", "size"),
            n_attained=("attained", "sum"),
            mean_frontier_set_size=("frontier_set_size", "mean"),
            median_frontier_set_size=("frontier_set_size", "median"),
            mean_coverage=("coverage", "mean"),
            mean_point_risk=("point_risk", "mean"),
        )
        .reset_index()
    )
    summary["attainment_rate"] = summary["n_attained"] / summary["n_cells"]
    combined = (
        frame.assign(dataset_name="Combined")
        .groupby(["dataset_name", "method_family", "coverage_target"], dropna=False)
        .agg(
            n_cells=("frontier_status", "size"),
            n_attained=("attained", "sum"),
            mean_frontier_set_size=("frontier_set_size", "mean"),
            median_frontier_set_size=("frontier_set_size", "median"),
            mean_coverage=("coverage", "mean"),
            mean_point_risk=("point_risk", "mean"),
        )
        .reset_index()
    )
    combined["attainment_rate"] = combined["n_attained"] / combined["n_cells"]
    out = pd.concat([summary, combined], ignore_index=True)
    out.insert(0, "tolerance_label", label)
    return out


def load_ess_summary(label: str, directory: Path) -> pd.DataFrame:
    path = directory / "ess_comparison_seed_level.csv"
    frame = pd.read_csv(path)
    frame = frame.loc[frame["run_label"].isin(RUN_LABELS)].copy()
    frame["dataset_name"] = frame["run_label"].map(RUN_LABELS)
    frame["disagree"] = frame["ess_brackets_disagree"].astype(str).str.lower().isin({"true", "1", "yes"})
    summary = (
        frame.groupby(["dataset_name", "coverage_target"], dropna=False)
        .agg(n_cells=("disagree", "size"), n_disagree=("disagree", "sum"))
        .reset_index()
    )
    summary["disagreement_rate"] = summary["n_disagree"] / summary["n_cells"]
    combined = (
        frame.assign(dataset_name="Combined")
        .groupby(["dataset_name", "coverage_target"], dropna=False)
        .agg(n_cells=("disagree", "size"), n_disagree=("disagree", "sum"))
        .reset_index()
    )
    combined["disagreement_rate"] = combined["n_disagree"] / combined["n_cells"]
    out = pd.concat([summary, combined], ignore_index=True)
    out.insert(0, "tolerance_label", label)
    return out


def write_markdown(frontier: pd.DataFrame, ess: pd.DataFrame, path: Path) -> None:
    combined_frontier = frontier.loc[frontier["dataset_name"].eq("Combined")].copy()
    combined_ess = ess.loc[ess["dataset_name"].eq("Combined")].copy()
    lines = [
        "# Coverage-Tolerance Sensitivity",
        "",
        "Purpose: assess whether the valid-frontier and ESS conclusions are artifacts of the audit-admissibility tolerance.",
        "Each row recomputes frontiers from the same all-policy results with a different tolerance in the rule coverage >= target - tolerance.",
        "",
        "## Combined Frontier Summary",
        "",
        "| Tolerance | Family | Coverage | Attained | Mean set size | Mean coverage | Mean point risk |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in combined_frontier.sort_values(["tolerance_label", "method_family", "coverage_target"]).itertuples(index=False):
        lines.append(
            "| {tol} | {family} | {cov:.1f} | {attained}/{cells} | {size:.3f} | {coverage:.3f} | {risk:.3f} |".format(
                tol=row.tolerance_label,
                family=row.method_family,
                cov=float(row.coverage_target),
                attained=int(row.n_attained),
                cells=int(row.n_cells),
                size=float(row.mean_frontier_set_size),
                coverage=float(row.mean_coverage),
                risk=float(row.mean_point_risk),
            )
        )
    lines.extend(
        [
            "",
            "## Combined R-ESS/FESS Disagreement",
            "",
            "| Tolerance | Coverage | Disagreements | Rate |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in combined_ess.sort_values(["tolerance_label", "coverage_target"]).itertuples(index=False):
        lines.append(
            "| {tol} | {cov:.1f} | {k}/{n} | {rate:.3f} |".format(
                tol=row.tolerance_label,
                cov=float(row.coverage_target),
                k=int(row.n_disagree),
                n=int(row.n_cells),
                rate=float(row.disagreement_rate),
            )
        )
    lines.extend(
        [
            "",
            "Interpretation: tolerance changes attainment and some bracket counts, so it should be treated as an explicit audit parameter.",
            "The main manuscript uses tolerance 0.03; ACS-complete analyses should rerun this table before re-review.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frontier-dir", action="append", required=True, help="label=directory")
    parser.add_argument("--ess-dir", action="append", required=True, help="label=directory")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    frontier_dirs = parse_mapping(args.frontier_dir)
    ess_dirs = parse_mapping(args.ess_dir)
    if set(frontier_dirs) != set(ess_dirs):
        raise ValueError("frontier-dir and ess-dir labels must match")

    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    frontier = pd.concat(
        [load_frontier_summary(label, directory) for label, directory in frontier_dirs.items()],
        ignore_index=True,
    )
    ess = pd.concat(
        [load_ess_summary(label, directory) for label, directory in ess_dirs.items()],
        ignore_index=True,
    )
    frontier.to_csv(output / "coverage_tolerance_frontier_summary.csv", index=False)
    ess.to_csv(output / "coverage_tolerance_ess_summary.csv", index=False)
    write_markdown(frontier, ess, output / "coverage_tolerance_sensitivity_summary.md")
    print(
        {
            "status": "ok",
            "output_dir": str(output),
            "n_frontier_rows": int(len(frontier)),
            "n_ess_rows": int(len(ess)),
        }
    )


if __name__ == "__main__":
    main()
