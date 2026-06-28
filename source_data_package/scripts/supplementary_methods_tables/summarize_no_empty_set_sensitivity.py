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


def load_seed_level(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for source_order, path in enumerate(paths):
        frame = pd.read_csv(path)
        frame["_source_order"] = source_order
        frames.append(frame)
    out = pd.concat(frames, ignore_index=True)
    out = out.loc[out["run_label"].isin(RUN_LABELS)].copy()
    out["dataset_name"] = out["run_label"].map(RUN_LABELS)

    latest_by_run = out.groupby("run_label")["_source_order"].transform("max")
    out = out.loc[out["_source_order"].eq(latest_by_run)].copy()
    for col in ["empty_set_rate", "one_label_rate", "two_label_rate", "mean_set_size", "coverage", "point_risk"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    # Practical display sensitivity: empty sets are counted as requiring at
    # least one displayed label. This is not a new conformal procedure.
    out["no_empty_display_set_size"] = out["mean_set_size"] + out["empty_set_rate"]
    out["no_empty_display_inflation"] = out["empty_set_rate"]
    return out


def summarize(frame: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    summary = (
        frame.groupby(group_cols, dropna=False)
        .agg(
            n_cells=("empty_set_rate", "size"),
            total_audit_n=("audit_n", "sum"),
            mean_empty_set_rate=("empty_set_rate", "mean"),
            max_empty_set_rate=("empty_set_rate", "max"),
            mean_one_label_rate=("one_label_rate", "mean"),
            mean_two_label_rate=("two_label_rate", "mean"),
            mean_set_size=("mean_set_size", "mean"),
            mean_no_empty_display_set_size=("no_empty_display_set_size", "mean"),
            mean_no_empty_display_inflation=("no_empty_display_inflation", "mean"),
            mean_coverage=("coverage", "mean"),
            mean_point_risk=("point_risk", "mean"),
        )
        .reset_index()
    )
    return summary


def add_combined(frame: pd.DataFrame) -> pd.DataFrame:
    task = summarize(frame, ["dataset_name", "run_label", "method_family", "coverage_target"])
    combined = frame.copy()
    combined["dataset_name"] = "Combined"
    combined["run_label"] = "combined"
    all_rows = summarize(combined, ["dataset_name", "run_label", "method_family", "coverage_target"])
    return pd.concat([task, all_rows], ignore_index=True)


def paired_family_gaps(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["dataset_name", "run_label", "coverage_target"]
    for key_values, sub in summary.groupby(keys, dropna=False):
        families = set(sub["method_family"])
        if {"llm_prompt_policy_frontier", "reference"}.issubset(families):
            llm = sub.loc[sub["method_family"].eq("llm_prompt_policy_frontier")].iloc[0]
            ref = sub.loc[sub["method_family"].eq("reference")].iloc[0]
            row = dict(zip(keys, key_values if isinstance(key_values, tuple) else (key_values,)))
            row.update(
                {
                    "llm_mean_set_size": float(llm.mean_set_size),
                    "reference_mean_set_size": float(ref.mean_set_size),
                    "raw_gap_llm_minus_reference": float(llm.mean_set_size - ref.mean_set_size),
                    "llm_no_empty_display_set_size": float(llm.mean_no_empty_display_set_size),
                    "reference_no_empty_display_set_size": float(ref.mean_no_empty_display_set_size),
                    "no_empty_gap_llm_minus_reference": float(
                        llm.mean_no_empty_display_set_size - ref.mean_no_empty_display_set_size
                    ),
                    "llm_empty_rate": float(llm.mean_empty_set_rate),
                    "reference_empty_rate": float(ref.mean_empty_set_rate),
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def write_markdown(summary: pd.DataFrame, gaps: pd.DataFrame, output_path: Path) -> None:
    combined = summary.loc[summary["dataset_name"].eq("Combined")].copy()
    combined_gaps = gaps.loc[gaps["dataset_name"].eq("Combined")].copy()
    lines = [
        "# No-Empty-Set Practical Sensitivity",
        "",
        "Purpose: quantify how much practical set-size summaries change if empty conformal prediction sets are counted as requiring at least one displayed label.",
        "This is a display/utility sensitivity, not a replacement conformal algorithm and not a new coverage guarantee.",
        "",
        "## Combined Composition and No-Empty Display Size",
        "",
        "| Family | Coverage | Empty rate | Mean set size | No-empty display size | Inflation | Coverage |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in combined.sort_values(["method_family", "coverage_target"]).itertuples(index=False):
        lines.append(
            "| {family} | {cov:.1f} | {empty:.3f} | {size:.3f} | {adj:.3f} | {infl:.3f} | {coverage:.3f} |".format(
                family=row.method_family,
                cov=float(row.coverage_target),
                empty=float(row.mean_empty_set_rate),
                size=float(row.mean_set_size),
                adj=float(row.mean_no_empty_display_set_size),
                infl=float(row.mean_no_empty_display_inflation),
                coverage=float(row.mean_coverage),
            )
        )
    lines.extend(
        [
            "",
            "## Combined LLM-Reference Gaps",
            "",
            "| Coverage | Raw gap | No-empty display gap | LLM empty rate | Reference empty rate |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for row in combined_gaps.sort_values("coverage_target").itertuples(index=False):
        lines.append(
            "| {cov:.1f} | {raw:.3f} | {adj:.3f} | {le:.3f} | {re:.3f} |".format(
                cov=float(row.coverage_target),
                raw=float(row.raw_gap_llm_minus_reference),
                adj=float(row.no_empty_gap_llm_minus_reference),
                le=float(row.llm_empty_rate),
                re=float(row.reference_empty_rate),
            )
        )
    lines.extend(
        [
            "",
            "Interpretation: empty sets are not driving the main frontier gap when the no-empty display adjustment is applied.",
            "ACS-complete analyses should rerun this table before external re-review.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-level-csv", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    frame = load_seed_level([Path(path).resolve() for path in args.seed_level_csv])
    summary = add_combined(frame)
    gaps = paired_family_gaps(summary)
    frame.to_csv(output / "no_empty_set_sensitivity_seed_level.csv", index=False)
    summary.to_csv(output / "no_empty_set_sensitivity_summary.csv", index=False)
    gaps.to_csv(output / "no_empty_set_sensitivity_family_gaps.csv", index=False)
    write_markdown(summary, gaps, output / "no_empty_set_sensitivity_summary.md")
    print(
        {
            "status": "ok",
            "output_dir": str(output),
            "n_seed_rows": int(len(frame)),
            "n_summary_rows": int(len(summary)),
            "n_gap_rows": int(len(gaps)),
        }
    )


if __name__ == "__main__":
    main()
