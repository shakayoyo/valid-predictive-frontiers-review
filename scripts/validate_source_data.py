from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "source_data_package" / "source_data"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def approx(value: str | float, expected: float, tol: float = 5e-4) -> bool:
    return abs(float(value) - expected) <= tol


def one_row(rows: list[dict[str, str]], **criteria: str) -> dict[str, str]:
    matches = [
        row
        for row in rows
        if all(str(row.get(key)) == str(value) for key, value in criteria.items())
    ]
    require(len(matches) == 1, f"expected one row for {criteria}, found {len(matches)}")
    return matches[0]


def check_rank_disagreement() -> None:
    rows = read_rows(SOURCE / "figure_4" / "cluster_bootstrap_disagreement_summary.csv")
    row = one_row(rows, statistic="risk_frontier_rank_disagreement", scope="combined")
    require(int(row["n_cells"]) == 970, "risk/frontier ranking cell count changed")
    require(int(row["n_disagree"]) == 775, "risk/frontier ranking disagreement count changed")
    require(approx(row["observed_rate"], 775 / 970), "risk/frontier ranking rate changed")
    require(approx(row["task_bootstrap95_low"], 0.7127422600619195), "task bootstrap low changed")
    require(approx(row["task_bootstrap95_high"], 0.8877686712983065), "task bootstrap high changed")


def check_ess_disagreement() -> None:
    rows = read_rows(SOURCE / "figure_4" / "ess_disagreement_by_task.csv")
    row = one_row(rows, dataset_name="Combined")
    require(int(row["n_cells"]) == 941, "R-ESS/FESS cell count changed")
    require(int(row["n_disagree"]) == 488, "R-ESS/FESS disagreement count changed")
    require(approx(row["disagreement_rate"], 488 / 941), "R-ESS/FESS rate changed")
    require(approx(row["wilson95_low"], 0.48666210560273243), "R-ESS/FESS Wilson low changed")
    require(approx(row["wilson95_high"], 0.5503811461229966), "R-ESS/FESS Wilson high changed")


def check_qwen_table() -> None:
    rows = read_rows(SOURCE / "table_1" / "frontier_versions_summary.csv")
    expected = {
        ("qwen25_7b_anes_promptselect_s10", "llm_prompt_policy_frontier", "0.8"): 1.855,
        ("qwen25_7b_anes_promptselect_s10", "reference", "0.8"): 0.919,
        ("qwen25_7b_anes_promptselect_s10", "llm_prompt_policy_frontier", "0.9"): 1.847,
        ("qwen25_7b_anes_promptselect_s10", "reference", "0.9"): 1.037,
        ("qwen25_7b_adult_promptselect_s10", "llm_prompt_policy_frontier", "0.8"): 1.803,
        ("qwen25_7b_adult_promptselect_s10", "reference", "0.8"): 1.106,
        ("qwen25_7b_adult_promptselect_s10", "llm_prompt_policy_frontier", "0.9"): 1.945,
        ("qwen25_7b_adult_promptselect_s10", "reference", "0.9"): 1.316,
    }
    for (run_label, method_family, coverage_target), expected_value in expected.items():
        row = one_row(
            rows,
            run_label=run_label,
            method_family=method_family,
            coverage_target=coverage_target,
        )
        require(
            approx(row["mean_frontier_set_size"], expected_value, tol=5e-3),
            f"unexpected Qwen table value for {(run_label, method_family, coverage_target)}",
        )


def main() -> None:
    check_rank_disagreement()
    check_ess_disagreement()
    check_qwen_table()
    print("source-data validation passed")


if __name__ == "__main__":
    main()

