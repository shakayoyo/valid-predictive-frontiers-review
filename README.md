# Valid Predictive Frontiers: Review Materials

This repository contains reviewer-facing code and source data for the manuscript
`Valid predictive frontiers for language-model predictions of human outcomes`.

The repository is intentionally compact. It contains the numerical source data
behind the main figures, main table, selected supplementary displays, and the
Python scripts used to generate those displays. Raw provider-specific language
model completions are not included; the released source data retain parsed
predictions, derived frontier summaries, equivalent-sample-size summaries, and
figure/table inputs needed to check the manuscript's central numerical claims.

## Contents

- `source_data_package/source_data/`: CSV and JSON source data for figures,
  tables, and supplementary summaries.
- `source_data_package/display_files/`: rendered figure files corresponding to
  the source data.
- `source_data_package/scripts/`: plotting and summary scripts used to build the
  displayed results.
- `scripts/validate_source_data.py`: lightweight claim checks against the
  included source data.
- `environment.yml` and `requirements.txt`: Python environment specifications.

## Quick Check

From the repository root:

```bash
python scripts/validate_source_data.py
```

The script checks the central counts used in the manuscript, including the
775/970 point-risk versus valid-set ranking disagreement, the 488/941
R-ESS/FESS disagreement, and the Qwen2.5-7B table values.

## Rebuilding Figures

Each figure directory under `source_data_package/scripts/` contains the script
used for that figure. For example:

```bash
python source_data_package/scripts/figure_2/plot_frontier_diagnostics.py \
  --input-dir source_data_package/source_data/figure_2 \
  --output-dir reproduced_figures/figure_2
```

The scripts are designed to run from a normal Python environment with the
packages listed in `requirements.txt`.

## Data Scope

The empirical datasets are public datasets distributed through standard Python
packages or public data interfaces. The language-model outputs in this
review-facing repository are cached derived data rather than a request to rerun
provider APIs. Provider-restricted raw text completions are excluded from this
repository; parsed outputs and derived numerical tables are included for
checking the reported results.

