# Frontier and FESS Uncertainty Bands

Purpose: summarize uncertainty in continuous frontier gaps and in finite-grid FESS/R-ESS bracket locations.
Positive frontier gaps mean the LLM prompt-policy frontier has larger conformal prediction sets than the reference frontier.
Positive FESS--R-ESS order means the frontier-equivalent bracket is larger than the risk-equivalent bracket on the displayed finite budget grid.

## Combined Summaries

| Quantity | Mean | Task bootstrap 95% | Task-seed bootstrap 95% |
|---|---:|---:|---:|
| LLM minus reference set-size gap | 0.284 | 0.121 to 0.438 | 0.129 to 0.447 |
| FESS minus R-ESS bracket order | -18.835 | -35.614 to -4.266 | -36.324 to -3.899 |

## Coverage-Level Frontier Gaps

| Scope | Mean LLM-reference set-size gap | Task bootstrap 95% | Task-seed bootstrap 95% |
|---|---:|---:|---:|
| coverage_0.8 | 0.298 | 0.131 to 0.466 | 0.135 to 0.463 |
| coverage_0.9 | 0.268 | 0.122 to 0.419 | 0.116 to 0.423 |

Interpretation: cell-level t intervals by task, coverage, and budget are written separately for display bands.
Combined task/task-seed bootstrap intervals are intentionally wider and should carry inferential language.
