# No-Empty-Set Practical Sensitivity

Purpose: quantify how much practical set-size summaries change if empty conformal prediction sets are counted as requiring at least one displayed label.
This is a display/utility sensitivity, not a replacement conformal algorithm and not a new coverage guarantee.

## Combined Composition and No-Empty Display Size

| Family | Coverage | Empty rate | Mean set size | No-empty display size | Inflation | Coverage |
|---|---:|---:|---:|---:|---:|---:|
| llm_prompt_policy_frontier | 0.8 | 0.012 | 1.451 | 1.462 | 0.012 | 0.886 |
| llm_prompt_policy_frontier | 0.9 | 0.001 | 1.636 | 1.637 | 0.001 | 0.952 |
| reference | 0.8 | 0.033 | 1.158 | 1.190 | 0.033 | 0.839 |
| reference | 0.9 | 0.006 | 1.375 | 1.381 | 0.006 | 0.921 |

## Combined LLM-Reference Gaps

| Coverage | Raw gap | No-empty display gap | LLM empty rate | Reference empty rate |
|---:|---:|---:|---:|---:|
| 0.8 | 0.293 | 0.272 | 0.012 | 0.033 |
| 0.9 | 0.260 | 0.256 | 0.001 | 0.006 |

Interpretation: empty sets are not driving the main frontier gap when the no-empty display adjustment is applied.
This table is the ACS-complete rerun used for external re-review.
