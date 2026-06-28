# Coverage-Tolerance Sensitivity

Purpose: assess whether the valid-frontier and ESS conclusions are artifacts of the audit-admissibility tolerance.
Each row recomputes frontiers from the same all-policy results with a different tolerance in the rule coverage >= target - tolerance.

## Combined Frontier Summary

| Tolerance | Family | Coverage | Attained | Mean set size | Mean coverage | Mean point risk |
|---|---|---:|---:|---:|---:|---:|
| tol00 | llm_prompt_policy_frontier | 0.8 | 498/510 | 1.463 | 0.894 | 0.286 |
| tol00 | llm_prompt_policy_frontier | 0.9 | 436/440 | 1.658 | 0.960 | 0.294 |
| tol00 | reference | 0.8 | 451/520 | 1.180 | 0.854 | 0.219 |
| tol00 | reference | 0.9 | 399/450 | 1.409 | 0.934 | 0.225 |
| tol01 | llm_prompt_policy_frontier | 0.8 | 500/510 | 1.459 | 0.892 | 0.286 |
| tol01 | llm_prompt_policy_frontier | 0.9 | 436/440 | 1.643 | 0.956 | 0.288 |
| tol01 | reference | 0.8 | 459/520 | 1.176 | 0.851 | 0.220 |
| tol01 | reference | 0.9 | 404/450 | 1.399 | 0.931 | 0.225 |
| tol03 | llm_prompt_policy_frontier | 0.8 | 503/510 | 1.449 | 0.886 | 0.287 |
| tol03 | llm_prompt_policy_frontier | 0.9 | 438/440 | 1.634 | 0.953 | 0.287 |
| tol03 | reference | 0.8 | 478/520 | 1.150 | 0.838 | 0.221 |
| tol03 | reference | 0.9 | 418/450 | 1.368 | 0.920 | 0.225 |
| tol05 | llm_prompt_policy_frontier | 0.8 | 503/510 | 1.444 | 0.882 | 0.289 |
| tol05 | llm_prompt_policy_frontier | 0.9 | 439/440 | 1.619 | 0.947 | 0.287 |
| tol05 | reference | 0.8 | 495/520 | 1.130 | 0.826 | 0.223 |
| tol05 | reference | 0.9 | 436/450 | 1.336 | 0.909 | 0.225 |

## Combined R-ESS/FESS Disagreement

| Tolerance | Coverage | Disagreements | Rate |
|---|---:|---:|---:|
| tol00 | 0.8 | 270/498 | 0.542 |
| tol00 | 0.9 | 213/436 | 0.489 |
| tol01 | 0.8 | 272/500 | 0.544 |
| tol01 | 0.9 | 205/436 | 0.470 |
| tol03 | 0.8 | 280/503 | 0.557 |
| tol03 | 0.9 | 208/438 | 0.475 |
| tol05 | 0.8 | 271/503 | 0.539 |
| tol05 | 0.9 | 209/439 | 0.476 |

Interpretation: tolerance changes attainment and some bracket counts, so it should be treated as an explicit audit parameter.
The main manuscript uses tolerance 0.03; this table is the ACS-complete sensitivity used for re-review.
