# Group-Conditional Coverage And Set-Size Diagnostics

Purpose: report descriptive group-conditional coverage, set size, and prediction-set composition for frontier-selected methods.
These are audit diagnostics, not group-conditional conformal guarantees.

| Task | Group | Family | Coverage | Total group audit n | Mean coverage | Mean set size | Empty | One label | Two labels | Mean point risk | Cells |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ACS income | 1 | llm_prompt_policy_frontier | 0.8 | 65526 | 0.851 | 1.520 | 0.000 | 0.480 | 0.520 | 0.422 | 50 |
| ACS income | 2 | llm_prompt_policy_frontier | 0.8 | 5457 | 0.812 | 1.235 | 0.000 | 0.765 | 0.235 | 0.330 | 50 |
| ACS income | 3 | llm_prompt_policy_frontier | 0.8 | 563 | 0.900 | 1.122 | 0.000 | 0.878 | 0.122 | 0.209 | 50 |
| ACS income | 5 | llm_prompt_policy_frontier | 0.8 | 227 | 0.785 | 1.000 | 0.000 | 1.000 | 0.000 | 0.215 | 50 |
| ACS income | 6 | llm_prompt_policy_frontier | 0.8 | 17668 | 0.914 | 1.711 | 0.000 | 0.289 | 0.711 | 0.394 | 50 |
| ACS income | 7 | llm_prompt_policy_frontier | 0.8 | 430 | 0.785 | 1.008 | 0.000 | 0.992 | 0.008 | 0.221 | 50 |
| ACS income | 8 | llm_prompt_policy_frontier | 0.8 | 12103 | 0.897 | 1.310 | 0.000 | 0.690 | 0.310 | 0.207 | 50 |
| ACS income | 9 | llm_prompt_policy_frontier | 0.8 | 4426 | 0.838 | 1.384 | 0.000 | 0.616 | 0.384 | 0.343 | 50 |
| ACS income | 1 | llm_prompt_policy_frontier | 0.9 | 65526 | 0.915 | 1.664 | 0.000 | 0.336 | 0.664 | 0.382 | 50 |
| ACS income | 2 | llm_prompt_policy_frontier | 0.9 | 5457 | 0.880 | 1.485 | 0.000 | 0.515 | 0.485 | 0.355 | 50 |
| ACS income | 3 | llm_prompt_policy_frontier | 0.9 | 563 | 0.926 | 1.339 | 0.000 | 0.661 | 0.339 | 0.201 | 50 |
| ACS income | 5 | llm_prompt_policy_frontier | 0.9 | 227 | 0.879 | 1.256 | 0.000 | 0.744 | 0.256 | 0.306 | 50 |
| ACS income | 6 | llm_prompt_policy_frontier | 0.9 | 17668 | 0.925 | 1.752 | 0.000 | 0.248 | 0.752 | 0.433 | 50 |
| ACS income | 7 | llm_prompt_policy_frontier | 0.9 | 430 | 0.824 | 1.225 | 0.000 | 0.775 | 0.225 | 0.330 | 50 |
| ACS income | 8 | llm_prompt_policy_frontier | 0.9 | 12103 | 0.944 | 1.533 | 0.000 | 0.467 | 0.533 | 0.309 | 50 |
| ACS income | 9 | llm_prompt_policy_frontier | 0.9 | 4426 | 0.917 | 1.569 | 0.000 | 0.431 | 0.569 | 0.353 | 50 |
| ACS income | 1 | reference | 0.8 | 62807 | 0.798 | 1.137 | 0.000 | 0.862 | 0.138 | 0.266 | 48 |
| ACS income | 2 | reference | 0.8 | 5229 | 0.807 | 1.143 | 0.000 | 0.857 | 0.143 | 0.253 | 48 |
| ACS income | 3 | reference | 0.8 | 542 | 0.898 | 1.121 | 0.000 | 0.879 | 0.121 | 0.155 | 48 |
| ACS income | 5 | reference | 0.8 | 218 | 0.829 | 1.107 | 0.000 | 0.893 | 0.107 | 0.214 | 48 |
| ACS income | 6 | reference | 0.8 | 16954 | 0.795 | 1.147 | 0.000 | 0.852 | 0.147 | 0.273 | 48 |
| ACS income | 7 | reference | 0.8 | 411 | 0.849 | 1.085 | 0.003 | 0.910 | 0.088 | 0.175 | 48 |
| ACS income | 8 | reference | 0.8 | 11597 | 0.855 | 1.107 | 0.000 | 0.892 | 0.108 | 0.184 | 48 |
| ACS income | 9 | reference | 0.8 | 4242 | 0.789 | 1.123 | 0.001 | 0.875 | 0.124 | 0.264 | 48 |
| ACS income | 1 | reference | 0.9 | 64056 | 0.888 | 1.357 | 0.000 | 0.643 | 0.357 | 0.263 | 49 |
| ACS income | 2 | reference | 0.9 | 5335 | 0.901 | 1.384 | 0.000 | 0.616 | 0.384 | 0.257 | 49 |
| ACS income | 3 | reference | 0.9 | 551 | 0.955 | 1.344 | 0.000 | 0.656 | 0.344 | 0.143 | 49 |
| ACS income | 5 | reference | 0.9 | 222 | 0.900 | 1.295 | 0.000 | 0.705 | 0.295 | 0.242 | 49 |
| ACS income | 6 | reference | 0.9 | 17275 | 0.892 | 1.384 | 0.000 | 0.616 | 0.384 | 0.272 | 49 |
| ACS income | 7 | reference | 0.9 | 421 | 0.923 | 1.251 | 0.000 | 0.749 | 0.251 | 0.185 | 49 |
| ACS income | 8 | reference | 0.9 | 11832 | 0.912 | 1.255 | 0.000 | 0.745 | 0.255 | 0.188 | 49 |
| ACS income | 9 | reference | 0.9 | 4328 | 0.875 | 1.329 | 0.000 | 0.671 | 0.329 | 0.261 | 49 |
| Adult income | Female | llm_prompt_policy_frontier | 0.8 | 2262 | 0.925 | 1.075 | 0.006 | 0.914 | 0.081 | 0.099 | 60 |
| Adult income | Male | llm_prompt_policy_frontier | 0.8 | 4238 | 0.833 | 1.163 | 0.026 | 0.785 | 0.189 | 0.230 | 60 |
| Adult income | Female | llm_prompt_policy_frontier | 0.9 | 1652 | 0.960 | 1.228 | 0.000 | 0.772 | 0.228 | 0.096 | 48 |
| Adult income | Male | llm_prompt_policy_frontier | 0.9 | 3108 | 0.945 | 1.451 | 0.000 | 0.548 | 0.451 | 0.229 | 48 |
| Adult income | Female | reference | 0.8 | 2277 | 0.898 | 1.088 | 0.007 | 0.897 | 0.096 | 0.121 | 66 |
| Adult income | Male | reference | 0.8 | 4263 | 0.783 | 1.147 | 0.038 | 0.777 | 0.185 | 0.277 | 66 |
| Adult income | Female | reference | 0.9 | 1877 | 0.940 | 1.202 | 0.001 | 0.796 | 0.203 | 0.119 | 60 |
| Adult income | Male | reference | 0.9 | 3523 | 0.909 | 1.472 | 0.004 | 0.519 | 0.476 | 0.272 | 60 |
