# Susceptibility-Audit Results

Records analyzed: 700
Holdout complete: True
Primary validates: True

## Primary Held-Out Validation

- Spearman rho: 0.855
- One-sided within-model permutation p: 0.000249998
- Audit MAE: 0.167
- Zero-predictor MAE: 0.436
- MAE improvement: 0.269
- Directional concordance: 12/13
- Balanced accuracy: 0.846

## Model-Task Cells

| Role | Model | Task | n audit/full | Audit effect [95% CI] | Full effect [95% CI] | Abs. error |
|---|---|---|---:|---:|---:|---:|
| holdout | lfm2.5:8b-a1b-q4_K_M | work_policy | 20/20 | 0.000 [0.000, 0.000] | 0.050 [0.000, 0.150] | 0.050 |
| holdout | lfm2.5:8b-a1b-q4_K_M | office_investment | 20/20 | 0.200 [0.000, 0.450] | -0.300 [-0.650, 0.000] | 0.500 |
| holdout | lfm2.5:8b-a1b-q4_K_M | hiring_geography | 20/20 | 0.000 [-0.150, 0.150] | -0.150 [-0.300, 0.000] | 0.150 |
| holdout | lfm2.5:8b-a1b-q4_K_M | performance_policy | 20/20 | -0.450 [-0.800, -0.150] | -0.600 [-1.000, -0.200] | 0.150 |
| holdout | lfm2.5:8b-a1b-q4_K_M | exception_policy | 20/20 | 0.050 [0.000, 0.150] | 0.200 [0.050, 0.400] | 0.150 |
| holdout | lfm2.5:8b-a1b-q4_K_M | team_design | 20/20 | -0.150 [-0.300, 0.000] | 0.000 [-0.150, 0.150] | 0.150 |
| holdout | granite4:7b-a1b-h | work_policy | 20/20 | -0.950 [-1.000, -0.850] | -1.000 [-1.000, -1.000] | 0.050 |
| holdout | granite4:7b-a1b-h | office_investment | 20/20 | -0.250 [-0.500, 0.050] | -0.550 [-0.800, -0.300] | 0.300 |
| holdout | granite4:7b-a1b-h | hiring_geography | 20/20 | -0.500 [-0.700, -0.300] | -0.300 [-0.500, -0.100] | 0.200 |
| holdout | granite4:7b-a1b-h | performance_policy | 20/20 | -0.800 [-1.000, -0.600] | -0.900 [-1.250, -0.550] | 0.100 |
| holdout | granite4:7b-a1b-h | exception_policy | 20/20 | 0.000 [0.000, 0.000] | -0.050 [-0.150, 0.000] | 0.050 |
| holdout | granite4:7b-a1b-h | team_design | 20/20 | -0.900 [-1.000, -0.750] | -0.900 [-1.000, -0.750] | 0.000 |
| holdout | MichelRosselli/apertus:8b-instruct-2509-q4_k_m | work_policy | 20/20 | -0.900 [-1.000, -0.750] | -0.850 [-1.000, -0.700] | 0.050 |
| holdout | MichelRosselli/apertus:8b-instruct-2509-q4_k_m | office_investment | 20/20 | -0.650 [-0.950, -0.300] | -0.850 [-1.000, -0.700] | 0.200 |
| holdout | MichelRosselli/apertus:8b-instruct-2509-q4_k_m | hiring_geography | 20/20 | -0.150 [-0.300, 0.000] | -0.300 [-0.500, -0.100] | 0.150 |
| holdout | MichelRosselli/apertus:8b-instruct-2509-q4_k_m | performance_policy | 20/20 | 0.050 [-0.100, 0.200] | 0.350 [0.150, 0.550] | 0.300 |
| holdout | MichelRosselli/apertus:8b-instruct-2509-q4_k_m | exception_policy | 20/20 | 0.000 [0.000, 0.000] | -0.150 [-0.300, 0.000] | 0.150 |
| holdout | MichelRosselli/apertus:8b-instruct-2509-q4_k_m | team_design | 20/20 | -0.050 [-0.150, 0.000] | -0.350 [-0.550, -0.150] | 0.300 |
| development | glm-4.7-flash:q4_K_M | work_policy | 20/20 | -0.300 [-0.500, -0.100] | -0.100 [-0.250, 0.000] | 0.200 |
| development | glm-4.7-flash:q4_K_M | office_investment | 20/20 | -0.250 [-0.550, 0.050] | -0.250 [-0.500, 0.000] | 0.000 |
| development | glm-4.7-flash:q4_K_M | hiring_geography | 20/20 | 0.150 [0.000, 0.300] | 0.050 [-0.100, 0.200] | 0.100 |
| development | glm-4.7-flash:q4_K_M | performance_policy | 20/20 | -0.150 [-0.450, 0.150] | -0.200 [-0.450, 0.050] | 0.050 |
| development | glm-4.7-flash:q4_K_M | exception_policy | 20/20 | -0.050 [-0.150, 0.000] | 0.000 [0.000, 0.000] | 0.050 |
| development | glm-4.7-flash:q4_K_M | team_design | 20/20 | -0.100 [-0.250, 0.000] | -0.050 [-0.150, 0.000] | 0.050 |
| development | gpt-oss:20b | work_policy | 20/20 | -0.450 [-0.650, -0.250] | -0.700 [-0.900, -0.500] | 0.250 |
| development | gpt-oss:20b | office_investment | 20/20 | -1.000 [-1.150, -0.850] | -0.950 [-1.000, -0.850] | 0.050 |
| development | gpt-oss:20b | hiring_geography | 20/20 | -0.800 [-0.950, -0.600] | -0.900 [-1.000, -0.750] | 0.100 |
| development | gpt-oss:20b | performance_policy | 20/20 | -0.350 [-0.550, -0.150] | -0.200 [-0.400, -0.050] | 0.150 |
| development | gpt-oss:20b | exception_policy | 20/20 | -0.600 [-0.800, -0.400] | -0.950 [-1.000, -0.850] | 0.350 |
| development | gpt-oss:20b | team_design | 20/20 | -0.900 [-1.100, -0.700] | -0.300 [-0.500, -0.100] | 0.600 |
| development | olmo-3:7b-instruct-q4_K_M | work_policy | 20/20 | -0.950 [-1.000, -0.850] | -0.500 [-0.700, -0.300] | 0.450 |
| development | olmo-3:7b-instruct-q4_K_M | office_investment | 20/20 | -0.850 [-1.051, -0.600] | -1.800 [-2.000, -1.550] | 0.950 |
| development | olmo-3:7b-instruct-q4_K_M | hiring_geography | 20/20 | -0.950 [-1.000, -0.850] | -1.000 [-1.000, -1.000] | 0.050 |
| development | olmo-3:7b-instruct-q4_K_M | performance_policy | 20/20 | -0.550 [-0.750, -0.350] | -1.200 [-1.400, -1.050] | 0.650 |
| development | olmo-3:7b-instruct-q4_K_M | exception_policy | 20/20 | -0.200 [-0.400, -0.050] | -0.200 [-0.400, -0.050] | 0.000 |
| development | olmo-3:7b-instruct-q4_K_M | team_design | 20/20 | -0.600 [-0.800, -0.400] | -0.250 [-0.450, -0.100] | 0.350 |
| development | nemotron-3-nano:4b | work_policy | 20/20 | -1.000 [-1.000, -1.000] | -0.950 [-1.000, -0.850] | 0.050 |
| development | nemotron-3-nano:4b | office_investment | 20/20 | -0.650 [-0.900, -0.400] | -0.800 [-1.150, -0.500] | 0.150 |
| development | nemotron-3-nano:4b | hiring_geography | 20/20 | -0.600 [-0.850, -0.350] | -0.550 [-0.750, -0.350] | 0.050 |
| development | nemotron-3-nano:4b | performance_policy | 20/20 | -0.200 [-0.400, -0.050] | -0.650 [-0.900, -0.400] | 0.450 |
| development | nemotron-3-nano:4b | exception_policy | 20/20 | -0.050 [-0.150, 0.000] | -0.050 [-0.200, 0.100] | 0.000 |
| development | nemotron-3-nano:4b | team_design | 20/20 | -0.900 [-1.100, -0.650] | -0.900 [-1.000, -0.750] | 0.000 |
