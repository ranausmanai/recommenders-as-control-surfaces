# Reviewer-Control Results

Records analyzed: 1800

## Feed-Free Defaults

| Model | Topic | A/B/C | Modal | Share | Non-ceiling? |
|---|---|---:|---:|---:|---:|
| gemma4:e4b | access_policy | 50/0/0 | A | 1.00 | no |
| gemma4:e4b | ai_regulation | 0/50/0 | B | 1.00 | no |
| gemma4:e4b | deploy_security | 0/50/0 | B | 1.00 | no |
| gemma4:e4b | remote_work | 0/50/0 | B | 1.00 | no |
| gemma4:e4b | ubi | 0/50/0 | B | 1.00 | no |
| gemma4:e4b | vendor_security | 0/50/0 | B | 1.00 | no |
| llama3.2:3b | access_policy | 40/10/0 | A | 0.80 | yes |
| llama3.2:3b | ai_regulation | 0/0/50 | C | 1.00 | no |
| llama3.2:3b | deploy_security | 0/43/7 | B | 0.86 | yes |
| llama3.2:3b | remote_work | 0/9/41 | C | 0.82 | yes |
| llama3.2:3b | ubi | 0/50/0 | B | 1.00 | no |
| llama3.2:3b | vendor_security | 0/50/0 | B | 1.00 | no |
| qwen3.5:2b | access_policy | 36/14/0 | A | 0.72 | yes |
| qwen3.5:2b | ai_regulation | 10/34/6 | B | 0.68 | yes |
| qwen3.5:2b | deploy_security | 13/37/0 | B | 0.74 | yes |
| qwen3.5:2b | remote_work | 7/43/0 | B | 0.86 | yes |
| qwen3.5:2b | ubi | 16/34/0 | B | 0.68 | yes |
| qwen3.5:2b | vendor_security | 14/36/0 | B | 0.72 | yes |
| qwen3.5:9b | access_policy | 0/50/0 | B | 1.00 | no |
| qwen3.5:9b | ai_regulation | 0/50/0 | B | 1.00 | no |
| qwen3.5:9b | deploy_security | 0/36/14 | B | 0.72 | yes |
| qwen3.5:9b | remote_work | 0/50/0 | B | 1.00 | no |
| qwen3.5:9b | ubi | 0/50/0 | B | 1.00 | no |
| qwen3.5:9b | vendor_security | 0/50/0 | B | 1.00 | no |

## Primary Comparisons

| Experiment | Model | Comparison | n | Mean ordinal diff | 95% CI | Raw p | Holm p |
|---|---|---|---:|---:|---:|---:|---:|
| matched_selection | gemma4:e4b | matched_rto - matched_remote | 50 | 0.000 | [0.000, 0.000] | 1 | 1 |
| matched_selection | llama3.2:3b | matched_rto - matched_remote | 50 | -0.680 | [-0.800, -0.540] | 1e-05 | 2e-05 |
| pure_order | gemma4:e4b | order_rto_last - order_remote_last | 50 | 0.000 | [0.000, 0.000] | 1 | 1 |
| pure_order | llama3.2:3b | order_rto_last - order_remote_last | 50 | -0.060 | [-0.140, 0.000] | 0.2512 | 0.5024 |

## Default-Direction Asymmetry

| Model | n | Default/share | Non-ceiling? | RTO shift | Remote shift | Difference | Holm p |
|---|---:|---:|---:|---:|---:|---:|---:|
| gemma4:e4b | 50 | B/1.00 | no | 0.000 | 0.000 | 0.000 | 1 |
| llama3.2:3b | 50 | C/0.82 | yes | 0.500 | 0.180 | 0.320 | 0.07866 |

Dose-response wording is restricted to **dose-dependent trend**. These controls do not test or claim a precise onset threshold.
