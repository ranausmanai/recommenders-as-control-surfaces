# Cross-Interface Audit Results

Feed records: 700
Retrieval records: 700

## Confirmatory Tests

- Cross-interface rho: 0.750
- Exact one-sided p: 0.0331349
- Cross-interface validates: True
- Portable-audit rho: 0.631
- Portable-audit p: 0.0706349
- Portable audit validates: False
- Disclosure attenuation: 0.025
- Disclosure sign-flip p: 0.28125
- Disclosure reduces steering: False

## Model-Level Scores

| Model | Audit | Feed | Retrieval | Disclosed | Attenuation | Balanced shift |
|---|---:|---:|---:|---:|---:|---:|
| lfm2.5:8b-a1b-q4_K_M | -0.058 | -0.133 | -0.108 | -0.067 | -0.008 | 0.358 |
| granite4:7b-a1b-h | -0.567 | -0.617 | -0.817 | -0.858 | -0.042 | 0.242 |
| MichelRosselli/apertus:8b-instruct-2509-q4_k_m | -0.283 | -0.358 | -0.558 | -0.517 | 0.042 | 0.283 |
| glm-4.7-flash:q4_K_M | -0.117 | -0.092 | -0.333 | -0.317 | 0.017 | 0.117 |
| gpt-oss:20b | -0.683 | -0.667 | -0.533 | -0.525 | 0.008 | 0.133 |
| olmo-3:7b-instruct-q4_K_M | -0.683 | -0.825 | -1.017 | -0.842 | 0.175 | 0.483 |
| nemotron-3-nano:4b | -0.567 | -0.650 | -0.825 | -0.842 | -0.017 | 0.117 |
