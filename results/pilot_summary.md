# Pilot Summary
## 01 — Algorithm fingerprint
- N samples: **135**, n_layers: 24, hidden: 1024
- Conditions: ['engagement_max', 'random', 'recency']
- Chance accuracy: 0.333
- **Best layer: L12**, balanced accuracy = **0.978**
- Margin above chance: +0.644
- Confusion matrix (rows = true, cols = pred): `[[42, 2, 1], [0, 44, 1], [0, 0, 45]]`
- Figure: `results/figures/01_fingerprint_accuracy_vs_layer.png`

## 02 — Drift geometry
- Best layer (inherited): L12
- Variance explained, top 5 PCs: ['0.308', '0.103', '0.075', '0.054', '0.042']
- Linear drift (PC1 > 50%): False
- Figures: `02_geometry_pc1_pc2.png`, `02_geometry_pc1_by_turn.png`

## 03 — Behavioral prediction
- N (PC1_t, dstance) pairs: **90**
- Horizon H = 5 turns
- **R^2 = 0.000**, slope = +0.001, intercept = +0.076
- Figure: `03_behavior_pc1_predicts_stance.png`

## 04 — Ji-Ma comparison
- Best layer (inherited): L12
- remote_work: cos(empirical drift PC1, static stance vec) = +0.047  (|.|=0.047)
- Figure: `04_jima_alignment.png`

## Verdict
- **Pilot shows separation.** Feed-condition is decodable above chance margin. Scale up.
