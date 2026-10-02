# Preregistered Structured-Decoding Amendment

## Reason for Amendment

The original susceptibility-audit protocol completed 629 of 700 jobs. The
remaining 71 repeatedly failed the exact-text decision parser despite three
unchanged queue passes. No retained decision choices, effect sizes, model-task
comparisons, or interim statistics were inspected before this amendment.

The incomplete artifact is frozen as
`results/susceptibility_audit_frozen_incomplete_629.jsonl`, with SHA-256:

`f46ed11f142d05f459b7603d174dbc732d4c6789eaa6057c9b2c26323663237c`

## Amendment

The scientific design is unchanged: seven models, five conditions, twenty
paired seeds, six decisions, the same post pools, exposure construction,
persona, model sampling settings, and decision wording.

Only the decision-output interface changes. Every decision is re-elicited
through Ollama's native JSON-schema constrained decoding and must return one
field:

`{"recommendation": "A" | "B" | "C"}`

This amendment is applied uniformly to all 700 jobs and all 4,200 decisions.
No decision answer from the incomplete run is reused.

Seeds 990--999 are reserved for compatibility smoke tests and are excluded
from every retained analysis.

For the 629 completed jobs, the exact frozen feed trajectory, including posts,
order, and recorded reactions, is reused. For the 71 jobs with no retained
record, the trajectory is regenerated under the original frozen exposure and
reaction protocol before structured decisions are requested. Trajectory source
is recorded for every job.

## Analysis

The original frozen held-out analysis is unchanged:

- primary Spearman association between five-post audit effects and full-feed
  effects across 18 held-out model-task cells;
- 100,000 within-model task-label permutations;
- validation requires positive rho and one-sided `p < 0.05`;
- all secondary prediction, classification, baseline-shift, confidence-
  interval, positive, null, and compliance results are reported.

The structured run is a complete replacement dataset. It is never combined
with old decision outcomes. The incomplete protocol and its failure counts are
reported as a transparent implementation limitation.
