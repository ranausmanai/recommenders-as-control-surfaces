# arXiv v2 Metadata

## Existing Record

`arXiv:2606.00914`

Use **Replace**. Do not create a new arXiv record.

## Title

Counterfactual Evidence Audits Predict LLM-Agent Susceptibility to Ranked Context

## Authors

Rana Muhammad Usman

## Abstract

LLM agents increasingly decide from evidence assembled by upstream systems:
retrievers choose documents, recommenders choose posts, and memory systems
choose prior events. Existing evaluations usually hold this evidence fixed.
They can therefore miss a failure mode in which individually ordinary items
form a systematically one-sided context. Measuring this risk by replaying long
agent trajectories for every candidate model and task is expensive.

We introduce a counterfactual evidence audit: expose an agent to two mirrored
sets of five documents, measure the difference in six downstream decisions,
and use that contrast to predict its response to disjoint 45-document
contexts. The protocol was frozen before evaluating three held-out open-weight
model families. Across 18 held-out model-task cells, five-document effects
predict full-context effects with Spearman rho=.855 (p<.001 under within-model
permutation), reduce mean absolute prediction error by 62% relative to a
zero-effect predictor, and recover the direction of 12 of 13 material
full-context effects. A preregistered classifier reaches 1.00 precision and
.846 balanced accuracy. A reviewer-requested post-hoc baseline that predicts
each held-out cell from the corresponding task mean in four development
families is substantially weaker (MAE .369 versus .167).

We first establish what the audit measures. In 1,800 reviewer-motivated
controls, selecting one-sided ordinary items from a single matched pool shifts
Llama 3.2-3B (-0.68, 95% CI [-0.80,-0.54], Holm p=2e-5), while permuting the
identical items is null; a saturated Gemma 4 is null under both. Across seven
open-weight families, susceptibility to the interactive feed predicts
susceptibility to a static RAG-style dossier (rho=.750, exact p=.033).
However, direct five-document prediction of RAG narrowly misses significance
(p=.071), and a provenance warning does not reliably attenuate steering
(p=.281). A separate study of three deployed Codex agent tiers finds strong
audit-to-full ranking (rho=.951, p<.001) but no individually significant
45-document effect after Holm correction, revealing a robustness boundary
rather than universal vulnerability.

Within this single synthetic remote-work domain, the result supports a
domain-specific triage procedure, not a universal steering claim: short
counterfactual audits can identify susceptible open-weight model-task pairs,
but risk depends on the model, task, corpus, and agent harness. Evidence
selection must be evaluated as part of the agent.

## Comments

19 pages, 1 figure. Accepted at the NeurIPS 2026 Workshop on Foundations of
Language Model Security (FLMSec 2026). Substantially revised after peer review:
introduces a preregistered five-document counterfactual evidence audit, 1,800
matched controls, held-out validation across seven open-weight families,
feed-to-RAG transfer, a disclosure-defense test, and a three-tier Codex
boundary study; narrows unsupported universal and asymmetry claims.

## Replacement Reason

Substantially revised after peer review and new preregistered experiments. The
revision reframes the paper around a counterfactual evidence audit, adds
matched causal controls, held-out multi-family validation, RAG transfer, and
frontier-agent boundary tests, and updates the title and conclusions.

## Categories

Keep the existing primary category `cs.AI` unless arXiv permits and you want a
moderator-approved change. Retain cross-lists `cs.CL` and `cs.CR`.
