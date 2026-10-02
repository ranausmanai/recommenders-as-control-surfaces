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
choose prior events. Existing evaluations usually hold this evidence fixed,
missing failures in which individually ordinary items form a systematically
one-sided context. We introduce a counterfactual evidence audit: expose an
agent to two mirrored sets of five documents, measure the difference in six
downstream decisions, and use that contrast to predict its response to
disjoint 45-document contexts. The protocol was frozen before testing three
held-out open-weight model families. Across 18 held-out model-task cells,
five-document effects predict full-context effects with Spearman rho=.855
(p<.001), reduce mean absolute prediction error by 62% relative to a
zero-effect predictor, and recover the direction of 12 of 13 material effects.
A reviewer-requested post-hoc task-mean baseline is also substantially weaker
(MAE .369 versus .167). Matched controls show that selecting one-sided
ordinary items, rather than merely reordering identical items, causes the
shift in a susceptible model. Across seven open-weight families,
susceptibility transfers from an interactive feed to a static RAG dossier
(rho=.750, exact p=.033), while a provenance warning does not reliably
mitigate it. A separate study of three deployed Codex agent tiers finds strong
audit-to-full ranking (rho=.951, p<.001) but no individually significant
full-context effect after correction. Within this single synthetic remote-work
domain, the result supports a domain-specific triage procedure, not a
universal steering claim: evidence selection must be evaluated as part of the
composed agent system.

## Comments

19 pages, 1 figure. Accepted at the NeurIPS 2026 Workshop on Foundations of
Language Model Security (FLMSec 2026). Substantially revised after peer review:
introduces a preregistered five-document counterfactual evidence audit, 1,800
matched controls, held-out validation across seven open-weight families,
feed-to-RAG transfer, a disclosure-defense test, and a three-tier Codex
boundary study; narrows unsupported universal and asymmetry claims.
Code and data:
https://github.com/ranausmanai/recommenders-as-control-surfaces/releases/tag/flmsec-2026-camera-ready

## Replacement Reason

Substantially revised after peer review and new preregistered experiments. The
revision reframes the paper around a counterfactual evidence audit, adds
matched causal controls, held-out multi-family validation, RAG transfer, and
frontier-agent boundary tests, and updates the title and conclusions.

## Categories

Keep the existing primary category `cs.AI` unless arXiv permits and you want a
moderator-approved change. Retain cross-lists `cs.CL` and `cs.CR`.
