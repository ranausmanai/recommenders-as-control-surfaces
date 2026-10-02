# NeurIPS 2026 Workshop Submission Metadata

## Primary Target

Foundations of Language Model Security (FLMSec), non-archival workshop track.

- Deadline: August 22, 2026, 23:59 AoE
- Format: up to 8 content pages excluding references
- Review copy: double-blind NeurIPS 2026 workshop format

## Title

Counterfactual Evidence Audits Predict LLM-Agent Susceptibility to Ranked Context

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
(p<.001), reduce mean absolute prediction error by 62% relative to a zero-
effect predictor, and recover the direction of 12 of 13 material effects.
A reviewer-requested post-hoc task-mean baseline is also substantially weaker
(MAE .369 versus .167). Matched controls show that selecting one-sided
ordinary items, rather than merely reordering identical items, causes the
shift in a susceptible model. Across seven open-weight families,
susceptibility transfers from an interactive feed to a static RAG dossier
(rho=.750, exact p=.033), while a provenance warning does not reliably
mitigate it. A separate study of three deployed Codex agent tiers finds strong
audit-to-full ranking (rho=.951, p<.001) but no individually significant
full-context effect after correction. Within this single synthetic remote-
work domain, the result supports a domain-specific triage procedure, not a
universal steering claim: evidence selection must be evaluated as part of the
composed agent system.

## Keywords

LLM agents, agent security, retrieval-augmented generation, evidence
selection, behavioral auditing, compositional security, counterfactual
evaluation, ranked context

## TL;DR

A five-document counterfactual audit prospectively predicts susceptibility to
disjoint 45-document ranked contexts across held-out LLM model-task pairs,
exposing the upstream evidence selector as part of the agent's security
boundary.

## Topic Selections

- Design of reproducible, generalizable evaluation methodologies
- Compositional security of LLM systems
- Model- and system-level attacks and defenses
- Secure-by-design LLM system architectures

## Contribution Summary

The paper introduces and prospectively validates a five-document
counterfactual audit that predicts susceptibility to disjoint 45-document
ranked contexts. Matched controls isolate evidence composition from pure
ordering, a static-RAG study tests interface transfer and a disclosure defense,
and a frontier-agent study identifies a robustness boundary.

## Submission Notes

- Submit the anonymous `paper.pdf`, not the named arXiv manuscript.
- The current FLMSec OpenReview form accepts a PDF but has no supplementary-
  material field; do not claim that an artifact ZIP was uploaded with it.
- Do not include an identifying repository URL in the review copy.
- Select a non-archival track where a workshop offers multiple publication
  modes.
- At least one author must accept FLMSec's reciprocal-reviewing requirement.
