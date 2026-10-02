# Preregistered Codex Frontier-Agent Audit

## Question

Do current hosted Codex agent models show directional evidence susceptibility,
and does the five-document counterfactual audit predict their response to a
45-document RAG dossier?

This experiment is a separate frontier-agent boundary study. It is not pooled
with open-weight Ollama models because Codex adds a proprietary agent harness
and does not expose sampling seeds or temperature.

## Models

- `gpt-5.6-sol`
- `gpt-5.6-terra`
- `gpt-5.6-luna`

These were visible in the authenticated Codex model cache on 2026-07-28.
`gpt-5.6-sol` was also returned by OpenAI's current latest-model resolver.
Full model slugs, Codex CLI version, timestamps, and command settings are
recorded.

An excluded prelaunch probe found that the Homebrew CLI (`0.143.0`) did not
support the current GPT-5.6 family. The experiment therefore pins the newer
Codex binary bundled with the installed ChatGPT application. No retained
outcome was generated before this compatibility choice.

## Isolation

Every rollout is a fresh `codex exec --ephemeral` process in an empty,
read-only directory. User configuration and repository rules are ignored.
Live search is disabled, tools are prohibited by the prompt, reasoning effort
is fixed to `low`, and the final response is constrained by a JSON Schema.

These results characterize the deployed Codex agent harness, not the raw API
model.

## Design

Each model receives 20 independent replicates in five conditions:

- `no_evidence`
- `audit_rto`
- `audit_remote`
- `full_rto`
- `full_remote`

The evidence-selection replicate controls document identity and order but is
not a model-sampling seed. Audit conditions contain five mirrored documents.
Full conditions contain the corresponding disjoint 45-document dossier. Each
call returns all six fixed workplace decisions as `A`, `B`, or `C`.

Total: 300 CLI calls and 1,800 decision labels.

Replicates 990--999 are reserved for excluded compatibility tests.

## Confirmatory Analysis

For each of 18 model-task cells:

- `audit_effect = mean(AuditRTO - AuditRemote)`
- `full_effect = mean(FullRTO - FullRemote)`

Choices use `A=0`, `B=1`, `C=2`.

1. Each full-effect null is tested by a 100,000-draw two-sided sign-flip test
   over the 20 paired document-selection replicates. Holm correction is applied
   across all 18 model-task tests.
2. Audit-to-full prediction is summarized by Spearman rho across the 18 cells,
   with 100,000 permutations that shuffle audit task labels within each model.
3. Paired-document bootstrap confidence intervals are reported for all audit
   and full effects.

All positive, null, model-specific, refusal, rate-limit, and formatting
outcomes are reported. The open-weight result is not used to select tasks,
conditions, or stopping rules.
