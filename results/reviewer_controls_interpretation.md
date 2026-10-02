# Reviewer-Control Interpretation

## Bottom Line

The strongest corrected result is:

> Ranker-controlled stance selection from one matched, topic-specific pool of
> ordinary posts significantly changes Llama 3.2-3B decisions, even after
> holding generator, relevance, exposure length, uniqueness, and rhetorical
> intensity fixed.

The experiment supports **selection/composition as the control surface**. It
does not support an order-only mechanism, and it does not show matched-pool
susceptibility in Gemma 4-e4b.

## Confirmatory Matched Selection

Llama 3.2-3B (`n=50` per condition):

- matched RTO-ranked: 0 A / 34 B / 16 C;
- matched balanced: 0 A / 0 B / 50 C;
- matched remote-ranked: 0 A / 0 B / 50 C.

The prespecified RTO-ranked versus remote-ranked ordinal difference was -0.68,
95% CI [-0.80, -0.54], Holm-corrected p = 0.00002.

Gemma 4-e4b chose B in all 150 matched-selection trials. The matched-pool
selection effect therefore does not generalize to Gemma.

## Pure Ordering

Every order condition used the exact same 50 post IDs within a seed.

Llama 3.2-3B:

- RTO-last: 0 A / 3 B / 47 C;
- interleaved: 0 A / 0 B / 50 C;
- remote-last: 0 A / 0 B / 50 C.

RTO-last versus remote-last: -0.06, 95% CI [-0.14, 0.00],
Holm-corrected p = 0.502.

Gemma chose B in all 150 order trials. There is no confirmatory order-only
effect in either model.

## Directly Measured Defaults

On remote work:

- Llama: 0 A / 9 B / 41 C; modal remote-first share 0.82.
- Gemma: 0 A / 50 B / 0 C; saturated hybrid default.
- Qwen 3.5-2B: 7 A / 43 B / 0 C; modal hybrid share 0.86.
- Qwen 3.5-9B: 0 A / 50 B / 0 C; saturated hybrid default.

Feed-free distributions for all four models and all six paper tasks are in
`results/reviewer_controls_report.md`.

## Directional Asymmetry

For Llama, RTO-ranked exposure moved ordinal position by 0.50 toward RTO;
remote-ranked exposure moved it by 0.18 toward remote. Their difference was
0.32, 95% CI [0.04, 0.58], raw p = 0.039 and Holm-corrected p = 0.079.

This is suggestive but not confirmatory after correction. The paper should not
present default-direction asymmetry as an established general law.

## What Changes in the Paper

1. Lead with matched benign-feed selection, not adversarial prose or hidden
   activation mechanisms.
2. State explicitly that ordinary contextual persuasion is the operative
   pathway; the systems contribution is that a ranker controls that pathway.
3. Separate selection/composition from ordering. Selection is significant;
   ordering alone is not.
4. Restrict the matched-pool result to Llama. Gemma's original adversarial-pool
   result does not survive this stricter benign-pool control.
5. Use measured feed-free defaults. Do not infer defaults from organic feeds.
6. Call asymmetry suggestive, not confirmed.
7. Describe the dose result as a dose-dependent trend, not a 2/5 threshold.
8. Consider retitling to remove "Against Their Defaults." A defensible title is:
   **Ranked Benign Feeds Steer LLM Agent Decisions Through Content Selection**.

## Scientific Assessment

The reviewer controls strengthen identification but narrow the claim. They
show that the Llama effect is not explained by mixed-topic relevance,
generator identity, duplicated posts, or separately crafted adversarial prose.
That is a meaningful result. At the same time, they show that:

- order alone is insufficient;
- model susceptibility is not universal;
- the asymmetry claim is weaker than the original title suggests.

The defensible contribution is a systems-safety result about
ranker-controlled evidence selection, with explicit model-specific boundaries.

