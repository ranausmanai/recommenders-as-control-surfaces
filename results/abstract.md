# Ranking Is Not Content: Mechanistic Signatures of Feed-Algorithm-Induced Drift in LLM Agents

When the same pool of social-media posts is presented to an LLM agent under
different ranking algorithms, the algorithm itself leaves a measurable,
**model-agnostic**, **topic-conditional**, and **causally-active**
geometric fingerprint in the residual stream. We expose **11 instruction-
tuned LLMs spanning 8 model families** (Qwen2.5 at 0.5B–7B, Qwen3.5-0.8B,
HuggingFaceTB SmolLM2 at 360M and 1.7B, 01.ai Yi-1.5-6B-Chat, Mistral-
lineage Zephyr-7B-beta, Stability StableLM-2-1.6B, TII Falcon3-3B,
TinyLlama-1.1B-Chat) to 100 posts per topic across five topics
(remote work, AI regulation, nuclear energy, universal basic income,
gene editing) under three feed policies (random, recency, ε-greedy
engagement-max bandit). Across ~2,500 multi-turn probe activations, a
linear logistic-regression probe recovers feed policy at **0.83–1.00
balanced accuracy** in 24 of 24 within-(model, topic) cells on the
RTX 4000 cross-family pipeline (chance 0.33). The fingerprint persists
across a **16× parameter range** (0.5B → 7.6B). The probe **does not
transfer across topics** (off-diagonal accuracy near chance in 33 of
35 cells across two transfer matrices). The empirical drift PC1 is
**orthogonal** (|cos| < 0.06) to a Ji-Ma 2026 contrastive static stance
vector across all measured topics. **Sham-controlled activation patching**
on Qwen2.5-3B at L17, scale 10, produces a target-minus-sham stance
shift of **+0.55 points on a [−2, +2] scale** (target moves stance
+0.475, random-direction sham moves it −0.075) — the policy-encoding
direction is not just present in the residual stream but causally used
when generating downstream probe text. A weaker scale (5.0) showed no
causal separation from sham, suggesting a magnitude threshold worth
characterizing further. These results extend prior single-turn residual-
steering and multi-turn jailbreak-drift work to the benign feed-
exposure regime, and establish — across families, scales, and a sham
control — that recommender choice mechanistically shapes LLM internal
state in a way that propagates to output behavior.
