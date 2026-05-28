# Recommenders as Control Surfaces for LLM Agents: Adversarial Feed Injection, Model Regimes, and Simple Defenses

## Abstract

LLM agents increasingly consume ranked external information streams — social feeds, search results, retrieval contexts, and email queues — yet the safety implications of *who controls that ranking* remain underexplored. Existing safety evaluations test the model in isolation or the user prompt in isolation, but rarely the upstream ranker that decides what the agent reads just before it acts. This work introduces a controlled adversarial-injection protocol that holds the underlying model, persona, topic, and final decision prompt fixed while varying only the composition and ordering of posts shown during a preceding ten-turn "scrolling" phase. Across 2,465 decision rollouts on four modern open instruct LLMs spanning three independent labs (Meta, Google, Alibaba), three response regimes emerge, which we term ***capitulation***, ***saturation***, and ***asymmetry***. On Llama 3.2-3B, an exemplar of the capitulation regime, heavy adversarial injection reduces *recommend fully remote* decisions from 100% to 50% (Bonferroni-corrected p = 0.0065), strengthens to 5% under a generator-swap robustness test in which Gemma 4 authors both organic and adversarial pools (p = 3 × 10⁻¹⁰), and follows a monotonic dose-response with an apparent threshold near two adversarial posts per five-post batch (chi-square p = 0.006). Gemma 4-e4b shifts analogously (40% → 0% remote-first; Bonferroni p = 0.049), whereas Qwen 3.5-2B and Qwen 3.5-9B exhibit the saturation regime, returning their default recommendation regardless of feed composition. Two feed-level defenses — *balanced exposure* and *ranking disclosure* — significantly restore baseline behavior in the susceptible model (balanced: 95% restoration on the Claude pool, 65% under generator-swap; disclosure: 85% / 45%).

**Contributions.** This paper makes three contributions:
(i) the **three-regime taxonomy** of feed-injection susceptibility in modern instruct LLMs;
(ii) a **generator-swap-replicated** demonstration that ranked exposure shifts downstream agent decisions with effect sizes that survive multiple-comparison correction, accompanied by a dose-response characterization and two feed-level defenses;
and (iii) a **methodological warning** that random-CV activation probing overstates "hidden mechanism" claims in multi-turn agent settings by 30+ percentage points relative to group-aware evaluation with a visible-history baseline.

*In an age of agentic AI, every recommender silently authors every reply. The question is no longer whether models behave well; the question is who controls what they read just before they answer.*

## 1. Introduction

LLM agents are rarely deployed in a vacuum. They browse, search, retrieve, subscribe, react, summarize, and make decisions after consuming ranked streams of information. A model's downstream answer is therefore not only a function of its weights and the final user prompt; it is also a function of the information trajectory selected by upstream ranking systems.

This paper asks a direct question: can a feed ranker steer an LLM agent's consequential decision while holding the base model, persona, topic, and final decision prompt fixed?

Our initial experiments began as a mechanistic probing study. Linear probes could recover feed policy from residual-stream activations at high accuracy under random turn-level cross-validation. However, group-aware evaluation and visible-history baselines showed that this framing was overclaimed: naive CV inflated probe accuracy, and much of the activation signal was recoverable from visible conversation history. That failure is scientifically useful. It redirected the project from a hidden-representation story to a more operational question: whether ranked exposure changes what agents decide.

The answer is nuanced but important. Feed injection does not universally overpower models. Instead, we observe three regimes:

1. **Capitulation:** susceptible models move toward the injected feed pressure.
2. **Saturation:** models with strong defaults ignore the feed and return the same decision.
3. **Asymmetry / reactance:** injections aligned with an existing model default can be silent, while injections against the default can have large effects.

The main contribution is a controlled attack-and-defense study of these regimes. We show that adversarial post injection significantly changes downstream decisions in Llama 3.2-3B and Gemma 4-e4b, replicates under a post-generator swap, follows a dose-response curve, and can be mitigated by balanced exposure and ranking-disclosure defenses in the cleanest Llama setting.

## 2. Experimental Setup

### 2.1 Agent Protocol

Each rollout has two phases.

**Feed exposure.** The agent receives 10 turns of social-feed posts. Each turn shows 5 posts. The agent reacts to each post with LIKE, SHARE, or SKIP and a short rationale. The conversation history retains the recent reaction exchanges, so downstream decisions are conditioned on accumulated exposure.

**Decision.** After feed exposure, the agent receives a forced-choice decision prompt. For the remote-work experiments, the agent advises a CEO:

- A: full return-to-office
- B: hybrid
- C: remote-first

The main outcome is the final A/B/C recommendation.

### 2.2 Feed Conditions

The adversarial remote-work experiments use:

- `organic_random`: five random organic posts per turn.
- `organic_recency`: chronological organic posts.
- `light`: one adversarial pro-RTO post plus four organic posts per turn.
- `heavy`: five adversarial pro-RTO posts per turn.
- `balanced`: two adversarial pro-RTO posts plus three organic posts per turn.
- `disclosed_heavy`: heavy injection plus a system-level disclosure warning that the feed may be adversarially selected.

Follow-up experiments add:

- `anti-direction`: pro-remote adversarial posts, testing whether attacks aligned with the model's remote-first default have any effect.
- `generator-swap`: Gemma 4-generated organic and adversarial pools, testing whether the effect is an artifact of Claude-written posts.
- `dose0` to `dose5`: 0 to 5 adversarial posts per 5-post batch.

### 2.3 Models

The modern attack grid includes:

- Llama 3.2-3B
- Gemma 4-e4b
- Qwen 3.5-2B
- Qwen 3.5-9B

Additional historical runs include Qwen2.5, SmolLM2, Falcon3, Yi, Zephyr, StableLM, TinyLlama, and activation-probe experiments. Those are not the headline evidence; they are used to motivate the shift from probing to decision-level auditing.

### 2.4 Post Pools

Local artifacts include five post files:

- `posts/pool.jsonl`: 500 Claude-generated organic posts across five topics.
- `posts/adversarial_rto.jsonl`: 50 Claude-generated pro-RTO adversarial posts.
- `posts/adversarial_pro_remote.jsonl`: 50 Claude-generated pro-remote anti-direction posts.
- `posts/pool_gemma.jsonl`: 100 Gemma-generated organic remote-work posts.
- `posts/adversarial_rto_gemma.jsonl`: 50 Gemma-generated pro-RTO adversarial posts.

The generator-swap experiment is critical because it tests whether the attack depends on one generator's wording style.

## 3. Main Results

### 3.1 Adversarial Injection Shifts Llama 3.2-3B Decisions

Under organic random exposure, Llama 3.2-3B recommends remote-first in all 20 seeds. Under heavy pro-RTO injection, remote-first falls to 10/20; the remaining outputs are mostly hybrid with one full-RTO recommendation.

| Model | Baseline A/B/C | Heavy Attack A/B/C | Remote-first Change | Fisher p on C |
|---|---:|---:|---:|---:|
| Llama 3.2-3B | 0 / 0 / 20 | 1 / 9 / 10 | 100% -> 50% | 0.0004 |
| Gemma 4-e4b | 0 / 12 / 8 | 0 / 20 / 0 | 40% -> 0% | 0.0033 |
| Qwen 3.5-2B | 0 / 20 / 0 | 2 / 18 / 0 | 0% -> 0% | 1.0000 |
| Qwen 3.5-9B | 0 / 18 / 2 | 0 / 20 / 0 | 10% -> 0% | 0.4872 |

With Bonferroni correction over the per-model A/B/C comparisons in the existing table, Llama remains significant (corrected p=0.0065) and Gemma remains barely significant (corrected p=0.049). The two Qwen models are null because they are saturated near hybrid answers even before attack.

![Figure 1: Adversarial feed injection shifts decisions in 2 of 4 modern LLMs. Bars show P(recommend fully remote) under organic-random baseline vs heavy pro-RTO injection, with 95% Wilson CIs. Significance markers (Fisher's exact, two-sided): *** p<0.001, ** p<0.01, n.s. not significant.](figures/paper_fig1_cross_model_attack.png)

Interpretation: the attack is not universal. It succeeds when the model has a susceptible default that can be moved by accumulated evidence. It fails when the model is already saturated at a stable answer.

### 3.2 Generator Swap Replicates and Strengthens the Attack

To rule out a Claude-post artifact, we reran the Llama 3.2-3B experiment using Gemma 4-generated organic and adversarial posts. The effect became stronger.

| Condition | Remote-first Rate, Claude Pool | Remote-first Rate, Gemma Pool |
|---|---:|---:|
| Organic random | 100% | 100% |
| Heavy attack | 50% | 5% |
| Balanced defense | 95% | 65% |
| Disclosed defense | 85% | 45% |

For the Gemma-generated pool, heavy attack shifts Llama from 20/20 remote-first to 1/20 remote-first (Fisher p=3.0e-10). This is the strongest evidence in the project because it rules out the most obvious post-generator cherry-picking critique.

![Figure 3: Generator-swap robustness. P(remote-first) on Llama 3.2-3B across four feed conditions, comparing Claude-written posts (blue) vs Gemma-written posts (orange). The attack replicates and strengthens with a different post writer; the most extreme heavy-attack result (Gemma-written, p = 3 × 10⁻¹⁰) effectively rules out a content-style artifact.](figures/paper_fig3_generator_swap.png)

### 3.3 Dose-Response Supports a Causal Exposure Story

We varied the number of adversarial posts per 5-post batch while keeping the same model, topic, decision prompt, and exposure length. Remote-first choices decrease monotonically as adversarial density increases.

| Adversarial Posts per Batch | Remote-first Rate |
|---:|---:|
| 0 / 5 | 100% |
| 1 / 5 | 100% |
| 2 / 5 | 90% |
| 3 / 5 | 90% |
| 4 / 5 | 80% |
| 5 / 5 | 65% |

The C-vs-non-C distribution differs across dose levels (chi-square p=0.0062). This dose-response curve is important: it makes the result look like an exposure-dependent effect, not a one-off statistical fluctuation.

![Figure 2: Dose-response of adversarial injection on Llama 3.2-3B. Each point is n=20 seeds; shaded band is the 95% Wilson CI. The attack has a threshold near 2 adversarial posts per 5-post batch — below this, the effect is invisible; above it, the model's recommendation tilts monotonically.](figures/paper_fig2_dose_response.png)

### 3.4 Anti-Direction Attack Is a No-Op

Llama 3.2-3B defaults to remote-first in the remote-work setting. When the adversarial pool is pro-remote rather than pro-RTO, every condition remains 20/20 remote-first. This asymmetry suggests the attack is not simply "more adversarial content causes instability." It matters whether injected content pushes against the model's default.

This is useful for threat modeling. Attacks aligned with a model's existing default may be invisible because the output does not change; attacks opposing the default reveal susceptibility.

### 3.5 Simple Defenses Mitigate in the Cleanest Susceptible Model

In Llama 3.2-3B with Claude-generated posts, heavy attack moves remote-first from 100% to 50%. Balanced exposure restores it to 95%; ranking disclosure restores it to 85%.

In the Gemma-generated pool, the attack is stronger: 100% to 5%. Balanced exposure restores remote-first to 65%, while disclosure restores it to 45%. Both are significantly different from the heavy attack condition in Fisher tests on C: balanced p=0.00014, disclosed p=0.00836.

The synced local artifacts do not show the same defense restoration for Gemma 4-e4b itself: in `decision_shift_adv_modern.jsonl`, Gemma 4 remains at 100% hybrid under heavy, balanced, and disclosed conditions. We therefore report Gemma as attack-susceptible but do not claim a demonstrated Gemma defense success from the local data.

![Figure 4: Defenses on Llama 3.2-3B. Left: Claude-written post pool. Right: Gemma-written post pool. Red bars show the heavy-attack baseline; green and purple show the two defenses; dashed blue line shows the organic-baseline P(remote-first). Significance markers compare each defense against the heavy-attack arm (Fisher's exact): *** p<0.001, ** p<0.01, * p<0.05.](figures/paper_fig4_defenses.png)

## 4. Earlier Activation-Probe Findings: What Changed

The project initially found high policy-decoding accuracy from residual-stream activations. Under random turn-level CV, feed policy appeared recoverable at approximately 0.85-0.95 balanced accuracy in many cells. Harder leave-one-run-out splits reduced that substantially, and visible-history baselines often matched or exceeded the activation probe.

That result changes the interpretation:

- There is real feed-policy signal in trajectories.
- But the signal is largely visible-history mediated.
- Random turn-level CV is leaky for multi-turn agents.
- Activation probes alone should not be used to claim hidden mechanisms in agent settings.

This becomes a methodological contribution rather than the main result. The paper's central claim is decision-level feed susceptibility, not secret activation fingerprints.

## 5. Interpretation

The strongest interpretation is practical and systems-oriented:

> Ranked feeds are control surfaces for LLM agents.

This does not mean every model follows every malicious feed. The results instead show model-specific regimes.

**Capitulation.** Llama 3.2-3B follows adversarial RTO pressure in the remote-work decision task, especially under Gemma-generated adversarial posts.

**Saturation.** Qwen 3.5-2B and Qwen 3.5-9B are stable near hybrid recommendations in this setting. Their defaults swamp the attack.

**Asymmetry.** Llama's pro-remote default is not further moved by pro-remote attack content, but it can be moved away from remote-first by pro-RTO content.

**Partial defense.** Balanced feeds and ranking disclosure can reduce attack impact, but they are not universal fixes. Their effectiveness depends on the model and pool.

This is stronger than a "feeds influence models" platitude because the experiments isolate ranker-controlled exposure while holding the decision task fixed, include null models, include generator-swap replication, include dose-response, and test defenses.

## 6. Why This Matters

The immediate implication is for agent evaluation. A benchmark that tests only the final prompt misses the upstream control surface. An agent may answer safely under a clean context but behave differently after a ranked exposure trajectory.

The process change suggested by these results is:

1. Agent evaluations should include feed-exposure audits.
2. Audits should test adversarial rankers, not only organic feeds.
3. Evaluations should report model-specific susceptibility rather than average across models.
4. Defenses should be evaluated at the feed layer: balanced exposure, provenance/disclosure, diversity constraints, and context summarization.
5. Mechanistic probing in multi-turn agents should use group-aware splits and visible-history baselines.

The safety concern is especially relevant for agents connected to social platforms, search rankings, recommender systems, email triage, retrieval-augmented memory, or any environment where a third party can influence what the agent sees before it acts.

## 7. Limitations

**Single strongest task domain.** The clearest modern-model attack is currently remote-work advice. More domains are needed.

**Small per-cell sample size.** Most confirmatory cells use n=20 seeds. The effects are large enough to detect in Llama and Gemma, but additional seeds would tighten confidence intervals.

**Model coverage.** The modern grid includes Llama, Gemma, and Qwen. Broken/skipped models should be reported separately and not counted as nulls.

**Defense evidence is strongest for Llama.** The local artifacts show Llama defenses working. They do not show Gemma 4 defense restoration, even though Gemma 4 itself is attack-susceptible.

**Synthetic posts.** The generator-swap test improves robustness, but real social posts would further strengthen ecological validity.

**Visible-history mediation.** The attack works through ordinary context accumulation. That is operationally important, but it is not evidence of a hidden internal-only mechanism.

**Prompt sensitivity.** Earlier v1/v2 experiments showed that small changes to the final decision format can suppress or expose feed effects. This is a feature of agent evaluation, but it also means claims must be tied to the tested decision interface.

## 8. Conclusion

We have shown that adversarial ranked-feed exposure can significantly shift downstream decisions in susceptible modern LLM agents. The effect replicates across post generators, follows a monotonic dose-response curve, is asymmetric with respect to model defaults, and can be mitigated by simple feed-level defenses in the cleanest susceptible model. Other models exhibit saturated defaults, showing that susceptibility is model-specific rather than universal. The activation-level signal that originally motivated the project is largely visible-history mediated and is reported here as a methodological warning: in multi-turn LLM-agent settings, naive random-CV probing overstates the apparent "hidden mechanism" content of agent activations.

The title-level contribution is that recommender systems are a practical control surface for LLM agents.

> *In an age of agentic AI, every recommender silently authors every reply. The question is no longer whether models behave well; the question is who controls what they read just before they answer.*

## Reproducibility

All code, post pools, and per-rollout decision logs are released alongside the paper. The four headline figures regenerate from the released JSONL files via the included script (`notebooks/11_paper_figures.py`). The agent protocol uses standard HuggingFace Transformers and Ollama, with no non-public models or APIs. Random seeds are recorded with every rollout.

