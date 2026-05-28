# Recommenders as Control Surfaces for LLM Agents: Adversarial Feed Injection, Model Regimes, and Simple Defenses

## Abstract

LLM agents increasingly consume ranked external information streams (social feeds, search results, retrieval contexts, and email queues), yet the safety implications of *who controls that ranking* remain underexplored. Existing safety evaluations test the model in isolation or the user prompt in isolation, but rarely the upstream ranker that decides what the agent reads just before it acts. This work introduces a controlled adversarial-injection protocol that holds the underlying model, persona, topic, and final decision prompt fixed while varying only the composition and ordering of posts shown during a preceding ten-turn "scrolling" phase. Across 2,465 decision rollouts on four modern open instruct LLMs spanning three independent labs (Meta, Google, Alibaba), three response regimes emerge, which we term ***capitulation***, ***saturation***, and ***asymmetry***. On Llama 3.2-3B, an exemplar of the capitulation regime, heavy adversarial injection reduces *recommend fully remote* decisions from 100% to 50% (Bonferroni-corrected p = 0.0065), strengthens to 5% under a generator-swap robustness test in which Gemma 4 authors both organic and adversarial pools (p = 3 × 10⁻¹⁰), and follows a monotonic dose-response with an apparent threshold near two adversarial posts per five-post batch (chi-square p = 0.006). Gemma 4-e4b shifts analogously (40% → 0% remote-first; Bonferroni p = 0.049), whereas Qwen 3.5-2B and Qwen 3.5-9B exhibit the saturation regime, returning their default recommendation regardless of feed composition. Two feed-level defenses, *balanced exposure* and *ranking disclosure*, significantly restore baseline behavior in the susceptible model (balanced: 95% restoration on the Claude pool, 65% under generator-swap; disclosure: 85% / 45%).

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

## 1.5 Related Work

**Prompt injection and indirect prompt injection.** Direct prompt injection attacks on LLMs were first systematized by Perez and Ribeiro (2022). Greshake et al. (2023) extended the threat model to *indirect* prompt injection, where adversarial content is embedded in third-party documents the LLM retrieves rather than in the user's own input. Liu et al. (2024) formalized the attack surface and benchmarked defenses. The present work occupies a similar threat model, in that adversarial content reaches the agent through a third-party channel, but the channel is a *ranker over benign content*, and the targeted output is a downstream multi-step decision rather than a single-turn jailbreak.

**Adversarial attacks on aligned LLMs.** Zou et al. (2023) demonstrated universal transferable suffixes that elicit harmful completions from safety-tuned models. Our setting differs in that the injected content is *not jailbreaking*; the adversarial posts are plausible, persuasive, and individually unremarkable. The attack therefore does not require bypassing safety training; it exploits ordinary in-context reasoning.

**Retrieval-augmented and agentic poisoning.** Zou et al. (2024) studied corruption of retrieval indices in retrieval-augmented generation pipelines. Debenedetti et al. (2024) introduced AgentDojo for evaluating prompt injection against tool-using LLM agents. The present work is closely related to both: ranked-feed injection is the recommender-side analog of RAG poisoning, and the consequential-decision outcome we measure is in the spirit of AgentDojo's task-success metrics.

**Probing and interpretability methodology.** Activation-probing work, including the tuned-lens framework of Belrose et al. (2023) and the linear-truth-direction results of Marks and Tegmark (2023), has produced strong results on single-turn classification of latent model state. Our methodological warning (§4) concerns a setting outside the typical probing literature, namely multi-turn agent trajectories, where standard random k-fold cross-validation systematically inflates accuracy, and a visible-history baseline often matches the probe.

**Recommender systems and behavioral influence.** The longer-running literature on algorithmic amplification and behavioral change in human users, surveyed by Narayanan (2023), provides the conceptual backdrop. The contribution here is that the *audience* of the recommender shifts from a human to an LLM agent, which changes both the threat model and the available defenses.

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

Six core feed conditions are used. Each turn presents five posts; the conditions differ only in how those five posts are selected from the underlying organic and adversarial pools described in Section 2.4.

The first two conditions are non-adversarial baselines. A *random baseline* draws all five posts uniformly at random from the organic pool. A *recency baseline* orders the organic pool by post identifier and serves the first five unseen posts each turn.

Three conditions inject adversarial content at varying intensities. The *light injection* condition replaces one of the five organic posts with an adversarial item; the *heavy injection* condition replaces all five. A *balanced* condition, used as a candidate defense, serves two adversarial posts together with three random organic posts.

A sixth condition, *disclosed heavy injection*, presents the same five adversarial posts as the heavy condition but prepends a one-sentence persona-level disclosure that the feed may have been adversarially selected.

Three follow-up conditions extend the protocol. An *anti-direction* attack reuses the heavy and defense templates but with a pro-remote adversarial pool, testing whether injection aligned with the model's existing default direction has any effect. A *generator-swap* variant of all six core conditions uses adversarial and organic pools authored by Gemma 4 in place of Claude, testing whether the observed effects depend on the post writer's style. A *dose-response* sweep varies the number of adversarial posts per five-post batch from zero through five, characterizing the attack as a function of injection density.

The internal software identifiers for each condition (used in the released code and rollout records) are listed in Appendix A.

### 2.3 Models

The modern attack grid uses four open instruct LLMs released in 2024–2025: Llama 3.2-3B (Meta), Gemma 4-e4b (Google), Qwen 3.5-2B, and Qwen 3.5-9B (Alibaba), all served locally via Ollama. The exact Ollama tags invoked in every rollout are recorded in each released JSONL record. We avoid gated weights so the protocol is reproducible without authentication.

### 2.4 Post Pools

Five post pools provide the underlying content from which the conditions in Section 2.2 are constructed.

Two pools are *organic*: an English-language pool of 500 synthetically authored posts spanning five topics (remote work, AI regulation, nuclear energy, basic income, and human gene editing), balanced across five stance levels and four intensity levels and generated by Claude (Anthropic); and a smaller 100-post organic pool restricted to the remote-work topic, generated by Gemma 4-e4b. The second pool exists to support the generator-swap robustness test.

Three pools are *adversarial*, each containing fifty posts crafted to advocate one side of the remote-work debate persuasively without explicit identity attacks or named individuals. Two are written by Claude: one pro-return-to-office, used in the main attack experiments, and one pro-remote, used as an anti-direction control. The third is written by Gemma 4-e4b, pro-return-to-office, used to test whether the observed attack effects depend on the writer's idiomatic style.

All five pools are released under CC-BY 4.0 as the Hugging Face dataset `ranausmans/feed-injection-pool`. The file-level layout is documented in Appendix A.

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

![Figure 2: Dose-response of adversarial injection on Llama 3.2-3B. Each point is n=20 seeds; shaded band is the 95% Wilson CI. The attack has a threshold near 2 adversarial posts per 5-post batch: below this, the effect is invisible; above it, the model's recommendation tilts monotonically.](figures/paper_fig2_dose_response.png)

### 3.4 Anti-Direction Attack Is a No-Op

Llama 3.2-3B defaults to remote-first in the remote-work setting. When the adversarial pool is pro-remote rather than pro-RTO, every condition remains 20/20 remote-first. This asymmetry suggests the attack is not simply "more adversarial content causes instability." It matters whether injected content pushes against the model's default.

This is useful for threat modeling. Attacks aligned with a model's existing default may be invisible because the output does not change; attacks opposing the default reveal susceptibility.

### 3.5 Simple Defenses Mitigate in the Cleanest Susceptible Model

In Llama 3.2-3B with Claude-generated posts, heavy attack moves remote-first from 100% to 50%. Balanced exposure restores it to 95%; ranking disclosure restores it to 85%.

In the Gemma-generated pool, the attack is stronger: 100% to 5%. Balanced exposure restores remote-first to 65%, while disclosure restores it to 45%. Both are significantly different from the heavy attack condition in Fisher tests on C: balanced p=0.00014, disclosed p=0.00836.

**Defense outcomes on Gemma 4.** The same defense conditions do not produce a comparable restoration on Gemma 4-e4b: under both balanced exposure and ranking disclosure, Gemma remains at 100% hybrid, matching the heavy-attack arm. Gemma is therefore reported as attack-susceptible without a demonstrated defense success in the present configuration. Possible explanations include Gemma's stronger default attractor toward the hybrid option (visible in its baseline distribution in §3.1) and a smaller effective dynamic range over which the defenses can operate.

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

The strongest interpretation is practical and systems-oriented: ranked feeds function as control surfaces for LLM agents, in the sense that the choice of ranker measurably shifts the agent's downstream behavior on a held-fixed decision task. This does not imply that every model follows every adversarial feed; the experimental results identify *model-specific regimes*.

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

All code, post pools, and per-rollout decision logs are released alongside the paper. The four headline figures regenerate from the released decision-rollout files via a single analysis script (see Appendix A for the file map). The agent protocol uses standard HuggingFace Transformers and Ollama, with no gated weights and no non-public APIs. Random seeds are recorded with every rollout. The post pools are available as the Hugging Face dataset [`ranausmans/feed-injection-pool`](https://huggingface.co/datasets/ranausmans/feed-injection-pool), and the per-rollout decision logs as [`ranausmans/feed-injection-rollouts`](https://huggingface.co/datasets/ranausmans/feed-injection-rollouts).

## Appendix A: Software identifiers, file layout, and code locations

For reproducibility, this appendix lists the mapping between the human-readable condition names used throughout the paper and the software identifiers used in the released code and rollout records.

**Condition identifiers.** The following mapping is used in the `condition` field of every rollout record:

| Paper label | Identifier in code and rollout records |
|---|---|
| random baseline | `organic_random` |
| recency baseline | `organic_recency` |
| light injection (1/5 adv.) | `light` |
| heavy injection (5/5 adv.) | `heavy` |
| balanced defense | `balanced` |
| disclosed heavy injection | `disclosed_heavy` |
| dose-response, k/5 adv. | `dose0`, `dose1`, …, `dose5` |

**Post-pool file layout.** The five post pools are released as five JSON-Lines files under the Hugging Face dataset repository:

| File | Contents |
|---|---|
| `pool.jsonl` | 500 Claude-generated organic posts (5 topics) |
| `adversarial_rto.jsonl` | 50 Claude pro-return-to-office adversarial posts |
| `adversarial_pro_remote.jsonl` | 50 Claude pro-remote adversarial posts (anti-direction control) |
| `pool_gemma.jsonl` | 100 Gemma-generated organic posts (remote-work topic) |
| `adversarial_rto_gemma.jsonl` | 50 Gemma pro-return-to-office adversarial posts |

**Rollout-record file layout.** The 2,465 decision rollouts are released under the Hugging Face dataset `ranausmans/feed-injection-rollouts`. The headline cross-model attack data resides in `decision_shift_adv_modern.jsonl`; the generator-swap, anti-direction, and dose-response data resides in `decision_shift_followup.jsonl`.

**Analysis script.** The four paper figures regenerate from the JSONL files via `notebooks/11_paper_figures.py` in the companion GitHub repository [`ranausmanai/recommenders-as-control-surfaces`](https://github.com/ranausmanai/recommenders-as-control-surfaces).

