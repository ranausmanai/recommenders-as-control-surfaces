# Recommenders as Control Surfaces for LLM Agents

**Adversarial Feed Injection, Model Regimes, and Simple Defenses**

> *In an age of agentic AI, every recommender silently authors every reply.
> The question is no longer whether models behave well; the question is who
> controls what they read just before they answer.*

This repository contains the full source, data, and analysis for the paper.

## TL;DR

We show that the choice of *ranking algorithm* used to curate a feed an LLM
agent scrolls can significantly shift the agent's downstream consequential
decisions — even when the underlying post pool, persona, and final
decision prompt are held fixed.

- **2,465** decision rollouts across **9 open instruct LLMs** from 5 model
  families and 3 major labs (Meta, Google, Alibaba).
- On **Llama 3.2-3B** (Meta), heavy adversarial injection drops
  "recommend fully remote" decisions from 100% to 50%
  (Fisher p = 0.0004; Bonferroni p = 0.0065).
- The effect **replicates** with a generator-swap (Gemma-written posts):
  100% → 5%, p = 3 × 10⁻¹⁰.
- It follows a **monotonic dose-response** with a threshold at ~2/5
  adversarial posts per batch.
- Two simple defenses (**balanced exposure** and **ranking disclosure**)
  significantly mitigate the attack on the susceptible model.
- The activation-probing framing the project started with **does not
  survive group-aware cross-validation** and a visible-history baseline —
  reported as a methodological warning.

📄 **Paper**: [`paper/paper.tex`](paper/paper.tex) — arXiv: *coming soon*

🤗 **Datasets**:
- Post pools: [`ranausmans/feed-injection-pool`](https://huggingface.co/datasets/ranausmans/feed-injection-pool)
- Decision rollouts: [`ranausmans/feed-injection-rollouts`](https://huggingface.co/datasets/ranausmans/feed-injection-rollouts)

## Reproduce the headline figures in 2 minutes

```bash
git clone https://github.com/ranausmanai/recommenders-as-control-surfaces
cd recommenders-as-control-surfaces
pip install -r requirements.txt
python3 notebooks/11_paper_figures.py
```

Outputs all four paper figures to `results/figures/paper_fig*.png` from the
raw JSONL rollouts under `results/`.

## Rerun all the statistics

```bash
python3 notebooks/10_rigorous_stats.py
```

Computes Bonferroni-corrected Fisher tests, Wilson 95% CIs, and the defense
triviality check on every model × condition cell.

## Repository layout

```
.
├── paper/                  # LaTeX source + figures + arXiv submission guide
│   ├── paper.tex
│   ├── references.bib
│   ├── figures/
│   └── SUBMISSION_GUIDE.md
├── src/                    # agent loop, feed policies, decision protocols
│   ├── agent_loop.py
│   ├── feed_policies.py
│   ├── decision_shift.py
│   ├── decision_shift_adv.py
│   ├── decision_shift_adv_ollama.py
│   ├── gen_adversarial_posts.py
│   ├── gen_pool_ollama.py
│   └── ...
├── notebooks/              # analysis + figure generation
│   ├── 10_rigorous_stats.py        # Bonferroni / Wilson CIs / Fisher
│   ├── 11_paper_figures.py         # generates all 4 paper figures
│   └── ...
├── posts/                  # 5 post pools (Claude + Gemma generators)
│   ├── pool.jsonl                          # 500 organic posts
│   ├── adversarial_rto.jsonl               # 50 Claude pro-RTO
│   ├── adversarial_pro_remote.jsonl        # 50 Claude pro-remote (anti-dir)
│   ├── pool_gemma.jsonl                    # 100 Gemma organic
│   └── adversarial_rto_gemma.jsonl         # 50 Gemma pro-RTO
├── results/                # rollouts, analyses, figures
│   ├── decision_shift*.jsonl       # raw decision rollouts (2,465 total)
│   ├── 10_rigorous_stats.json
│   ├── 10_decision_tables.md
│   ├── top_conf_paper_draft.md     # markdown draft (mirrors paper.tex)
│   └── figures/
├── requirements.txt
├── LICENSE
├── CITATION.cff
└── README.md
```

## Cite

If you use this work, please cite:

```bibtex
@misc{rana2026recommenders,
  title  = {Recommenders as Control Surfaces for {LLM} Agents:
            Adversarial Feed Injection, Model Regimes, and Simple Defenses},
  author = {Rana Muhammad Usman},
  year   = {2026},
  eprint = {arXiv:XXXX.XXXXX},
  archivePrefix={arXiv},
  primaryClass={cs.CR}
}
```

(arXiv number will be filled in once the preprint is live.)

## License

MIT for code, CC-BY 4.0 for the post pools and decision rollouts. See
[`LICENSE`](LICENSE).

## Contact

Rana Muhammad Usman — `usmanashrafrana@gmail.com`
