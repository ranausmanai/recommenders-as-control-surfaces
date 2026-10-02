# Counterfactual Evidence Audits Predict LLM-Agent Susceptibility to Ranked Context

Code, data, frozen protocols, and analyses for the FLMSec 2026 paper and
arXiv:2606.00914 v2.

## Finding

Upstream selectors determine which evidence an LLM agent sees. We test whether
a small counterfactual audit can predict how strongly a model-task pair will
respond to a much larger one-sided context.

- Across 18 held-out open-weight model-task cells, five-document audit effects
  predict disjoint 45-document effects with Spearman rho `.855`
  (`p=.00025`).
- The audit has MAE `.167`, versus `.436` for a zero-effect predictor and
  `.369` for a reviewer-requested development-task baseline.
- Matched controls separate composition from pure order: selection shifts a
  susceptible Llama 3.2-3B agent, while reordering identical items is null.
- Model-level susceptibility transfers from an interactive feed to a static
  RAG dossier (rho `.750`, exact `p=.033`), but direct short-audit-to-RAG
  prediction remains nonsignificant (`p=.071`).
- A provenance warning is not a reliable defense. Three deployed Codex agent
  tiers show strong short-to-full ranking but no individually significant
  full-context effect after correction.

The validated claim is deliberately narrow: this is a domain-specific triage
procedure demonstrated on one synthetic remote-work corpus, not a universal
steering law or safety certificate.

## Artifact Release

The release contains exactly **3,500 confirmatory jobs** and **12,000 decision
labels**, plus frozen preregistrations, source pool, schema, queue logs,
manifests, analyses, reports, and figure code. See [`ARTIFACTS.md`](ARTIFACTS.md).

Validate an extracted release with:

```bash
python3 src/artifact_release_check.py
```

Expected output:

```text
ARTIFACT_RELEASE_OK jobs=3500 labels=12000 incomplete=629 pool=100
```

## Reproduce Analyses

```bash
pip install -r requirements.txt
python3 src/reviewer_controls_analyze.py
python3 src/susceptibility_audit_analyze.py
python3 src/cross_interface_audit_analyze.py
python3 src/codex_frontier_audit_analyze.py
python3 src/reviewer_requested_baseline.py
python3 notebooks/15_audit_paper_figures.py
```

## Repository Layout

```text
experiments/    Frozen preregistrations, amendment, and JSON schema
posts/          Matched synthetic evidence pool
results/        Retained rollouts, manifests, logs, analyses, and reports
src/            Experiment runners, queue managers, and analyzers
notebooks/      Figure generation
paper_arxiv_v2/ Full arXiv manuscript and appendices
paper_neurips_workshop/camera_ready/  FLMSec camera-ready source
output/         Built PDFs, source archives, and artifact release archives
```

## Citation

```bibtex
@misc{usman2026counterfactual,
  title         = {Counterfactual Evidence Audits Predict {LLM}-Agent
                   Susceptibility to Ranked Context},
  author        = {Rana Muhammad Usman},
  year          = {2026},
  eprint        = {2606.00914},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI}
}
```

## License

MIT for code; CC BY 4.0 for post pools and rollout data. See [`LICENSE`](LICENSE).

Contact: `usmanashrafrana@gmail.com`
