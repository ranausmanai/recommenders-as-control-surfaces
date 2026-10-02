# Counterfactual Evidence Audits: Artifact Release

This release accompanies **Counterfactual Evidence Audits Predict LLM-Agent
Susceptibility to Ranked Context** (FLMSec 2026; arXiv:2606.00914 v2).

## Confirmatory Inventory

The four retained studies contain exactly 3,500 jobs and 12,000 decision
labels:

| Study | Data | Jobs | Labels |
|---|---|---:|---:|
| Matched controls | `results/reviewer_controls.jsonl` | 1,800 | 1,800 |
| Structured susceptibility audit | `results/susceptibility_audit_structured.jsonl` | 700 | 4,200 |
| Cross-interface RAG audit | `results/cross_interface_audit.jsonl` | 700 | 4,200 |
| Codex frontier boundary | `results/codex_frontier_audit.jsonl` | 300 | 1,800 |

The archive also includes the frozen preregistrations, structured-output
schema, 100-item matched source pool, queue logs, launch manifests, analysis
JSON, human-readable reports, relevant runner/analyzer source, and figure
script. The incomplete 629-job free-text audit is preserved under
`results/susceptibility_audit_frozen_incomplete_629.jsonl` and is never mixed
with the complete structured replacement.

## Integrity and the Analysis Amendment

Run from the extracted archive root:

```bash
python3 src/artifact_release_check.py
```

The checker validates file checksums, record counts, unique job keys, label
counts, source-pool size, headline analysis values, and launch hashes.

The immutable matched-control launch manifest records the hash of the original
version-1 analyzer. After all 1,800 jobs completed, a topic-filtering bug was
found in secondary no-feed comparisons and corrected without changing any
rollout or either primary test. The reason and scope are recorded in
`results/reviewer_controls_analysis_amendment.md`. Consequently, the current
corrected `src/reviewer_controls_analyze.py` intentionally does not match the
version-1 analyzer hash in the immutable launch manifest. Every other mapped
launch hash is checked against the released file.

## Reproduce Analyses

With the Python dependencies from `requirements.txt` installed:

```bash
python3 src/reviewer_controls_analyze.py
python3 src/susceptibility_audit_analyze.py
python3 src/cross_interface_audit_analyze.py
python3 src/codex_frontier_audit_analyze.py
python3 src/reviewer_requested_baseline.py
python3 notebooks/15_audit_paper_figures.py
```

The queue logs document execution. Compatibility smoke tests, abandoned
exploratory runs, and the incomplete current-model replication are excluded
from confirmatory claims and from this release archive.

Code is released under the MIT license. Post pools and rollout data are
released under CC BY 4.0 as stated in `LICENSE`.
