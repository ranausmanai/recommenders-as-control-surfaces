# arXiv Replacement Package for 2606.00914

New title: **Counterfactual Evidence Audits Predict LLM-Agent Susceptibility
to Ranked Context**

This is the full methodology manuscript with appendices. It supersedes the
original framing while preserving the same research lineage, so upload it as
a replacement version under arXiv:2606.00914 rather than as a new paper.

Suggested arXiv comments:

> 19 pages, 1 figure. Accepted at the NeurIPS 2026 Workshop on Foundations of
> Language Model Security (FLMSec 2026). Substantially revised after peer
> review: introduces a preregistered five-document counterfactual evidence
> audit, 1,800 matched controls, held-out validation across seven open-weight
> families, feed-to-RAG transfer, a disclosure-defense test, and a three-tier
> Codex boundary study; narrows unsupported universal and asymmetry claims.

Suggested replacement reason:

> Substantially revised after peer review and new preregistered experiments.
> The revision reframes the paper around a counterfactual evidence audit,
> adds matched causal controls, held-out multi-family validation, RAG transfer,
> and frontier-agent boundary tests, and updates the title and conclusions.

Build locally with `tectonic paper.tex` and upload the generated source archive
using **Replace** on the existing arXiv record.

The source archive also contains the validated data/code release under
`anc/counterfactual_evidence_audits_artifacts_v1.zip`. arXiv should display it
as ancillary material after the replacement is announced.

Before submission, publish the current release at
<https://github.com/ranausmanai/recommenders-as-control-surfaces>, which is the
canonical repository linked from the manuscript. Do not link the older
Hugging Face datasets unless they are replaced by a release matching the
3,500-job confirmatory artifact.
