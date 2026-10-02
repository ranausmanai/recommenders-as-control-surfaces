# Ancillary artifact release

`counterfactual_evidence_audits_artifacts_v1.zip` contains the complete data,
code, frozen protocols, source pool, queue logs, launch manifests, analyses,
reports, and figure script described in the paper's reproducibility statement.
The canonical maintained repository is
<https://github.com/ranausmanai/recommenders-as-control-surfaces>.

After extracting it, run:

```bash
python3 src/artifact_release_check.py
```

Expected output:

```text
ARTIFACT_RELEASE_OK jobs=3500 labels=12000 incomplete=629 pool=100
```
