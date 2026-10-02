# Reviewer-Control Analysis Amendment

Date: 2026-07-26

This amendment was written after all 1,800 preregistered jobs completed and
during final output validation, before reporting results.

## Bug

The frozen version-1 analysis paired records by `(model, condition, seed)` but
did not filter by topic. This affects comparisons involving the `no_feed`
condition because no-feed records exist for six topics. Later topic records
overwrote the remote-work no-feed record for the same model and seed.

Consequently, these version-1 outputs were invalid:

- secondary matched-feed comparisons against `no_feed`;
- the default-direction asymmetry table.

The two primary tests were unaffected because their condition names occur only
in the remote-work experiment:

- `matched_rto` versus `matched_remote`;
- `order_rto_last` versus `order_remote_last`.

Feed-free A/B/C tables were also unaffected because those summaries already
grouped by topic.

## Correction

Analysis version 2 explicitly filters paired comparisons and asymmetry
triplets to `topic == "remote_work"`.

No rollout, condition, seed, outcome, exclusion rule, or statistical test was
changed. The immutable launch manifest retains the hash of the frozen
version-1 analysis, making this post-run correction auditable.

