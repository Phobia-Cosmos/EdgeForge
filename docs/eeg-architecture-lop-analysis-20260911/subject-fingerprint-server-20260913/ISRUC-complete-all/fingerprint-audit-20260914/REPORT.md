# ISRUC fingerprint audit (2026-09-14)

The audit re-read 85,520 epoch rows from 98 subjects and 4,276 sequences.

PCA is fit to column-standardized handcrafted features; the first two components explain 0.5024 of variance.

All-feature held-out logistic identity accuracy is 0.7328; the split holds out complete sequences within each subject.

The ablation table in `audit-summary.json` shows which feature groups carry identity information. This is a data-level fingerprint audit, not a CNN embedding or an LoP test.

PCA axis signs are arbitrary: multiplying one component by -1 gives the same projection. Distances, explained variance and clustering are the interpretable quantities.
