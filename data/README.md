# EEG experiment data

`eeg-mini/isruc-v0.17.1/` is a small, publicly shareable subset of the open
ISRUC-Sleep dataset for CPU pipeline and data-contract tests. The repository
owner explicitly confirmed that this subset may be uploaded. It contains five
processed data/label pairs (100 epochs, about 9.3 MiB) split into source,
target and retention roles. See its `DATASET_CARD.md` and `manifest.json` for
provenance, limitations and SHA-256 hashes.

The complete ISRUC and FACED datasets remain under
`/home/undefined/Disk/datasets/`; do not copy full datasets, checkpoints or
experiment outputs into this directory. Recreate another mini split with
`scripts/create-eeg-mini-split.py`.
