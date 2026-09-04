# ISRUC-Sleep Medium Experiment Subset v0.18.0

## Source and authorization

This subset is derived from the public [ISRUC-Sleep dataset](https://sleeptight.isr.uc.pt/ISRUC_Sleep/). The EdgeForge repository owner explicitly confirmed that this processed subset may be uploaded. Downstream users remain responsible for the original dataset terms; the EdgeForge source-code license does not relicense upstream data.

Please cite: Khalighi, S., Sousa, T., Santos, J. M., and Nunes, U. (2016). ISRUC-Sleep: A comprehensive public dataset for sleep researchers. *Computer Methods and Programs in Biomedicine*, 124, 180-192.

## Contents and split

Each data file is a NumPy `float32` tensor with shape `(20, 8, 3000)`. Each paired label is an `int64` vector with shape `(20,)`. The subset contains 100 data files, 100 labels and 2,000 sleep epochs. Its total size is 192,106,261 bytes (about 183.2 MiB); the largest file is 1,920,128 bytes.

| Role | Subjects | Files | Epochs | Aggregate label counts |
| --- | --- | ---: | ---: | --- |
| source | `1,3,4,6,7,9,10,21` | 40 | 800 | `0:161, 1:211, 2:234, 3:79, 4:115` |
| target | `2,11,12,13,14,15,16,17` | 40 | 800 | `0:152, 1:207, 2:268, 3:75, 4:98` |
| retention | `5,18,19,20` | 20 | 400 | `0:84, 1:129, 2:121, 3:46, 4:20` |

Subject IDs do not overlap across roles. For target adaptation, use the first 10 epochs of every file for training and the last 10 for held-out evaluation. The target adaptation half has label counts `0:68, 1:96, 2:133, 3:56, 4:47`; the evaluation half has `0:84, 1:111, 2:135, 3:19, 4:51`.

## Selection protocol

For each subject, `scripts/create-eeg-mini-split.py` ranks candidate 20-epoch files by:

1. Minimum class coverage across the first and second 10-epoch halves.
2. Full-file class coverage.
3. Label entropy.
4. Lower original file index as the deterministic tie breaker.

The top five paired files are included. This makes the subset useful for small experiments but deliberately changes the canonical class distribution. Use the complete ISRUC-Sleep data and a prespecified subject protocol for publication-grade conclusions.

## Intended use

- Multi-subject CPU/GPU development and architecture comparison.
- Preliminary continual target streams and LoP metric integration.
- Loader, compiler, runtime and reproducibility validation.
- Controlled shift generation without downloading the complete dataset.

The smaller `data/eeg-mini/isruc-v0.17.1/` remains the recommended fast smoke/CI fixture.

## Limitations

- This is a diversity-selected development subset, not an unbiased cohort sample.
- Five files per subject do not preserve a complete-night temporal trajectory.
- The data are insufficient for clinical, demographic, fairness or formal LoP claims.
- Multiple random training seeds do not create new independent subjects; inference must treat subject and seed correctly.
- Retention measures old-data behavior and is not a substitute for the fresh-vs-checkpoint LoP outcome.

## Reproduction

```sh
PYTHONPATH=src /home/undefined/Disk/python-envs/brainuicl/bin/python \
  scripts/create-eeg-mini-split.py \
  --source-root /path/to/isruc_group1_npy_float32 \
  --output-root data/eeg-medium/isruc-v0.18.0 \
  --source-subjects 1 3 4 6 7 9 10 21 \
  --target-subjects 2 11 12 13 14 15 16 17 \
  --retention-subjects 5 18 19 20 \
  --files-per-subject 5 \
  --selection-strategy label-diversity \
  --public-release
```

`manifest.json` records all relative paths and source/output SHA-256 hashes.
