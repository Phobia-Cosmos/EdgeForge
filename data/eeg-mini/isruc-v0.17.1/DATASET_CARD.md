# ISRUC-Sleep CPU Mini Split v0.17.1

## Source and authorization

This is a processed subset of the public [ISRUC-Sleep dataset](https://sleeptight.isr.uc.pt/ISRUC_Sleep/). The EdgeForge repository owner explicitly confirmed that this mini subset may be uploaded. Downstream users must still review and comply with the original dataset terms; the EdgeForge code license does not relicense the upstream data.

Please cite: Khalighi, S., Sousa, T., Santos, J. M., and Nunes, U. (2016). ISRUC-Sleep: A comprehensive public dataset for sleep researchers. *Computer Methods and Programs in Biomedicine*, 124, 180-192.

## Contents

Each data file is a NumPy `float32` tensor with shape `(20, 8, 3000)`. Each paired label file is an `int64` vector with shape `(20,)`. The five files contain 100 sleep epochs and occupy about 9.3 MiB.

| Role | Dataset subject | File | Label counts |
| --- | ---: | ---: | --- |
| source | 1 | 3 | `0:1, 1:4, 2:7, 3:8` |
| source | 3 | 16 | `0:1, 1:4, 2:1, 3:2, 4:12` |
| source | 4 | 0 | `0:3, 1:8, 2:9` |
| target | 2 | 45 | `0:2, 1:7, 2:8, 4:3` |
| retention | 5 | 30 | `0:2, 1:12, 2:3, 4:3` |

Subjects do not overlap across source, target and retention roles. `manifest.json` records the selection strategy, relative paths and SHA-256 hashes. The deterministic `label-diversity` strategy maximizes the minimum label coverage of the first and second 10-epoch halves, then full-file coverage and label entropy. It uses labels only to construct a useful tiny fixture and therefore is not an unbiased sampling protocol.

## Intended use

- CPU smoke tests for loaders, model registries and compiler/runtime paths.
- Initial architecture comparisons under a fixed data contract.
- Demonstrations of source training, target adaptation/evaluation and retention metrics.

For the current benchmark, the first 10 target epochs form the adaptation half and the last 10 form the held-out evaluation half. Both halves contain three label classes. This chronological split must not be called a statistically representative train/test split.

## Limitations

- Only five source-dataset subjects and one 20-epoch file per subject are included.
- The subset is intentionally selected for label diversity and does not preserve population class frequencies.
- It is insufficient for publication-grade accuracy, generalization, fairness, clinical or LoP conclusions.
- Accuracy values are highly discrete. Use the complete canonical dataset, subject-level splits and multiple independent seeds for scientific experiments.
- Retention is a separate old-data diagnostic and must not be used as the LoP outcome.

## Reproduction

```sh
PYTHONPATH=src /home/undefined/Disk/python-envs/brainuicl/bin/python \
  scripts/create-eeg-mini-split.py \
  --source-root /path/to/isruc_group1_npy_float32 \
  --output-root data/eeg-mini/isruc-v0.17.1 \
  --source-subjects 1 3 4 \
  --target-subject 2 \
  --retention-subject 5 \
  --selection-strategy label-diversity \
  --public-release
```
