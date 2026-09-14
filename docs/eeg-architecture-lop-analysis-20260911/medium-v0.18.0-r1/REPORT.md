# EdgeForge EEG architecture / LoP pilot report

Run: `medium-v0.18.0-r1`  
Dataset: `eeg-medium/isruc-v0.18.0`  
Status: completed pilot; `scientific_conclusion_allowed=false`

## Summary

The medium subset contains 100 data/label pairs, 2,000 epochs and 20 disjoint subjects. Every pair matched the published manifest shape and SHA-256 contract. The benchmark completed 15 runs: five architectures × three seeds.

| architecture | parameters | source acc | target checkpoint acc | target fresh acc | mean fresh gap | gap SD | retention acc | CPU ms/batch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `lop_mlp` | 1,542,533 | 0.2925 | 0.3375 | 0.2975 | -0.0400 | 0.0283 | 0.3025 | 0.322 |
| `eegnet` | 2,261 | 0.2929 | 0.3375 | 0.2208 | -0.1167 | 0.1250 | 0.3025 | 3.388 |
| `tcn` | 5,429 | 0.2429 | 0.2550 | 0.2408 | -0.0142 | 0.1551 | 0.2850 | 2.127 |
| `transformer` | 13,029 | 0.2829 | 0.3375 | 0.3375 | +0.0000 | 0.0000 | 0.3025 | 0.734 |
| `brainuicl` | 543,264 | 0.1837 | 0.1583 | 0.3175 | +0.1592 | 0.1325 | 0.1742 | 3.591 |

Per-seed fresh gaps:

- `lop_mlp`: `-0.0600, -0.0600, +0.0000`
- `eegnet`: `+0.0000, -0.0600, -0.2900`
- `tcn`: `+0.0600, +0.1275, -0.2300`
- `transformer`: `+0.0000, +0.0000, +0.0000`
- `brainuicl`: `+0.2100, -0.0225, +0.2900`

## What this run supports

- The uploaded data can be loaded by the EdgeForge EEG benchmark.
- The five registered architectures can be instantiated under a common `(8, 3000) → 5-class` contract.
- The fresh/checkpoint comparison, retention evaluation and latency collection are operational.
- Under this pilot budget, `brainuicl` has the largest positive mean fresh gap, while `lop_mlp` and `eegnet` are negative and `transformer` is neutral.

## What this run does not support

- It does not establish LoP. The data card states that this is a diversity-selected development subset.
- It does not provide the required 8–10 continuous subject stages; target subjects are pooled into one adaptation split by the benchmark.
- It does not provide a seed-cluster bootstrap gate over matched stage transitions.
- It does not test Deep Linear, exact `64→100→100→5` activation variants, AdaLin, G-Mixup, MNIST/CIFAR, Type-1/Type-2 mechanisms or full BWT/generalization-gap analysis.
- Accuracy is discrete and the source training budget is intentionally small; the observed gaps are screening signals, not rankings.

## Recommended next run

Use the same medium release to build a continuous subject runner with source pretraining on the eight source subjects, target order `2,11,12,13,14,15,16,17`, retention on `5,18,19,20`, budgets `0,5,10,25,50`, and seeds `4321,4322,4323`. Save a checkpoint and fresh probe at each transition, then run `evaluate-raeeg-lop-gate.py`. Keep this pilot as the architecture smoke baseline.
