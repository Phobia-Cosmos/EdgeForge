# Synthetic EEG LoP positive-control experiment

Status: completed controlled validation; `scientific_conclusion_allowed=false`.

This experiment is an EdgeForge detection positive control, not a claim about ISRUC or real EEG. The generated epochs have shape `(8,3000)` with 16 latent features arranged as 8 channels × 2 temporal half-windows. Source labels use latent feature 0; target subjects use an ordered set of orthogonal latent axes. The positive-control arm pretrains a warm model and then sets its encoder gradient scale to zero before the target stream. The no-lesion arm uses the same data, model, optimizer and seeds without this intervention.

## Dataset and protocol

- Dataset manifest: `/home/undefined/UbuntuData/datasets/edgeforge-synthetic-eeg-lop-positive-control-v1/manifest.json`
- Source subjects: `1,2,3`, 256 epochs each
- Ordered target subjects: `4 → 5 → 6 → 7 → 8 → 9 → 10 → 11`, 256 epochs each
- Each target stage: first 128 epochs for adaptation and last 128 for held-out evaluation
- Retention subjects: `12,13`, 128 epochs each
- Seeds: `4321,4322,4323`
- Source pretraining: 500 Adam updates, learning rate `1e-2`
- Target probe budgets: `0,5,10,25,50`, Adam learning rate `5e-3`
- Warm state is carried across all eight target transitions; fresh is reinitialized at each transition.

## Gate results

| budget | positive-control status | positive transitions | no-lesion status | no-lesion positive transitions |
| ---: | --- | ---: | --- | ---: |
| 5 | `blocked-inconsistent-direction` | 3/8 | `blocked-inconsistent-direction` | 2/8 |
| 10 | `blocked-inconsistent-direction` | 4/8 | `blocked-inconsistent-direction` | 3/8 |
| 25 | `candidate` | 8/8 | `blocked-inconsistent-direction` | 6/8 |
| 50 | `candidate` | 8/8 | `blocked-inconsistent-direction` | 6/8 |

At budget 50, the positive-control gate uses 3 seeds and 8 transitions. The transition means range from `+0.1276` to `+0.3646`, and every seed-level value is strictly positive; the seed-cluster bootstrap lower bounds are also positive. The no-lesion control has negative transitions at `5→6` and `6→7`, so it fails the all-transition requirement.

Gate artifacts:

- [positive-control gate JSON](audit-v1/gate/positive_control.json)
- [positive-control gate Markdown](audit-v1/gate/positive_control.md)
- [no-lesion gate JSON](audit-v1/gate/no_lesion.json)
- [audit summary](audit-v1/audit-summary.json)

## Interpretation

The gate correctly distinguishes the injected warm-model plasticity lesion from the unlesioned control under identical data and seeds. This validates the metric direction and the continuous-stage audit path. It does not show that the lesion is a naturally occurring mechanism, nor does it upgrade the real ISRUC experiment to a LoP conclusion.

## Reproduction

```sh
PYTHONPATH=src /home/undefined/UbuntuData/python-envs/research/bin/python \
  scripts/generate-synthetic-lop-positive-control.py \
  --output /home/undefined/UbuntuData/datasets/edgeforge-synthetic-eeg-lop-positive-control-v1

PYTHONPATH=src /home/undefined/UbuntuData/python-envs/research/bin/python \
  scripts/run-synthetic-lop-positive-control.py \
  --data-root /home/undefined/UbuntuData/datasets/edgeforge-synthetic-eeg-lop-positive-control-v1 \
  --output-root /home/undefined/UbuntuData/ai-storage/EdgeForge/eeg-architecture-lop/synthetic-positive-control-v1

PYTHONPATH=src /home/undefined/UbuntuData/python-envs/research/bin/python \
  scripts/audit-eeg-continuous-lop.py \
  --summary /home/undefined/UbuntuData/ai-storage/EdgeForge/eeg-architecture-lop/synthetic-positive-control-v1/summary.json \
  --output-dir /home/undefined/UbuntuData/ai-storage/EdgeForge/eeg-architecture-lop/synthetic-positive-control-v1/audit-v1 \
  --budget 50
```
