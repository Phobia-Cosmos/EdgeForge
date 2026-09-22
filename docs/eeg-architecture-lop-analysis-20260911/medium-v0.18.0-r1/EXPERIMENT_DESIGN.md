# EdgeForge EEG architecture / LoP pilot

## Purpose

Compare the registered EEG architectures under one fixed ISRUC data contract and a matched fresh-versus-checkpoint target adaptation probe. This is a development pilot for architecture screening; it is not a publication-grade LoP claim.

## Dataset

- Release: `eeg-medium/isruc-v0.18.0`
- Contract: each file is `(20, 8, 3000)` float32; labels are `(20,)` int64.
- Source: subjects `1,3,4,6,7,9,10,21` (40 files, 800 epochs).
- Target: subjects `2,11,12,13,14,15,16,17` (40 files, 800 epochs).
- Retention: subjects `5,18,19,20` (20 files, 400 epochs).
- Target split: first 10 epochs of every file for adaptation; last 10 for held-out evaluation.
- All 100 data/label pairs were checked against the manifest SHA-256 values.

## Fixed protocol

- Architectures: `lop_mlp`, `eegnet`, `tcn`, `transformer`, `brainuicl`.
- Seeds: `4321`, `4322`, `4323`.
- Source training: 3 epochs, Adam, learning rate `2e-3`.
- Target adaptation: 5 Adam steps, learning rate `1e-3`, batch size `32`.
- Evaluation: source accuracy, target checkpoint accuracy, target fresh accuracy, macro-F1/loss, retention accuracy and CPU latency.
- Fresh control: same architecture, same target batches, same seed-derived batch order, new initialization.
- Scientific conclusion flag: always `false`.

## Command

```sh
PYTHONPATH=src /home/undefined/UbuntuData/python-envs/research/bin/python \
  /tmp/benchmark-eeg-architectures.py \
  --data-root data/eeg-medium/isruc-v0.18.0 \
  --output-root /home/undefined/UbuntuData/ai-storage/EdgeForge/eeg-architecture-lop/medium-v0.18.0-r1 \
  --architectures lop_mlp eegnet tcn transformer brainuicl \
  --seeds 4321 4322 4323 \
  --epochs 3 --adapt-steps 5 --batch-size 32 \
  --lr 2e-3 --adapt-lr 1e-3 --latency-repeats 5
```

## Interpretation boundary

`fresh_gap = accuracy_fresh - accuracy_checkpoint`. A positive value is only a pilot indicator that the fresh initialization adapted better under this finite budget. The run has one aggregated target adaptation rather than the required 8–10 ordered stages, and the uploaded medium subset is explicitly a development subset. Therefore these values must not be promoted to a formal LoP or scientific conclusion.
