# Continuous EEG architecture LoP experiment

## Stream

- Dataset: `eeg-medium/isruc-v0.18.0`
- Source subjects: `1,3,4,6,7,9,10,21`
- Ordered target subjects: `2,11,12,13,14,15,16,17`
- Retention subjects: `5,18,19,20`
- Per-file target split: epochs `0..9` for adaptation and `10..19` for held-out evaluation

## Models and controls

- Architectures: `lop_mlp`, `eegnet`, `tcn`, `transformer`, `brainuicl`
- Seeds: `4321,4322,4323`
- Source pretraining: 3 epochs, Adam, learning rate `2e-3`
- Probe budgets: `0,5,10,25,50` target updates
- Probe optimizer: Adam, learning rate `1e-3`, batch size `32`
- Warm state: source-pretrained model carried through each target stage; after 50 steps its state becomes the next stage's warm checkpoint.
- Fresh state: new random initialization at every target stage, using the same target batches as warm.
- Primary descriptive outcome: final fresh-gap at 50 steps and fresh-vs-warm AULC gap.
- Separate stability outcome: retention accuracy/loss/MF1 on the fixed retention subjects.

## Reproduction

```sh
PYTHONPATH=src /home/undefined/UbuntuData/python-envs/research/bin/python \
  scripts/run-eeg-architecture-continuous-lop.py \
  --data-root data/eeg-medium/isruc-v0.18.0 \
  --output-root /home/undefined/UbuntuData/ai-storage/EdgeForge/eeg-architecture-lop/continuous-medium-v0.18.0-r1 \
  --architectures lop_mlp eegnet tcn transformer brainuicl \
  --seeds 4321 4322 4323 \
  --budgets 0 5 10 25 50 --epochs 3 --batch-size 32 \
  --lr 2e-3 --adapt-lr 1e-3 --device cpu
```

This is a continuous development pilot. The release is a diversity-selected subset, and the run does not by itself authorize a scientific LoP conclusion.
