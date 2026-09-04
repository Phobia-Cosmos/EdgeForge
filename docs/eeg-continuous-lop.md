# Continuous EEG LoP integration

EdgeForge now includes a local adapter and one-command audit for the continuous EEG architecture experiment. The runner compares a warm model carried through an ordered subject stream with a fresh re-initialization at every subject, using the same target batches and fixed adaptation budgets. The adapter does not modify the raw summary or dataset.

## Commands

Run the continuous experiment with the research environment:

```sh
PYTHONPATH=src /home/undefined/UbuntuData/python-envs/research/bin/python \
  scripts/run-eeg-architecture-continuous-lop.py \
  --data-root data/eeg-medium/isruc-v0.18.0 \
  --output-root /home/undefined/UbuntuData/ai-storage/EdgeForge/eeg-architecture-lop/continuous-medium-v0.18.0-r2 \
  --architectures lop_mlp eegnet tcn transformer brainuicl \
  --seeds 4321 4322 4323 --budgets 0 5 10 25 50 \
  --epochs 3 --batch-size 32 --lr 2e-3 --adapt-lr 1e-3 --device cpu
```

Convert the summary and run the standard LoP gate for every architecture in one local command:

```sh
PYTHONPATH=src /home/undefined/UbuntuData/python-envs/research/bin/python \
  scripts/audit-eeg-continuous-lop.py \
  --summary /home/undefined/UbuntuData/ai-storage/EdgeForge/eeg-architecture-lop/continuous-medium-v0.18.0-r2/summary.json \
  --output-dir /home/undefined/UbuntuData/ai-storage/EdgeForge/eeg-architecture-lop/continuous-medium-v0.18.0-r2/audit-v1
```

The adapter emits one standard trajectory result per architecture/seed and writes `trajectory-catalog.json`. The primary metric is `task.plasticity.fresh_gap` at the selected budget, with `metric_role=outcome` and explicit source/target stage indices. Retention metrics are tagged `metric_role=retention` and are reported only as an old-task stability inventory; they cannot satisfy the LoP gate.

The current development run has 15 trajectories, 120 stages, and 600 budget-level observations. Its gate uses three seeds and eight transitions per architecture. No architecture passes the strict requirement that every seed at every transition has a positive fresh-gap with a strictly positive seed-cluster bootstrap lower bound. The gate therefore returns `insufficient-direction` or `blocked-inconsistent-direction`, while always recording `scientific_conclusion_allowed=false`.

The same one-command audit can be repeated at budgets 5, 10, 25, and 50. In the current run, BrainUICL has aggregate fresh-gap means of +0.0875 and +0.0408 at budgets 5 and 10, while TCN has +0.0200 and +0.0208 at budgets 10 and 25. These aggregate positives still contain zero or negative transition/seed cells and do not pass the strict gate. The complete sweep is stored under `audit-budgets/budget-{5,10,25,50}` in the result archive.

Generate a subject-level data profile and exploratory gap associations:

```sh
PYTHONPATH=src /home/undefined/UbuntuData/python-envs/research/bin/python \
  scripts/analyze-eeg-continuous-lop.py \
  --summary /home/undefined/UbuntuData/ai-storage/EdgeForge/eeg-architecture-lop/continuous-medium-v0.18.0-r2/summary.json \
  --data-root data/eeg-medium/isruc-v0.18.0 \
  --output-dir /home/undefined/UbuntuData/ai-storage/EdgeForge/eeg-architecture-lop/continuous-medium-v0.18.0-r2/data-analysis-v1
```

The profile includes signal RMS, per-channel RMS, temporal-difference RMS, class entropy, train/evaluation label divergence, and distance from the source-subject feature center. It also checks source accuracy against a constant-class baseline and joins these features with fresh-gap at each budget. Correlations are exploratory only.

This experiment uses the diversity-selected ISRUC medium development subset and compact CPU model settings. It is an auditable protocol validation and architecture comparison, not a formal cohort-level LoP claim.
