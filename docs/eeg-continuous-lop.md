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

Visualize the EEG drift directly (the command only reads the local ISRUC subset and writes derived figures outside Git):

```sh
PYTHONPATH=src /home/undefined/UbuntuData/python-envs/research/bin/python \
  scripts/visualize-eeg-drift.py \
  --data-root data/eeg-medium/isruc-v0.18.0 \
  --output-dir /home/undefined/UbuntuData/ai-storage/EdgeForge/eeg-architecture-lop/continuous-medium-v0.18.0-r2/visualizations-v1
```

For another subject split, pass `--source-subjects`, `--target-subjects`, and `--retention-subjects`. This keeps the visualized cohorts aligned with the experiment summary instead of assuming the medium development split.

The output contains raw/per-epoch-standardized waveforms, channel-averaged mean-per-epoch Welch spectra, an epoch-level RMS heatmap with the adaptation/evaluation boundary, PCA of log-RMS and band-power features, and an adaptation/evaluation label-prior heatmap. `visualization-summary.json` additionally records the target RMS max/min ratio, per-subject delta/theta/alpha/sigma/beta fractions, and Jensen–Shannon spectral distance to the target median. These are drift diagnostics, not LoP outcomes; a visual separation can be caused by amplitude scale, frequency content, or label prior and must be paired with equal-budget fresh/warm probes.

For a controlled, label-preserving perturbation pilot, first derive conditions outside the repository:

```sh
PYTHONPATH=src /home/undefined/UbuntuData/python-envs/research/bin/python \
  scripts/prepare-eeg-lop-conditions.py \
  --input-root data/eeg-medium/isruc-v0.18.0 \
  --output-root /home/undefined/UbuntuData/datasets/edgeforge-isruc-medium-rms-equalized-v1 \
  --condition rms_equalized

PYTHONPATH=src /home/undefined/UbuntuData/python-envs/research/bin/python \
  scripts/prepare-eeg-lop-conditions.py \
  --input-root data/eeg-medium/isruc-v0.18.0 \
  --output-root /home/undefined/UbuntuData/datasets/edgeforge-isruc-medium-target-snr20-v1 \
  --condition target_snr20_noise
```

Run the same architecture/seed/budget grid on each derived root, then compare with `scripts/summarize-eeg-condition-experiment.py`. In the current TCN pilot, RMS equalization raised mean fresh-gap from `+0.0208` to `+0.1058` at budget 25 and from `-0.0075` to `+0.1092` at budget 50; 20 dB target noise yielded `+0.0683` and `+0.1042`. Both still contain negative/zero stage-seed cells and remain `blocked-inconsistent-direction`. This is evidence of transfer/preprocessing sensitivity, not an induced LoP result.

The utility checks are intentionally strict: RMS equalization has file-level correlation ≈1.0 and only changes the acquisition-gain scalar; target SNR-20 noise has minimum file-level correlation ≈0.995, leaves labels/epoch boundaries unchanged and does not clip samples. The additional `target_gain_drift10` condition applies a smooth ±10% envelope and is checked separately. Stronger perturbations (lower SNR, aggressive time warps, channel permutations or label edits) should not be called “usable EEG” without a separate signal-quality and task-utility gate.

To make the perturbation dose response auditable, run:

```sh
PYTHONPATH=src /home/undefined/UbuntuData/python-envs/research/bin/python \
  scripts/plot-eeg-lop-dose-curve.py \
  --baseline /home/undefined/UbuntuData/ai-storage/EdgeForge/eeg-architecture-lop/continuous-medium-v0.18.0-r2/summary.json \
  --condition rms_equalized=/home/undefined/UbuntuData/ai-storage/EdgeForge/eeg-architecture-lop/condition-rms-equalized-tcn-v1/summary.json \
  --condition target_snr20_noise=/home/undefined/UbuntuData/ai-storage/EdgeForge/eeg-architecture-lop/condition-target-snr20-tcn-v1/summary.json \
  --condition target_snr15_noise=/home/undefined/UbuntuData/ai-storage/EdgeForge/eeg-architecture-lop/condition-target-snr15-tcn-v1/summary.json \
  --condition target_snr10_noise=/home/undefined/UbuntuData/ai-storage/EdgeForge/eeg-architecture-lop/condition-target-snr10-tcn-v1/summary.json \
  --condition target_gain_drift10=/home/undefined/UbuntuData/ai-storage/EdgeForge/eeg-architecture-lop/condition-target-gain-drift10-tcn-v1/summary.json \
  --output-dir /home/undefined/UbuntuData/ai-storage/EdgeForge/eeg-architecture-lop/condition-dose-curve-tcn-v1
```

The continuous runner can optionally record compact checkpoint diagnostics at every budget:

```sh
PYTHONPATH=src /home/undefined/UbuntuData/python-envs/research/bin/python \
  scripts/run-eeg-architecture-continuous-lop.py \
  --data-root /home/undefined/UbuntuData/datasets/edgeforge-isruc-medium-rms-equalized-v1 \
  --output-root /home/undefined/UbuntuData/ai-storage/EdgeForge/eeg-architecture-lop/rms-equalized-tcn-source-epochs10-diagnostics-v1 \
  --architectures tcn --seeds 4321 4322 4323 --epochs 10 --device cpu \
  --checkpoint-diagnostics --diagnostic-max-observations 128

PYTHONPATH=src /home/undefined/UbuntuData/python-envs/research/bin/python \
  scripts/summarize-eeg-checkpoint-diagnostics.py \
  --summary /home/undefined/UbuntuData/ai-storage/EdgeForge/eeg-architecture-lop/rms-equalized-tcn-source-epochs10-diagnostics-v1/summary.json \
  --output-dir /home/undefined/UbuntuData/ai-storage/EdgeForge/eeg-architecture-lop/rms-equalized-tcn-source-epochs10-diagnostics-v1/diagnostics-report
```

The 2026-09-06 run used 10 source epochs with RMS equalization. Its budget-25 and budget-50 gates both remained `blocked-inconsistent-direction`; the warm embedding effective-rank changed from about 3.09 at budget 0 to 2.99 at budget 50, and the exploratory warm-initial-rank/final-fresh-gap Pearson was -0.112 over 24 stage-seed pairs. These diagnostics do not establish a causal LoP mechanism.

To compare learning strategies under the same target stream, select `--adaptation-strategy plain`, `source_replay`, `l2_sp`, or `replay_l2_sp`. Source replay mixes the target and source losses as `(1-r)·target_loss + r·source_loss` with `r=0.25` by default. L2-SP adds `0.5·lambda·||θ−θ₀||²` using the parameters at the start of each probe as `θ₀`; the default `lambda` is `1e-4`. The source replay batches are deterministic and shared by warm/fresh probes. Each run records the strategy, mixture losses, regularization penalty, and retention metrics.

Example strategy run:

```sh
PYTHONPATH=src /home/undefined/UbuntuData/python-envs/research/bin/python \
  scripts/run-eeg-architecture-continuous-lop.py \
  --data-root data/eeg-medium/isruc-v0.18.0 \
  --output-root /home/undefined/UbuntuData/ai-storage/EdgeForge/eeg-architecture-lop/strategy-source-replay-tcn-v1 \
  --architectures tcn --seeds 4321 4322 4323 \
  --adaptation-strategy source_replay --replay-ratio 0.25 \
  --budgets 0 5 10 25 50 --epochs 3 --device cpu
```

Pair matched summaries with `scripts/summarize-eeg-adaptation-strategies.py` using `--strategy plain=...`, `source_replay=...`, `l2_sp=...`, and `replay_l2_sp=...`. The utility rejects mismatched protocol fingerprints and reports mean fresh-gap, positive/zero/negative cells, paired changes versus plain, and warm retention changes. It always writes `scientific_conclusion_allowed=false`; a strategy is not a LoP result unless the existing strict all-seed/all-transition gate passes.

This experiment uses the diversity-selected ISRUC medium development subset and compact CPU model settings. It is an auditable protocol validation and architecture comparison, not a formal cohort-level LoP claim.
