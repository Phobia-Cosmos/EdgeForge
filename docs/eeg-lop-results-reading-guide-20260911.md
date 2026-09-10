# EEG LoP results reading guide

Use the generated archive at `/home/undefined/UbuntuData/ai-storage/EdgeForge/eeg-architecture-lop/`. Image directories retain their PNG files and `EXPLANATION.md`; JSON, reports, manifests and logs were moved to `analysis-configs-20260911/` with the same relative paths. `organization-manifest.json` is the reversible move map.

1. Start with `full-isruc-visualizations-v3`: waveform, normalized overlay, spectra, RMS trajectory, subject distance, gain-invariant PCA and label prior. These answer whether the cohorts differ in signal scale, shape, frequency content or label composition. They do not establish LoP.
2. Read the clean `continuous-medium-v0.18.0-r2` gate by architecture, seed and target transition. The outcome is `fresh_gap = accuracy_fresh - accuracy_warm`; positive means fresh adapts better. The strict gate requires at least three seeds, every transition and positive bootstrap lower bound. The current architectures are blocked.
3. Compare the paired perturbation reports for RMS equalization, SNR20, gain/baseline drift and cross-talk. Larger mean gaps with mixed cells indicate transfer sensitivity, not natural LoP.
4. Read `rms-equalized-tcn-source-epochs10-diagnostics-v1/diagnostics-report`. Effective/stable rank, gradient reachability, parameter movement and near-zero activation are mechanism diagnostics. The existing run has no monotonic rank collapse and no classifier near-zero increase.
5. Use synthetic gradient lesion, label reversal and polarity as controls. They validate the gate or isolate negative transfer; they are not ISRUC evidence.
6. Read the real channel-polarity pilot and `analysis-configs-20260911/state-conditioned-steering-pilot-20260911/summary.json`. The polarity pilot is blocked. The short steering pilot selected filters that were already inactive, so the four conditions were indistinguishable. A useful next run needs longer source training, multiple target transitions and a dormant-unit reset counterfactual.

The local machine produced the medium-subset quality, visualization and CPU pilots. The remote machine produced the complete ISRUC visualizations; Slurm job 196292 did not run because A100 node193 was `DOWN* / Not responding`. Do not report that job as a completed full GPU LoP experiment.
