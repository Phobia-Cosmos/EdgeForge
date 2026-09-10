# EEG state-conditioned steering pilot

This experiment is a controlled mechanism probe for the EEG LoP question. It trains EEGNet on source subjects, measures the temporal ReLU filter coverage, chooses low-coverage filters, and constructs four target-train conditions: clean, equal-RMS band-limited random perturbation, spectrum-matched random-phase perturbation, and a perturbation optimized against the selected filters. Perturbations are constrained to 0.5–40 Hz and 5% of the epoch RMS; labels and held-out evaluation epochs are unchanged.

Run it with:

```sh
PYTHONPATH=src /home/undefined/UbuntuData/python-envs/research/bin/python \
  scripts/run-eeg-state-conditioned-steering.py \
  --data-root data/eeg-medium/isruc-v0.18.0 \
  --output-root /home/undefined/UbuntuData/ai-storage/EdgeForge/eeg-architecture-lop/state-conditioned-steering-pilot-20260911
```

The first pilot used three source subjects, one target subject, one retention subject, one source epoch pass and budgets 0/5/10. Four low-coverage filters already had active fraction 0 on the calibration batch, so the optimizer could not suppress them further. All four conditions therefore had the same fresh-gap: 0.0 at budget 5 and −0.2 at budget 10. This is a negative mechanism result, not an LoP finding: the pilot is too short and the selected units are already inactive. The next useful run should use more source epochs and at least two target transitions, then add the dormant-unit reset counterfactual.

For EEGNet, `dead` means a ReLU filter whose pre-activation is non-positive for essentially all calibration samples. For TCN and BrainUICL, the main nonlinearities are GELU; report near-zero, low variance and low gradient reachability instead of permanent dead-unit claims. A causal LoP interpretation requires the state change to precede a stable multi-seed/multi-transition positive fresh-gap and to be reversed by reset/reactivation.

`summary.json` is stored in the experiment archive under `analysis-configs-20260911/state-conditioned-steering-pilot-20260911/` because the archive organization keeps image directories human-readable. The organization map is `analysis-configs-20260911/organization-manifest.json`.
