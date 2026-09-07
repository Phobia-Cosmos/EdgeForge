EdgeForge 0.19.1 records the calibration rerun after fixing the ISRUC input-unit scale.
The processed float32 amplitudes are multiplied by 100000.0 at every train, eval and retention boundary.
Calibration matrix: five architectures, three independent seeds, random-a order, first 12 target subjects.
All 15 trajectories completed and produced 1,080/1,080 structural budget cells.
The original strict audit is retained externally under analysis/calibration; it reported 852 valid cells because
the old global 10-seed/50-stage and per-stage fresh-gain rules were applied to a deliberately smaller calibration phase.
The phase-aware audit is analysis/calibration-v2: candidate-evidence-ready, with 1,080 valid cells and learning/sample gates passed.
Calibration is a protocol-readiness result only. It does not authorize a scientific LoP conclusion.
The formal architecture set remains tcn, transformer and brainuicl; lop_mlp and eegnet remain calibration controls.
The source/fresh summaries, checkpoints and full manifests are stored outside Git in the versioned experiment output root.
Calibration lock: `/home/undefined/ai-storage/EdgeForge/eeg-lop-full/v0.19.1/protocol/calibration-lock.json`.
Lock digest: `2385b24bcd11c17147e59e7df5fda1793e9b724fa1266f252047d3208ef9de4e`.
