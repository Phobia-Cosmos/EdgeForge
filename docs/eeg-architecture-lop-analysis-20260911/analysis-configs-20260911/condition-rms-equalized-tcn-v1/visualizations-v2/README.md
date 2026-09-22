# EEG drift visualizations

These figures visualize signal-scale, spectral, feature-space, within-stream and label-prior drift in the ISRUC medium development subset. They are descriptive diagnostics, not LoP evidence.

- `eeg-drift-waveforms.png`: raw channel-0 waveforms on a common y-scale (RMS is shown in each title) alongside per-epoch standardized shape.
- `eeg-drift-normalized-overlay.png`: normalized first-epoch waveforms overlaid to compare shape directly after removing gain.
- `eeg-drift-spectra.png`: mean per-epoch, channel-averaged Welch power spectra for target subjects.
- `eeg-drift-rms-trajectory.png`: epoch-level RMS heatmap; the dashed line separates adaptation and held-out halves.
- `eeg-drift-subject-distance.png`: pairwise standardized feature distance between target subjects.
- `eeg-drift-pca.png`: PCA of log-RMS and band-power features.
- `eeg-drift-labels.png`: adaptation/evaluation label fractions.
- `visualization-summary.json`: reproducible paths and quantitative RMS/spectral/feature-drift summaries.
