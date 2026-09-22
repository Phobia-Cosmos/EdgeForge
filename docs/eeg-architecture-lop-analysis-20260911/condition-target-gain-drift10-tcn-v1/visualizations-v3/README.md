# EEG drift visualizations

These figures visualize signal-scale, spectral, feature-space, within-stream and label-prior drift in the ISRUC cohort. They are descriptive diagnostics, not LoP evidence. Arrays are read as-is at 100 Hz; this script does not apply a band-pass filter or resampling. Each epoch is 3000 samples (30 seconds).

- `eeg-drift-waveforms.png`: complete 30-second representative epoch on a per-subject raw y-scale plus standardized shape.
- `eeg-subject-{subject}-all-epochs.png`: all epochs and all 8 channels for one subject; color is robust subject-normalized amplitude, black lines mark files and white dotted lines mark adaptation/evaluation.
- `eeg-drift-normalized-overlay.png`: complete normalized 30-second representative epoch overlaid across subjects after removing gain.
- `eeg-drift-spectra.png`: mean per-epoch, channel-averaged Welch power spectra for target subjects.
- `eeg-drift-rms-trajectory.png`: epoch-level RMS heatmap with every file boundary and per-file adaptation/evaluation split.
- `eeg-drift-subject-distance.png`: pairwise standardized feature distance between target subjects.
- `eeg-drift-pca.png`: raw-scale PCA of 8 log channel-RMS and 5 log band-power features; separation is scale/power-sensitive.
- `eeg-drift-pca-gain-invariant.png`: PCA after per-epoch RMS normalization; remaining separation reflects channel balance, temporal roughness and spectral composition.
- `eeg-drift-labels.png`: class fractions (class count divided by epochs in that file split), separately for adaptation and held-out evaluation.
- `visualization-summary.json`: reproducible metadata and quantitative RMS/spectral/feature-drift summaries.
