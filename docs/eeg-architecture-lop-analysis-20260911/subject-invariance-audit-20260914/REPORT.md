# Subject-invariance audit

This read-only audit aggregates each 20-epoch sequence to one median profile and computes one-way sequence-level ICC per feature or network dimension. Higher ICC means that sequence profiles from the same subject are more reproducible relative to between-subject variation; it is not a biometric authentication score.

`subject-invariance-comparison.png` compares median ICC and sequence-disjoint identity probes for the expanded waveform bank and each supplied BrainUICL tap. `waveform-feature-family-icc.png` shows which waveform feature families are repeatable.

The top-k probe orders dimensions by ICC before fitting the classifier. If top-k accuracy is not higher than the all-dimension result, extra dimensions are either useful jointly or not harmful in this pilot; if it is higher, unstable dimensions are likely adding drift/noise. This selection is descriptive and must be confirmed on session-disjoint data.

BrainUICL stage names refer to EEG/EOG CNN branches, fusion, Transformer outputs, classifier input and logits. A trained checkpoint is required before interpreting a tap as a learned invariant representation; random-init results only measure architectural signal preservation.
