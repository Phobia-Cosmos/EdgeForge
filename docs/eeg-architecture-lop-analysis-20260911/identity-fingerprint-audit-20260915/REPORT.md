# EEG subject fingerprint analysis

The analysis is read-only and descriptive. Each subject profile contains gain-aware RMS, gain-invariant channel balance, roughness, relative band power, spectral entropy and signed connectivity summaries.

Identity probes use within-subject sequence/trial-disjoint splits: the first half of groups is used to form a subject profile and the second half is held out. A high score means the recorded subject pattern is repeatable across groups; it does not mean a single epoch is a reliable biometric identifier.

The saved `subject_profiles` are intended as constraints for later modifications: sample drift/noise from the subject's own MAD and reject a transformed epoch when its profile leaves the subject median/MAD envelope. Preserve signed connectivity because RMS and PSD alone cannot preserve reference/polarity structure.

All results remain `scientific_conclusion_allowed=false`; identity separability is not evidence of LoP.

## ISRUC

Samples: 85520; subjects: 98; feature dimension: 83.
Nearest-centroid identity accuracy: 0.2372; logistic probe accuracy: 0.7325; between/within profile distance ratio: 0.9542171062710783.
See `ISRUC/subject-fingerprint-pca.png` and `ISRUC/subject-fingerprint-summary.json`.

## FACED

Samples: 3444; subjects: 123; feature dimension: 323.
Nearest-centroid identity accuracy: 0.9413; logistic probe accuracy: 0.9861; between/within profile distance ratio: 2.592782526913515.
See `FACED/subject-fingerprint-pca.png` and `FACED/subject-fingerprint-summary.json`.
