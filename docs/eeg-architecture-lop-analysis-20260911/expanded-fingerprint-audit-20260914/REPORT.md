# Expanded ISRUC fingerprint analysis

The CPU pilot processed 100 sequences, 2000 epochs and 20 subjects.

The 83-D baseline was expanded to 295 dimensions with robust time-domain/Hjorth, absolute spectral, spectral-shape, PSD-slope and pairwise signed-correlation features.

All expanded features held-out Logistic identity accuracy: 0.8875; first/second subject-profile correlation: 0.8454.

Feature-group ablations are in `expanded-fingerprint-summary.json`. This is a descriptive identity-stability audit, not an LoP result or a biometric authentication claim.
