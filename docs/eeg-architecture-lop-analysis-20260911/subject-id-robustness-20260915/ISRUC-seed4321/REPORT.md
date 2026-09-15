# ISRUC subject identity granularity/open-set probe

This run contains 4,276 complete groups (85,520 epochs). It enrolls 78 subjects and reserves 20 subjects as unseen identities.

Known subjects use half of their complete groups for enrollment and half for held-out identity testing. Unseen subjects are never present in the probe training classes; their concrete subject id is therefore not a valid prediction target. Unknown rejection is reported separately.

| unit | stage | known top-1 | known top-5 | chance | unknown rejection | unknown AUROC |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| sequence | raw_waveform | 0.0383 | 0.1479 | 0.0128 | 0.0449 | 0.5130 |
| sequence | eeg_branch | 0.7546 | 0.9368 | 0.0128 | 0.1371 | 0.6627 |
| sequence | transformer_layer1 | 0.5290 | 0.8480 | 0.0128 | 0.1935 | 0.5862 |
| sequence | classifier_input | 0.3034 | 0.6317 | 0.0128 | 0.1071 | 0.5196 |

## Epoch-count aggregation curve

| k epochs per group | stage | known top-1 | known top-5 |
| ---: | --- | ---: | ---: |
| 1 | raw_waveform | 0.0481 | 0.1479 |
| 1 | eeg_branch | 0.3660 | 0.6821 |
| 1 | transformer_layer1 | 0.4020 | 0.7030 |
| 1 | classifier_input | 0.1856 | 0.4710 |
| 5 | raw_waveform | 0.0470 | 0.1421 |
| 5 | eeg_branch | 0.5870 | 0.8579 |
| 5 | transformer_layer1 | 0.4142 | 0.7639 |
| 5 | classifier_input | 0.2140 | 0.5081 |
| 20 | raw_waveform | 0.0383 | 0.1479 |
| 20 | eeg_branch | 0.7546 | 0.9368 |
| 20 | transformer_layer1 | 0.5290 | 0.8480 |
| 20 | classifier_input | 0.3034 | 0.6317 |

A high known top-1 means a held-out group from an enrolled subject can be assigned its concrete subject id. For an unseen subject, a classifier trained only on known ids cannot produce the true new id; unknown rejection/AUROC instead quantify whether it can detect that the sample is outside enrollment.

This is a subject-information audit, not a LoP outcome. Transformer epoch taps are sequence-contextual and should not be described as a strictly isolated single-epoch measurement.
