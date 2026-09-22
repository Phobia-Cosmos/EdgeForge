# FACED subject identity granularity/open-set probe

This run contains 984 complete groups (19,680 epochs). It enrolls 98 subjects and reserves 25 subjects as unseen identities.

Known subjects use half of their complete groups for enrollment and half for held-out identity testing. Unseen subjects are never present in the probe training classes; their concrete subject id is therefore not a valid prediction target. Unknown rejection is reported separately.

| unit | stage | known top-1 | known top-5 | chance | unknown rejection | unknown AUROC |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| sequence | raw_waveform | 0.0485 | 0.1199 | 0.0102 | 1.0000 | 0.5076 |
| sequence | eeg_branch | 0.8622 | 0.9617 | 0.0102 | 0.9750 | 0.8251 |
| sequence | transformer_layer1 | 0.6378 | 0.8546 | 0.0102 | 0.9550 | 0.7033 |
| sequence | classifier_input | 0.2781 | 0.6276 | 0.0102 | 0.6450 | 0.5381 |

## Epoch-count aggregation curve

| k epochs per group | stage | known top-1 | known top-5 |
| ---: | --- | ---: | ---: |
| 1 | raw_waveform | 0.0128 | 0.0536 |
| 1 | eeg_branch | 0.1352 | 0.3827 |
| 1 | transformer_layer1 | 0.2092 | 0.4949 |
| 1 | classifier_input | 0.1148 | 0.3444 |
| 5 | raw_waveform | 0.0204 | 0.0791 |
| 5 | eeg_branch | 0.5944 | 0.8342 |
| 5 | transformer_layer1 | 0.4821 | 0.7500 |
| 5 | classifier_input | 0.1786 | 0.4490 |
| 20 | raw_waveform | 0.0485 | 0.1199 |
| 20 | eeg_branch | 0.8622 | 0.9617 |
| 20 | transformer_layer1 | 0.6378 | 0.8546 |
| 20 | classifier_input | 0.2781 | 0.6276 |

A high known top-1 means a held-out group from an enrolled subject can be assigned its concrete subject id. For an unseen subject, a classifier trained only on known ids cannot produce the true new id; unknown rejection/AUROC instead quantify whether it can detect that the sample is outside enrollment.

This is a subject-information audit, not a LoP outcome. Transformer epoch taps are sequence-contextual and should not be described as a strictly isolated single-epoch measurement.
