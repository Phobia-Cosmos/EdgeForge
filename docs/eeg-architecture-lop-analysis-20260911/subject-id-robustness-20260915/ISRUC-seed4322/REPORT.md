# ISRUC subject identity granularity/open-set probe

This run contains 4,276 complete groups (85,520 epochs). It enrolls 78 subjects and reserves 20 subjects as unseen identities.

Known subjects use half of their complete groups for enrollment and half for held-out identity testing. Unseen subjects are never present in the probe training classes; their concrete subject id is therefore not a valid prediction target. Unknown rejection is reported separately.

| unit | stage | known top-1 | known top-5 | chance | unknown rejection | unknown AUROC |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| sequence | raw_waveform | 0.0371 | 0.1439 | 0.0128 | 0.1343 | 0.5441 |
| sequence | eeg_branch | 0.7715 | 0.9443 | 0.0128 | 0.2319 | 0.7231 |
| sequence | transformer_layer1 | 0.5563 | 0.8463 | 0.0128 | 0.2377 | 0.5834 |
| sequence | classifier_input | 0.3016 | 0.6114 | 0.0128 | 0.0643 | 0.5020 |

## Epoch-count aggregation curve

| k epochs per group | stage | known top-1 | known top-5 |
| ---: | --- | ---: | ---: |
| 1 | raw_waveform | 0.0360 | 0.1392 |
| 1 | eeg_branch | 0.3898 | 0.6978 |
| 1 | transformer_layer1 | 0.3712 | 0.7088 |
| 1 | classifier_input | 0.1787 | 0.4397 |
| 5 | raw_waveform | 0.0383 | 0.1473 |
| 5 | eeg_branch | 0.6096 | 0.8706 |
| 5 | transformer_layer1 | 0.4327 | 0.7581 |
| 5 | classifier_input | 0.2193 | 0.5174 |
| 20 | raw_waveform | 0.0371 | 0.1439 |
| 20 | eeg_branch | 0.7715 | 0.9443 |
| 20 | transformer_layer1 | 0.5563 | 0.8463 |
| 20 | classifier_input | 0.3016 | 0.6114 |

A high known top-1 means a held-out group from an enrolled subject can be assigned its concrete subject id. For an unseen subject, a classifier trained only on known ids cannot produce the true new id; unknown rejection/AUROC instead quantify whether it can detect that the sample is outside enrollment.

This is a subject-information audit, not a LoP outcome. Transformer epoch taps are sequence-contextual and should not be described as a strictly isolated single-epoch measurement.
