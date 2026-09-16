# ISRUC subject identity granularity/open-set probe

This run contains 4,276 complete groups (85,520 epochs). It enrolls 78 subjects and reserves 20 subjects as unseen identities.

Known subjects use half of their complete groups for enrollment and half for held-out identity testing. Unseen subjects are never present in the probe training classes; their concrete subject id is therefore not a valid prediction target. Unknown rejection is reported separately.

| unit | stage | known top-1 | known top-5 | chance | unknown rejection | unknown AUROC |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| sequence | raw_waveform | 0.0445 | 0.1601 | 0.0128 | 0.0233 | 0.4289 |
| sequence | eeg_branch | 0.7572 | 0.9318 | 0.0128 | 0.1515 | 0.6609 |
| sequence | transformer_layer1 | 0.5491 | 0.8474 | 0.0128 | 0.1678 | 0.5728 |
| sequence | classifier_input | 0.3139 | 0.6439 | 0.0128 | 0.0828 | 0.5212 |

## Epoch-count aggregation curve

| k epochs per group | stage | known top-1 | known top-5 |
| ---: | --- | ---: | ---: |
| 1 | raw_waveform | 0.0341 | 0.1491 |
| 1 | eeg_branch | 0.3590 | 0.6954 |
| 1 | transformer_layer1 | 0.3572 | 0.6919 |
| 1 | classifier_input | 0.1832 | 0.4827 |
| 5 | raw_waveform | 0.0382 | 0.1405 |
| 5 | eeg_branch | 0.5919 | 0.8705 |
| 5 | transformer_layer1 | 0.4249 | 0.7682 |
| 5 | classifier_input | 0.2162 | 0.5341 |
| 20 | raw_waveform | 0.0445 | 0.1601 |
| 20 | eeg_branch | 0.7572 | 0.9318 |
| 20 | transformer_layer1 | 0.5491 | 0.8474 |
| 20 | classifier_input | 0.3139 | 0.6439 |

A high known top-1 means a held-out group from an enrolled subject can be assigned its concrete subject id. For an unseen subject, a classifier trained only on known ids cannot produce the true new id; unknown rejection/AUROC instead quantify whether it can detect that the sample is outside enrollment.

This is a subject-information audit, not a LoP outcome. Transformer epoch taps are sequence-contextual and should not be described as a strictly isolated single-epoch measurement.
