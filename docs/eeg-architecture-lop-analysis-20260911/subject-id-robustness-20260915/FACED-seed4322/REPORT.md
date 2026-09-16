# FACED subject identity granularity/open-set probe

This run contains 984 complete groups (19,680 epochs). It enrolls 98 subjects and reserves 25 subjects as unseen identities.

Known subjects use half of their complete groups for enrollment and half for held-out identity testing. Unseen subjects are never present in the probe training classes; their concrete subject id is therefore not a valid prediction target. Unknown rejection is reported separately.

| unit | stage | known top-1 | known top-5 | chance | unknown rejection | unknown AUROC |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| sequence | raw_waveform | 0.0281 | 0.0969 | 0.0102 | 1.0000 | 0.5002 |
| sequence | eeg_branch | 0.8648 | 0.9694 | 0.0102 | 0.9900 | 0.7589 |
| sequence | transformer_layer1 | 0.6071 | 0.8801 | 0.0102 | 0.9150 | 0.5826 |
| sequence | classifier_input | 0.2577 | 0.5663 | 0.0102 | 0.6550 | 0.5231 |

## Epoch-count aggregation curve

| k epochs per group | stage | known top-1 | known top-5 |
| ---: | --- | ---: | ---: |
| 1 | raw_waveform | 0.0128 | 0.0638 |
| 1 | eeg_branch | 0.1505 | 0.4209 |
| 1 | transformer_layer1 | 0.2347 | 0.5153 |
| 1 | classifier_input | 0.1199 | 0.3240 |
| 5 | raw_waveform | 0.0230 | 0.0714 |
| 5 | eeg_branch | 0.5102 | 0.8036 |
| 5 | transformer_layer1 | 0.3954 | 0.6964 |
| 5 | classifier_input | 0.1735 | 0.3827 |
| 20 | raw_waveform | 0.0281 | 0.0969 |
| 20 | eeg_branch | 0.8648 | 0.9694 |
| 20 | transformer_layer1 | 0.6071 | 0.8801 |
| 20 | classifier_input | 0.2577 | 0.5663 |

A high known top-1 means a held-out group from an enrolled subject can be assigned its concrete subject id. For an unseen subject, a classifier trained only on known ids cannot produce the true new id; unknown rejection/AUROC instead quantify whether it can detect that the sample is outside enrollment.

This is a subject-information audit, not a LoP outcome. Transformer epoch taps are sequence-contextual and should not be described as a strictly isolated single-epoch measurement.
