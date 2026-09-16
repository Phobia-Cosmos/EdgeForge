# FACED subject identity granularity/open-set probe

This run contains 984 complete groups (19,680 epochs). It enrolls 98 subjects and reserves 25 subjects as unseen identities.

Known subjects use half of their complete groups for enrollment and half for held-out identity testing. Unseen subjects are never present in the probe training classes; their concrete subject id is therefore not a valid prediction target. Unknown rejection is reported separately.

| unit | stage | known top-1 | known top-5 | chance | unknown rejection | unknown AUROC |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| sequence | raw_waveform | 0.0434 | 0.1301 | 0.0102 | 1.0000 | 0.5311 |
| sequence | eeg_branch | 0.8495 | 0.9694 | 0.0102 | 0.9650 | 0.7716 |
| sequence | transformer_layer1 | 0.6403 | 0.8546 | 0.0102 | 0.9200 | 0.6466 |
| sequence | classifier_input | 0.2653 | 0.5587 | 0.0102 | 0.6850 | 0.5273 |

## Epoch-count aggregation curve

| k epochs per group | stage | known top-1 | known top-5 |
| ---: | --- | ---: | ---: |
| 1 | raw_waveform | 0.0230 | 0.0612 |
| 1 | eeg_branch | 0.1709 | 0.3852 |
| 1 | transformer_layer1 | 0.2449 | 0.5306 |
| 1 | classifier_input | 0.1403 | 0.3342 |
| 5 | raw_waveform | 0.0179 | 0.0689 |
| 5 | eeg_branch | 0.5536 | 0.7985 |
| 5 | transformer_layer1 | 0.4260 | 0.7117 |
| 5 | classifier_input | 0.1811 | 0.3673 |
| 20 | raw_waveform | 0.0434 | 0.1301 |
| 20 | eeg_branch | 0.8495 | 0.9694 |
| 20 | transformer_layer1 | 0.6403 | 0.8546 |
| 20 | classifier_input | 0.2653 | 0.5587 |

A high known top-1 means a held-out group from an enrolled subject can be assigned its concrete subject id. For an unseen subject, a classifier trained only on known ids cannot produce the true new id; unknown rejection/AUROC instead quantify whether it can detect that the sample is outside enrollment.

This is a subject-information audit, not a LoP outcome. Transformer epoch taps are sequence-contextual and should not be described as a strictly isolated single-epoch measurement.
