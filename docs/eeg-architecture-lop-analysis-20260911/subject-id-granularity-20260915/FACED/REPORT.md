# FACED subject identity granularity/open-set probe

This run contains 984 complete groups (19,680 epochs). It enrolls 98 subjects and reserves 25 subjects as unseen identities.

Known subjects use half of their complete groups for enrollment and half for held-out identity testing. Unseen subjects are never present in the probe training classes; their concrete subject id is therefore not a valid prediction target. Unknown rejection is reported separately.

| unit | stage | known top-1 | known top-5 | chance | unknown rejection | unknown AUROC |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| epoch | raw_waveform | 0.0165 | 0.0653 | 0.0102 | 0.0420 | 0.4773 |
| epoch | raw_waveform_gain_normalized | 0.0162 | 0.0653 | 0.0102 | 0.0415 | 0.4760 |
| epoch | eeg_branch | 0.4427 | 0.7444 | 0.0102 | 0.0777 | 0.6021 |
| epoch | eog_branch | 0.4427 | 0.7444 | 0.0102 | 0.0777 | 0.6021 |
| epoch | fusion | 0.4427 | 0.7444 | 0.0102 | 0.0777 | 0.6021 |
| epoch | transformer_layer1 | 0.6682 | 0.8820 | 0.0102 | 0.8525 | 0.6795 |
| epoch | transformer_layer2 | 0.5811 | 0.8288 | 0.0102 | 0.8417 | 0.6273 |
| epoch | transformer_layer3 | 0.5321 | 0.7990 | 0.0102 | 0.8385 | 0.6178 |
| epoch | classifier_input | 0.2437 | 0.5406 | 0.0102 | 0.1520 | 0.5422 |
| epoch | logits | 0.1355 | 0.3703 | 0.0102 | 0.0432 | 0.5012 |
| sequence | raw_waveform | 0.0485 | 0.1199 | 0.0102 | 1.0000 | 0.5077 |
| sequence | raw_waveform_gain_normalized | 0.0485 | 0.1199 | 0.0102 | 1.0000 | 0.5090 |
| sequence | eeg_branch | 0.8622 | 0.9617 | 0.0102 | 0.9750 | 0.8251 |
| sequence | eog_branch | 0.8622 | 0.9617 | 0.0102 | 0.9750 | 0.8251 |
| sequence | fusion | 0.8622 | 0.9617 | 0.0102 | 0.9750 | 0.8251 |
| sequence | transformer_layer1 | 0.6378 | 0.8546 | 0.0102 | 0.9550 | 0.7033 |
| sequence | transformer_layer2 | 0.5434 | 0.7679 | 0.0102 | 0.9600 | 0.6509 |
| sequence | transformer_layer3 | 0.4872 | 0.7398 | 0.0102 | 0.9450 | 0.6136 |
| sequence | classifier_input | 0.2755 | 0.6276 | 0.0102 | 0.6450 | 0.5381 |
| sequence | logits | 0.2474 | 0.5510 | 0.0102 | 0.0500 | 0.5542 |

## Epoch-count aggregation curve

| k epochs per group | stage | known top-1 | known top-5 |
| ---: | --- | ---: | ---: |
| 1 | raw_waveform | 0.0128 | 0.0536 |
| 1 | eeg_branch | 0.1352 | 0.3827 |
| 1 | transformer_layer1 | 0.2066 | 0.4949 |
| 1 | classifier_input | 0.1148 | 0.3444 |
| 2 | raw_waveform | 0.0204 | 0.0867 |
| 2 | eeg_branch | 0.3010 | 0.6097 |
| 2 | transformer_layer1 | 0.3597 | 0.6607 |
| 2 | classifier_input | 0.1633 | 0.3648 |
| 5 | raw_waveform | 0.0204 | 0.0791 |
| 5 | eeg_branch | 0.5944 | 0.8342 |
| 5 | transformer_layer1 | 0.4821 | 0.7500 |
| 5 | classifier_input | 0.1786 | 0.4490 |
| 10 | raw_waveform | 0.0306 | 0.1122 |
| 10 | eeg_branch | 0.7526 | 0.9337 |
| 10 | transformer_layer1 | 0.5383 | 0.8189 |
| 10 | classifier_input | 0.2755 | 0.5918 |
| 20 | raw_waveform | 0.0485 | 0.1199 |
| 20 | eeg_branch | 0.8622 | 0.9617 |
| 20 | transformer_layer1 | 0.6378 | 0.8546 |
| 20 | classifier_input | 0.2755 | 0.6276 |

A high known top-1 means a held-out group from an enrolled subject can be assigned its concrete subject id. For an unseen subject, a classifier trained only on known ids cannot produce the true new id; unknown rejection/AUROC instead quantify whether it can detect that the sample is outside enrollment.

This is a subject-information audit, not a LoP outcome. Transformer epoch taps are sequence-contextual and should not be described as a strictly isolated single-epoch measurement.
