# ISRUC subject identity granularity/open-set probe

This run contains 4,276 complete groups (85,520 epochs). It enrolls 78 subjects and reserves 20 subjects as unseen identities.

Known subjects use half of their complete groups for enrollment and half for held-out identity testing. Unseen subjects are never present in the probe training classes; their concrete subject id is therefore not a valid prediction target. Unknown rejection is reported separately.

| unit | stage | known top-1 | known top-5 | chance | unknown rejection | unknown AUROC |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| epoch | raw_waveform | 0.0328 | 0.1372 | 0.0128 | 0.0627 | 0.5178 |
| epoch | raw_waveform_gain_normalized | 0.0191 | 0.0835 | 0.0128 | 0.0643 | 0.5166 |
| epoch | eeg_branch | 0.5198 | 0.8294 | 0.0128 | 0.0755 | 0.5921 |
| epoch | eog_branch | 0.3449 | 0.6569 | 0.0128 | 0.0612 | 0.5363 |
| epoch | fusion | 0.4906 | 0.7935 | 0.0128 | 0.0812 | 0.5940 |
| epoch | transformer_layer1 | 0.5857 | 0.8667 | 0.0128 | 0.0866 | 0.5833 |
| epoch | transformer_layer2 | 0.5132 | 0.8334 | 0.0128 | 0.0725 | 0.5570 |
| epoch | transformer_layer3 | 0.4709 | 0.8144 | 0.0128 | 0.0663 | 0.5454 |
| epoch | classifier_input | 0.2932 | 0.6211 | 0.0128 | 0.0632 | 0.5228 |
| epoch | logits | 0.0625 | 0.2317 | 0.0128 | 0.0460 | 0.4974 |
| sequence | raw_waveform | 0.0383 | 0.1473 | 0.0128 | 0.0438 | 0.5132 |
| sequence | raw_waveform_gain_normalized | 0.0232 | 0.0876 | 0.0128 | 0.3007 | 0.4986 |
| sequence | eeg_branch | 0.7541 | 0.9379 | 0.0128 | 0.1452 | 0.6633 |
| sequence | eog_branch | 0.5226 | 0.8057 | 0.0128 | 0.0899 | 0.5620 |
| sequence | fusion | 0.6056 | 0.8852 | 0.0128 | 0.1014 | 0.5864 |
| sequence | transformer_layer1 | 0.5284 | 0.8474 | 0.0128 | 0.1912 | 0.5871 |
| sequence | transformer_layer2 | 0.4948 | 0.8144 | 0.0128 | 0.1567 | 0.5622 |
| sequence | transformer_layer3 | 0.4652 | 0.7993 | 0.0128 | 0.1555 | 0.5465 |
| sequence | classifier_input | 0.3022 | 0.6311 | 0.0128 | 0.1060 | 0.5197 |
| sequence | logits | 0.0777 | 0.2570 | 0.0128 | 0.0507 | 0.4952 |

## Epoch-count aggregation curve

| k epochs per group | stage | known top-1 | known top-5 |
| ---: | --- | ---: | ---: |
| 1 | raw_waveform | 0.0476 | 0.1473 |
| 1 | eeg_branch | 0.3648 | 0.6827 |
| 1 | transformer_layer1 | 0.4037 | 0.7013 |
| 1 | classifier_input | 0.1845 | 0.4716 |
| 2 | raw_waveform | 0.0325 | 0.1410 |
| 2 | eeg_branch | 0.5249 | 0.8196 |
| 2 | transformer_layer1 | 0.4066 | 0.7454 |
| 2 | classifier_input | 0.2100 | 0.5180 |
| 5 | raw_waveform | 0.0464 | 0.1421 |
| 5 | eeg_branch | 0.5858 | 0.8573 |
| 5 | transformer_layer1 | 0.4194 | 0.7651 |
| 5 | classifier_input | 0.2135 | 0.5087 |
| 10 | raw_waveform | 0.0389 | 0.1421 |
| 10 | eeg_branch | 0.7082 | 0.9188 |
| 10 | transformer_layer1 | 0.4959 | 0.8144 |
| 10 | classifier_input | 0.2691 | 0.5858 |
| 20 | raw_waveform | 0.0383 | 0.1473 |
| 20 | eeg_branch | 0.7541 | 0.9379 |
| 20 | transformer_layer1 | 0.5284 | 0.8474 |
| 20 | classifier_input | 0.3022 | 0.6311 |

A high known top-1 means a held-out group from an enrolled subject can be assigned its concrete subject id. For an unseen subject, a classifier trained only on known ids cannot produce the true new id; unknown rejection/AUROC instead quantify whether it can detect that the sample is outside enrollment.

This is a subject-information audit, not a LoP outcome. Transformer epoch taps are sequence-contextual and should not be described as a strictly isolated single-epoch measurement.
