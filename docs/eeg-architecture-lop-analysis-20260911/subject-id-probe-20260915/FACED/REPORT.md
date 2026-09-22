# FACED subject identity probe

This closed-set experiment uses 984 complete sequences/trials from 123 subjects. The target is subject id only; sleep/emotion labels are not loaded.

Enrollment uses a deterministic random half of complete groups per subject and evaluation uses the other half. A high score means a new group from a known subject can be assigned to that subject; it does not identify an unseen subject without an enrollment profile.

| representation | top-1 | balanced top-1 | top-5 | chance | feature dim |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_waveform | 0.0407 | 0.0407 | 0.1037 | 0.0081 | 4096 |
| raw_waveform_gain_normalized | 0.0427 | 0.0427 | 0.1057 | 0.0081 | 4096 |
| eeg_branch | 0.8313 | 0.8313 | 0.9533 | 0.0081 | 512 |
| eog_branch | 0.8313 | 0.8313 | 0.9533 | 0.0081 | 512 |
| fusion | 0.8313 | 0.8313 | 0.9533 | 0.0081 | 512 |
| transformer_layer1 | 0.5772 | 0.5772 | 0.8313 | 0.0081 | 512 |
| transformer_layer2 | 0.4776 | 0.4776 | 0.7480 | 0.0081 | 512 |
| transformer_layer3 | 0.4268 | 0.4268 | 0.6931 | 0.0081 | 512 |
| classifier_input | 0.2480 | 0.2480 | 0.5488 | 0.0081 | 128 |
| logits | 0.0264 | 0.0264 | 0.0833 | 0.0081 | 20 |

The raw waveform stages use sequence-level median mean-pooled samples (128 points per channel), with and without per-epoch gain normalization. The learned stages are sequence-level medians of 20 epoch tokens.

This is an identity separability audit, not a biometric authentication claim and not a LoP outcome. Repeat with session-disjoint and class-balanced splits before interpreting the representation as subject-invariant biology.
