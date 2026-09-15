# ISRUC subject identity probe

This closed-set experiment uses 300 complete sequences/trials from 98 subjects. The target is subject id only; sleep/emotion labels are not loaded.

Enrollment uses a deterministic random half of complete groups per subject and evaluation uses the other half. A high score means a new group from a known subject can be assigned to that subject; it does not identify an unseen subject without an enrollment profile.

| representation | top-1 | balanced top-1 | top-5 | chance | feature dim |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_waveform | 0.0153 | 0.0153 | 0.0510 | 0.0102 | 1024 |
| raw_waveform_gain_normalized | 0.0204 | 0.0204 | 0.1173 | 0.0102 | 1024 |
| eeg_branch | 0.2551 | 0.2551 | 0.4337 | 0.0102 | 512 |
| eog_branch | 0.1735 | 0.1735 | 0.4031 | 0.0102 | 512 |
| fusion | 0.2653 | 0.2653 | 0.4184 | 0.0102 | 512 |
| transformer_layer1 | 0.2296 | 0.2296 | 0.3776 | 0.0102 | 512 |
| transformer_layer2 | 0.1939 | 0.1939 | 0.3827 | 0.0102 | 512 |
| transformer_layer3 | 0.2143 | 0.2143 | 0.3469 | 0.0102 | 512 |
| classifier_input | 0.1735 | 0.1735 | 0.3367 | 0.0102 | 128 |
| logits | 0.0612 | 0.0612 | 0.1837 | 0.0102 | 20 |

The raw waveform stages use sequence-level median mean-pooled samples (128 points per channel), with and without per-epoch gain normalization. The learned stages are sequence-level medians of 20 epoch tokens.

This is an identity separability audit, not a biometric authentication claim and not a LoP outcome. Repeat with session-disjoint and class-balanced splits before interpreting the representation as subject-invariant biology.
