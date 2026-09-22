# ISRUC subject identity probe

This closed-set experiment uses 4,276 complete sequences/trials from 98 subjects. The target is subject id only; sleep/emotion labels are not loaded.

Enrollment uses a deterministic random half of complete groups per subject and evaluation uses the other half. A high score means a new group from a known subject can be assigned to that subject; it does not identify an unseen subject without an enrollment profile.

| representation | top-1 | balanced top-1 | top-5 | chance | feature dim |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_waveform | 0.0337 | 0.0313 | 0.1303 | 0.0102 | 1024 |
| raw_waveform_gain_normalized | 0.0245 | 0.0241 | 0.0832 | 0.0102 | 1024 |
| eeg_branch | 0.7195 | 0.7176 | 0.9168 | 0.0102 | 512 |
| eog_branch | 0.4908 | 0.4903 | 0.7768 | 0.0102 | 512 |
| fusion | 0.5933 | 0.5932 | 0.8623 | 0.0102 | 512 |
| transformer_layer1 | 0.5143 | 0.5128 | 0.8332 | 0.0102 | 512 |
| transformer_layer2 | 0.4621 | 0.4608 | 0.7934 | 0.0102 | 512 |
| transformer_layer3 | 0.4344 | 0.4329 | 0.7662 | 0.0102 | 512 |
| classifier_input | 0.2791 | 0.2755 | 0.5661 | 0.0102 | 128 |
| logits | 0.0457 | 0.0446 | 0.1617 | 0.0102 | 20 |

The raw waveform stages use sequence-level median mean-pooled samples (128 points per channel), with and without per-epoch gain normalization. The learned stages are sequence-level medians of 20 epoch tokens.

This is an identity separability audit, not a biometric authentication claim and not a LoP outcome. Repeat with session-disjoint and class-balanced splits before interpreting the representation as subject-invariant biology.
