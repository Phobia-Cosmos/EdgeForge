# ISRUC subject identity probe

This closed-set experiment uses 4,276 complete sequences/trials from 98 subjects. The target is subject id only; sleep/emotion labels are not loaded.

Enrollment uses a deterministic random half of complete groups per subject and evaluation uses the other half. A high score means a new group from a known subject can be assigned to that subject; it does not identify an unseen subject without an enrollment profile.

| representation | top-1 | balanced top-1 | top-5 | chance | feature dim |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_waveform | 0.0337 | 0.0313 | 0.1303 | 0.0102 | 1024 |
| raw_waveform_gain_normalized | 0.0245 | 0.0241 | 0.0832 | 0.0102 | 1024 |
| eeg_branch | 0.7093 | 0.7083 | 0.9224 | 0.0102 | 512 |
| eog_branch | 0.4940 | 0.4934 | 0.7833 | 0.0102 | 512 |
| fusion | 0.5901 | 0.5906 | 0.8577 | 0.0102 | 512 |
| transformer_layer1 | 0.5125 | 0.5122 | 0.8068 | 0.0102 | 512 |
| transformer_layer2 | 0.4695 | 0.4691 | 0.7680 | 0.0102 | 512 |
| transformer_layer3 | 0.4381 | 0.4368 | 0.7449 | 0.0102 | 512 |
| classifier_input | 0.2689 | 0.2666 | 0.5508 | 0.0102 | 128 |
| logits | 0.0425 | 0.0404 | 0.1414 | 0.0102 | 20 |

The raw waveform stages use sequence-level median mean-pooled samples (128 points per channel), with and without per-epoch gain normalization. The learned stages are sequence-level medians of 20 epoch tokens.

This is an identity separability audit, not a biometric authentication claim and not a LoP outcome. Repeat with session-disjoint and class-balanced splits before interpreting the representation as subject-invariant biology.
