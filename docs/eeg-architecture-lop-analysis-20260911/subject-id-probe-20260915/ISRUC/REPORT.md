# ISRUC subject identity probe

This closed-set experiment uses 4,276 complete sequences/trials from 98 subjects. The target is subject id only; sleep/emotion labels are not loaded.

Enrollment uses a deterministic random half of complete groups per subject and evaluation uses the other half. A high score means a new group from a known subject can be assigned to that subject; it does not identify an unseen subject without an enrollment profile.

| representation | top-1 | balanced top-1 | top-5 | chance | feature dim |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_waveform | 0.0347 | 0.0331 | 0.1275 | 0.0102 | 1024 |
| raw_waveform_gain_normalized | 0.0189 | 0.0190 | 0.0772 | 0.0102 | 1024 |
| eeg_branch | 0.7223 | 0.7209 | 0.9210 | 0.0102 | 512 |
| eog_branch | 0.4806 | 0.4802 | 0.7699 | 0.0102 | 512 |
| fusion | 0.5698 | 0.5694 | 0.8512 | 0.0102 | 512 |
| transformer_layer1 | 0.5129 | 0.5123 | 0.8230 | 0.0102 | 512 |
| transformer_layer2 | 0.4570 | 0.4562 | 0.7884 | 0.0102 | 512 |
| transformer_layer3 | 0.4261 | 0.4253 | 0.7703 | 0.0102 | 512 |
| classifier_input | 0.2657 | 0.2624 | 0.5841 | 0.0102 | 128 |
| logits | 0.0471 | 0.0446 | 0.1673 | 0.0102 | 20 |

The raw waveform stages use sequence-level median mean-pooled samples (128 points per channel), with and without per-epoch gain normalization. The learned stages are sequence-level medians of 20 epoch tokens.

This is an identity separability audit, not a biometric authentication claim and not a LoP outcome. Repeat with session-disjoint and class-balanced splits before interpreting the representation as subject-invariant biology.
