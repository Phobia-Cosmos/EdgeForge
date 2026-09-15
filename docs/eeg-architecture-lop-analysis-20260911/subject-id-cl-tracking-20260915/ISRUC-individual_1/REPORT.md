# ISRUC subject identity granularity/open-set probe

This run contains 4,276 complete groups (85,520 epochs). It enrolls 78 subjects and reserves 20 subjects as unseen identities.

Known subjects use half of their complete groups for enrollment and half for held-out identity testing. Unseen subjects are never present in the probe training classes; their concrete subject id is therefore not a valid prediction target. Unknown rejection is reported separately.

| unit | stage | known top-1 | known top-5 | chance | unknown rejection | unknown AUROC |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| sequence | eeg_branch | 0.7645 | 0.9420 | 0.0128 | 0.1417 | 0.6669 |
| sequence | transformer_layer1 | 0.5638 | 0.8596 | 0.0128 | 0.2270 | 0.5945 |
| sequence | transformer_layer2 | 0.5267 | 0.8364 | 0.0128 | 0.1993 | 0.5810 |
| sequence | transformer_layer3 | 0.4988 | 0.8057 | 0.0128 | 0.1763 | 0.5626 |
| sequence | classifier_input | 0.3155 | 0.6311 | 0.0128 | 0.0749 | 0.5315 |
| sequence | logits | 0.0806 | 0.2564 | 0.0128 | 0.0426 | 0.4999 |

A high known top-1 means a held-out group from an enrolled subject can be assigned its concrete subject id. For an unseen subject, a classifier trained only on known ids cannot produce the true new id; unknown rejection/AUROC instead quantify whether it can detect that the sample is outside enrollment.

This is a subject-information audit, not a LoP outcome. Transformer epoch taps are sequence-contextual and should not be described as a strictly isolated single-epoch measurement.
