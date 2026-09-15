# ISRUC subject identity granularity/open-set probe

This run contains 4,276 complete groups (85,520 epochs). It enrolls 78 subjects and reserves 20 subjects as unseen identities.

Known subjects use half of their complete groups for enrollment and half for held-out identity testing. Unseen subjects are never present in the probe training classes; their concrete subject id is therefore not a valid prediction target. Unknown rejection is reported separately.

| unit | stage | known top-1 | known top-5 | chance | unknown rejection | unknown AUROC |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| sequence | eeg_branch | 0.7715 | 0.9513 | 0.0128 | 0.1855 | 0.6919 |
| sequence | transformer_layer1 | 0.5615 | 0.8666 | 0.0128 | 0.4435 | 0.5966 |
| sequence | transformer_layer2 | 0.5249 | 0.8213 | 0.0128 | 0.4159 | 0.5845 |
| sequence | transformer_layer3 | 0.5058 | 0.7999 | 0.0128 | 0.3859 | 0.5765 |
| sequence | classifier_input | 0.3306 | 0.6502 | 0.0128 | 0.1083 | 0.5221 |
| sequence | logits | 0.0829 | 0.2726 | 0.0128 | 0.0507 | 0.4941 |

A high known top-1 means a held-out group from an enrolled subject can be assigned its concrete subject id. For an unseen subject, a classifier trained only on known ids cannot produce the true new id; unknown rejection/AUROC instead quantify whether it can detect that the sample is outside enrollment.

This is a subject-information audit, not a LoP outcome. Transformer epoch taps are sequence-contextual and should not be described as a strictly isolated single-epoch measurement.
