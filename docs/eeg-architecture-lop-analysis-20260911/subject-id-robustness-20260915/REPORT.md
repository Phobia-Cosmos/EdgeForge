# Subject-ID feature robustness across splits

This report aggregates deterministic subject/group splits. The source checkpoint remains sleep/emotion-supervised; subject ID is decoded by a separate frozen-representation probe. Complete sequences/trials are kept disjoint between enrollment and evaluation.

## FACED

Runs: 3 seeds; 984 complete groups; 98 enrolled and 25 unseen subjects per run.

| representation | known top-1 mean ± std | unknown AUROC mean ± std |
| --- | ---: | ---: |
| raw_waveform | 0.0400 ± 0.0106 | 0.5130 ± 0.0162 |
| eeg_branch | 0.8588 ± 0.0082 | 0.7852 ± 0.0351 |
| transformer_layer1 | 0.6284 ± 0.0185 | 0.6442 ± 0.0604 |
| classifier_input | 0.2670 ± 0.0103 | 0.5295 ± 0.0078 |

Epoch aggregation:

| epochs | raw_waveform | eeg_branch | transformer_layer1 | classifier_input |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.0162 | 0.1522 | 0.2296 | 0.1250 |
| 5 | 0.0204 | 0.5527 | 0.4345 | 0.1777 |
| 20 | 0.0400 | 0.8588 | 0.6284 | 0.2670 |

## ISRUC

Runs: 3 seeds; 4,276 complete groups; 78 enrolled and 20 unseen subjects per run.

| representation | known top-1 mean ± std | unknown AUROC mean ± std |
| --- | ---: | ---: |
| raw_waveform | 0.0400 ± 0.0040 | 0.4953 ± 0.0596 |
| eeg_branch | 0.7611 ± 0.0091 | 0.6822 ± 0.0354 |
| transformer_layer1 | 0.5448 ± 0.0141 | 0.5808 ± 0.0071 |
| classifier_input | 0.3063 ± 0.0066 | 0.5143 ± 0.0107 |

Epoch aggregation:

| epochs | raw_waveform | eeg_branch | transformer_layer1 | classifier_input |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.0394 | 0.3716 | 0.3768 | 0.1825 |
| 5 | 0.0411 | 0.5962 | 0.4239 | 0.2165 |
| 20 | 0.0400 | 0.7611 | 0.5448 | 0.3063 |

## Interpretation boundary

Known-subject top-1 answers whether a held-out complete group can be assigned to an already enrolled identity. Unknown AUROC answers whether confidence tends to separate unseen from enrolled subjects; it does not provide the unseen subject's concrete ID. The raw thresholded rejection rate is secondary because the source runs did not save known-test false-rejection rate.

These experiments establish repeatable subject information, not biometric uniqueness and not LoP. The next control must hold sleep/emotion class fixed, followed by checkpoint-wise tracking during continual learning.
