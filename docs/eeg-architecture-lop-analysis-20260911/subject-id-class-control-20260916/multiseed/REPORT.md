# Subject-ID class-conditioned control (multi-seed)

This report fixes the task label (sleep stage for ISRUC or emotion class for FACED) and samples equal train/test epoch counts per subject. The checkpoint is frozen and the identity decoder is an external class-balanced Ridge probe.

## FACED

Runs: 3 split seeds; 984 complete groups; 123 subjects; 9 task classes.

Class-histogram-only baseline:

| metric | mean ± sample std |
| --- | ---: |
| accuracy_top1 | 0.0007 ± 0.0012 |
| balanced_accuracy_top1 | 0.0007 ± 0.0012 |

Class-conditioned epoch probe (macro balanced accuracy):

| representation | mean ± sample std |
| --- | ---: |
| raw_waveform | 0.0261 ± 0.0004 |
| eeg_branch | 0.2563 ± 0.0140 |
| transformer_layer1 | 0.2560 ± 0.0416 |
| classifier_input | 0.1680 ± 0.0077 |

The histogram baseline tests whether subject identity can be explained by different task-label proportions. The conditioned probe removes that explanation by evaluating within a single task class. Above-chance values still may reflect session, electrode placement, impedance, gain, or acquisition-device cues; they are not biometric-authentication claims and are not LoP outcomes.

## ISRUC

Runs: 3 split seeds; 4,276 complete groups; 98 subjects; 5 task classes.

Class-histogram-only baseline:

| metric | mean ± sample std |
| --- | ---: |
| accuracy_top1 | 0.0231 ± 0.0021 |
| balanced_accuracy_top1 | 0.0235 ± 0.0025 |

Class-conditioned epoch probe (macro balanced accuracy):

| representation | mean ± sample std |
| --- | ---: |
| raw_waveform | 0.0145 ± 0.0029 |
| eeg_branch | 0.2629 ± 0.0041 |
| transformer_layer1 | 0.2450 ± 0.0122 |
| classifier_input | 0.1297 ± 0.0074 |

The histogram baseline tests whether subject identity can be explained by different task-label proportions. The conditioned probe removes that explanation by evaluating within a single task class. Above-chance values still may reflect session, electrode placement, impedance, gain, or acquisition-device cues; they are not biometric-authentication claims and are not LoP outcomes.

## Next use in continual learning

Apply the same frozen probe to checkpoints before training and after each subject transition. Track identity accuracy/AUROC jointly with task accuracy, effective rank, gradient coverage, and fresh-subject adaptation gap. A change in identity decodability alone does not establish plasticity loss; LoP requires a corresponding degradation in learning on a fresh task or subject.
