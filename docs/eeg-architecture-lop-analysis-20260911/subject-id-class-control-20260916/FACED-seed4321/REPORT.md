# FACED subject-ID class-conditioned control

This run uses 984 complete groups, 19,680 epochs and 123 subjects. Subject ID is the probe target; the original task label is used only to construct controls.

A classifier that sees only each sequence's 9-D task-label histogram obtains top-1 `0.0020` (chance `0.0081`). This estimates identity leakage from class composition alone.

Class-conditioned probes hold the task class fixed and balance train/test epoch counts across included subjects:

| representation | macro balanced subject-ID accuracy | class-to-class std |
| --- | ---: | ---: |
| raw_waveform | 0.0258 | 0.0275 |
| eeg_branch | 0.2722 | 0.1222 |
| transformer_layer1 | 0.2954 | 0.0716 |
| classifier_input | 0.1761 | 0.0454 |

Accuracy above each row's recorded chance means subject information remains decodable even after fixing the sleep/emotion class. It still may contain session, electrode, impedance, gain or acquisition-device cues, so this is not proof of immutable biometric identity.

This control is not an LoP outcome. Its role is to define a cleaner identity signal that can later be tracked over continual-learning checkpoints.
