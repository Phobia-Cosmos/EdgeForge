# ISRUC subject-ID class-conditioned control

This run uses 4,276 complete groups, 85,520 epochs and 98 subjects. Subject ID is the probe target; the original task label is used only to construct controls.

A classifier that sees only each sequence's 5-D task-label histogram obtains top-1 `0.0254` (chance `0.0102`). This estimates identity leakage from class composition alone.

Class-conditioned probes hold the task class fixed and balance train/test epoch counts across included subjects:

| representation | macro balanced subject-ID accuracy | class-to-class std |
| --- | ---: | ---: |
| raw_waveform | 0.0157 | 0.0072 |
| eeg_branch | 0.2642 | 0.0647 |
| transformer_layer1 | 0.2393 | 0.0712 |
| classifier_input | 0.1291 | 0.0139 |

Accuracy above each row's recorded chance means subject information remains decodable even after fixing the sleep/emotion class. It still may contain session, electrode, impedance, gain or acquisition-device cues, so this is not proof of immutable biometric identity.

This control is not an LoP outcome. Its role is to define a cleaner identity signal that can later be tracked over continual-learning checkpoints.
