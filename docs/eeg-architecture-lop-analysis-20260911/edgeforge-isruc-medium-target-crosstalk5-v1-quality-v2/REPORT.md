# EEG condition quality audit

Clean: `/home/undefined/Desktop/EdgeForge/data/eeg-medium/isruc-v0.18.0`
Derived: `/home/undefined/UbuntuData/datasets/edgeforge-isruc-medium-target-crosstalk5-v1`

| check | value |
| --- | --- |
| files | 100 |
| labels byte-identical | `True` |
| shape and finite | `True` |
| all-file correlation min / median | 0.998801 / 1.000000 |
| all-file spectral JS max / median | 0.000134 / 0.000000 |
| all-file RMS ratio min / max | 0.925803 / 1.000000 |
| changed data files | 40 |
| changed-file correlation min / median | 0.9988011337209064 / 0.9991920869397203 |
| changed-file spectral JS max / median | 0.00013368761574383825 / 8.73707358550746e-06 |
| changed-file RMS ratio min / max | 0.9258033018150097 / 0.9764795210647613 |
| preferred waveform gate (corr >= 0.98) | `True` |

Group-level details are stored in `quality-audit.json`; changed-file statistics isolate the intended target-only perturbation. This audit measures preservation of the recorded signal representation; it does not establish clinical validity, label quality, or LoP.
