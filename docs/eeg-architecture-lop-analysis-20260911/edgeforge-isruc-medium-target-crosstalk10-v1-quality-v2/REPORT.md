# EEG condition quality audit

Clean: `/home/undefined/Desktop/EdgeForge/data/eeg-medium/isruc-v0.18.0`
Derived: `/home/undefined/UbuntuData/datasets/edgeforge-isruc-medium-target-crosstalk10-v1`

| check | value |
| --- | --- |
| files | 100 |
| labels byte-identical | `True` |
| shape and finite | `True` |
| all-file correlation min / median | 0.994758 / 1.000000 |
| all-file spectral JS max / median | 0.000624 / 0.000000 |
| all-file RMS ratio min / max | 0.853435 / 1.000000 |
| changed data files | 40 |
| changed-file correlation min / median | 0.994758346391336 / 0.9965652205151413 |
| changed-file spectral JS max / median | 0.0006236392073333263 / 3.549260145518929e-05 |
| changed-file RMS ratio min / max | 0.8534349242928866 / 0.9544401987368656 |
| preferred waveform gate (corr >= 0.98) | `True` |

Group-level details are stored in `quality-audit.json`; changed-file statistics isolate the intended target-only perturbation. This audit measures preservation of the recorded signal representation; it does not establish clinical validity, label quality, or LoP.
