# EEG condition quality audit

Clean: `/home/undefined/Desktop/EdgeForge/data/eeg-medium/isruc-v0.18.0`
Derived: `/home/undefined/UbuntuData/datasets/edgeforge-isruc-medium-target-baseline-drift10-v1`

| check | value |
| --- | --- |
| files | 100 |
| labels byte-identical | `True` |
| shape and finite | `True` |
| all-file correlation min / median | 0.997500 / 1.000000 |
| all-file spectral JS max / median | 0.000000 / 0.000000 |
| all-file RMS ratio min / max | 0.999828 / 1.006258 |
| changed data files | 40 |
| changed-file correlation min / median | 0.9974995729339341 / 0.9975090146309026 |
| changed-file spectral JS max / median | 1.757896939125203e-08 / 5.1021227243097655e-09 |
| changed-file RMS ratio min / max | 0.9998283343288599 / 1.0062584902522398 |
| preferred waveform gate (corr >= 0.98) | `True` |

Group-level details are stored in `quality-audit.json`; changed-file statistics isolate the intended target-only perturbation. This audit measures preservation of the recorded signal representation; it does not establish clinical validity, label quality, or LoP.
