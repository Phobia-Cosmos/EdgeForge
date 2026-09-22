# EEG condition quality audit

Clean: `/home/undefined/Desktop/EdgeForge/data/eeg-medium/isruc-v0.18.0`
Derived: `/home/undefined/UbuntuData/datasets/edgeforge-isruc-medium-target-baseline-drift5-v1`

| check | value |
| --- | --- |
| files | 100 |
| labels byte-identical | `True` |
| shape and finite | `True` |
| all-file correlation min / median | 0.999375 / 1.000000 |
| all-file spectral JS max / median | 0.000000 / 0.000000 |
| all-file RMS ratio min / max | 0.999289 / 1.002511 |
| changed data files | 40 |
| changed-file correlation min / median | 0.9993748051107033 / 0.9993755411814206 |
| changed-file spectral JS max / median | 2.3650056135693376e-08 / 4.459613123231065e-09 |
| changed-file RMS ratio min / max | 0.9992889221992597 / 1.0025108857570508 |
| preferred waveform gate (corr >= 0.98) | `True` |

Group-level details are stored in `quality-audit.json`; changed-file statistics isolate the intended target-only perturbation. This audit measures preservation of the recorded signal representation; it does not establish clinical validity, label quality, or LoP.
