# EEG condition quality audit

Clean: `/home/undefined/Desktop/EdgeForge/data/eeg-medium/isruc-v0.18.0`
Derived: `/home/undefined/UbuntuData/datasets/induction-target_channel_rotation`

| check | value |
| --- | --- |
| files | 100 |
| labels byte-identical | `True` |
| shape and finite | `True` |
| all-file correlation min / median | -0.460025 / 1.000000 |
| all-file spectral JS max / median | 0.000000 / 0.000000 |
| all-file RMS ratio min / max | 1.000000 / 1.000000 |
| changed data files | 40 |
| changed-file correlation min / median | -0.4600253648765207 / -0.028194818090114655 |
| changed-file spectral JS max / median | 2.0392725730289385e-08 / 8.509988269622681e-09 |
| changed-file RMS ratio min / max | 0.9999999818228431 / 0.9999999849997718 |
| preferred waveform gate (corr >= 0.98) | `False` |

Group-level details are stored in `quality-audit.json`; changed-file statistics isolate the intended target-only perturbation. This audit measures preservation of the recorded signal representation; it does not establish clinical validity, label quality, or LoP.
