# EEG condition quality audit

Clean: `/home/undefined/Desktop/EdgeForge/data/eeg-medium/isruc-v0.18.0`
Derived: `/home/undefined/UbuntuData/datasets/induction-target_montage_swap`

| check | value |
| --- | --- |
| files | 100 |
| labels byte-identical | `True` |
| shape and finite | `True` |
| all-file correlation min / median | -0.000021 / 1.000000 |
| all-file spectral JS max / median | 0.000000 / 0.000000 |
| all-file RMS ratio min / max | 1.000000 / 1.000000 |
| changed data files | 40 |
| changed-file correlation min / median | -2.0524947933623004e-05 / 1.3359671113753495e-08 |
| changed-file spectral JS max / median | 2.215562133756066e-08 / 5.548352888240515e-09 |
| changed-file RMS ratio min / max | 0.9999999999999997 / 1.0000000000000004 |
| preferred waveform gate (corr >= 0.98) | `False` |

Group-level details are stored in `quality-audit.json`; changed-file statistics isolate the intended target-only perturbation. This audit measures preservation of the recorded signal representation; it does not establish clinical validity, label quality, or LoP.
