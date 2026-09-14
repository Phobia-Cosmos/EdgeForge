# EEG condition quality audit

Clean: `/home/undefined/Desktop/EdgeForge/data/eeg-medium/isruc-v0.18.0`
Derived: `/home/undefined/UbuntuData/datasets/induction-rms-baseline10`

| check | value |
| --- | --- |
| files | 100 |
| labels byte-identical | `True` |
| shape and finite | `True` |
| all-file correlation min / median | 0.997497 / 1.000000 |
| all-file spectral JS max / median | 0.000000 / 0.000000 |
| all-file RMS ratio min / max | 0.039038 / 1.809488 |
| changed data files | 100 |
| changed-file correlation min / median | 0.9974970470854382 / 0.9999999999999984 |
| changed-file spectral JS max / median | 1.7837100685369478e-08 / 5.218191656553017e-09 |
| changed-file RMS ratio min / max | 0.039037913031534206 / 1.8094884159143874 |
| preferred waveform gate (corr >= 0.98) | `True` |

Group-level details are stored in `quality-audit.json`; changed-file statistics isolate the intended target-only perturbation. This audit measures preservation of the recorded signal representation; it does not establish clinical validity, label quality, or LoP.
