# EEG condition quality audit

Clean: `/home/undefined/Desktop/EdgeForge/data/eeg-medium/isruc-v0.18.0`
Derived: `/home/undefined/UbuntuData/datasets/induction-target_time_reverse`

| check | value |
| --- | --- |
| files | 100 |
| labels byte-identical | `True` |
| shape and finite | `True` |
| all-file correlation min / median | -0.065012 / 1.000000 |
| all-file spectral JS max / median | 0.009872 / 0.000000 |
| all-file RMS ratio min / max | 1.000000 / 1.000000 |
| changed data files | 40 |
| changed-file correlation min / median | -0.06501227098332085 / -0.002698979892218431 |
| changed-file spectral JS max / median | 0.009872430004179478 / 0.0010437810560688376 |
| changed-file RMS ratio min / max | 0.9999999999999996 / 1.0000000000000007 |
| preferred waveform gate (corr >= 0.98) | `False` |

Group-level details are stored in `quality-audit.json`; changed-file statistics isolate the intended target-only perturbation. This audit measures preservation of the recorded signal representation; it does not establish clinical validity, label quality, or LoP.
