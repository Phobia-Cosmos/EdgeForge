# EEG condition quality audit

Clean: `/home/undefined/Desktop/EdgeForge/data/eeg-medium/isruc-v0.18.0`
Derived: `/home/undefined/UbuntuData/datasets/induction-target_common_average_reference`

| check | value |
| --- | --- |
| files | 100 |
| labels byte-identical | `True` |
| shape and finite | `True` |
| all-file correlation min / median | 0.697591 / 1.000000 |
| all-file spectral JS max / median | 0.006376 / 0.000000 |
| all-file RMS ratio min / max | 0.697591 / 1.000000 |
| changed data files | 40 |
| changed-file correlation min / median | 0.6975914750471226 / 0.8210713330903856 |
| changed-file spectral JS max / median | 0.0063762119971215725 / 0.0012213703012093902 |
| changed-file RMS ratio min / max | 0.697591278635742 / 0.9561925896760453 |
| preferred waveform gate (corr >= 0.98) | `False` |

Group-level details are stored in `quality-audit.json`; changed-file statistics isolate the intended target-only perturbation. This audit measures preservation of the recorded signal representation; it does not establish clinical validity, label quality, or LoP.
