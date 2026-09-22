# EEG condition quality audit

Clean: `/home/undefined/Desktop/EdgeForge/data/eeg-medium/isruc-v0.18.0`
Derived: `/home/undefined/UbuntuData/datasets/edgeforge-isruc-medium-target-snr10-v1`

| check | value |
| --- | --- |
| files | 100 |
| labels byte-identical | `True` |
| shape and finite | `True` |
| correlation min / median | 0.953366 / 1.000000 |
| spectral JS max / median | 0.023373 / 0.000000 |
| RMS ratio min / max | 1.000000 / 1.049991 |
| preferred waveform gate (corr >= 0.98) | `False` |

This audit measures preservation of the recorded signal representation; it does not establish clinical validity, label quality, or LoP.
