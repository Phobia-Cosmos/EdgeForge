# Metric-driven EEG LoP induction plan

Architecture: `tcn`; budget: `25`

| condition | score | mean fresh-gap | +/0/- | quality | mechanism |
| --- | ---: | ---: | --- | --- | --- |
| `rms_equalized` | 0.5994 | 0.105833332054317 | 19/3/2 | `True` | acquisition_gain_equalization |
| `target_baseline_drift10` | 0.4940 | 0.10083333309739828 | 16/5/3 | `True` | low_frequency_drift |
| `target_snr15_noise` | 0.4398 | 0.09249999901900689 | 15/4/5 | `True` | band_limited_noise |
| `target_gain_drift10` | 0.3927 | 0.07083333283662796 | 14/4/6 | `True` | smooth_gain |
| `target_crosstalk10` | 0.3860 | 0.04416666738688946 | 14/4/6 | `True` | cross_channel_mixing |
| `clean` | -0.0990 | 0.02083333395421505 | 15/2/7 | `False` | baseline |

Next conditions: `rms_equalized`, `target_baseline_drift10`

This ranks candidates; it does not claim LoP. A condition is accepted only after the quality, strict fresh-gap and mechanism counterfactual gates pass.
