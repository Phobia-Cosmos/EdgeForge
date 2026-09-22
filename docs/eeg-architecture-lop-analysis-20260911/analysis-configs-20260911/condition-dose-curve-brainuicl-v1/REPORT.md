# EEG LoP dose curves

The curves summarize fresh-gap across the same seed/transition cells. Error bars are descriptive normal-approximation 95% intervals, not a causal or population-level confidence claim.

| condition | dose axis | dose | budget | mean | 95% CI | positive | zero | negative | gate |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |
| raw | noise_rms_ratio | 0.0000 | 0 | -0.0275 | [-0.0996, +0.0446] | 12 | 3 | 9 | `n/a` |
| raw | noise_rms_ratio | 0.0000 | 5 | +0.0875 | [+0.0176, +0.1574] | 15 | 3 | 6 | `blocked-inconsistent-direction` |
| raw | noise_rms_ratio | 0.0000 | 10 | +0.0408 | [-0.0217, +0.1034] | 13 | 6 | 5 | `blocked-inconsistent-direction` |
| raw | noise_rms_ratio | 0.0000 | 25 | -0.0325 | [-0.1257, +0.0607] | 12 | 0 | 12 | `blocked-inconsistent-direction` |
| raw | noise_rms_ratio | 0.0000 | 50 | -0.0517 | [-0.1292, +0.0259] | 9 | 0 | 15 | `blocked-inconsistent-direction` |
| target_crosstalk5 | cross_talk_fraction | 0.0500 | 0 | -0.0242 | [-0.0804, +0.0321] | 8 | 6 | 10 | `n/a` |
| target_crosstalk5 | cross_talk_fraction | 0.0500 | 5 | +0.0292 | [-0.0166, +0.0750] | 11 | 6 | 7 | `n/a` |
| target_crosstalk5 | cross_talk_fraction | 0.0500 | 10 | +0.0300 | [-0.0196, +0.0796] | 14 | 4 | 6 | `n/a` |
| target_crosstalk5 | cross_talk_fraction | 0.0500 | 25 | +0.0042 | [-0.0808, +0.0892] | 10 | 2 | 12 | `blocked-inconsistent-direction` |
| target_crosstalk5 | cross_talk_fraction | 0.0500 | 50 | -0.0425 | [-0.1208, +0.0358] | 8 | 1 | 15 | `blocked-inconsistent-direction` |
| target_crosstalk10 | cross_talk_fraction | 0.1000 | 0 | -0.0558 | [-0.1249, +0.0133] | 6 | 2 | 16 | `n/a` |
| target_crosstalk10 | cross_talk_fraction | 0.1000 | 5 | +0.0550 | [-0.0099, +0.1199] | 14 | 6 | 4 | `n/a` |
| target_crosstalk10 | cross_talk_fraction | 0.1000 | 10 | +0.0192 | [-0.0486, +0.0870] | 14 | 4 | 6 | `n/a` |
| target_crosstalk10 | cross_talk_fraction | 0.1000 | 25 | +0.0117 | [-0.0625, +0.0858] | 9 | 4 | 11 | `blocked-inconsistent-direction` |
| target_crosstalk10 | cross_talk_fraction | 0.1000 | 50 | -0.0208 | [-0.1055, +0.0638] | 12 | 0 | 12 | `blocked-inconsistent-direction` |
| target_baseline_drift5 | baseline_drift_fraction | 0.0500 | 0 | -0.0117 | [-0.0714, +0.0481] | 9 | 4 | 11 | `n/a` |
| target_baseline_drift5 | baseline_drift_fraction | 0.0500 | 5 | +0.0767 | [+0.0297, +0.1237] | 13 | 7 | 4 | `n/a` |
| target_baseline_drift5 | baseline_drift_fraction | 0.0500 | 10 | +0.0642 | [+0.0168, +0.1115] | 14 | 6 | 4 | `n/a` |
| target_baseline_drift5 | baseline_drift_fraction | 0.0500 | 25 | +0.0367 | [-0.0375, +0.1108] | 12 | 2 | 10 | `blocked-inconsistent-direction` |
| target_baseline_drift5 | baseline_drift_fraction | 0.0500 | 50 | -0.0225 | [-0.0993, +0.0543] | 9 | 2 | 13 | `blocked-inconsistent-direction` |
| target_baseline_drift10 | baseline_drift_fraction | 0.1000 | 0 | -0.0250 | [-0.0907, +0.0407] | 9 | 3 | 12 | `n/a` |
| target_baseline_drift10 | baseline_drift_fraction | 0.1000 | 5 | +0.0600 | [+0.0012, +0.1188] | 15 | 5 | 4 | `n/a` |
| target_baseline_drift10 | baseline_drift_fraction | 0.1000 | 10 | +0.0667 | [+0.0169, +0.1165] | 15 | 5 | 4 | `n/a` |
| target_baseline_drift10 | baseline_drift_fraction | 0.1000 | 25 | +0.0083 | [-0.0895, +0.1062] | 13 | 1 | 10 | `blocked-inconsistent-direction` |
| target_baseline_drift10 | baseline_drift_fraction | 0.1000 | 50 | -0.0892 | [-0.1739, -0.0044] | 8 | 1 | 15 | `blocked-inconsistent-direction` |

`target_snr*` dose is noise RMS / signal RMS, so lower SNR appears farther right. Cross-talk and baseline-drift conditions use their configured fractions as separate axes. `target_gain_drift10` is plotted on its own gain-drift axis; `rms_equalized` is a calibration condition and is retained in JSON but not placed on a nuisance-dose axis.

All results remain descriptive and `scientific_conclusion_allowed=false`.
