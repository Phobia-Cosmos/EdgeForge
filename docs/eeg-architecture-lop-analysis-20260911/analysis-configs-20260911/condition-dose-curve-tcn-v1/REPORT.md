# EEG LoP dose curves

The curves summarize fresh-gap across the same seed/transition cells. Error bars are descriptive normal-approximation 95% intervals, not a causal or population-level confidence claim.

| condition | dose axis | dose | budget | mean | 95% CI | positive | zero | negative | gate |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |
| raw | noise_rms_ratio | 0.0000 | 0 | -0.0267 | [-0.0985, +0.0451] | 10 | 6 | 8 | `n/a` |
| raw | noise_rms_ratio | 0.0000 | 5 | -0.0525 | [-0.1206, +0.0156] | 11 | 4 | 9 | `blocked-inconsistent-direction` |
| raw | noise_rms_ratio | 0.0000 | 10 | +0.0200 | [-0.0335, +0.0735] | 13 | 6 | 5 | `blocked-inconsistent-direction` |
| raw | noise_rms_ratio | 0.0000 | 25 | +0.0208 | [-0.0488, +0.0904] | 15 | 2 | 7 | `blocked-inconsistent-direction` |
| raw | noise_rms_ratio | 0.0000 | 50 | -0.0075 | [-0.0643, +0.0493] | 11 | 6 | 7 | `blocked-inconsistent-direction` |
| rms_equalized | separate | n/a | 0 | +0.0550 | [-0.0136, +0.1236] | 11 | 6 | 7 | `n/a` |
| rms_equalized | separate | n/a | 10 | +0.0708 | [+0.0209, +0.1208] | 15 | 5 | 4 | `blocked-inconsistent-direction` |
| rms_equalized | separate | n/a | 25 | +0.1058 | [+0.0500, +0.1616] | 19 | 3 | 2 | `blocked-inconsistent-direction` |
| rms_equalized | separate | n/a | 50 | +0.1092 | [+0.0475, +0.1708] | 15 | 5 | 4 | `blocked-inconsistent-direction` |
| target_snr20_noise | noise_rms_ratio | 0.1000 | 0 | +0.0217 | [-0.0396, +0.0830] | 11 | 5 | 8 | `n/a` |
| target_snr20_noise | noise_rms_ratio | 0.1000 | 10 | +0.0375 | [-0.0351, +0.1101] | 13 | 4 | 7 | `blocked-inconsistent-direction` |
| target_snr20_noise | noise_rms_ratio | 0.1000 | 25 | +0.0683 | [+0.0163, +0.1204] | 15 | 5 | 4 | `blocked-inconsistent-direction` |
| target_snr20_noise | noise_rms_ratio | 0.1000 | 50 | +0.1042 | [+0.0463, +0.1621] | 16 | 4 | 4 | `blocked-inconsistent-direction` |
| target_snr15_noise | noise_rms_ratio | 0.1778 | 0 | +0.0133 | [-0.0469, +0.0736] | 10 | 6 | 8 | `n/a` |
| target_snr15_noise | noise_rms_ratio | 0.1778 | 25 | +0.0925 | [+0.0139, +0.1711] | 15 | 4 | 5 | `blocked-inconsistent-direction` |
| target_snr15_noise | noise_rms_ratio | 0.1778 | 50 | +0.0533 | [-0.0050, +0.1117] | 14 | 5 | 5 | `blocked-inconsistent-direction` |
| target_snr10_noise | noise_rms_ratio | 0.3162 | 0 | +0.0125 | [-0.0518, +0.0768] | 8 | 7 | 9 | `n/a` |
| target_snr10_noise | noise_rms_ratio | 0.3162 | 25 | +0.0758 | [+0.0042, +0.1475] | 15 | 3 | 6 | `blocked-inconsistent-direction` |
| target_snr10_noise | noise_rms_ratio | 0.3162 | 50 | +0.0692 | [+0.0131, +0.1252] | 14 | 6 | 4 | `blocked-inconsistent-direction` |
| target_gain_drift10 | gain_drift_fraction | 0.1000 | 0 | -0.0275 | [-0.0740, +0.0190] | 5 | 9 | 10 | `n/a` |
| target_gain_drift10 | gain_drift_fraction | 0.1000 | 25 | +0.0708 | [+0.0181, +0.1236] | 14 | 4 | 6 | `blocked-inconsistent-direction` |
| target_gain_drift10 | gain_drift_fraction | 0.1000 | 50 | -0.0308 | [-0.0817, +0.0200] | 10 | 5 | 9 | `blocked-inconsistent-direction` |

`target_snr*` dose is noise RMS / signal RMS, so lower SNR appears farther right. `target_gain_drift10` is plotted on its own gain-drift axis; `rms_equalized` is a calibration condition and is retained in JSON but not placed on either nuisance-dose axis.

All results remain descriptive and `scientific_conclusion_allowed=false`.
