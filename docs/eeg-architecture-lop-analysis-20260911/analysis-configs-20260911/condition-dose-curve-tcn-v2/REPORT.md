# EEG LoP dose curves

The curves summarize fresh-gap across the same seed/transition cells. Error bars are descriptive normal-approximation 95% intervals, not a causal or population-level confidence claim.

| condition | dose axis | dose | budget | mean | 95% CI | positive | zero | negative | gate |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |
| raw | noise_rms_ratio | 0.0000 | 0 | -0.0267 | [-0.0985, +0.0451] | 10 | 6 | 8 | `n/a` |
| raw | noise_rms_ratio | 0.0000 | 5 | -0.0525 | [-0.1206, +0.0156] | 11 | 4 | 9 | `blocked-inconsistent-direction` |
| raw | noise_rms_ratio | 0.0000 | 10 | +0.0200 | [-0.0335, +0.0735] | 13 | 6 | 5 | `blocked-inconsistent-direction` |
| raw | noise_rms_ratio | 0.0000 | 25 | +0.0208 | [-0.0488, +0.0904] | 15 | 2 | 7 | `blocked-inconsistent-direction` |
| raw | noise_rms_ratio | 0.0000 | 50 | -0.0075 | [-0.0643, +0.0493] | 11 | 6 | 7 | `blocked-inconsistent-direction` |
| target_crosstalk5 | cross_talk_fraction | 0.0500 | 0 | -0.0142 | [-0.0771, +0.0487] | 6 | 5 | 13 | `n/a` |
| target_crosstalk5 | cross_talk_fraction | 0.0500 | 5 | +0.0700 | [-0.0039, +0.1439] | 13 | 3 | 8 | `n/a` |
| target_crosstalk5 | cross_talk_fraction | 0.0500 | 10 | +0.0783 | [+0.0209, +0.1358] | 14 | 6 | 4 | `n/a` |
| target_crosstalk5 | cross_talk_fraction | 0.0500 | 25 | +0.0375 | [-0.0243, +0.0993] | 15 | 2 | 7 | `blocked-inconsistent-direction` |
| target_crosstalk5 | cross_talk_fraction | 0.0500 | 50 | +0.0242 | [-0.0315, +0.0798] | 10 | 6 | 8 | `blocked-inconsistent-direction` |
| target_crosstalk10 | cross_talk_fraction | 0.1000 | 0 | +0.0033 | [-0.0574, +0.0641] | 9 | 6 | 9 | `n/a` |
| target_crosstalk10 | cross_talk_fraction | 0.1000 | 5 | -0.0017 | [-0.0665, +0.0632] | 10 | 6 | 8 | `n/a` |
| target_crosstalk10 | cross_talk_fraction | 0.1000 | 10 | +0.0525 | [-0.0143, +0.1193] | 13 | 7 | 4 | `n/a` |
| target_crosstalk10 | cross_talk_fraction | 0.1000 | 25 | +0.0442 | [-0.0167, +0.1050] | 14 | 4 | 6 | `blocked-inconsistent-direction` |
| target_crosstalk10 | cross_talk_fraction | 0.1000 | 50 | +0.0825 | [+0.0114, +0.1536] | 17 | 3 | 4 | `blocked-inconsistent-direction` |
| target_baseline_drift5 | baseline_drift_fraction | 0.0500 | 0 | +0.0258 | [-0.0519, +0.1036] | 10 | 6 | 8 | `n/a` |
| target_baseline_drift5 | baseline_drift_fraction | 0.0500 | 5 | +0.0242 | [-0.0514, +0.0997] | 11 | 5 | 8 | `n/a` |
| target_baseline_drift5 | baseline_drift_fraction | 0.0500 | 10 | +0.0692 | [-0.0029, +0.1413] | 15 | 4 | 5 | `n/a` |
| target_baseline_drift5 | baseline_drift_fraction | 0.0500 | 25 | +0.0700 | [+0.0025, +0.1375] | 14 | 4 | 6 | `blocked-inconsistent-direction` |
| target_baseline_drift5 | baseline_drift_fraction | 0.0500 | 50 | +0.0500 | [-0.0244, +0.1244] | 11 | 5 | 8 | `blocked-inconsistent-direction` |
| target_baseline_drift10 | baseline_drift_fraction | 0.1000 | 0 | +0.0133 | [-0.0647, +0.0914] | 11 | 2 | 11 | `n/a` |
| target_baseline_drift10 | baseline_drift_fraction | 0.1000 | 5 | -0.0233 | [-0.0947, +0.0480] | 8 | 7 | 9 | `n/a` |
| target_baseline_drift10 | baseline_drift_fraction | 0.1000 | 10 | +0.0283 | [-0.0366, +0.0932] | 12 | 5 | 7 | `n/a` |
| target_baseline_drift10 | baseline_drift_fraction | 0.1000 | 25 | +0.1008 | [+0.0369, +0.1648] | 16 | 5 | 3 | `blocked-inconsistent-direction` |
| target_baseline_drift10 | baseline_drift_fraction | 0.1000 | 50 | +0.0350 | [-0.0439, +0.1139] | 11 | 5 | 8 | `blocked-inconsistent-direction` |

`target_snr*` dose is noise RMS / signal RMS, so lower SNR appears farther right. Cross-talk and baseline-drift conditions use their configured fractions as separate axes. `target_gain_drift10` is plotted on its own gain-drift axis; `rms_equalized` is a calibration condition and is retained in JSON but not placed on a nuisance-dose axis.

All results remain descriptive and `scientific_conclusion_allowed=false`.
