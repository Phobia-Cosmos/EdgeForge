# EEG condition LoP comparison

This is a paired descriptive comparison of the same TCN seeds and subject transitions. Positive fresh-gap is the LoP outcome direction, but mixed cells remain inconclusive.

## Aggregate fresh-gap

| condition | budget | mean | positive | zero | negative | min | max | gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| raw | 0 | -0.0267 | 10 | 6 | 8 | -0.4200 | +0.2800 | `n/a` |
| raw | 5 | -0.0525 | 11 | 4 | 9 | -0.4200 | +0.2400 | `n/a` |
| raw | 10 | +0.0200 | 13 | 6 | 5 | -0.2600 | +0.3600 | `blocked-inconsistent-direction` |
| raw | 25 | +0.0208 | 15 | 2 | 7 | -0.2400 | +0.4400 | `blocked-inconsistent-direction` |
| raw | 50 | -0.0075 | 11 | 6 | 7 | -0.3800 | +0.2000 | `blocked-inconsistent-direction` |
| rms_equalized | 0 | +0.0550 | 11 | 6 | 7 | -0.1800 | +0.4200 | `n/a` |
| rms_equalized | 10 | +0.0708 | 15 | 5 | 4 | -0.1600 | +0.3400 | `blocked-inconsistent-direction` |
| rms_equalized | 25 | +0.1058 | 19 | 3 | 2 | -0.2200 | +0.4400 | `blocked-inconsistent-direction` |
| rms_equalized | 50 | +0.1092 | 15 | 5 | 4 | -0.2200 | +0.3800 | `blocked-inconsistent-direction` |
| target_snr20_noise | 0 | +0.0217 | 11 | 5 | 8 | -0.2200 | +0.2800 | `n/a` |
| target_snr20_noise | 10 | +0.0375 | 13 | 4 | 7 | -0.4000 | +0.3600 | `blocked-inconsistent-direction` |
| target_snr20_noise | 25 | +0.0683 | 15 | 5 | 4 | -0.2200 | +0.3600 | `blocked-inconsistent-direction` |
| target_snr20_noise | 50 | +0.1042 | 16 | 4 | 4 | -0.1800 | +0.3800 | `blocked-inconsistent-direction` |
| target_snr15_noise | 0 | +0.0133 | 10 | 6 | 8 | -0.2200 | +0.3000 | `n/a` |
| target_snr15_noise | 25 | +0.0925 | 15 | 4 | 5 | -0.2400 | +0.4400 | `blocked-inconsistent-direction` |
| target_snr15_noise | 50 | +0.0533 | 14 | 5 | 5 | -0.2200 | +0.3600 | `blocked-inconsistent-direction` |
| target_snr10_noise | 0 | +0.0125 | 8 | 7 | 9 | -0.2200 | +0.3400 | `n/a` |
| target_snr10_noise | 25 | +0.0758 | 15 | 3 | 6 | -0.2400 | +0.4400 | `blocked-inconsistent-direction` |
| target_snr10_noise | 50 | +0.0692 | 14 | 6 | 4 | -0.2200 | +0.3600 | `blocked-inconsistent-direction` |
| target_gain_drift10 | 0 | -0.0275 | 5 | 9 | 10 | -0.2200 | +0.2400 | `n/a` |
| target_gain_drift10 | 25 | +0.0708 | 14 | 4 | 6 | -0.2200 | +0.3600 | `blocked-inconsistent-direction` |
| target_gain_drift10 | 50 | -0.0308 | 10 | 5 | 9 | -0.3000 | +0.2400 | `blocked-inconsistent-direction` |

## Paired change versus raw

| condition | budget | mean fresh-gap change | cells improved | cells worsened |
| --- | ---: | ---: | ---: | ---: |
| rms_equalized | 0 | +0.0817 | 11 | 10 |
| rms_equalized | 10 | +0.0508 | 9 | 8 |
| rms_equalized | 25 | +0.0850 | 14 | 1 |
| rms_equalized | 50 | +0.1167 | 15 | 5 |
| target_snr20_noise | 0 | +0.0483 | 11 | 10 |
| target_snr20_noise | 10 | +0.0175 | 8 | 11 |
| target_snr20_noise | 25 | +0.0475 | 10 | 6 |
| target_snr20_noise | 50 | +0.1117 | 17 | 3 |
| target_snr15_noise | 0 | +0.0400 | 10 | 10 |
| target_snr15_noise | 25 | +0.0717 | 11 | 4 |
| target_snr15_noise | 50 | +0.0608 | 12 | 6 |
| target_snr10_noise | 0 | +0.0392 | 10 | 11 |
| target_snr10_noise | 25 | +0.0550 | 11 | 5 |
| target_snr10_noise | 50 | +0.0767 | 13 | 4 |
| target_gain_drift10 | 0 | -0.0008 | 7 | 12 |
| target_gain_drift10 | 25 | +0.0500 | 11 | 5 |
| target_gain_drift10 | 50 | -0.0233 | 8 | 12 |

## Utility interpretation

`rms_equalized` preserves waveform correlation and removes only subject-level gain differences. `target_snr20_noise` keeps labels/epoch boundaries unchanged and adds a mild 20 dB band-limited perturbation. Neither condition passed the strict all-transition/all-seed gate in this pilot, so the results should be interpreted as transfer/preprocessing sensitivity rather than induced natural LoP.
