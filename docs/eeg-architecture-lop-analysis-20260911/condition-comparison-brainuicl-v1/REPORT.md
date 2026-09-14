# EEG condition LoP comparison

This is a paired descriptive comparison of the same `brainuicl` seeds and subject transitions. Positive fresh-gap is the LoP outcome direction, but mixed cells remain inconclusive.

## Aggregate fresh-gap

| condition | budget | mean | positive | zero | negative | min | max | gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| raw | 0 | -0.0275 | 12 | 3 | 9 | -0.3200 | +0.2000 | `n/a` |
| raw | 5 | +0.0875 | 15 | 3 | 6 | -0.3000 | +0.4000 | `n/a` |
| raw | 10 | +0.0408 | 13 | 6 | 5 | -0.4000 | +0.3600 | `blocked-inconsistent-direction` |
| raw | 25 | -0.0325 | 12 | 0 | 12 | -0.3600 | +0.4000 | `blocked-inconsistent-direction` |
| raw | 50 | -0.0517 | 9 | 0 | 15 | -0.4200 | +0.3200 | `blocked-inconsistent-direction` |
| rms_equalized | 0 | +0.0017 | 11 | 4 | 9 | -0.4000 | +0.3400 | `n/a` |
| rms_equalized | 25 | +0.0583 | 15 | 2 | 7 | -0.3400 | +0.4400 | `blocked-inconsistent-direction` |
| rms_equalized | 50 | +0.0675 | 12 | 2 | 10 | -0.2000 | +0.4400 | `blocked-inconsistent-direction` |

## Paired change versus raw

| condition | budget | mean fresh-gap change | cells improved | cells worsened |
| --- | ---: | ---: | ---: | ---: |
| rms_equalized | 0 | +0.0292 | 14 | 9 |
| rms_equalized | 25 | +0.0908 | 14 | 7 |
| rms_equalized | 50 | +0.1192 | 16 | 8 |

## Utility interpretation

`rms_equalized` preserves waveform correlation and removes only subject-level gain differences. `target_snr20_noise` keeps labels/epoch boundaries unchanged and adds a mild 20 dB band-limited perturbation. Neither condition passed the strict all-transition/all-seed gate in this pilot, so the results should be interpreted as transfer/preprocessing sensitivity rather than induced natural LoP.
