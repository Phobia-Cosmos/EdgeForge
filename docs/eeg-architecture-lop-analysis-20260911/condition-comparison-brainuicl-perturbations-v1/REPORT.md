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
| target_crosstalk5_brainuicl | 0 | -0.0242 | 8 | 6 | 10 | -0.3200 | +0.2000 | `n/a` |
| target_crosstalk5_brainuicl | 5 | +0.0292 | 11 | 6 | 7 | -0.3000 | +0.2400 | `n/a` |
| target_crosstalk5_brainuicl | 10 | +0.0300 | 14 | 4 | 6 | -0.3600 | +0.2400 | `n/a` |
| target_crosstalk5_brainuicl | 25 | +0.0042 | 10 | 2 | 12 | -0.3600 | +0.4000 | `blocked-inconsistent-direction` |
| target_crosstalk5_brainuicl | 50 | -0.0425 | 8 | 1 | 15 | -0.3800 | +0.3600 | `blocked-inconsistent-direction` |
| target_crosstalk10_brainuicl | 0 | -0.0558 | 6 | 2 | 16 | -0.4000 | +0.2400 | `n/a` |
| target_crosstalk10_brainuicl | 5 | +0.0550 | 14 | 6 | 4 | -0.4000 | +0.3600 | `n/a` |
| target_crosstalk10_brainuicl | 10 | +0.0192 | 14 | 4 | 6 | -0.3600 | +0.2800 | `n/a` |
| target_crosstalk10_brainuicl | 25 | +0.0117 | 9 | 4 | 11 | -0.2800 | +0.3600 | `blocked-inconsistent-direction` |
| target_crosstalk10_brainuicl | 50 | -0.0208 | 12 | 0 | 12 | -0.5800 | +0.3400 | `blocked-inconsistent-direction` |
| target_baseline_drift5_brainuicl | 0 | -0.0117 | 9 | 4 | 11 | -0.3000 | +0.2400 | `n/a` |
| target_baseline_drift5_brainuicl | 5 | +0.0767 | 13 | 7 | 4 | -0.1400 | +0.3600 | `n/a` |
| target_baseline_drift5_brainuicl | 10 | +0.0642 | 14 | 6 | 4 | -0.1400 | +0.3600 | `n/a` |
| target_baseline_drift5_brainuicl | 25 | +0.0367 | 12 | 2 | 10 | -0.3200 | +0.4000 | `blocked-inconsistent-direction` |
| target_baseline_drift5_brainuicl | 50 | -0.0225 | 9 | 2 | 13 | -0.3800 | +0.3800 | `blocked-inconsistent-direction` |
| target_baseline_drift10_brainuicl | 0 | -0.0250 | 9 | 3 | 12 | -0.3000 | +0.2600 | `n/a` |
| target_baseline_drift10_brainuicl | 5 | +0.0600 | 15 | 5 | 4 | -0.4000 | +0.3800 | `n/a` |
| target_baseline_drift10_brainuicl | 10 | +0.0667 | 15 | 5 | 4 | -0.2000 | +0.3400 | `n/a` |
| target_baseline_drift10_brainuicl | 25 | +0.0083 | 13 | 1 | 10 | -0.4400 | +0.4200 | `blocked-inconsistent-direction` |
| target_baseline_drift10_brainuicl | 50 | -0.0892 | 8 | 1 | 15 | -0.5000 | +0.3000 | `blocked-inconsistent-direction` |

## Paired change versus raw

| condition | budget | mean fresh-gap change | cells improved | cells worsened |
| --- | ---: | ---: | ---: | ---: |
| target_crosstalk5_brainuicl | 0 | +0.0033 | 10 | 10 |
| target_crosstalk5_brainuicl | 5 | -0.0583 | 7 | 15 |
| target_crosstalk5_brainuicl | 10 | -0.0108 | 9 | 10 |
| target_crosstalk5_brainuicl | 25 | +0.0367 | 12 | 9 |
| target_crosstalk5_brainuicl | 50 | +0.0092 | 15 | 9 |
| target_crosstalk10_brainuicl | 0 | -0.0283 | 7 | 14 |
| target_crosstalk10_brainuicl | 5 | -0.0325 | 9 | 10 |
| target_crosstalk10_brainuicl | 10 | -0.0217 | 10 | 11 |
| target_crosstalk10_brainuicl | 25 | +0.0442 | 13 | 9 |
| target_crosstalk10_brainuicl | 50 | +0.0308 | 14 | 10 |
| target_baseline_drift5_brainuicl | 0 | +0.0158 | 12 | 9 |
| target_baseline_drift5_brainuicl | 5 | -0.0108 | 8 | 8 |
| target_baseline_drift5_brainuicl | 10 | +0.0233 | 11 | 7 |
| target_baseline_drift5_brainuicl | 25 | +0.0692 | 15 | 6 |
| target_baseline_drift5_brainuicl | 50 | +0.0292 | 13 | 9 |
| target_baseline_drift10_brainuicl | 0 | +0.0025 | 9 | 12 |
| target_baseline_drift10_brainuicl | 5 | -0.0275 | 10 | 9 |
| target_baseline_drift10_brainuicl | 10 | +0.0258 | 10 | 9 |
| target_baseline_drift10_brainuicl | 25 | +0.0408 | 12 | 7 |
| target_baseline_drift10_brainuicl | 50 | -0.0375 | 9 | 15 |

## Utility interpretation

All compared conditions preserve the same subject order, labels and epoch boundaries. Refer to each derived condition manifest and quality-audit report for the exact waveform-correlation and spectral checks. No condition in this comparison passed the strict all-transition/all-seed gate, so the results should be interpreted as transfer/preprocessing sensitivity rather than induced natural LoP.
