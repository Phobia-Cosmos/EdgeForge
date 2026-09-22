# EEG condition LoP comparison

This is a paired descriptive comparison of the same `tcn` seeds and subject transitions. Positive fresh-gap is the LoP outcome direction, but mixed cells remain inconclusive.

## Aggregate fresh-gap

| condition | budget | mean | positive | zero | negative | min | max | gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| raw | 0 | -0.0267 | 10 | 6 | 8 | -0.4200 | +0.2800 | `n/a` |
| raw | 5 | -0.0525 | 11 | 4 | 9 | -0.4200 | +0.2400 | `n/a` |
| raw | 10 | +0.0200 | 13 | 6 | 5 | -0.2600 | +0.3600 | `blocked-inconsistent-direction` |
| raw | 25 | +0.0208 | 15 | 2 | 7 | -0.2400 | +0.4400 | `blocked-inconsistent-direction` |
| raw | 50 | -0.0075 | 11 | 6 | 7 | -0.3800 | +0.2000 | `blocked-inconsistent-direction` |
| target_crosstalk5 | 0 | -0.0142 | 6 | 5 | 13 | -0.2200 | +0.3800 | `n/a` |
| target_crosstalk5 | 5 | +0.0700 | 13 | 3 | 8 | -0.2200 | +0.3800 | `n/a` |
| target_crosstalk5 | 10 | +0.0783 | 14 | 6 | 4 | -0.1600 | +0.4400 | `n/a` |
| target_crosstalk5 | 25 | +0.0375 | 15 | 2 | 7 | -0.2400 | +0.3600 | `blocked-inconsistent-direction` |
| target_crosstalk5 | 50 | +0.0242 | 10 | 6 | 8 | -0.2400 | +0.4000 | `blocked-inconsistent-direction` |
| target_baseline_drift5 | 0 | +0.0258 | 10 | 6 | 8 | -0.4200 | +0.3800 | `n/a` |
| target_baseline_drift5 | 5 | +0.0242 | 11 | 5 | 8 | -0.4000 | +0.3600 | `n/a` |
| target_baseline_drift5 | 10 | +0.0692 | 15 | 4 | 5 | -0.4000 | +0.3400 | `n/a` |
| target_baseline_drift5 | 25 | +0.0700 | 14 | 4 | 6 | -0.2400 | +0.4400 | `blocked-inconsistent-direction` |
| target_baseline_drift5 | 50 | +0.0500 | 11 | 5 | 8 | -0.2400 | +0.4400 | `blocked-inconsistent-direction` |

## Paired change versus raw

| condition | budget | mean fresh-gap change | cells improved | cells worsened |
| --- | ---: | ---: | ---: | ---: |
| target_crosstalk5 | 0 | +0.0125 | 8 | 12 |
| target_crosstalk5 | 5 | +0.1225 | 13 | 7 |
| target_crosstalk5 | 10 | +0.0583 | 11 | 7 |
| target_crosstalk5 | 25 | +0.0167 | 8 | 6 |
| target_crosstalk5 | 50 | +0.0317 | 7 | 10 |
| target_baseline_drift5 | 0 | +0.0525 | 10 | 9 |
| target_baseline_drift5 | 5 | +0.0767 | 11 | 8 |
| target_baseline_drift5 | 10 | +0.0492 | 9 | 9 |
| target_baseline_drift5 | 25 | +0.0492 | 10 | 6 |
| target_baseline_drift5 | 50 | +0.0575 | 10 | 6 |

## Utility interpretation

All compared conditions preserve the same subject order, labels and epoch boundaries. Refer to each derived condition manifest and quality-audit report for the exact waveform-correlation and spectral checks. No condition in this comparison passed the strict all-transition/all-seed gate, so the results should be interpreted as transfer/preprocessing sensitivity rather than induced natural LoP.
