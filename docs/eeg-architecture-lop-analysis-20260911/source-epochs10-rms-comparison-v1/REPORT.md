# EEG condition LoP comparison

This is a paired descriptive comparison of the same `tcn` seeds and subject transitions. Positive fresh-gap is the LoP outcome direction, but mixed cells remain inconclusive.

## Aggregate fresh-gap

| condition | budget | mean | positive | zero | negative | min | max | gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| raw | 0 | +0.0208 | 10 | 6 | 8 | -0.4000 | +0.3800 | `n/a` |
| raw | 25 | +0.0708 | 15 | 5 | 4 | -0.2200 | +0.3600 | `blocked-inconsistent-direction` |
| raw | 50 | +0.0942 | 15 | 5 | 4 | -0.2600 | +0.3800 | `blocked-inconsistent-direction` |
| rms_equalized_epochs10 | 0 | +0.0058 | 8 | 8 | 8 | -0.1800 | +0.3400 | `n/a` |
| rms_equalized_epochs10 | 5 | +0.0525 | 9 | 10 | 5 | -0.2200 | +0.4200 | `n/a` |
| rms_equalized_epochs10 | 10 | +0.0900 | 14 | 7 | 3 | -0.1600 | +0.4000 | `n/a` |
| rms_equalized_epochs10 | 25 | +0.0975 | 17 | 4 | 3 | -0.2200 | +0.4400 | `blocked-inconsistent-direction` |
| rms_equalized_epochs10 | 50 | +0.0850 | 17 | 3 | 4 | -0.2200 | +0.3400 | `blocked-inconsistent-direction` |

## Paired change versus raw

| condition | budget | mean fresh-gap change | cells improved | cells worsened |
| --- | ---: | ---: | ---: | ---: |
| rms_equalized_epochs10 | 0 | -0.0150 | 6 | 9 |
| rms_equalized_epochs10 | 25 | +0.0267 | 10 | 6 |
| rms_equalized_epochs10 | 50 | -0.0092 | 10 | 9 |

## Utility interpretation

All compared conditions preserve the same subject order, labels and epoch boundaries. Refer to each derived condition manifest and quality-audit report for the exact waveform-correlation and spectral checks. No condition in this comparison passed the strict all-transition/all-seed gate, so the results should be interpreted as transfer/preprocessing sensitivity rather than induced natural LoP.
