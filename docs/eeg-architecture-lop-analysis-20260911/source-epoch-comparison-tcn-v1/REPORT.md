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
| source_epochs10 | 0 | +0.0208 | 10 | 6 | 8 | -0.4000 | +0.3800 | `n/a` |
| source_epochs10 | 25 | +0.0708 | 15 | 5 | 4 | -0.2200 | +0.3600 | `blocked-inconsistent-direction` |
| source_epochs10 | 50 | +0.0942 | 15 | 5 | 4 | -0.2600 | +0.3800 | `blocked-inconsistent-direction` |

## Paired change versus raw

| condition | budget | mean fresh-gap change | cells improved | cells worsened |
| --- | ---: | ---: | ---: | ---: |
| source_epochs10 | 0 | +0.0475 | 11 | 11 |
| source_epochs10 | 25 | +0.0500 | 11 | 6 |
| source_epochs10 | 50 | +0.1017 | 15 | 5 |

## Utility interpretation

All compared conditions preserve the same subject order, labels and epoch boundaries. Refer to each derived condition manifest and quality-audit report for the exact waveform-correlation and spectral checks. No condition in this comparison passed the strict all-transition/all-seed gate, so the results should be interpreted as transfer/preprocessing sensitivity rather than induced natural LoP.
