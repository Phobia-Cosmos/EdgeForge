# Continuous EEG LoP gate

- Status: `blocked-inconsistent-direction`
- Outcome: `task.plasticity.fresh_gap`
- Seeds: 3/3
- Transitions: 8/2
- Design consistent: `True`
- Scientific conclusion allowed: `False`

| Transition | Mean | Seeds | Positive | Negative | Zero | 95% CI |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `0->1` | 0.1 | 3 | 3 | 0 | 0 | [0.02, 0.26000001] |
| `1->2` | 0.0 | 3 | 1 | 1 | 1 | [-0.23999998, 0.23999998] |
| `2->3` | 0.21333334 | 3 | 3 | 0 | 0 | [0.08, 0.36000001] |
| `3->4` | 0.02666666 | 3 | 2 | 1 | 0 | [-0.12, 0.09999999] |
| `4->5` | 0.0 | 3 | 1 | 2 | 0 | [-0.02, 0.04000001] |
| `5->6` | 0.09999999 | 3 | 3 | 0 | 0 | [0.09999999, 0.09999999] |
| `6->7` | -0.09333333 | 3 | 1 | 2 | 0 | [-0.25999999, 0.07999998] |
| `7->8` | -0.17333333 | 3 | 1 | 2 | 0 | [-0.38, 0.08] |

## Reasons

- fresh-gap direction is mixed across seeds/stages; positive and non-positive outcomes cannot support a stable LoP direction

Retention/forgetting metrics are inventory only and are not substituted for fresh-gap.
