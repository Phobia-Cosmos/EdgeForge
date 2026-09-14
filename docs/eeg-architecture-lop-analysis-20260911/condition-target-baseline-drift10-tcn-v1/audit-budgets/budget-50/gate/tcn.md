# Continuous EEG LoP gate

- Status: `blocked-inconsistent-direction`
- Outcome: `task.plasticity.fresh_gap`
- Seeds: 3/3
- Transitions: 8/2
- Design consistent: `True`
- Scientific conclusion allowed: `False`

| Transition | Mean | Seeds | Positive | Negative | Zero | 95% CI |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `0->1` | -0.04666667 | 3 | 0 | 1 | 2 | [-0.14, 0] |
| `1->2` | -0.07999999 | 3 | 0 | 1 | 2 | [-0.23999998, 0] |
| `2->3` | 0.36000001 | 3 | 3 | 0 | 0 | [0.36000001, 0.36000001] |
| `3->4` | 0.18 | 3 | 3 | 0 | 0 | [0.09999999, 0.34] |
| `4->5` | -0.00666667 | 3 | 1 | 2 | 0 | [-0.04000001, 0.04000001] |
| `5->6` | -0.11333334 | 3 | 1 | 2 | 0 | [-0.30000001, 0.09999999] |
| `6->7` | 0.14666666 | 3 | 3 | 0 | 0 | [0.07999998, 0.25999999] |
| `7->8` | -0.16 | 3 | 0 | 2 | 1 | [-0.25999999, 0] |

## Reasons

- fresh-gap direction is mixed across seeds/stages; positive and non-positive outcomes cannot support a stable LoP direction

Retention/forgetting metrics are inventory only and are not substituted for fresh-gap.
