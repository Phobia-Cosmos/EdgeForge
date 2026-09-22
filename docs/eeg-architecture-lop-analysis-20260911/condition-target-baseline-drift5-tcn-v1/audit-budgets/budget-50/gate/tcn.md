# Continuous EEG LoP gate

- Status: `blocked-inconsistent-direction`
- Outcome: `task.plasticity.fresh_gap`
- Seeds: 3/3
- Transitions: 8/2
- Design consistent: `True`
- Scientific conclusion allowed: `False`

| Transition | Mean | Seeds | Positive | Negative | Zero | 95% CI |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `0->1` | -0.07333334 | 3 | 1 | 1 | 1 | [-0.24000001, 0.02] |
| `1->2` | 0.0 | 3 | 0 | 0 | 3 | [0, 0] |
| `2->3` | 0.18666668 | 3 | 2 | 1 | 0 | [-0.16, 0.36000001] |
| `3->4` | 0.25333333 | 3 | 3 | 0 | 0 | [0.09999999, 0.44] |
| `4->5` | -0.02666667 | 3 | 0 | 3 | 0 | [-0.04000001, -0.02] |
| `5->6` | 0.19333333 | 3 | 3 | 0 | 0 | [0.09999999, 0.23999999] |
| `6->7` | 0.05999999 | 3 | 2 | 0 | 1 | [0, 0.09999999] |
| `7->8` | -0.19333333 | 3 | 0 | 3 | 0 | [-0.22, -0.14] |

## Reasons

- fresh-gap direction is mixed across seeds/stages; positive and non-positive outcomes cannot support a stable LoP direction

Retention/forgetting metrics are inventory only and are not substituted for fresh-gap.
