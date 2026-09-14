# Continuous EEG LoP gate

- Status: `blocked-inconsistent-direction`
- Outcome: `task.plasticity.fresh_gap`
- Seeds: 3/3
- Transitions: 8/2
- Design consistent: `True`
- Scientific conclusion allowed: `False`

| Transition | Mean | Seeds | Positive | Negative | Zero | 95% CI |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `0->1` | 0.00666667 | 3 | 1 | 1 | 1 | [-0.24000001, 0.26000001] |
| `1->2` | -0.07999999 | 3 | 0 | 1 | 2 | [-0.23999998, 0] |
| `2->3` | 0.36000001 | 3 | 3 | 0 | 0 | [0.36000001, 0.36000001] |
| `3->4` | 0.21333333 | 3 | 3 | 0 | 0 | [0.09999999, 0.44] |
| `4->5` | 0.06 | 3 | 1 | 1 | 1 | [-0.02, 0.2] |
| `5->6` | 0.19333333 | 3 | 3 | 0 | 0 | [0.09999999, 0.23999999] |
| `6->7` | 0.07999998 | 3 | 3 | 0 | 0 | [0.07999998, 0.07999998] |
| `7->8` | -0.09333333 | 3 | 1 | 2 | 0 | [-0.22, 0.16] |

## Reasons

- fresh-gap direction is mixed across seeds/stages; positive and non-positive outcomes cannot support a stable LoP direction

Retention/forgetting metrics are inventory only and are not substituted for fresh-gap.
