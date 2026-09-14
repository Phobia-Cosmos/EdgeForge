# Continuous EEG LoP gate

- Status: `blocked-inconsistent-direction`
- Outcome: `task.plasticity.fresh_gap`
- Seeds: 3/3
- Transitions: 8/2
- Design consistent: `True`
- Scientific conclusion allowed: `False`

| Transition | Mean | Seeds | Positive | Negative | Zero | 95% CI |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `0->1` | 0.11333333 | 3 | 3 | 0 | 0 | [0.02, 0.16] |
| `1->2` | 0.06 | 3 | 1 | 0 | 2 | [0, 0.18000001] |
| `2->3` | 0.30666668 | 3 | 3 | 0 | 0 | [0.20000002, 0.36000001] |
| `3->4` | 0.21333333 | 3 | 3 | 0 | 0 | [0.09999999, 0.44] |
| `4->5` | 0.00666668 | 3 | 2 | 1 | 0 | [-0.02, 0.02000001] |
| `5->6` | 0.06666666 | 3 | 2 | 0 | 1 | [0, 0.09999999] |
| `6->7` | 0.01999999 | 3 | 1 | 1 | 1 | [-0.02000001, 0.07999998] |
| `7->8` | -0.00666667 | 3 | 2 | 1 | 0 | [-0.22, 0.16] |

## Reasons

- fresh-gap direction is mixed across seeds/stages; positive and non-positive outcomes cannot support a stable LoP direction

Retention/forgetting metrics are inventory only and are not substituted for fresh-gap.
