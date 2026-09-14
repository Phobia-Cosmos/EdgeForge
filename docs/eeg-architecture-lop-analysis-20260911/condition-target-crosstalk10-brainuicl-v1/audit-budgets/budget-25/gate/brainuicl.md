# Continuous EEG LoP gate

- Status: `blocked-inconsistent-direction`
- Outcome: `task.plasticity.fresh_gap`
- Seeds: 3/3
- Transitions: 8/2
- Design consistent: `True`
- Scientific conclusion allowed: `False`

| Transition | Mean | Seeds | Positive | Negative | Zero | 95% CI |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `0->1` | 0.10666667 | 3 | 1 | 1 | 1 | [-0.04, 0.36000001] |
| `1->2` | -1e-08 | 3 | 2 | 1 | 0 | [-0.16, 0.07999998] |
| `2->3` | 0.22666668 | 3 | 2 | 0 | 1 | [0, 0.36000001] |
| `3->4` | -0.10000001 | 3 | 1 | 2 | 0 | [-0.22000003, 0.09999999] |
| `4->5` | -0.15333334 | 3 | 0 | 3 | 0 | [-0.28, -0.04000001] |
| `5->6` | 0.00666667 | 3 | 1 | 1 | 1 | [-0.14, 0.16000001] |
| `6->7` | 0.12666667 | 3 | 2 | 0 | 1 | [0, 0.28] |
| `7->8` | -0.12000001 | 3 | 0 | 3 | 0 | [-0.24000001, -0.02000001] |

## Reasons

- fresh-gap direction is mixed across seeds/stages; positive and non-positive outcomes cannot support a stable LoP direction

Retention/forgetting metrics are inventory only and are not substituted for fresh-gap.
