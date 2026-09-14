# Continuous EEG LoP gate

- Status: `blocked-inconsistent-direction`
- Outcome: `task.plasticity.fresh_gap`
- Seeds: 3/3
- Transitions: 8/2
- Design consistent: `True`
- Scientific conclusion allowed: `False`

| Transition | Mean | Seeds | Positive | Negative | Zero | 95% CI |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `0->1` | 0.10666668 | 3 | 1 | 2 | 0 | [-0.04, 0.38000001] |
| `1->2` | -0.17333334 | 3 | 1 | 2 | 0 | [-0.38, 0.07999998] |
| `2->3` | 0.13333335 | 3 | 2 | 0 | 1 | [0, 0.20000002] |
| `3->4` | -0.08666666 | 3 | 0 | 2 | 1 | [-0.13999999, 0] |
| `4->5` | -0.24 | 3 | 0 | 3 | 0 | [-0.26000001, -0.22000001] |
| `5->6` | 0.20000001 | 3 | 3 | 0 | 0 | [0.16000001, 0.28000001] |
| `6->7` | -0.00666667 | 3 | 2 | 1 | 0 | [-0.18000001, 0.12] |
| `7->8` | -0.11333334 | 3 | 0 | 3 | 0 | [-0.18000001, -0.04000002] |

## Reasons

- fresh-gap direction is mixed across seeds/stages; positive and non-positive outcomes cannot support a stable LoP direction

Retention/forgetting metrics are inventory only and are not substituted for fresh-gap.
