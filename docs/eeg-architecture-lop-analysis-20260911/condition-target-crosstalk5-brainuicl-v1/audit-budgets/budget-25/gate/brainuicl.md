# Continuous EEG LoP gate

- Status: `blocked-inconsistent-direction`
- Outcome: `task.plasticity.fresh_gap`
- Seeds: 3/3
- Transitions: 8/2
- Design consistent: `True`
- Scientific conclusion allowed: `False`

| Transition | Mean | Seeds | Positive | Negative | Zero | 95% CI |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `0->1` | 0.24666667 | 3 | 2 | 0 | 1 | [0, 0.40000001] |
| `1->2` | -0.00666668 | 3 | 2 | 1 | 0 | [-0.16, 0.07999998] |
| `2->3` | 0.22000001 | 3 | 3 | 0 | 0 | [0.16000001, 0.30000002] |
| `3->4` | 0.09999999 | 3 | 2 | 1 | 0 | [-0.16000003, 0.36] |
| `4->5` | -0.24666666 | 3 | 0 | 3 | 0 | [-0.35999998, -0.10000001] |
| `5->6` | -0.15333333 | 3 | 0 | 2 | 1 | [-0.31999999, 0] |
| `6->7` | -0.01333334 | 3 | 1 | 2 | 0 | [-0.12, 0.09999999] |
| `7->8` | -0.11333334 | 3 | 0 | 3 | 0 | [-0.16, -0.08000001] |

## Reasons

- fresh-gap direction is mixed across seeds/stages; positive and non-positive outcomes cannot support a stable LoP direction

Retention/forgetting metrics are inventory only and are not substituted for fresh-gap.
