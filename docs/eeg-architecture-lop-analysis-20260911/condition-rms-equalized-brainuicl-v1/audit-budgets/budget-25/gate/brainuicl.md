# Continuous EEG LoP gate

- Status: `blocked-inconsistent-direction`
- Outcome: `task.plasticity.fresh_gap`
- Seeds: 3/3
- Transitions: 8/2
- Design consistent: `True`
- Scientific conclusion allowed: `False`

| Transition | Mean | Seeds | Positive | Negative | Zero | 95% CI |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `0->1` | 0.12666668 | 3 | 2 | 1 | 0 | [-0.03999999, 0.40000001] |
| `1->2` | 0.09333332 | 3 | 3 | 0 | 0 | [0.07999998, 0.11999999] |
| `2->3` | 0.25333335 | 3 | 3 | 0 | 0 | [0.20000002, 0.36000001] |
| `3->4` | 0.18 | 3 | 2 | 1 | 0 | [-0.34, 0.44] |
| `4->5` | -0.03333333 | 3 | 1 | 2 | 0 | [-0.12, 0.04000001] |
| `5->6` | -0.17999999 | 3 | 1 | 2 | 0 | [-0.31999999, 0.02000001] |
| `6->7` | 0.06666667 | 3 | 2 | 0 | 1 | [0, 0.18] |
| `7->8` | -0.04 | 3 | 1 | 1 | 1 | [-0.16, 0.03999999] |

## Reasons

- fresh-gap direction is mixed across seeds/stages; positive and non-positive outcomes cannot support a stable LoP direction

Retention/forgetting metrics are inventory only and are not substituted for fresh-gap.
