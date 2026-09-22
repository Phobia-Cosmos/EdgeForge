# Continuous EEG LoP gate

- Status: `blocked-inconsistent-direction`
- Outcome: `task.plasticity.fresh_gap`
- Seeds: 3/3
- Transitions: 8/2
- Design consistent: `True`
- Scientific conclusion allowed: `False`

| Transition | Mean | Seeds | Positive | Negative | Zero | 95% CI |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `0->1` | -0.05333333 | 3 | 0 | 1 | 2 | [-0.16, 0] |
| `1->2` | 0.03333333 | 3 | 1 | 1 | 1 | [-0.06, 0.16] |
| `2->3` | -0.28666668 | 3 | 0 | 3 | 0 | [-0.36000001, -0.14] |
| `3->4` | 0.07333333 | 3 | 2 | 1 | 0 | [-0.28, 0.38] |
| `4->5` | 0.04 | 3 | 1 | 2 | 0 | [-0.03999999, 0.18] |
| `5->6` | -0.11333333 | 3 | 0 | 2 | 1 | [-0.17999999, 0] |
| `6->7` | -0.04666667 | 3 | 0 | 3 | 0 | [-0.09999999, -0.02000001] |
| `7->8` | 0.11333332 | 3 | 3 | 0 | 0 | [0.01999998, 0.16] |

## Reasons

- fresh-gap direction is mixed across seeds/stages; positive and non-positive outcomes cannot support a stable LoP direction

Retention/forgetting metrics are inventory only and are not substituted for fresh-gap.
