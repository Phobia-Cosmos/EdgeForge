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
| `1->2` | 0.03333332 | 3 | 2 | 1 | 0 | [-0.04000002, 0.07999998] |
| `2->3` | 0.28666668 | 3 | 3 | 0 | 0 | [0.20000002, 0.36000001] |
| `3->4` | 0.00666666 | 3 | 2 | 1 | 0 | [-0.10000002, 0.09999999] |
| `4->5` | -0.23333333 | 3 | 0 | 3 | 0 | [-0.32000001, -0.17999999] |
| `5->6` | -0.02 | 3 | 1 | 2 | 0 | [-0.14, 0.16000001] |
| `6->7` | 0.04 | 3 | 2 | 0 | 1 | [0, 0.09999999] |
| `7->8` | -0.06666667 | 3 | 0 | 3 | 0 | [-0.12, -0.02000001] |

## Reasons

- fresh-gap direction is mixed across seeds/stages; positive and non-positive outcomes cannot support a stable LoP direction

Retention/forgetting metrics are inventory only and are not substituted for fresh-gap.
