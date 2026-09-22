# Continuous EEG LoP gate

- Status: `blocked-inconsistent-direction`
- Outcome: `task.plasticity.fresh_gap`
- Seeds: 3/3
- Transitions: 8/2
- Design consistent: `True`
- Scientific conclusion allowed: `False`

| Transition | Mean | Seeds | Positive | Negative | Zero | 95% CI |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `0->1` | -0.08 | 3 | 1 | 1 | 1 | [-0.26000001, 0.02] |
| `1->2` | 0.0 | 3 | 0 | 0 | 3 | [0, 0] |
| `2->3` | 0.10666667 | 3 | 2 | 1 | 0 | [-0.08000001, 0.36000001] |
| `3->4` | 0.10666666 | 3 | 2 | 0 | 1 | [0, 0.22] |
| `4->5` | -0.06666666 | 3 | 1 | 1 | 1 | [-0.21999999, 0.02000001] |
| `5->6` | 0.04 | 3 | 2 | 1 | 0 | [-0.08, 0.09999999] |
| `6->7` | 0.09999999 | 3 | 3 | 0 | 0 | [0.09999999, 0.09999999] |
| `7->8` | -0.04666667 | 3 | 2 | 1 | 0 | [-0.22, 0.03999999] |

## Reasons

- fresh-gap direction is mixed across seeds/stages; positive and non-positive outcomes cannot support a stable LoP direction

Retention/forgetting metrics are inventory only and are not substituted for fresh-gap.
