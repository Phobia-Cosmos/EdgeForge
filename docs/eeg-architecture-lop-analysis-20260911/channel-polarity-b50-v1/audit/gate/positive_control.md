# Continuous EEG LoP gate

- Status: `blocked-inconsistent-direction`
- Outcome: `task.plasticity.fresh_gap`
- Seeds: 3/3
- Transitions: 8/2
- Design consistent: `True`
- Scientific conclusion allowed: `False`

| Transition | Mean | Seeds | Positive | Negative | Zero | 95% CI |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `0->1` | 0.67708333 | 3 | 3 | 0 | 0 | [0.671875, 0.6875] |
| `1->2` | 0.609375 | 3 | 3 | 0 | 0 | [0.5546875, 0.671875] |
| `2->3` | 0.4296875 | 3 | 3 | 0 | 0 | [0.3984375, 0.4765625] |
| `3->4` | 0.3203125 | 3 | 3 | 0 | 0 | [0.2109375, 0.40625] |
| `4->5` | 0.0546875 | 3 | 2 | 1 | 0 | [-0.046875, 0.1953125] |
| `5->6` | -0.05208333 | 3 | 1 | 2 | 0 | [-0.140625, 0.0625] |
| `6->7` | -0.1875 | 3 | 0 | 3 | 0 | [-0.265625, -0.0859375] |
| `7->8` | -0.16666667 | 3 | 0 | 3 | 0 | [-0.2265625, -0.125] |

## Reasons

- fresh-gap direction is mixed across seeds/stages; positive and non-positive outcomes cannot support a stable LoP direction

Retention/forgetting metrics are inventory only and are not substituted for fresh-gap.
