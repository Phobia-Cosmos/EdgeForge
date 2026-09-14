# Continuous EEG LoP gate

- Status: `blocked-inconsistent-direction`
- Outcome: `task.plasticity.fresh_gap`
- Seeds: 3/3
- Transitions: 8/2
- Design consistent: `True`
- Scientific conclusion allowed: `False`

| Transition | Mean | Seeds | Positive | Negative | Zero | 95% CI |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `0->1` | 0.13020833 | 3 | 3 | 0 | 0 | [0.09895833, 0.17513021] |
| `1->2` | -0.06770833 | 3 | 1 | 2 | 0 | [-0.13072917, 0.015625] |
| `2->3` | 0.0625 | 3 | 3 | 0 | 0 | [0.015625, 0.11263021] |
| `3->4` | 0.12760417 | 3 | 3 | 0 | 0 | [0.04166667, 0.2890625] |
| `4->5` | 0.07552083 | 3 | 2 | 1 | 0 | [-0.01302083, 0.2109375] |
| `5->6` | 0.03645833 | 3 | 2 | 1 | 0 | [-0.00820312, 0.09375] |
| `6->7` | 0.02083333 | 3 | 2 | 1 | 0 | [-0.015625, 0.046875] |
| `7->8` | -0.05729167 | 3 | 0 | 3 | 0 | [-0.0703125, -0.04427083] |

## Reasons

- fresh-gap direction is mixed across seeds/stages; positive and non-positive outcomes cannot support a stable LoP direction

Retention/forgetting metrics are inventory only and are not substituted for fresh-gap.
