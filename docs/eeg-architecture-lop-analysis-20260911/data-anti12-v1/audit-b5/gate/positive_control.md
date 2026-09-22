# Continuous EEG LoP gate

- Status: `blocked-inconsistent-direction`
- Outcome: `task.plasticity.fresh_gap`
- Seeds: 3/3
- Transitions: 8/2
- Design consistent: `True`
- Scientific conclusion allowed: `False`

| Transition | Mean | Seeds | Positive | Negative | Zero | 95% CI |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `0->1` | 0.40364583 | 3 | 2 | 0 | 1 | [0, 0.8203125] |
| `1->2` | 0.26302083 | 3 | 3 | 0 | 0 | [0.046875, 0.3828125] |
| `2->3` | -0.30208333 | 3 | 1 | 2 | 0 | [-0.7734375, 0.15625] |
| `3->4` | 0.55989583 | 3 | 2 | 1 | 0 | [-0.0546875, 0.8828125] |
| `4->5` | -0.2109375 | 3 | 1 | 2 | 0 | [-0.3828125, 0.1015625] |
| `5->6` | -0.2734375 | 3 | 1 | 2 | 0 | [-0.828125, 0.0625] |
| `6->7` | -0.05729167 | 3 | 1 | 2 | 0 | [-0.109375, 0.015625] |
| `7->8` | -0.0234375 | 3 | 2 | 1 | 0 | [-0.796875, 0.515625] |

## Reasons

- fresh-gap direction is mixed across seeds/stages; positive and non-positive outcomes cannot support a stable LoP direction

Retention/forgetting metrics are inventory only and are not substituted for fresh-gap.
