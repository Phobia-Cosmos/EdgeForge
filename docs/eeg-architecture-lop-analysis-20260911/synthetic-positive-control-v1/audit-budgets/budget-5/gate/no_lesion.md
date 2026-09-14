# Continuous EEG LoP gate

- Status: `blocked-inconsistent-direction`
- Outcome: `task.plasticity.fresh_gap`
- Seeds: 3/3
- Transitions: 8/2
- Design consistent: `True`
- Scientific conclusion allowed: `False`

| Transition | Mean | Seeds | Positive | Negative | Zero | 95% CI |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `0->1` | 0.06510417 | 3 | 2 | 1 | 0 | [-0.02994792, 0.1328125] |
| `1->2` | -0.0390625 | 3 | 1 | 1 | 1 | [-0.11178385, 0.015625] |
| `2->3` | 0.12239583 | 3 | 3 | 0 | 0 | [0.0390625, 0.18007812] |
| `3->4` | 0.08854167 | 3 | 2 | 1 | 0 | [-0.0390625, 0.1875] |
| `4->5` | -0.046875 | 3 | 1 | 2 | 0 | [-0.125, 0.015625] |
| `5->6` | -0.04947917 | 3 | 1 | 2 | 0 | [-0.1953125, 0.07148437] |
| `6->7` | 0.00260417 | 3 | 2 | 1 | 0 | [-0.0546875, 0.03125] |
| `7->8` | 0.140625 | 3 | 3 | 0 | 0 | [0.1015625, 0.20019531] |

## Reasons

- fresh-gap direction is mixed across seeds/stages; positive and non-positive outcomes cannot support a stable LoP direction

Retention/forgetting metrics are inventory only and are not substituted for fresh-gap.
