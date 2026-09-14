# Continuous EEG LoP gate

- Status: `blocked-inconsistent-direction`
- Outcome: `task.plasticity.fresh_gap`
- Seeds: 3/3
- Transitions: 8/2
- Design consistent: `True`
- Scientific conclusion allowed: `False`

| Transition | Mean | Seeds | Positive | Negative | Zero | 95% CI |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `0->1` | 0.29947917 | 3 | 3 | 0 | 0 | [0.2734375, 0.328125] |
| `1->2` | 0.15364583 | 3 | 3 | 0 | 0 | [0.140625, 0.1640625] |
| `2->3` | 0.28645833 | 3 | 3 | 0 | 0 | [0.25, 0.3515625] |
| `3->4` | 0.1796875 | 3 | 3 | 0 | 0 | [0.1796875, 0.1796875] |
| `4->5` | 0.11458333 | 3 | 2 | 1 | 0 | [-0.015625, 0.1796875] |
| `5->6` | -0.11197917 | 3 | 0 | 3 | 0 | [-0.1875, -0.046875] |
| `6->7` | 0.0078125 | 3 | 2 | 1 | 0 | [-0.0546875, 0.0703125] |
| `7->8` | 0.04427083 | 3 | 3 | 0 | 0 | [0.0390625, 0.0546875] |

## Reasons

- fresh-gap direction is mixed across seeds/stages; positive and non-positive outcomes cannot support a stable LoP direction

Retention/forgetting metrics are inventory only and are not substituted for fresh-gap.
