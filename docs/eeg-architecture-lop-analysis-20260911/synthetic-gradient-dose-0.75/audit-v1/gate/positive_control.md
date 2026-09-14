# Continuous EEG LoP gate

- Status: `blocked-inconsistent-direction`
- Outcome: `task.plasticity.fresh_gap`
- Seeds: 3/3
- Transitions: 8/2
- Design consistent: `True`
- Scientific conclusion allowed: `False`

| Transition | Mean | Seeds | Positive | Negative | Zero | 95% CI |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `0->1` | 0.296875 | 3 | 3 | 0 | 0 | [0.2734375, 0.3203125] |
| `1->2` | 0.15364583 | 3 | 3 | 0 | 0 | [0.140625, 0.1640625] |
| `2->3` | 0.29166667 | 3 | 3 | 0 | 0 | [0.25, 0.359375] |
| `3->4` | 0.1875 | 3 | 3 | 0 | 0 | [0.1796875, 0.1953125] |
| `4->5` | 0.11458333 | 3 | 2 | 1 | 0 | [-0.0078125, 0.1796875] |
| `5->6` | -0.10416667 | 3 | 0 | 3 | 0 | [-0.171875, -0.046875] |
| `6->7` | 0.0 | 3 | 2 | 1 | 0 | [-0.0546875, 0.046875] |
| `7->8` | 0.03645833 | 3 | 3 | 0 | 0 | [0.015625, 0.0625] |

## Reasons

- fresh-gap direction is mixed across seeds/stages; positive and non-positive outcomes cannot support a stable LoP direction

Retention/forgetting metrics are inventory only and are not substituted for fresh-gap.
