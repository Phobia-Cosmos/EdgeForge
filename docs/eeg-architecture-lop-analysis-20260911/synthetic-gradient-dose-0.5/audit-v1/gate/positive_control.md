# Continuous EEG LoP gate

- Status: `blocked-inconsistent-direction`
- Outcome: `task.plasticity.fresh_gap`
- Seeds: 3/3
- Transitions: 8/2
- Design consistent: `True`
- Scientific conclusion allowed: `False`

| Transition | Mean | Seeds | Positive | Negative | Zero | 95% CI |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `0->1` | 0.29427083 | 3 | 3 | 0 | 0 | [0.265625, 0.3203125] |
| `1->2` | 0.15104167 | 3 | 3 | 0 | 0 | [0.140625, 0.1640625] |
| `2->3` | 0.29166667 | 3 | 3 | 0 | 0 | [0.25, 0.34375] |
| `3->4` | 0.19270833 | 3 | 3 | 0 | 0 | [0.1796875, 0.2109375] |
| `4->5` | 0.125 | 3 | 3 | 0 | 0 | [0.0078125, 0.1953125] |
| `5->6` | -0.10416667 | 3 | 0 | 3 | 0 | [-0.1953125, -0.046875] |
| `6->7` | 0.00260417 | 3 | 2 | 1 | 0 | [-0.046875, 0.0390625] |
| `7->8` | 0.02864583 | 3 | 2 | 1 | 0 | [-0.0078125, 0.0625] |

## Reasons

- fresh-gap direction is mixed across seeds/stages; positive and non-positive outcomes cannot support a stable LoP direction

Retention/forgetting metrics are inventory only and are not substituted for fresh-gap.
