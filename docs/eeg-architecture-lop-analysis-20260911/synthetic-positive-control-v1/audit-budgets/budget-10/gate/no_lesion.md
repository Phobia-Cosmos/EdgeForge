# Continuous EEG LoP gate

- Status: `blocked-inconsistent-direction`
- Outcome: `task.plasticity.fresh_gap`
- Seeds: 3/3
- Transitions: 8/2
- Design consistent: `True`
- Scientific conclusion allowed: `False`

| Transition | Mean | Seeds | Positive | Negative | Zero | 95% CI |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `0->1` | 0.12239583 | 3 | 3 | 0 | 0 | [0.03255208, 0.18489583] |
| `1->2` | 0.03385417 | 3 | 2 | 1 | 0 | [-0.03613281, 0.09375] |
| `2->3` | 0.1484375 | 3 | 3 | 0 | 0 | [0.1015625, 0.171875] |
| `3->4` | 0.10416667 | 3 | 2 | 1 | 0 | [-0.0390625, 0.20833333] |
| `4->5` | -0.00260417 | 3 | 1 | 2 | 0 | [-0.0703125, 0.0859375] |
| `5->6` | -0.06770833 | 3 | 1 | 2 | 0 | [-0.1796875, 0.05501302] |
| `6->7` | 0.01822917 | 3 | 2 | 1 | 0 | [-0.0078125, 0.03125] |
| `7->8` | 0.16927083 | 3 | 3 | 0 | 0 | [0.13802083, 0.21419271] |

## Reasons

- fresh-gap direction is mixed across seeds/stages; positive and non-positive outcomes cannot support a stable LoP direction

Retention/forgetting metrics are inventory only and are not substituted for fresh-gap.
