# Continuous EEG LoP gate

- Status: `blocked-inconsistent-direction`
- Outcome: `task.plasticity.fresh_gap`
- Seeds: 3/3
- Transitions: 8/2
- Design consistent: `True`
- Scientific conclusion allowed: `False`

| Transition | Mean | Seeds | Positive | Negative | Zero | 95% CI |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `0->1` | 0.30729167 | 3 | 3 | 0 | 0 | [0.25579427, 0.34375] |
| `1->2` | 0.19791667 | 3 | 3 | 0 | 0 | [0.1796875, 0.2109375] |
| `2->3` | 0.20833333 | 3 | 3 | 0 | 0 | [0.15338542, 0.296875] |
| `3->4` | 0.14583333 | 3 | 3 | 0 | 0 | [0.0078125, 0.2421875] |
| `4->5` | 0.13541667 | 3 | 3 | 0 | 0 | [0.11979167, 0.1640625] |
| `5->6` | -0.09114583 | 3 | 0 | 3 | 0 | [-0.140625, -0.04238281] |
| `6->7` | -0.0234375 | 3 | 1 | 2 | 0 | [-0.046875, 0.015625] |
| `7->8` | 0.08333333 | 3 | 3 | 0 | 0 | [0.07154948, 0.09375] |

## Reasons

- fresh-gap direction is mixed across seeds/stages; positive and non-positive outcomes cannot support a stable LoP direction

Retention/forgetting metrics are inventory only and are not substituted for fresh-gap.
