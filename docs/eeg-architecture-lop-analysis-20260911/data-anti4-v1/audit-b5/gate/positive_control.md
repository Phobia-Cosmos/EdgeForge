# Continuous EEG LoP gate

- Status: `blocked-inconsistent-direction`
- Outcome: `task.plasticity.fresh_gap`
- Seeds: 3/3
- Transitions: 8/2
- Design consistent: `True`
- Scientific conclusion allowed: `False`

| Transition | Mean | Seeds | Positive | Negative | Zero | 95% CI |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `0->1` | 0.50520833 | 3 | 3 | 0 | 0 | [0.1484375, 0.765625] |
| `1->2` | 0.13541667 | 3 | 2 | 0 | 1 | [0, 0.3359375] |
| `2->3` | -0.3828125 | 3 | 0 | 3 | 0 | [-0.6875, -0.109375] |
| `3->4` | 0.52864583 | 3 | 3 | 0 | 0 | [0.2421875, 0.765625] |
| `4->5` | -0.03125 | 3 | 2 | 1 | 0 | [-0.2578125, 0.15625] |
| `5->6` | 0.16927083 | 3 | 2 | 0 | 1 | [0, 0.328125] |
| `6->7` | 0.0 | 3 | 2 | 1 | 0 | [-0.2734375, 0.140625] |
| `7->8` | -0.30729167 | 3 | 0 | 3 | 0 | [-0.7890625, -0.0234375] |

## Reasons

- fresh-gap direction is mixed across seeds/stages; positive and non-positive outcomes cannot support a stable LoP direction

Retention/forgetting metrics are inventory only and are not substituted for fresh-gap.
