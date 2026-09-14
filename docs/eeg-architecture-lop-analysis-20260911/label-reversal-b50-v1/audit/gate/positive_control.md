# Continuous EEG LoP gate

- Status: `blocked-inconsistent-direction`
- Outcome: `task.plasticity.fresh_gap`
- Seeds: 3/3
- Transitions: 8/2
- Design consistent: `True`
- Scientific conclusion allowed: `False`

| Transition | Mean | Seeds | Positive | Negative | Zero | 95% CI |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `0->1` | 0.66666667 | 3 | 3 | 0 | 0 | [0.6484375, 0.6796875] |
| `1->2` | 0.56770833 | 3 | 3 | 0 | 0 | [0.5390625, 0.6015625] |
| `2->3` | 0.390625 | 3 | 3 | 0 | 0 | [0.2890625, 0.4453125] |
| `3->4` | 0.296875 | 3 | 3 | 0 | 0 | [0.15625, 0.421875] |
| `4->5` | -0.00520833 | 3 | 1 | 2 | 0 | [-0.0234375, 0.03125] |
| `5->6` | -0.05208333 | 3 | 0 | 3 | 0 | [-0.0859375, -0.0234375] |
| `6->7` | -0.1796875 | 3 | 0 | 3 | 0 | [-0.2734375, -0.1015625] |
| `7->8` | -0.16145833 | 3 | 0 | 3 | 0 | [-0.203125, -0.125] |

## Reasons

- fresh-gap direction is mixed across seeds/stages; positive and non-positive outcomes cannot support a stable LoP direction

Retention/forgetting metrics are inventory only and are not substituted for fresh-gap.
