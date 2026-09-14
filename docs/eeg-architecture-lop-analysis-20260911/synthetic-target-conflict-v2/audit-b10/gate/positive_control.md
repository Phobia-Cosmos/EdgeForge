# Continuous EEG LoP gate

- Status: `blocked-inconsistent-direction`
- Outcome: `task.plasticity.fresh_gap`
- Seeds: 3/3
- Transitions: 8/2
- Design consistent: `True`
- Scientific conclusion allowed: `False`

| Transition | Mean | Seeds | Positive | Negative | Zero | 95% CI |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `0->1` | 0.77604167 | 3 | 3 | 0 | 0 | [0.3984375, 0.96875] |
| `1->2` | 0.96614583 | 3 | 3 | 0 | 0 | [0.9140625, 1] |
| `2->3` | 0.76822917 | 3 | 3 | 0 | 0 | [0.578125, 0.9375] |
| `3->4` | 0.6640625 | 3 | 3 | 0 | 0 | [0.421875, 0.8828125] |
| `4->5` | 0.1796875 | 3 | 2 | 0 | 1 | [0, 0.421875] |
| `5->6` | 0.00260417 | 3 | 1 | 0 | 2 | [0, 0.0078125] |
| `6->7` | -0.2734375 | 3 | 0 | 2 | 1 | [-0.6875, 0] |
| `7->8` | -0.26041667 | 3 | 0 | 2 | 1 | [-0.7265625, 0] |

## Reasons

- fresh-gap direction is mixed across seeds/stages; positive and non-positive outcomes cannot support a stable LoP direction

Retention/forgetting metrics are inventory only and are not substituted for fresh-gap.
