# Continuous EEG LoP gate

- Status: `blocked-inconsistent-direction`
- Outcome: `task.plasticity.fresh_gap`
- Seeds: 3/3
- Transitions: 8/2
- Design consistent: `True`
- Scientific conclusion allowed: `False`

| Transition | Mean | Seeds | Positive | Negative | Zero | 95% CI |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `0->1` | 0.21875 | 3 | 3 | 0 | 0 | [0.18997396, 0.24479167] |
| `1->2` | 0.18229167 | 3 | 3 | 0 | 0 | [0.16529948, 0.2109375] |
| `2->3` | 0.1953125 | 3 | 3 | 0 | 0 | [0.12669271, 0.2578125] |
| `3->4` | 0.14322917 | 3 | 3 | 0 | 0 | [0.0078125, 0.24479167] |
| `4->5` | 0.1484375 | 3 | 3 | 0 | 0 | [0.1328125, 0.15885417] |
| `5->6` | -0.09895833 | 3 | 0 | 3 | 0 | [-0.15625, -0.06497396] |
| `6->7` | 0.00260417 | 3 | 1 | 1 | 1 | [-0.03125, 0.0546875] |
| `7->8` | 0.08072917 | 3 | 3 | 0 | 0 | [0.07291667, 0.08854167] |

## Reasons

- fresh-gap direction is mixed across seeds/stages; positive and non-positive outcomes cannot support a stable LoP direction

Retention/forgetting metrics are inventory only and are not substituted for fresh-gap.
