# Continuous EEG LoP gate

- Status: `candidate`
- Outcome: `task.plasticity.fresh_gap`
- Seeds: 3/3
- Transitions: 8/2
- Design consistent: `True`
- Scientific conclusion allowed: `False`

| Transition | Mean | Seeds | Positive | Negative | Zero | 95% CI |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `0->1` | 0.67447917 | 3 | 3 | 0 | 0 | [0.65625, 0.703125] |
| `1->2` | 0.65104167 | 3 | 3 | 0 | 0 | [0.609375, 0.703125] |
| `2->3` | 0.5234375 | 3 | 3 | 0 | 0 | [0.46875, 0.6015625] |
| `3->4` | 0.48697917 | 3 | 3 | 0 | 0 | [0.359375, 0.5703125] |
| `4->5` | 0.53385417 | 3 | 3 | 0 | 0 | [0.46875, 0.578125] |
| `5->6` | 0.53645833 | 3 | 3 | 0 | 0 | [0.484375, 0.640625] |
| `6->7` | 0.39583333 | 3 | 3 | 0 | 0 | [0.25, 0.4921875] |
| `7->8` | 0.171875 | 3 | 3 | 0 | 0 | [0.1171875, 0.2109375] |

## Reasons

- all gate checks passed; this remains a candidate only

Retention/forgetting metrics are inventory only and are not substituted for fresh-gap.
