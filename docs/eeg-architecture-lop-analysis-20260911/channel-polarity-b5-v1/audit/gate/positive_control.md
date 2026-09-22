# Continuous EEG LoP gate

- Status: `candidate`
- Outcome: `task.plasticity.fresh_gap`
- Seeds: 3/3
- Transitions: 8/2
- Design consistent: `True`
- Scientific conclusion allowed: `False`

| Transition | Mean | Seeds | Positive | Negative | Zero | 95% CI |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `0->1` | 0.44791667 | 3 | 3 | 0 | 0 | [0.3671875, 0.515625] |
| `1->2` | 0.52604167 | 3 | 3 | 0 | 0 | [0.4609375, 0.6171875] |
| `2->3` | 0.54427083 | 3 | 3 | 0 | 0 | [0.4921875, 0.6015625] |
| `3->4` | 0.453125 | 3 | 3 | 0 | 0 | [0.421875, 0.5] |
| `4->5` | 0.57291667 | 3 | 3 | 0 | 0 | [0.53125, 0.640625] |
| `5->6` | 0.6015625 | 3 | 3 | 0 | 0 | [0.53125, 0.6875] |
| `6->7` | 0.52604167 | 3 | 3 | 0 | 0 | [0.3515625, 0.6484375] |
| `7->8` | 0.4921875 | 3 | 3 | 0 | 0 | [0.3203125, 0.609375] |

## Reasons

- all gate checks passed; this remains a candidate only

Retention/forgetting metrics are inventory only and are not substituted for fresh-gap.
