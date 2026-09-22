# Continuous EEG LoP gate

- Status: `candidate`
- Outcome: `task.plasticity.fresh_gap`
- Seeds: 3/3
- Transitions: 8/2
- Design consistent: `True`
- Scientific conclusion allowed: `False`

| Transition | Mean | Seeds | Positive | Negative | Zero | 95% CI |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `0->1` | 0.59375 | 3 | 3 | 0 | 0 | [0.515625, 0.65625] |
| `1->2` | 0.58333333 | 3 | 3 | 0 | 0 | [0.5, 0.65625] |
| `2->3` | 0.4921875 | 3 | 3 | 0 | 0 | [0.4140625, 0.5703125] |
| `3->4` | 0.4609375 | 3 | 3 | 0 | 0 | [0.40625, 0.5234375] |
| `4->5` | 0.60677083 | 3 | 3 | 0 | 0 | [0.5390625, 0.671875] |
| `5->6` | 0.65625 | 3 | 3 | 0 | 0 | [0.6171875, 0.71875] |
| `6->7` | 0.44010417 | 3 | 3 | 0 | 0 | [0.3984375, 0.5078125] |
| `7->8` | 0.45052083 | 3 | 3 | 0 | 0 | [0.3828125, 0.4921875] |

## Reasons

- all gate checks passed; this remains a candidate only

Retention/forgetting metrics are inventory only and are not substituted for fresh-gap.
