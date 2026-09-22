# Continuous EEG LoP gate

- Status: `candidate`
- Outcome: `task.plasticity.fresh_gap`
- Seeds: 3/3
- Transitions: 8/2
- Design consistent: `True`
- Scientific conclusion allowed: `False`

| Transition | Mean | Seeds | Positive | Negative | Zero | 95% CI |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `0->1` | 0.47916667 | 3 | 3 | 0 | 0 | [0.4375, 0.53125] |
| `1->2` | 0.58333333 | 3 | 3 | 0 | 0 | [0.515625, 0.6640625] |
| `2->3` | 0.50260417 | 3 | 3 | 0 | 0 | [0.4296875, 0.609375] |
| `3->4` | 0.46614583 | 3 | 3 | 0 | 0 | [0.3984375, 0.546875] |
| `4->5` | 0.58072917 | 3 | 3 | 0 | 0 | [0.5234375, 0.640625] |
| `5->6` | 0.62760417 | 3 | 3 | 0 | 0 | [0.578125, 0.6953125] |
| `6->7` | 0.4296875 | 3 | 3 | 0 | 0 | [0.3046875, 0.5859375] |
| `7->8` | 0.4609375 | 3 | 3 | 0 | 0 | [0.359375, 0.5390625] |

## Reasons

- all gate checks passed; this remains a candidate only

Retention/forgetting metrics are inventory only and are not substituted for fresh-gap.
