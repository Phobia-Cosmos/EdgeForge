# Continuous EEG LoP gate

- Status: `candidate`
- Outcome: `task.plasticity.fresh_gap`
- Seeds: 3/3
- Transitions: 8/2
- Design consistent: `True`
- Scientific conclusion allowed: `False`

| Transition | Mean | Seeds | Positive | Negative | Zero | 95% CI |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `0->1` | 0.52083333 | 3 | 3 | 0 | 0 | [0.3984375, 0.6328125] |
| `1->2` | 0.609375 | 3 | 3 | 0 | 0 | [0.5625, 0.6953125] |
| `2->3` | 0.52864583 | 3 | 3 | 0 | 0 | [0.4921875, 0.5859375] |
| `3->4` | 0.50260417 | 3 | 3 | 0 | 0 | [0.4453125, 0.5390625] |
| `4->5` | 0.5625 | 3 | 3 | 0 | 0 | [0.5234375, 0.59375] |
| `5->6` | 0.6328125 | 3 | 3 | 0 | 0 | [0.5546875, 0.7578125] |
| `6->7` | 0.51302083 | 3 | 3 | 0 | 0 | [0.375, 0.625] |
| `7->8` | 0.48958333 | 3 | 3 | 0 | 0 | [0.3359375, 0.65625] |

## Reasons

- all gate checks passed; this remains a candidate only

Retention/forgetting metrics are inventory only and are not substituted for fresh-gap.
