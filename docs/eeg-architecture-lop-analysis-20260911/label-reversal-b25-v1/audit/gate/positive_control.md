# Continuous EEG LoP gate

- Status: `candidate`
- Outcome: `task.plasticity.fresh_gap`
- Seeds: 3/3
- Transitions: 8/2
- Design consistent: `True`
- Scientific conclusion allowed: `False`

| Transition | Mean | Seeds | Positive | Negative | Zero | 95% CI |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `0->1` | 0.65364583 | 3 | 3 | 0 | 0 | [0.625, 0.671875] |
| `1->2` | 0.61979167 | 3 | 3 | 0 | 0 | [0.5625, 0.6796875] |
| `2->3` | 0.44010417 | 3 | 3 | 0 | 0 | [0.390625, 0.4921875] |
| `3->4` | 0.50520833 | 3 | 3 | 0 | 0 | [0.390625, 0.6328125] |
| `4->5` | 0.5234375 | 3 | 3 | 0 | 0 | [0.4609375, 0.625] |
| `5->6` | 0.55989583 | 3 | 3 | 0 | 0 | [0.5234375, 0.578125] |
| `6->7` | 0.37760417 | 3 | 3 | 0 | 0 | [0.2578125, 0.4765625] |
| `7->8` | 0.19010417 | 3 | 3 | 0 | 0 | [0.109375, 0.34375] |

## Reasons

- all gate checks passed; this remains a candidate only

Retention/forgetting metrics are inventory only and are not substituted for fresh-gap.
