# Continuous EEG LoP gate

- Status: `candidate`
- Outcome: `task.plasticity.fresh_gap`
- Seeds: 3/3
- Transitions: 8/2
- Design consistent: `True`
- Scientific conclusion allowed: `False`

| Transition | Mean | Seeds | Positive | Negative | Zero | 95% CI |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `0->1` | 0.36197917 | 3 | 3 | 0 | 0 | [0.34114583, 0.3828125] |
| `1->2` | 0.17708333 | 3 | 3 | 0 | 0 | [0.1546224, 0.1953125] |
| `2->3` | 0.35677083 | 3 | 3 | 0 | 0 | [0.34088542, 0.3671875] |
| `3->4` | 0.29166667 | 3 | 3 | 0 | 0 | [0.25260417, 0.3359375] |
| `4->5` | 0.36458333 | 3 | 3 | 0 | 0 | [0.3203125, 0.4453125] |
| `5->6` | 0.1484375 | 3 | 3 | 0 | 0 | [0.1328125, 0.1796875] |
| `6->7` | 0.15364583 | 3 | 3 | 0 | 0 | [0.0859375, 0.1953125] |
| `7->8` | 0.12760417 | 3 | 3 | 0 | 0 | [0.08072917, 0.18541667] |

## Reasons

- all gate checks passed; this remains a candidate only

Retention/forgetting metrics are inventory only and are not substituted for fresh-gap.
