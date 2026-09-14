# Continuous EEG LoP gate

- Status: `candidate`
- Outcome: `task.plasticity.fresh_gap`
- Seeds: 3/3
- Transitions: 8/2
- Design consistent: `True`
- Scientific conclusion allowed: `False`

| Transition | Mean | Seeds | Positive | Negative | Zero | 95% CI |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `0->1` | 0.22916667 | 3 | 3 | 0 | 0 | [0.2093099, 0.2421875] |
| `1->2` | 0.11197917 | 3 | 3 | 0 | 0 | [0.03789063, 0.1640625] |
| `2->3` | 0.22395833 | 3 | 3 | 0 | 0 | [0.1484375, 0.26438802] |
| `3->4` | 0.234375 | 3 | 3 | 0 | 0 | [0.19010417, 0.296875] |
| `4->5` | 0.234375 | 3 | 3 | 0 | 0 | [0.15364583, 0.375] |
| `5->6` | 0.08333333 | 3 | 3 | 0 | 0 | [0.0390625, 0.1171875] |
| `6->7` | 0.11197917 | 3 | 3 | 0 | 0 | [0.0234375, 0.17447917] |
| `7->8` | 0.0859375 | 3 | 3 | 0 | 0 | [0.0390625, 0.14921875] |

## Reasons

- all gate checks passed; this remains a candidate only

Retention/forgetting metrics are inventory only and are not substituted for fresh-gap.
