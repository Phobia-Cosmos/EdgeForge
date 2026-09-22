# Continuous EEG LoP gate

- Status: `blocked-inconsistent-direction`
- Outcome: `task.plasticity.fresh_gap`
- Seeds: 3/3
- Transitions: 8/2
- Design consistent: `True`
- Scientific conclusion allowed: `False`

| Transition | Mean | Seeds | Positive | Negative | Zero | 95% CI |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `0->1` | 0.06 | 3 | 2 | 0 | 1 | [0, 0.16] |
| `1->2` | 0.21333332 | 3 | 3 | 0 | 0 | [0.11999999, 0.27999999] |
| `2->3` | 0.04000001 | 3 | 1 | 1 | 1 | [-0.08, 0.20000002] |
| `3->4` | 0.22666667 | 3 | 2 | 0 | 1 | [0, 0.34] |
| `4->5` | 0.04 | 3 | 1 | 1 | 1 | [-0.07999998, 0.2] |
| `5->6` | 0.19333333 | 3 | 3 | 0 | 0 | [0.09999999, 0.23999999] |
| `6->7` | 0.04666665 | 3 | 2 | 1 | 0 | [-0.02000001, 0.07999998] |
| `7->8` | 0.05333333 | 3 | 1 | 1 | 1 | [-0.22, 0.38] |

## Reasons

- fresh-gap direction is mixed across seeds/stages; positive and non-positive outcomes cannot support a stable LoP direction

Retention/forgetting metrics are inventory only and are not substituted for fresh-gap.
