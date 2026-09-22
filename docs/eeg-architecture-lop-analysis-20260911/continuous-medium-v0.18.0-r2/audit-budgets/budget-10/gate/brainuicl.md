# Continuous EEG LoP gate

- Status: `blocked-inconsistent-direction`
- Outcome: `task.plasticity.fresh_gap`
- Seeds: 3/3
- Transitions: 8/2
- Design consistent: `True`
- Scientific conclusion allowed: `False`

| Transition | Mean | Seeds | Positive | Negative | Zero | 95% CI |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `0->1` | -0.05333333 | 3 | 1 | 1 | 1 | [-0.40000001, 0.24000001] |
| `1->2` | 0.02666666 | 3 | 1 | 0 | 2 | [0, 0.07999998] |
| `2->3` | 0.18666668 | 3 | 2 | 0 | 1 | [0, 0.36000001] |
| `3->4` | 0.04666666 | 3 | 2 | 1 | 0 | [-0.02000001, 0.09999999] |
| `4->5` | -0.02 | 3 | 1 | 1 | 1 | [-0.08, 0.02000001] |
| `5->6` | -0.07333334 | 3 | 1 | 2 | 0 | [-0.23999999, 0.08] |
| `6->7` | 0.15333333 | 3 | 3 | 0 | 0 | [0.07999998, 0.28] |
| `7->8` | 0.06 | 3 | 2 | 0 | 1 | [0, 0.16] |

## Reasons

- fresh-gap direction is mixed across seeds/stages; positive and non-positive outcomes cannot support a stable LoP direction

Retention/forgetting metrics are inventory only and are not substituted for fresh-gap.
