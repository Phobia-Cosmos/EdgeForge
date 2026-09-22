# Continuous EEG LoP gate

- Status: `blocked-inconsistent-direction`
- Outcome: `task.plasticity.fresh_gap`
- Seeds: 3/3
- Transitions: 8/2
- Design consistent: `True`
- Scientific conclusion allowed: `False`

| Transition | Mean | Seeds | Positive | Negative | Zero | 95% CI |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `0->1` | 0.08 | 3 | 1 | 0 | 2 | [0, 0.24000001] |
| `1->2` | 0.18666666 | 3 | 3 | 0 | 0 | [0.16, 0.23999998] |
| `2->3` | -0.30666668 | 3 | 0 | 3 | 0 | [-0.36000001, -0.2] |
| `3->4` | 0.11333333 | 3 | 2 | 1 | 0 | [-0.06, 0.3] |
| `4->5` | -0.08 | 3 | 0 | 3 | 0 | [-0.18000001, -0.02000001] |
| `5->6` | 0.05333333 | 3 | 1 | 1 | 1 | [-0.08, 0.23999999] |
| `6->7` | 0.0 | 3 | 1 | 1 | 1 | [-0.02000001, 0.02000001] |
| `7->8` | 0.10666666 | 3 | 2 | 0 | 1 | [0, 0.16] |

## Reasons

- fresh-gap direction is mixed across seeds/stages; positive and non-positive outcomes cannot support a stable LoP direction

Retention/forgetting metrics are inventory only and are not substituted for fresh-gap.
