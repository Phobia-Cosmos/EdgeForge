# Continuous EEG LoP gate

- Status: `blocked-inconsistent-direction`
- Outcome: `task.plasticity.fresh_gap`
- Seeds: 3/3
- Transitions: 8/2
- Design consistent: `True`
- Scientific conclusion allowed: `False`

| Transition | Mean | Seeds | Positive | Negative | Zero | 95% CI |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `0->1` | 0.0 | 3 | 0 | 0 | 3 | [0, 0] |
| `1->2` | -0.05333333 | 3 | 0 | 1 | 2 | [-0.16, 0] |
| `2->3` | -0.36000001 | 3 | 0 | 3 | 0 | [-0.36000001, -0.36000001] |
| `3->4` | 0.04 | 3 | 1 | 1 | 1 | [-0.09999999, 0.22] |
| `4->5` | 0.01333334 | 3 | 1 | 0 | 2 | [0, 0.04000001] |
| `5->6` | -0.06000002 | 3 | 0 | 3 | 0 | [-0.06000002, -0.06000002] |
| `6->7` | 0.00666667 | 3 | 1 | 0 | 2 | [0, 0.02000001] |
| `7->8` | 0.0 | 3 | 0 | 0 | 3 | [0, 0] |

## Reasons

- fresh-gap direction is mixed across seeds/stages; positive and non-positive outcomes cannot support a stable LoP direction

Retention/forgetting metrics are inventory only and are not substituted for fresh-gap.
