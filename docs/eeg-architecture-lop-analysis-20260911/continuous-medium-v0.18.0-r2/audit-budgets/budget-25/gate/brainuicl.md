# Continuous EEG LoP gate

- Status: `blocked-inconsistent-direction`
- Outcome: `task.plasticity.fresh_gap`
- Seeds: 3/3
- Transitions: 8/2
- Design consistent: `True`
- Scientific conclusion allowed: `False`

| Transition | Mean | Seeds | Positive | Negative | Zero | 95% CI |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `0->1` | 0.17333334 | 3 | 2 | 1 | 0 | [-0.14, 0.40000001] |
| `1->2` | -0.05333334 | 3 | 1 | 2 | 0 | [-0.14000002, 0.02] |
| `2->3` | 0.30000002 | 3 | 3 | 0 | 0 | [0.20000002, 0.36000001] |
| `3->4` | -0.12666668 | 3 | 1 | 2 | 0 | [-0.24000001, 0.09999999] |
| `4->5` | -0.33333335 | 3 | 0 | 3 | 0 | [-0.36000003, -0.30000003] |
| `5->6` | -0.23333333 | 3 | 0 | 3 | 0 | [-0.31999999, -0.14] |
| `6->7` | 0.07333333 | 3 | 3 | 0 | 0 | [0.02000001, 0.09999999] |
| `7->8` | -0.06000001 | 3 | 2 | 1 | 0 | [-0.24000001, 0.03999999] |

## Reasons

- fresh-gap direction is mixed across seeds/stages; positive and non-positive outcomes cannot support a stable LoP direction

Retention/forgetting metrics are inventory only and are not substituted for fresh-gap.
