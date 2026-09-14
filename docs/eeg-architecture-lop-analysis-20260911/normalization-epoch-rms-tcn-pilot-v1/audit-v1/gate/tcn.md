# Continuous EEG LoP gate

- Status: `blocked-inconsistent-direction`
- Outcome: `task.plasticity.fresh_gap`
- Seeds: 3/3
- Transitions: 8/2
- Design consistent: `True`
- Scientific conclusion allowed: `False`

| Transition | Mean | Seeds | Positive | Negative | Zero | 95% CI |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `0->1` | 0.04 | 3 | 2 | 1 | 0 | [-0.06, 0.09999999] |
| `1->2` | 0.05333333 | 3 | 2 | 1 | 0 | [-0.02000004, 0.16000003] |
| `2->3` | -0.11333333 | 3 | 1 | 2 | 0 | [-0.22, 0.01999998] |
| `3->4` | -0.12 | 3 | 0 | 3 | 0 | [-0.13999999, -0.10000002] |
| `4->5` | -0.14666669 | 3 | 0 | 3 | 0 | [-0.20000005, -0.12] |
| `5->6` | -0.11333333 | 3 | 0 | 3 | 0 | [-0.22, -0.02000001] |
| `6->7` | 0.02 | 3 | 2 | 1 | 0 | [-0.01999998, 0.06] |
| `7->8` | -0.04666666 | 3 | 1 | 2 | 0 | [-0.14000002, 0.02000001] |

## Reasons

- fresh-gap direction is mixed across seeds/stages; positive and non-positive outcomes cannot support a stable LoP direction

Retention/forgetting metrics are inventory only and are not substituted for fresh-gap.
