# Continuous EEG LoP gate

- Status: `blocked-inconsistent-direction`
- Outcome: `task.plasticity.fresh_gap`
- Seeds: 3/3
- Transitions: 8/2
- Design consistent: `True`
- Scientific conclusion allowed: `False`

| Transition | Mean | Seeds | Positive | Negative | Zero | 95% CI |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `0->1` | 0.17333333 | 3 | 3 | 0 | 0 | [0.07999998, 0.22000003] |
| `1->2` | 0.05333332 | 3 | 3 | 0 | 0 | [0.02000001, 0.07999998] |
| `2->3` | -0.09333333 | 3 | 1 | 2 | 0 | [-0.22000003, 0.10000002] |
| `3->4` | -0.01333334 | 3 | 1 | 2 | 0 | [-0.12, 0.18000001] |
| `4->5` | -0.04666667 | 3 | 1 | 2 | 0 | [-0.13999999, 0.01999998] |
| `5->6` | -0.11333333 | 3 | 0 | 3 | 0 | [-0.28000003, -0.01999998] |
| `6->7` | -0.02 | 3 | 1 | 2 | 0 | [-0.06, 0.02000001] |
| `7->8` | -0.01333333 | 3 | 1 | 1 | 1 | [-0.13999999, 0.09999999] |

## Reasons

- fresh-gap direction is mixed across seeds/stages; positive and non-positive outcomes cannot support a stable LoP direction

Retention/forgetting metrics are inventory only and are not substituted for fresh-gap.
