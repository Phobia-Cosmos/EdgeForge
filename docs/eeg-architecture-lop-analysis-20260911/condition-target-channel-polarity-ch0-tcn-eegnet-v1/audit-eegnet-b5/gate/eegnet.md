# Continuous EEG LoP gate

- Status: `blocked-inconsistent-direction`
- Outcome: `task.plasticity.fresh_gap`
- Seeds: 3/3
- Transitions: 8/2
- Design consistent: `True`
- Scientific conclusion allowed: `False`

| Transition | Mean | Seeds | Positive | Negative | Zero | 95% CI |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `0->1` | -0.13333334 | 3 | 0 | 2 | 1 | [-0.24000001, 0] |
| `1->2` | 0.04 | 3 | 1 | 1 | 1 | [-0.04000001, 0.16] |
| `2->3` | -0.14666667 | 3 | 0 | 3 | 0 | [-0.2, -0.08000001] |
| `3->4` | -0.11333333 | 3 | 0 | 1 | 2 | [-0.34, 0] |
| `4->5` | -0.01333333 | 3 | 1 | 2 | 0 | [-0.24, 0.22] |
| `5->6` | -0.18666667 | 3 | 0 | 3 | 0 | [-0.23999999, -0.16000001] |
| `6->7` | -0.02666667 | 3 | 1 | 2 | 0 | [-0.18000001, 0.18000001] |
| `7->8` | 0.05333333 | 3 | 2 | 0 | 1 | [0, 0.14] |

## Reasons

- fresh-gap direction is mixed across seeds/stages; positive and non-positive outcomes cannot support a stable LoP direction

Retention/forgetting metrics are inventory only and are not substituted for fresh-gap.
