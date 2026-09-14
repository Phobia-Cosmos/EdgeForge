# Continuous EEG LoP gate

- Status: `blocked-inconsistent-direction`
- Outcome: `task.plasticity.fresh_gap`
- Seeds: 3/3
- Transitions: 8/2
- Design consistent: `True`
- Scientific conclusion allowed: `False`

| Transition | Mean | Seeds | Positive | Negative | Zero | 95% CI |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `0->1` | -0.1 | 3 | 0 | 2 | 1 | [-0.16, 0] |
| `1->2` | 0.07999999 | 3 | 1 | 0 | 2 | [0, 0.23999998] |
| `2->3` | -0.10666666 | 3 | 0 | 2 | 1 | [-0.16, 0] |
| `3->4` | 0.0 | 3 | 0 | 0 | 3 | [0, 0] |
| `4->5` | -0.00666667 | 3 | 0 | 1 | 2 | [-0.02, 0] |
| `5->6` | -0.01333333 | 3 | 1 | 1 | 1 | [-0.17999999, 0.14] |
| `6->7` | -0.06 | 3 | 0 | 1 | 2 | [-0.18000001, 0] |
| `7->8` | 0.05333333 | 3 | 1 | 1 | 1 | [-0.22, 0.38] |

## Reasons

- fresh-gap direction is mixed across seeds/stages; positive and non-positive outcomes cannot support a stable LoP direction

Retention/forgetting metrics are inventory only and are not substituted for fresh-gap.
