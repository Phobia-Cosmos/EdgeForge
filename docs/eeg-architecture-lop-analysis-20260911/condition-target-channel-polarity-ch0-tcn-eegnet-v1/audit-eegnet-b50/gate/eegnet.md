# Continuous EEG LoP gate

- Status: `blocked-inconsistent-direction`
- Outcome: `task.plasticity.fresh_gap`
- Seeds: 3/3
- Transitions: 8/2
- Design consistent: `True`
- Scientific conclusion allowed: `False`

| Transition | Mean | Seeds | Positive | Negative | Zero | 95% CI |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `0->1` | 0.16000001 | 3 | 2 | 0 | 1 | [0, 0.24000001] |
| `1->2` | 0.00666666 | 3 | 1 | 1 | 1 | [-0.14000002, 0.16] |
| `2->3` | -0.12666667 | 3 | 0 | 2 | 1 | [-0.36000001, 0] |
| `3->4` | -0.03999999 | 3 | 1 | 2 | 0 | [-0.07999998, 0.02000001] |
| `4->5` | -0.18666667 | 3 | 1 | 2 | 0 | [-0.33999999, 0.05999999] |
| `5->6` | 0.01333333 | 3 | 1 | 1 | 1 | [-0.06000002, 0.09999999] |
| `6->7` | 0.0 | 3 | 0 | 0 | 3 | [0, 0] |
| `7->8` | 0.11999999 | 3 | 2 | 0 | 1 | [0, 0.19999999] |

## Reasons

- fresh-gap direction is mixed across seeds/stages; positive and non-positive outcomes cannot support a stable LoP direction

Retention/forgetting metrics are inventory only and are not substituted for fresh-gap.
