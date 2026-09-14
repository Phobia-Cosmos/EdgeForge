# Continuous EEG LoP gate

- Status: `blocked-inconsistent-direction`
- Outcome: `task.plasticity.fresh_gap`
- Seeds: 3/3
- Transitions: 8/2
- Design consistent: `True`
- Scientific conclusion allowed: `False`

| Transition | Mean | Seeds | Positive | Negative | Zero | 95% CI |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `0->1` | -0.02 | 3 | 2 | 1 | 0 | [-0.24000001, 0.16] |
| `1->2` | -0.05333333 | 3 | 1 | 1 | 1 | [-0.23999998, 0.07999998] |
| `2->3` | 0.0 | 3 | 1 | 1 | 1 | [-0.16, 0.16] |
| `3->4` | 0.09999999 | 3 | 3 | 0 | 0 | [0.09999999, 0.09999999] |
| `4->5` | -0.01333333 | 3 | 2 | 1 | 0 | [-0.09999999, 0.04000001] |
| `5->6` | 0.16 | 3 | 2 | 0 | 1 | [0, 0.23999999] |
| `6->7` | 0.08666665 | 3 | 3 | 0 | 0 | [0.07999998, 0.09999999] |
| `7->8` | 0.03333333 | 3 | 1 | 1 | 1 | [-0.14, 0.23999999] |

## Reasons

- fresh-gap direction is mixed across seeds/stages; positive and non-positive outcomes cannot support a stable LoP direction

Retention/forgetting metrics are inventory only and are not substituted for fresh-gap.
