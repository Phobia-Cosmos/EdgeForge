# Continuous EEG LoP gate

- Status: `blocked-inconsistent-direction`
- Outcome: `task.plasticity.fresh_gap`
- Seeds: 3/3
- Transitions: 8/2
- Design consistent: `True`
- Scientific conclusion allowed: `False`

| Transition | Mean | Seeds | Positive | Negative | Zero | 95% CI |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `0->1` | -0.13333334 | 3 | 0 | 1 | 2 | [-0.40000001, 0] |
| `1->2` | -0.04666667 | 3 | 1 | 1 | 1 | [-0.3, 0.16] |
| `2->3` | -0.31333334 | 3 | 0 | 3 | 0 | [-0.36000001, -0.22] |
| `3->4` | 0.07333334 | 3 | 2 | 1 | 0 | [-0.23999999, 0.34] |
| `4->5` | 0.06 | 3 | 2 | 1 | 0 | [-0.08, 0.24] |
| `5->6` | -0.00666667 | 3 | 1 | 1 | 1 | [-0.16000001, 0.14] |
| `6->7` | -0.03999999 | 3 | 1 | 2 | 0 | [-0.09999999, 0.02000001] |
| `7->8` | 0.1 | 3 | 2 | 0 | 1 | [0, 0.16] |

## Reasons

- fresh-gap direction is mixed across seeds/stages; positive and non-positive outcomes cannot support a stable LoP direction

Retention/forgetting metrics are inventory only and are not substituted for fresh-gap.
