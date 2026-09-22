# Continuous EEG LoP gate

- Status: `blocked-inconsistent-direction`
- Outcome: `task.plasticity.fresh_gap`
- Seeds: 3/3
- Transitions: 8/2
- Design consistent: `True`
- Scientific conclusion allowed: `False`

| Transition | Mean | Seeds | Positive | Negative | Zero | 95% CI |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `0->1` | 0.24666667 | 3 | 2 | 0 | 1 | [0, 0.40000001] |
| `1->2` | -0.02000001 | 3 | 1 | 2 | 0 | [-0.06, 0.02] |
| `2->3` | 0.18666668 | 3 | 3 | 0 | 0 | [0.02000001, 0.34000001] |
| `3->4` | 0.06000001 | 3 | 2 | 1 | 0 | [-0.25999999, 0.42] |
| `4->5` | -0.33333335 | 3 | 0 | 3 | 0 | [-0.44000003, -0.26000001] |
| `5->6` | -0.10666666 | 3 | 1 | 2 | 0 | [-0.23999999, 0.16000001] |
| `6->7` | 0.09333334 | 3 | 3 | 0 | 0 | [0.02000001, 0.24] |
| `7->8` | -0.06000001 | 3 | 1 | 2 | 0 | [-0.22, 0.23999999] |

## Reasons

- fresh-gap direction is mixed across seeds/stages; positive and non-positive outcomes cannot support a stable LoP direction

Retention/forgetting metrics are inventory only and are not substituted for fresh-gap.
