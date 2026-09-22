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
| `1->2` | 0.05333332 | 3 | 2 | 1 | 0 | [-0.16, 0.23999998] |
| `2->3` | 0.20000001 | 3 | 3 | 0 | 0 | [0.2, 0.20000002] |
| `3->4` | 0.25333333 | 3 | 3 | 0 | 0 | [0.09999999, 0.44] |
| `4->5` | -0.01333333 | 3 | 0 | 2 | 1 | [-0.02, 0] |
| `5->6` | 0.06666666 | 3 | 2 | 1 | 0 | [-0.06000002, 0.16000001] |
| `6->7` | 0.08666666 | 3 | 2 | 1 | 0 | [-0.02000001, 0.25999999] |
| `7->8` | 0.14666666 | 3 | 3 | 0 | 0 | [0.03999999, 0.23999999] |

## Reasons

- fresh-gap direction is mixed across seeds/stages; positive and non-positive outcomes cannot support a stable LoP direction

Retention/forgetting metrics are inventory only and are not substituted for fresh-gap.
