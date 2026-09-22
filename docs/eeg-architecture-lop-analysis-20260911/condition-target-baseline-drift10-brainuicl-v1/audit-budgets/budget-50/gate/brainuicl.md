# Continuous EEG LoP gate

- Status: `blocked-inconsistent-direction`
- Outcome: `task.plasticity.fresh_gap`
- Seeds: 3/3
- Transitions: 8/2
- Design consistent: `True`
- Scientific conclusion allowed: `False`

| Transition | Mean | Seeds | Positive | Negative | Zero | 95% CI |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `0->1` | 0.0 | 3 | 1 | 1 | 1 | [-0.02, 0.02000001] |
| `1->2` | -0.22666668 | 3 | 0 | 3 | 0 | [-0.34000003, -0.02000001] |
| `2->3` | 0.15333335 | 3 | 3 | 0 | 0 | [0.06, 0.20000002] |
| `3->4` | -0.22666667 | 3 | 0 | 3 | 0 | [-0.40000001, -0.03999999] |
| `4->5` | -0.38000001 | 3 | 0 | 3 | 0 | [-0.49999997, -0.16000001] |
| `5->6` | 0.14666668 | 3 | 3 | 0 | 0 | [0.04000002, 0.30000001] |
| `6->7` | -0.01333333 | 3 | 1 | 2 | 0 | [-0.03999999, 0.02000001] |
| `7->8` | -0.16666668 | 3 | 0 | 3 | 0 | [-0.22, -0.10000002] |

## Reasons

- fresh-gap direction is mixed across seeds/stages; positive and non-positive outcomes cannot support a stable LoP direction

Retention/forgetting metrics are inventory only and are not substituted for fresh-gap.
