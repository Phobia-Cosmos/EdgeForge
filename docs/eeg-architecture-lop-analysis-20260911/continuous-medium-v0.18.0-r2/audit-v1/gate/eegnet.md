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
| `1->2` | -0.02 | 3 | 0 | 1 | 2 | [-0.06, 0] |
| `2->3` | -0.21333334 | 3 | 0 | 3 | 0 | [-0.36000001, -0.04] |
| `3->4` | 0.05333332 | 3 | 2 | 1 | 0 | [-0.02000001, 0.09999999] |
| `4->5` | -0.09999999 | 3 | 1 | 2 | 0 | [-0.27999999, 0.02] |
| `5->6` | -0.00666668 | 3 | 1 | 2 | 0 | [-0.06000002, 0.09999999] |
| `6->7` | -0.03333333 | 3 | 1 | 2 | 0 | [-0.08000001, 0.02000001] |
| `7->8` | 0.0 | 3 | 0 | 0 | 3 | [0, 0] |

## Reasons

- fresh-gap direction is mixed across seeds/stages; positive and non-positive outcomes cannot support a stable LoP direction

Retention/forgetting metrics are inventory only and are not substituted for fresh-gap.
