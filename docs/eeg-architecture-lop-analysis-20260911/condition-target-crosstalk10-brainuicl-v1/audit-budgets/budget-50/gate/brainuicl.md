# Continuous EEG LoP gate

- Status: `blocked-inconsistent-direction`
- Outcome: `task.plasticity.fresh_gap`
- Seeds: 3/3
- Transitions: 8/2
- Design consistent: `True`
- Scientific conclusion allowed: `False`

| Transition | Mean | Seeds | Positive | Negative | Zero | 95% CI |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `0->1` | 0.18666667 | 3 | 2 | 1 | 0 | [-0.02, 0.34000001] |
| `1->2` | -0.13333335 | 3 | 0 | 3 | 0 | [-0.20000002, -0.06] |
| `2->3` | 0.16000002 | 3 | 3 | 0 | 0 | [0.14000002, 0.20000002] |
| `3->4` | -0.14000001 | 3 | 1 | 2 | 0 | [-0.58000001, 0.18] |
| `4->5` | -0.21333334 | 3 | 0 | 3 | 0 | [-0.36000003, -0.03999999] |
| `5->6` | 0.04666668 | 3 | 3 | 0 | 0 | [0.02000001, 0.10000001] |
| `6->7` | 0.00666666 | 3 | 2 | 1 | 0 | [-0.18000001, 0.12] |
| `7->8` | -0.08 | 3 | 1 | 2 | 0 | [-0.25999999, 0.13999999] |

## Reasons

- fresh-gap direction is mixed across seeds/stages; positive and non-positive outcomes cannot support a stable LoP direction

Retention/forgetting metrics are inventory only and are not substituted for fresh-gap.
