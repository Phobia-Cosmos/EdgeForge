# Continuous EEG LoP gate

- Status: `blocked-inconsistent-direction`
- Outcome: `task.plasticity.fresh_gap`
- Seeds: 3/3
- Transitions: 8/2
- Design consistent: `True`
- Scientific conclusion allowed: `False`

| Transition | Mean | Seeds | Positive | Negative | Zero | 95% CI |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `0->1` | 0.02 | 3 | 1 | 2 | 0 | [-0.20000002, 0.36000001] |
| `1->2` | -0.00666668 | 3 | 2 | 1 | 0 | [-0.14000002, 0.07999998] |
| `2->3` | 0.13333335 | 3 | 2 | 0 | 1 | [0, 0.20000002] |
| `3->4` | 0.23999999 | 3 | 2 | 1 | 0 | [-0.16000003, 0.44] |
| `4->5` | -0.00666667 | 3 | 1 | 1 | 1 | [-0.03999999, 0.02] |
| `5->6` | 0.11333334 | 3 | 2 | 1 | 0 | [-0.06, 0.24000001] |
| `6->7` | 0.08 | 3 | 1 | 2 | 0 | [-0.02000001, 0.28] |
| `7->8` | -0.03333335 | 3 | 1 | 2 | 0 | [-0.10000002, 0.02] |

## Reasons

- fresh-gap direction is mixed across seeds/stages; positive and non-positive outcomes cannot support a stable LoP direction

Retention/forgetting metrics are inventory only and are not substituted for fresh-gap.
