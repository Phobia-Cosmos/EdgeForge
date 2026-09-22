# Continuous EEG LoP gate

- Status: `blocked-inconsistent-direction`
- Outcome: `task.plasticity.fresh_gap`
- Seeds: 3/3
- Transitions: 8/2
- Design consistent: `True`
- Scientific conclusion allowed: `False`

| Transition | Mean | Seeds | Positive | Negative | Zero | 95% CI |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `0->1` | 0.10000001 | 3 | 1 | 2 | 0 | [-0.02, 0.34000001] |
| `1->2` | -0.21999999 | 3 | 0 | 3 | 0 | [-0.31999999, -0.06] |
| `2->3` | 0.22000002 | 3 | 3 | 0 | 0 | [0.10000002, 0.36000001] |
| `3->4` | -0.24666668 | 3 | 0 | 3 | 0 | [-0.38000003, -0.12] |
| `4->5` | -0.21333334 | 3 | 0 | 3 | 0 | [-0.32000001, -0.12] |
| `5->6` | 0.07333334 | 3 | 2 | 0 | 1 | [0, 0.16000001] |
| `6->7` | 0.01333334 | 3 | 2 | 1 | 0 | [-0.1, 0.12] |
| `7->8` | -0.06666668 | 3 | 0 | 3 | 0 | [-0.10000002, -0.02000001] |

## Reasons

- fresh-gap direction is mixed across seeds/stages; positive and non-positive outcomes cannot support a stable LoP direction

Retention/forgetting metrics are inventory only and are not substituted for fresh-gap.
