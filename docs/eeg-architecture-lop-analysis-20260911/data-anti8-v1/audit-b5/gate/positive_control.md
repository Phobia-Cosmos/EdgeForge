# Continuous EEG LoP gate

- Status: `blocked-inconsistent-direction`
- Outcome: `task.plasticity.fresh_gap`
- Seeds: 3/3
- Transitions: 8/2
- Design consistent: `True`
- Scientific conclusion allowed: `False`

| Transition | Mean | Seeds | Positive | Negative | Zero | 95% CI |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `0->1` | 0.45572917 | 3 | 3 | 0 | 0 | [0.015625, 0.8046875] |
| `1->2` | 0.25260417 | 3 | 3 | 0 | 0 | [0.0390625, 0.390625] |
| `2->3` | -0.29427083 | 3 | 1 | 2 | 0 | [-0.765625, 0.1328125] |
| `3->4` | 0.52604167 | 3 | 2 | 1 | 0 | [-0.0234375, 0.8203125] |
| `4->5` | -0.13802083 | 3 | 1 | 2 | 0 | [-0.359375, 0.1171875] |
| `5->6` | -0.02083333 | 3 | 2 | 1 | 0 | [-0.390625, 0.2109375] |
| `6->7` | -0.25 | 3 | 0 | 3 | 0 | [-0.484375, -0.0625] |
| `7->8` | -0.17708333 | 3 | 1 | 2 | 0 | [-0.7734375, 0.25] |

## Reasons

- fresh-gap direction is mixed across seeds/stages; positive and non-positive outcomes cannot support a stable LoP direction

Retention/forgetting metrics are inventory only and are not substituted for fresh-gap.
