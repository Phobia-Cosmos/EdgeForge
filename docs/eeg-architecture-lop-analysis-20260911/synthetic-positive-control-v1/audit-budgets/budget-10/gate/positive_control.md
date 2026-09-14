# Continuous EEG LoP gate

- Status: `blocked-inconsistent-direction`
- Outcome: `task.plasticity.fresh_gap`
- Seeds: 3/3
- Transitions: 8/2
- Design consistent: `True`
- Scientific conclusion allowed: `False`

| Transition | Mean | Seeds | Positive | Negative | Zero | 95% CI |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `0->1` | 0.1875 | 3 | 3 | 0 | 0 | [0.16927083, 0.2125651] |
| `1->2` | 0.0078125 | 3 | 2 | 1 | 0 | [-0.06985677, 0.09375] |
| `2->3` | 0.1171875 | 3 | 3 | 0 | 0 | [0.03125, 0.18951823] |
| `3->4` | 0.140625 | 3 | 3 | 0 | 0 | [0.06770833, 0.28125] |
| `4->5` | 0.11458333 | 3 | 2 | 1 | 0 | [0.0234375, 0.265625] |
| `5->6` | 0.08854167 | 3 | 3 | 0 | 0 | [0.04772135, 0.1328125] |
| `6->7` | 0.0625 | 3 | 2 | 1 | 0 | [-0.015625, 0.1171875] |
| `7->8` | 0.00260417 | 3 | 1 | 1 | 1 | [-0.02083333, 0.0328776] |

## Reasons

- fresh-gap direction is mixed across seeds/stages; positive and non-positive outcomes cannot support a stable LoP direction

Retention/forgetting metrics are inventory only and are not substituted for fresh-gap.
