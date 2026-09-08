# Full EEG LoP experiment analysis

- Status: `blocked`
- Phase: `calibration`
- Candidate evidence ready: `False`
- Scientific conclusion allowed: `False`
- Valid cells: 852/1080
- Bootstrap: `two-way-seed-subject-cluster-percentile`

Passing this report means candidate evidence is complete enough for review. It is not an automatic scientific conclusion.

## Audit gates

| Gate | Passed |
| --- | --- |
| Completeness | `True` |
| Design consistency | `True` |
| Source and fresh learning adequacy | `False` |
| Minimum seeds and target stages | `False` |
| Every expected cell valid | `False` |

## Architecture by budget

Only valid cells appear in this table.

| Architecture | Budget | n | Seeds | Subjects | Mean gap | Median gap | Positive fraction | Two-way cluster 95% CI | Holm p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| `brainuicl` | 0 | 36 | 3 | 12 | -0.260781 | -0.303516 | 0.139 | [-0.359718, -0.144936] | n/a |
| `brainuicl` | 5 | 36 | 3 | 12 | -0.340891 | -0.356798 | 0.028 | [-0.402112, -0.270116] | 1 |
| `brainuicl` | 10 | 36 | 3 | 12 | -0.32033 | -0.331755 | 0.000 | [-0.382353, -0.252042] | 1 |
| `brainuicl` | 25 | 36 | 3 | 12 | -0.227268 | -0.200028 | 0.000 | [-0.287822, -0.17117] | 1 |
| `brainuicl` | 50 | 36 | 3 | 12 | -0.13103 | -0.138174 | 0.056 | [-0.171735, -0.0735948] | 1 |
| `brainuicl` | 100 | 36 | 3 | 12 | -0.0668751 | -0.0653846 | 0.083 | [-0.0927773, -0.037934] | 1 |
| `eegnet` | 0 | 36 | 3 | 12 | -0.23998 | -0.263698 | 0.139 | [-0.375278, -0.0982521] | n/a |
| `eegnet` | 5 | 36 | 3 | 12 | -0.369725 | -0.411572 | 0.083 | [-0.481513, -0.261206] | 1 |
| `eegnet` | 10 | 36 | 3 | 12 | -0.349481 | -0.369919 | 0.000 | [-0.417188, -0.278205] | 1 |
| `eegnet` | 25 | 36 | 3 | 12 | -0.319615 | -0.353022 | 0.000 | [-0.374206, -0.257201] | 1 |
| `eegnet` | 50 | 36 | 3 | 12 | -0.211449 | -0.181502 | 0.000 | [-0.269039, -0.159259] | 1 |
| `eegnet` | 100 | 36 | 3 | 12 | -0.121174 | -0.110692 | 0.028 | [-0.1533, -0.0892205] | 1 |
| `tcn` | 0 | 34 | 3 | 12 | -0.340583 | -0.374061 | 0.088 | [-0.461132, -0.198196] | n/a |
| `tcn` | 5 | 34 | 3 | 12 | -0.40964 | -0.427173 | 0.029 | [-0.508596, -0.306759] | 1 |
| `tcn` | 10 | 34 | 3 | 12 | -0.361553 | -0.366935 | 0.000 | [-0.442371, -0.292778] | 1 |
| `tcn` | 25 | 34 | 3 | 12 | -0.308839 | -0.297674 | 0.000 | [-0.351787, -0.264556] | 1 |
| `tcn` | 50 | 34 | 3 | 12 | -0.22069 | -0.22381 | 0.000 | [-0.273047, -0.174914] | 1 |
| `tcn` | 100 | 34 | 3 | 12 | -0.153179 | -0.124419 | 0.000 | [-0.227726, -0.102671] | 1 |
| `transformer` | 0 | 36 | 3 | 12 | -0.412694 | -0.377381 | 0.000 | [-0.494911, -0.341254] | n/a |
| `transformer` | 5 | 36 | 3 | 12 | -0.357664 | -0.367054 | 0.000 | [-0.408309, -0.307604] | 1 |
| `transformer` | 10 | 36 | 3 | 12 | -0.331036 | -0.328304 | 0.000 | [-0.360816, -0.299258] | 1 |
| `transformer` | 25 | 36 | 3 | 12 | -0.279783 | -0.284168 | 0.000 | [-0.314771, -0.241953] | 1 |
| `transformer` | 50 | 36 | 3 | 12 | -0.185435 | -0.178116 | 0.000 | [-0.221501, -0.152689] | 1 |
| `transformer` | 100 | 36 | 3 | 12 | -0.144118 | -0.127368 | 0.000 | [-0.18201, -0.107016] | 1 |

## Order effects

Differences match the same seed and target subject across orders. The slope interaction is `(left order stage slope) - (right order stage slope)`.

| Architecture | Budget | Contrast | Pairs | Mean difference | Two-way cluster 95% CI | Stage-slope interaction | Holm p |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |

## Audit issues

- `fresh_learning_insufficient`: 9
- `source_learning_insufficient`: 3
