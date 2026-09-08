# Full EEG LoP experiment analysis

- Status: `candidate-evidence-ready`
- Phase: `calibration`
- Candidate evidence ready: `True`
- Scientific conclusion allowed: `False`
- Valid cells: 1080/1080
- Bootstrap: `two-way-seed-subject-cluster-percentile`

Passing this report means candidate evidence is complete enough for review. It is not an automatic scientific conclusion.

## Audit gates

| Gate | Passed |
| --- | --- |
| Completeness | `True` |
| Design consistency | `True` |
| Source and fresh learning adequacy | `True` |
| Minimum seeds and target stages | `True` |
| Every expected cell valid | `True` |

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
| `lop_mlp` | 0 | 36 | 3 | 12 | -0.171685 | -0.182558 | 0.056 | [-0.222936, -0.113451] | n/a |
| `lop_mlp` | 5 | 36 | 3 | 12 | -0.0693486 | -0.0537523 | 0.111 | [-0.105098, -0.0348528] | 1 |
| `lop_mlp` | 10 | 36 | 3 | 12 | -0.0388149 | -0.0293381 | 0.278 | [-0.0712917, -0.0101703] | 1 |
| `lop_mlp` | 25 | 36 | 3 | 12 | -0.0611786 | -0.0576412 | 0.083 | [-0.0834285, -0.0385837] | 1 |
| `lop_mlp` | 50 | 36 | 3 | 12 | -0.0682032 | -0.0690868 | 0.028 | [-0.0877512, -0.0470881] | 1 |
| `lop_mlp` | 100 | 36 | 3 | 12 | -0.0749728 | -0.0717608 | 0.028 | [-0.0953653, -0.0548594] | 1 |
| `tcn` | 0 | 36 | 3 | 12 | -0.317495 | -0.352326 | 0.139 | [-0.452298, -0.168758] | n/a |
| `tcn` | 5 | 36 | 3 | 12 | -0.407342 | -0.400604 | 0.028 | [-0.507641, -0.302777] | 1 |
| `tcn` | 10 | 36 | 3 | 12 | -0.364917 | -0.376744 | 0.000 | [-0.428789, -0.298363] | 1 |
| `tcn` | 25 | 36 | 3 | 12 | -0.317589 | -0.302326 | 0.000 | [-0.366387, -0.270438] | 1 |
| `tcn` | 50 | 36 | 3 | 12 | -0.235032 | -0.227049 | 0.000 | [-0.296103, -0.182817] | 1 |
| `tcn` | 100 | 36 | 3 | 12 | -0.171592 | -0.127076 | 0.000 | [-0.249286, -0.107098] | 1 |
| `transformer` | 0 | 36 | 3 | 12 | -0.412694 | -0.377381 | 0.000 | [-0.496843, -0.338649] | n/a |
| `transformer` | 5 | 36 | 3 | 12 | -0.357664 | -0.367054 | 0.000 | [-0.404138, -0.307978] | 1 |
| `transformer` | 10 | 36 | 3 | 12 | -0.331036 | -0.328304 | 0.000 | [-0.360155, -0.301628] | 1 |
| `transformer` | 25 | 36 | 3 | 12 | -0.279783 | -0.284168 | 0.000 | [-0.313959, -0.242892] | 1 |
| `transformer` | 50 | 36 | 3 | 12 | -0.185435 | -0.178116 | 0.000 | [-0.219126, -0.152778] | 1 |
| `transformer` | 100 | 36 | 3 | 12 | -0.144118 | -0.127368 | 0.000 | [-0.180729, -0.106388] | 1 |

## Order effects

Differences match the same seed and target subject across orders. The slope interaction is `(left order stage slope) - (right order stage slope)`.

| Architecture | Budget | Contrast | Pairs | Mean difference | Two-way cluster 95% CI | Stage-slope interaction | Holm p |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |

## Audit issues

- `warning:fresh_learning_insufficient`: 9
- `warning:source_learning_insufficient`: 3
