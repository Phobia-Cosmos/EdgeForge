# EEG adaptation strategy LoP comparison

Matched `tcn` comparison. Positive fresh-gap is the LoP direction; a positive mean is insufficient when any seed/transition cell is zero or negative.

| strategy | budget | fresh-gap mean | + / 0 / - | warm accuracy | warm retention | gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| plain | 25 | +0.0208 | 15 / 2 / 7 | 0.2300 | 0.2268 | `blocked-inconsistent-direction` |
| plain | 50 | -0.0075 | 11 / 6 / 7 | 0.2442 | 0.2194 | `blocked-inconsistent-direction` |
| source_replay | 25 | +0.0992 | 15 / 4 / 5 | 0.2208 | 0.2192 | `blocked-inconsistent-direction` |
| source_replay | 50 | -0.0192 | 3 / 13 / 8 | 0.2050 | 0.2080 | `blocked-inconsistent-direction` |
| l2_sp | 25 | +0.0592 | 14 / 3 / 7 | 0.2108 | 0.2220 | `blocked-inconsistent-direction` |
| l2_sp | 50 | +0.0367 | 15 / 4 / 5 | 0.2000 | 0.2025 | `blocked-inconsistent-direction` |
| replay_l2_sp | 25 | +0.0508 | 12 / 8 / 4 | 0.2558 | 0.2293 | `blocked-inconsistent-direction` |
| replay_l2_sp | 50 | -0.0375 | 7 / 5 / 12 | 0.2225 | 0.2183 | `blocked-inconsistent-direction` |

| strategy | budget | fresh-gap Δ vs plain | improved / worsened cells | warm-retention Δ |
| --- | ---: | ---: | ---: | ---: |
| source_replay | 25 | +0.0783 | 12 / 9 | -0.0076 |
| source_replay | 50 | -0.0117 | 4 / 13 | -0.0114 |
| l2_sp | 25 | +0.0383 | 9 / 6 | -0.0048 |
| l2_sp | 50 | +0.0442 | 13 / 7 | -0.0169 |
| replay_l2_sp | 25 | +0.0300 | 9 / 11 | +0.0025 |
| replay_l2_sp | 50 | -0.0300 | 7 / 13 | -0.0010 |

Retention is reported as a separate stability inventory and cannot satisfy the LoP gate. All conclusions remain developmental (`scientific_conclusion_allowed=false`).
