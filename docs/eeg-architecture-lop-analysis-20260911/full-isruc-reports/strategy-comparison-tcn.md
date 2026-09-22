# EEG adaptation strategy LoP comparison

Matched `tcn` comparison. Positive fresh-gap is the LoP direction; a positive mean is insufficient when any seed/transition cell is zero or negative.

| strategy | budget | fresh-gap mean | + / 0 / - | warm accuracy | warm retention | gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| plain | 5 | +0.0720 | 35 / 8 / 17 | 0.1827 | 0.1835 | `blocked-inconsistent-direction` |
| plain | 10 | +0.0877 | 34 / 10 / 16 | 0.1994 | 0.1964 | `blocked-inconsistent-direction` |
| plain | 25 | +0.1159 | 42 / 4 / 14 | 0.1914 | 0.1837 | `blocked-inconsistent-direction` |
| plain | 50 | +0.0655 | 33 / 6 / 21 | 0.2097 | 0.1997 | `blocked-inconsistent-direction` |
| source_replay | 5 | +0.0686 | 32 / 13 / 15 | 0.1837 | 0.1997 | `blocked-inconsistent-direction` |
| source_replay | 10 | +0.1105 | 39 / 6 / 15 | 0.1837 | 0.1927 | `blocked-inconsistent-direction` |
| source_replay | 25 | +0.1142 | 40 / 5 / 15 | 0.2075 | 0.1996 | `blocked-inconsistent-direction` |
| source_replay | 50 | +0.0360 | 28 / 19 / 13 | 0.2069 | 0.2019 | `blocked-inconsistent-direction` |
| l2_sp | 5 | +0.0779 | 38 / 7 / 15 | 0.1768 | 0.1888 | `blocked-inconsistent-direction` |
| l2_sp | 10 | +0.0937 | 38 / 10 / 12 | 0.1934 | 0.1959 | `blocked-inconsistent-direction` |
| l2_sp | 25 | +0.1185 | 38 / 11 / 11 | 0.1890 | 0.1937 | `blocked-inconsistent-direction` |
| l2_sp | 50 | +0.0666 | 34 / 5 / 21 | 0.2075 | 0.2051 | `blocked-inconsistent-direction` |
| replay_l2_sp | 5 | +0.0459 | 28 / 16 / 16 | 0.2061 | 0.2094 | `blocked-inconsistent-direction` |
| replay_l2_sp | 10 | +0.1056 | 36 / 9 / 15 | 0.1911 | 0.1995 | `blocked-inconsistent-direction` |
| replay_l2_sp | 25 | +0.1256 | 41 / 10 / 9 | 0.1974 | 0.2064 | `blocked-inconsistent-direction` |
| replay_l2_sp | 50 | +0.0699 | 30 / 27 / 3 | 0.1716 | 0.1900 | `blocked-inconsistent-direction` |

| strategy | budget | fresh-gap Δ vs plain | improved / worsened cells | warm-retention Δ |
| --- | ---: | ---: | ---: | ---: |
| source_replay | 5 | -0.0034 | 21 / 22 | +0.0162 |
| source_replay | 10 | +0.0228 | 25 / 22 | -0.0036 |
| source_replay | 25 | -0.0017 | 23 / 24 | +0.0159 |
| source_replay | 50 | -0.0295 | 28 / 30 | +0.0023 |
| l2_sp | 5 | +0.0059 | 23 / 20 | +0.0053 |
| l2_sp | 10 | +0.0060 | 22 / 21 | -0.0004 |
| l2_sp | 25 | +0.0027 | 19 / 21 | +0.0100 |
| l2_sp | 50 | +0.0011 | 25 / 22 | +0.0055 |
| replay_l2_sp | 5 | -0.0260 | 16 / 28 | +0.0259 |
| replay_l2_sp | 10 | +0.0179 | 23 / 24 | +0.0031 |
| replay_l2_sp | 25 | +0.0098 | 25 / 18 | +0.0226 |
| replay_l2_sp | 50 | +0.0044 | 32 / 27 | -0.0097 |

Retention is reported as a separate stability inventory and cannot satisfy the LoP gate. All conclusions remain developmental (`scientific_conclusion_allowed=false`).
