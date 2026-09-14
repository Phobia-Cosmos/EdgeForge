# EEG architecture LoP comparison

Matched comparison with `tcn` as the paired baseline. Positive fresh-gap is the LoP direction; aggregate means do not override mixed-direction cells.

| architecture | budget | fresh-gap mean | + / 0 / - | warm accuracy | warm retention | gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| tcn | 5 | +0.0720 | 35 / 8 / 17 | 0.1827 | 0.1835 | `blocked-inconsistent-direction` |
| tcn | 10 | +0.0877 | 34 / 10 / 16 | 0.1994 | 0.1964 | `blocked-inconsistent-direction` |
| tcn | 25 | +0.1159 | 42 / 4 / 14 | 0.1914 | 0.1837 | `blocked-inconsistent-direction` |
| tcn | 50 | +0.0655 | 33 / 6 / 21 | 0.2097 | 0.1997 | `blocked-inconsistent-direction` |
| brainuicl | 5 | -0.2061 | 7 / 0 / 53 | 0.5426 | 0.5474 | `blocked-inconsistent-direction` |
| brainuicl | 10 | -0.1949 | 9 / 0 / 51 | 0.5380 | 0.5241 | `blocked-inconsistent-direction` |
| brainuicl | 25 | -0.3312 | 3 / 0 / 57 | 0.6668 | 0.6202 | `blocked-inconsistent-direction` |
| brainuicl | 50 | -0.4088 | 1 / 0 / 59 | 0.7164 | 0.6573 | `blocked-inconsistent-direction` |

| architecture | budget | fresh-gap delta vs tcn | improved / worsened cells | warm-retention delta |
| --- | ---: | ---: | ---: | ---: |
| brainuicl | 5 | -0.2781 | 3 / 56 | +0.3639 |
| brainuicl | 10 | -0.2826 | 4 / 56 | +0.3278 |
| brainuicl | 25 | -0.4471 | 1 / 59 | +0.4365 |
| brainuicl | 50 | -0.4743 | 1 / 59 | +0.4576 |

Retention is a separate stability inventory and cannot satisfy the LoP gate. All conclusions remain developmental (`scientific_conclusion_allowed=false`).
