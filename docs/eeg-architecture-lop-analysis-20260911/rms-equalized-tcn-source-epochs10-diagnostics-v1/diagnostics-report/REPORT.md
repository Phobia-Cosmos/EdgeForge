# EEG checkpoint diagnostics

These summaries join representation geometry, activation sparsity, gradient magnitude and parameter movement with the fixed-budget fresh-gap. They are descriptive mechanism diagnostics; the LoP gate remains the required outcome test.

| arm | budget | rows | embedding ER | embedding ER norm | block1 ER | gradient norm | gradient nonzero | parameter relative update | classifier near-zero |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fresh | 0 | 24 | 4.39139 | 0.137231 | 1.99371 | 1.71226 | 0.999248 | 0 | 0 |
| fresh | 5 | 24 | 5.5615 | 0.173797 | 1.99579 | 1.63835 | 0.999578 | 0.0209331 | 0 |
| fresh | 10 | 24 | 6.00145 | 0.187545 | 1.99854 | 1.59996 | 0.99977 | 0.037528 | 0 |
| fresh | 25 | 24 | 11.3691 | 0.355283 | 2.034 | 2.72875 | 0.999954 | 0.0745904 | 0 |
| fresh | 50 | 24 | 3.45867 | 0.108083 | 2.15714 | 121.942 | 0.999992 | 0.117081 | 0 |
| warm | 0 | 24 | 3.09069 | 0.0965839 | 2.19589 | 1421.66 | 0.595828 | 0 | 0 |
| warm | 5 | 24 | 2.90926 | 0.0909144 | 2.12236 | 1554.15 | 0.575459 | 0.0171378 | 0 |
| warm | 10 | 24 | 3.26948 | 0.102171 | 2.13698 | 814.051 | 0.58473 | 0.0277842 | 0 |
| warm | 25 | 24 | 3.08357 | 0.0963615 | 2.18896 | 1178.43 | 0.588299 | 0.0491681 | 0 |
| warm | 50 | 24 | 2.98936 | 0.0934175 | 2.21382 | 1467.01 | 0.595559 | 0.0746477 | 0 |

Warm initial embedding rank versus final-budget fresh-gap Pearson: `-0.111604` over 24 stage-seed pairs.

A rank, gradient or activation trend is not sufficient to label LoP; it must be paired with a stable multi-seed, multi-transition fresh-gap direction and retention checks.
