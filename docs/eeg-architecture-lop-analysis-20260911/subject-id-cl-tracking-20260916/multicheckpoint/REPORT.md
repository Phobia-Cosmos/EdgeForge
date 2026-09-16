# ISRUC continual-learning subject-ID checkpoint tracking (2026-09-16)

The checkpoint source objective remains sleep-stage prediction; subject ID is an external audit target. The 2026-09-15 rows use the open-set robustness protocol, while the 2026-09-16 rows use the all-subject closed-set probe. They must not be compared as if they were the same split or decoder; use the within-protocol trajectory for checkpoint comparisons.

### 80-percent known subjects + 20-percent unknown subjects; group-disjoint known test

| checkpoint | EEG branch | Transformer L1 | Transformer L2 | Transformer L3 | classifier input | logits |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ISRUC-individual_1 | 0.7645 | 0.5638 | 0.5267 | 0.4988 | 0.3155 | 0.0806 |
| ISRUC-individual_38 | 0.7715 | 0.5615 | 0.5249 | 0.5058 | 0.3306 | 0.0829 |

### all subjects enrolled; group-disjoint closed-set test

| checkpoint | EEG branch | Transformer L1 | Transformer L2 | Transformer L3 | classifier input | logits |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ISRUC-individual_11 | 0.7195 | 0.5143 | 0.4621 | 0.4344 | 0.2791 | 0.0457 |
| ISRUC-individual_24 | 0.7093 | 0.5125 | 0.4695 | 0.4381 | 0.2689 | 0.0425 |
| ISRUC-individual_48 | 0.6918 | 0.4806 | 0.4182 | 0.4039 | 0.2560 | 0.0527 |


The table measures representation decodability, not biometric uniqueness. An increase or decrease across checkpoints is not by itself LoP; it must be paired with task accuracy, fresh-subject adaptation, retention, effective rank/spectrum, and gradient-coverage measurements.

The ISRUC dataset currently exposes processed subject/group files but no explicit recording/session identifier in the canonical copy. Therefore this remains group-disjoint rather than session-disjoint; a session-level rerun is required before making biological invariance claims.
