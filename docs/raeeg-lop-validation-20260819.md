# RA-EEG LoP Pipeline Validation — 2026-08-19

## Scope

本次验证的目标是确认：BrainUICL 历史结果迁移后，EdgeForge 能否从同一套数据集/模型/checkpoint 契约重放一个多阶段 LoP probe，并保存研究指标、source digest、Experiment Bundle、SQLite 任务记录与版本化日志。它不是一次完整的 LoP 科研实验。

## Inputs

- Dataset：ISRUC-Group-I
- Dataset manifest digest：`5344651092da22c1fa3dc068064e6c3c8f5ef9e87178fe8f77f2fb89d67d5346`
- Model：BrainUICL，seed 4321，checkpoint stages 0/10/25
- Protocol：`raeeg-lop-posthoc-v1`
- Probe：supervised-oracle fixed-budget heldout
- Worker：`worker-4070s-lop`，x86_64，RTX 4070 SUPER，EdgeForge `0.10.1`
- Config：`config/raeeg-lop-edgeforge-v0.10.1.json`

## Execution evidence

- Task：`b3598e30f1db4ea3b22f4527f6e400cd`
- Status：`succeeded`
- Exit code：`0`
- Elapsed：约 `2529.65 ms`
- Experiment Bundle digest：`24ff34db216123baeff947cadd86fd989201bedb870dc64e86f34c97116d991a`
- Source result digest：`6314253d4076854aa5e42ea45b70786e9922595acabbe3475426cd27e675c7c4`
- Persisted model experiments：9（8 curated imports + 1 LoP replay）
- Persisted Experiment Metrics：8396
- Persisted Artifacts：9

## Observed smoke metrics

| Stage | Subject | Plasticity ACC gain | Transformer effective rank |
| ---: | ---: | ---: | ---: |
| 0 | 64 | +0.05 | 10.6028 |
| 10 | 91 | -0.05 | 8.9730 |
| 25 | 10 | 0.00 | 7.7558 |

Summary mean plasticity ACC gain is `0.00`; Transformer effective-rank versus future plasticity smoke Pearson is `0.5704744175`. These values are descriptive for this three-stage, one-seed smoke only. No causal or population-level LoP conclusion is allowed.

## Archive

The persistent archive is `logs/archive/v0.10.1/lop-validation-20260819/`. It includes control/Worker JSONL logs, the SQLite database, a WAL-checkpointed `edgeforge-backup.db`, task receipts and the Artifact store. The raw EEG dataset and source repository remain outside EdgeForge.

## Next validation gate

Before discussing LoP as a research result, run the same contract with at least three seeds, a pre-registered stage/subject selection, full evaluation budget, a non-adaptive baseline, and confidence intervals. Only then connect effective/stable rank to plasticity through a versioned analysis artifact; the current smoke must not be promoted by Capability Gate.
