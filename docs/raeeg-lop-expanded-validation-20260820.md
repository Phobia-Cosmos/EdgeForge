# RA-EEG LoP Expanded Validation — 2026-08-20

## Scope

本轮在已迁移的 BrainUICL/RA-EEG 结果基础上，扩大 LoP probe 的 sequence cap 和评测预算，验证 EdgeForge 对更强单 seed 实验的重放、指标归一化和失败记录能力。它仍不是多 seed 科研结论。

## Configuration

- Dataset：ISRUC-Group-I
- Dataset manifest：`5344651092da22c1fa3dc068064e6c3c8f5ef9e87178fe8f77f2fb89d67d5346`
- Model/checkpoint seed：4321
- Available probe checkpoint stages：0/10/25；`individual_49` 是最终 checkpoint，不满足 probe 的“stage 后继 subject”约束
- Sequence cap：16
- Batch：4
- Probe steps：0/10/20
- Evaluation：`monitor-max-batches=0`、`eval-max-batches=0`，即完整 loader
- Protocol：`raeeg-lop-posthoc-v1`、supervised-oracle fixed-budget heldout

## First failed attempt (preserved)

任务 `6353124725024e7a8e4bed771e82a420` 使用了 `stages=0,10,25,49`，被 probe 正确拒绝：`stages must be in 0..48`。原因是 stage 49 没有下一 subject，不能作为 plasticity probe stage。该 failed task 保留在归档 SQLite 和日志中，说明配置校验生效。

## Successful run

任务 `270882c477e94d399abb6c28a1b3f591` 成功，EdgeForge 和 Worker 版本均为 `0.10.1`，运行时约 `3108.92 ms`。

| Stage | Subject | Plasticity ACC gain | Transformer effective rank |
| ---: | ---: | ---: | ---: |
| 0 | 64 | -0.01 | 26.6664 |
| 10 | 91 | +0.09 | 25.9615 |
| 25 | 10 | +0.23 | 13.1070 |

Summary：mean plasticity ACC gain `0.1033333333`；mean transformer effective rank `21.91160393`；rank 与 future plasticity 的 Pearson `-0.9281156074`。

与前一轮 cap=4 的 smoke（Pearson `0.5704744175`、mean gain `0.00`）不同，说明小样本/预算改变会显著改变观测相关性；这恰恰证明不能把任意一次 smoke 当作 LoP 机制结论。

## Multi-seed gate

本地真实 checkpoint 文件名只发现 seed `4321`，没有 seed 4322/4323 的对应 source checkpoint 和 CL milestone checkpoint。因此多 seed LoP 验证当前状态为 `blocked-by-checkpoint`，不是 PASS，也不是把 seed 4321 复制成其它 seed。下一步需要生成或接入真实 seed-specific BrainUICL checkpoint，再复用相同 EdgeForge manifest。

## Archive

归档目录：`logs/archive/v0.10.1/lop-expanded-validation-20260820/`。其中包含 failed/succeeded 两个任务回执、控制面/Worker 日志、SQLite、WAL checkpoint 后的 `edgeforge-backup.db` 和 Artifact。结果文件位于 BrainUICL 工作区的 `experiments/edgeforge_runs/lop-edgeforge-v0.10.1-expanded-seed4321/metrics.json`。
