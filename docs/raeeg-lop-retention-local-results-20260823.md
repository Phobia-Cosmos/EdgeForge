# 0.15.0 本地 retention 对照结果

日期：2026-08-23。本页记录固定预算 probe 的旧任务保持诊断，不把单 cell 数值升级为 LoP、BWT 或方法结论。原始 EEG、BrainUICL 源码和 checkpoint 仍保留在共享存储，EdgeForge 只保存派生证据和 digest。

## Protocol

目标集：ISRUC subject 2，seed 4321，pretrain checkpoint，target max-files=2，训练/held-out 划分为 1/1 个 sequence，固定预算 steps `0,2`。Retention 集：ISRUC subject 1，max-files=1；它只在 step 0 和每个 probe checkpoint 上评估，未参与 optimizer loss、梯度或参数更新。Checkpoint 与 fresh control 都使用显式 pretrain checkpoint，以保持两条曲线的状态来源一致。

## Observed smoke values

| quantity | checkpoint | fresh control |
| --- | ---: | ---: |
| retention ACC initial | 0.25 | 0.25 |
| retention ACC final | 0.15 | 0.15 |
| retention ACC drop | 0.10 | 0.10 |
| retention MF1 drop | 0.0254545 | 0.0254545 |
| retention loss delta | 1.96686 | 1.79018 |

这些数值只说明 retention 路径能够在新任务更新前后产生可审计的 old-task stability 读数；它们不代表完整任务流中的 forgetting/BWT，也不能替代 `plasticity.acc_gain` 或 fresh gap。由于只有一个 stage 和一个 seed，matrix audit 保持 `blocked-incomplete-evidence`/`scientific_conclusion_allowed=false`。

## Bundle contract

真实 matrix smoke 的 bundle 同时包含 `metric_role=predictor`（effective rank）、`metric_role=outcome`（plasticity/fresh gap）和 `metric_role=retention`（`task.forgetting.*`）。无标签诊断若启用仍单独使用 `metric_role=diagnostic`。Trajectory grouping 会绑定 split 与 retention design digest，避免不同旧任务集合被合并为同一条 seed trajectory。

下一步是把同一 retention manifest 复用于真实 seed 4322/4323 的连续 checkpoint，并与 clean/shift、fresh initialization、固定 probe budget 共同运行；在这些 checkpoint 到位前，不输出正式 LoP 结论。

`config/raeeg-lop-matrix-v15-isruc-multiseed-retention-preflight.json` 已把这个门槛固定下来：12 个 cell 中 seed 4321 的 4 个 stage 为 `ready`，seed 4322/4323 的 8 个 stage 因三份模型参数文件不存在而 `blocked`。预检不会复制 seed 4321，也不会把 blocked cell 纳入 trajectory 或 audit。
