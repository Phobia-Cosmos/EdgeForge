# EdgeForge 实验结果归档

这个目录保存可复现和汇报所需的轻量实验产物。原始 EEG 数据位于 `/home/undefined/Disk/datasets/brainuicl/processed/isruc_group1_npy_float32`，训练 checkpoint 位于外部版本化实验目录；二者不复制到 Git 仓库。仓库内的结果包含分析 JSON/Markdown、实验 manifest、协议锁、架构 summary、逐 seed run 元数据、运行日志和可视化图表。

## 当前版本

`eeg-lop-full/v0.19.1/` 是 ISRUC 全量角色划分上的 calibration 归档。它使用 30 个 source subjects、50 个 target subjects 和 18 个 retention subjects；calibration 使用 5 个架构、3 个 seed、`random-a` 的前 12 个 target stages，产生 15 条 trajectory 和 1,080 个 stage-budget cells。

`analysis/calibration-v2/analysis.json` 的状态是 `candidate-evidence-ready`，完整性、设计一致性、样本量和正式架构训练充分性门禁通过；`scientific_conclusion_allowed` 保持 `false`。这份结果用于检查输入尺度、训练链路、fresh/warm 对照和审计流程，不能单独作为 LoP 结论。

## 重新生成图表

在仓库根目录运行：

```bash
/home/undefined/Disk/python-envs/brainuicl/bin/python \
  scripts/plot-eeg-lop-results.py \
  --result-root results/eeg-lop-full/v0.19.1
```

图表会写入 `eeg-lop-full/v0.19.1/plots/`：

- `source-accuracy-by-architecture.png`：source held-out accuracy 和多数类/门禁线；
- `target-learning-curves.png`：正式架构 warm/fresh target accuracy 曲线；
- `fresh-gap-by-budget.png`：各架构按 budget 的 fresh-gap 及 bootstrap 区间；
- `plot-manifest.json`：图表来源和生成时间。

图表属于描述性 calibration 输出，脚本和 manifest 始终记录 `scientific_conclusion_allowed=false`。

## 运行结果结构

`analysis/` 保存分析器输出，`runs/calibration/random-a/<architecture>/` 保存架构 summary、metadata 和逐 seed `run.json`，`protocol/calibration-lock.json` 固定正式实验协议，`role-view/PLAN.json` 保存 subject split、顺序和 label profile，`logs/` 保存 runner 日志。

完整的 50-stage、10-seed、`random-a/random-b` confirmatory 和 data-composition 结果尚未归档；它们完成后应创建新的版本目录，不覆盖 `v0.19.1`。
