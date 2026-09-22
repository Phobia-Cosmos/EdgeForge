# EEG LoP 训练指标汇总（2026-09-12）

本记录汇总当前已经产生的可复现实验，不把描述性诊断升级为 LoP 科学结论。主要的层级诊断来自 `rms-equalized-tcn-source-epochs10-diagnostics-v1`：TCN、3 个 seed、8 个 target stage、5 个 adaptation budget，每个 arm/budget 有 24 个 stage-seed cell，calibration 最多 128 个 observation。

## 当前协议边界

连续 medium 基准的 source subjects 为 `1,3,4,6,7,9,10,21`，target 顺序为 `2,11,12,13,14,15,16,17`，retention subjects 为 `5,18,19,20`。每个 target 文件的前 10 个 30 秒 epoch 用于 adaptation，后 10 个 epoch 用于 held-out evaluation；warm 模型跨 target stage 传递，fresh 模型在每个 stage 重新初始化。该协议适合作为固定预算的 pilot，但不是最终的 subject-level 泛化基准，因为同一文件的前后两半仍共享 sequence-local context 和 class-prior 结构。

## TCN 层级诊断均值

下表是跨 8 个 stage 和 3 个 seed 的均值。`embedding ER` 是 embedding 表征矩阵的 effective rank，`ER norm` 除以 `min(N,D)`，`block1 ER` 是首个卷积块表征的 effective rank，`grad NZ` 是梯度非零比例，`relative update` 是参数全局相对更新量。这里的 `sigma_max` 仅是激活表征矩阵的最大奇异值代理，不是参数矩阵完整谱。

| arm | budget | embedding ER | ER norm | block1 ER | grad NZ | relative update | classifier near-zero |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| warm | 0 | 3.091 | 0.0966 | 2.196 | 0.596 | 0.0000 | 0 |
| warm | 5 | 2.909 | 0.0909 | 2.122 | 0.575 | 0.0171 | 0 |
| warm | 10 | 3.269 | 0.1022 | 2.137 | 0.585 | 0.0278 | 0 |
| warm | 25 | 3.084 | 0.0964 | 2.189 | 0.588 | 0.0492 | 0 |
| warm | 50 | 2.989 | 0.0934 | 2.214 | 0.596 | 0.0746 | 0 |
| fresh | 0 | 4.391 | 0.1372 | 1.994 | 0.999 | 0.0000 | 0 |
| fresh | 5 | 5.562 | 0.1738 | 1.996 | 1.000 | 0.0209 | 0 |
| fresh | 10 | 6.001 | 0.1875 | 1.999 | 1.000 | 0.0375 | 0 |
| fresh | 25 | 11.369 | 0.3553 | 2.034 | 1.000 | 0.0746 | 0 |
| fresh | 50 | 3.459 | 0.1081 | 2.157 | 1.000 | 0.1171 | 0 |

warm embedding stable rank约为 `1.16–1.23`，首个卷积块的 `sigma_max` 约为 `141–188`，随 budget 非单调变化；fresh embedding stable rank 约从 `0.56` 增至 `1.45` 后回落，首个卷积块的 `sigma_max` 约从 `0.044` 增至 `0.184`。这些变化更像 calibration 表征和适应预算的即时变化，不是随 stream age 单调坍缩。

## LoP 结果对照

连续 medium v0.18.0-r2 的最终 budget=50 fresh-gap 均值为：`lop_mlp 0.000`、`eegnet −0.020`、`tcn −0.008`、`transformer 0.000`、`brainuicl −0.052`。所有架构均未通过“所有 transition、所有 seed、bootstrap 下界都严格为正”的 gate。TCN 在 budget=25 的均值约为 `+0.021`，但仍包含负方向 cell；RMS equalization 在单独比较中 budget=25/50 的均值约为 `+0.106/+0.109`，同样没有通过严格 gate。

warm 初始 embedding rank 与最终 fresh-gap 的探索性 Pearson 相关为 `−0.112`（24 个 stage-seed cell）。它既不显著也不构成因果证据。warm 梯度非零比例、参数更新和激活 rank 也没有与 fresh-gap 稳定同向变化。

## 哪些指标当前有效

- `fresh_gap = Acc_fresh − Acc_warm`：唯一可以作为 LoP 主 outcome 的指标，但当前只证明不同协议下的 transfer sensitivity，尚未证明自然 LoP。
- held-out accuracy/loss、AULC gap 和 retention：用于区分适应速度、最终性能和旧任务稳定性；不能相互替代。
- RMS、Welch PSD、gain-invariant PCA、subject/sequence 距离：有效地描述采集尺度、频谱和个体漂移，但不是 LoP 证据。
- effective/stable rank、singular-value summary、gradient reachability：适合作为机制候选，只有在状态指标先于 fresh-gap 变化、跨 seed/transition 重复，并且 reset/ReDo 可以反事实恢复时才有效。

## 哪些指标当前无效或不足

- 单次 rank 下降、`sigma_max` 变化或 PCA 分离：都可能只是 amplitude、calibration 数量、feature axis 或 subject composition 的结果。
- 梯度范数变大/变小：当前 warm 梯度范数约 `814–1554` 且高度波动，不能解释为“梯度消失”；fresh 梯度接近全非零也不代表模型没有 LoP。
- classifier-input near-zero fraction：当前所有 cell 都为 `0`，没有检测到该层的整体失活；对 TCN 的 GELU 也不能使用永久 dead-ReLU 术语。
- accuracy 下降、旧任务 retention 下降或输入 RMS 变大：可以是负迁移、遗忘或信号质量问题，单独不能替代 fresh-gap gate。

## 指标变化图

已根据同一份 240 行结构化诊断数据生成图形化结果，位于 [training-metrics-v1](/home/undefined/UbuntuData/ai-storage/EdgeForge/eeg-architecture-lop/training-metrics-v1/)：

- [training-metrics-by-budget.png](/home/undefined/UbuntuData/ai-storage/EdgeForge/eeg-architecture-lop/training-metrics-v1/training-metrics-by-budget.png)：warm/fresh 的 effective rank、normalized rank、block1 rank、梯度范数、梯度非零比例和参数相对更新，带跨 stage-seed 的均值±标准差。
- [fresh-gap-by-budget.png](/home/undefined/UbuntuData/ai-storage/EdgeForge/eeg-architecture-lop/training-metrics-v1/fresh-gap-by-budget.png)：固定 adaptation budget 下的 fresh-gap 均值、离散度和正 gap 比例。
- [metric-vs-fresh-gap.png](/home/undefined/UbuntuData/ai-storage/EdgeForge/eeg-architecture-lop/training-metrics-v1/metric-vs-fresh-gap.png)：rank/梯度候选指标与 fresh-gap 的散点关联及 Pearson 相关。
- [stage-heatmaps.png](/home/undefined/UbuntuData/ai-storage/EdgeForge/eeg-architecture-lop/training-metrics-v1/stage-heatmaps.png)：按 target stage × adaptation budget 展示 warm rank、block1 rank、梯度非零比例、梯度范数和 fresh-gap，便于发现被总体均值掩盖的局部异常。

这些图是描述性诊断，不改变 LoP 判据。当前图中 warm rank 没有单调下降，梯度非零比例没有整体跌落，指标与 fresh-gap 的相关也接近零；它们用于排查候选机制，而不是单独宣布 LoP。绘图脚本为 [`plot-eeg-training-metrics.py`](/home/undefined/Desktop/EdgeForge/scripts/plot-eeg-training-metrics.py)。

下一次正式实验应改为 subject-level sequence split：source subject 的全部训练 sequence 用于预训练；每个 target subject 用互不重叠的 adaptation sequences 和 evaluation sequences；retention subject 完全不参与更新。然后在相同 checkpoint 上重新采集每层 ER/stable rank、奇异值摘要、ReLU dead/active 或 GELU near-zero、梯度可达性和参数更新，并做 `state(t−1) → fresh_gap(t)` 的滞后分析。只有这套协议通过严格 gate，才值得讨论 rank/谱是否参与 LoP。
