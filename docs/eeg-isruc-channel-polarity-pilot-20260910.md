# ISRUC medium channel-polarity pilot

本实验直接从本地 ISRUC medium 派生数据，不修改 clean 数据。target 的 channel 0 在所有 epoch/file 上乘以 `-1`，标签、epoch 边界、subject 顺序和训练/评估切分保持不变。

## 数据质量

- 派生目录：`/home/undefined/UbuntuData/datasets/edgeforge-isruc-medium-target-channel-polarity-ch0-v1`
- 100 个数据文件，输入 shape 和标签文件数量不变。
- 被翻转通道的相关系数为 `-1`，RMS ratio 为 `1.0`，PSD 相对误差为 `0`。
- 配对图：`/home/undefined/UbuntuData/ai-storage/EdgeForge/eeg-architecture-lop/condition-target-channel-polarity-ch0-v1/visualizations-v1/`

## LoP gate

Protocol：TCN/EEGNet，3 seeds（4321/4322/4323），8 target transitions，budgets `0/5/10/25/50`，fresh-vs-warm accuracy gap。

| Architecture | Budget | Mean fresh-gap | Positive transitions | Gate |
| --- | ---: | ---: | ---: | --- |
| TCN | 5 | +0.025 | 1/8 | blocked-inconsistent-direction |
| TCN | 10 | +0.071 | 1/8 | blocked-inconsistent-direction |
| TCN | 25 | +0.078 | 4/8 | blocked-inconsistent-direction |
| TCN | 50 | +0.022 | 3/8 | blocked-inconsistent-direction |
| EEGNet | 5 | -0.066 | 0/8 | blocked-inconsistent-direction |
| EEGNet | 10 | -0.038 | 0/8 | blocked-inconsistent-direction |
| EEGNet | 25 | +0.016 | 2/8 | blocked-inconsistent-direction |
| EEGNet | 50 | -0.007 | 0/8 | blocked-inconsistent-direction |

## 结论

真实 ISRUC 的单通道极性变化会改变 signed cross-channel structure，但在当前 medium cohort、subject order 和网络下没有稳定 fresh-better LoP。它不能复制 synthetic channel-polarity 正控的结果，也不能仅凭 accuracy 下降称为 LoP。后续应转向 EEGNet ReLU 的 state-conditioned dormant-unit steering，并保留 clean、matched-random 和频谱匹配对照；TCN/BrainUICL 的 GELU 只能报告 near-zero/低梯度，不能称永久 dead。
