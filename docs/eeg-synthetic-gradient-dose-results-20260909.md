# Synthetic EEG LoP gradient-dose results

本实验用于验证 EdgeForge 的 LoP gate 是否能够检测“warm 模型表示可塑性受损”这一已知正控机制。它不是 ISRUC 真实数据的科学结论，也不代表自然 EEG 中已经发现 LoP。

## 实验设计

- 数据：`edgeforge-synthetic-eeg-lop-positive-control-v1`。
- source subjects：3 个；target transition：8 个；retention subjects：2 个。
- seeds：4321、4322、4323。
- warm 模型先在 source 上训练，随后只对 positive-control arm 的 encoder 反向梯度乘以 `warm_encoder_gradient_scale`。
- fresh 模型在每个 target stage 重新随机初始化并保持完全可训练。
- LoP outcome：`fresh_gap = accuracy_fresh − accuracy_warm`，budget=50，严格 gate 要求所有 seed/transition 同方向且 bootstrap 下界为正。

## 结果

| warm encoder gradient scale | budget 10 平均 fresh-gap | budget 25 平均 fresh-gap | budget 50 平均 fresh-gap | budget 50 正/零/负单元 | 正 transition 数 | gate |
|---:|---:|---:|---:|---:|---:|---|
| 0.00 | +0.0902 | +0.1644 | +0.2477 | 24 / 0 / 0 | 8/8 | candidate |
| 0.25 | +0.0742 | +0.1025 | +0.1279 | 20 / 0 / 4 | 6/8 | blocked-inconsistent-direction |
| 0.50 | +0.0710 | +0.1022 | +0.1227 | 19 / 0 / 5 | 5/8 | blocked-inconsistent-direction |
| 0.75 | +0.0732 | +0.1003 | +0.1221 | 19 / 0 / 5 | 5/8 | blocked-inconsistent-direction |
| 1.00 | +0.0710 | +0.1016 | +0.1217 | 19 / 0 / 5 | 5/8 | blocked-inconsistent-direction |

其中 `candidate` 表示通过 requirement gate；由于实验是人为构造的正控，结果仍标记为 `scientific_conclusion_allowed=false`。

## 解释

1. `scale=0.0` 能稳定产生 LoP：warm encoder 完全不能针对新的 latent axis 更新，fresh 模型经过少量 target 训练后持续优于 warm 模型。
2. 只保留 25% encoder 梯度时，平均 gap 仍为正，但 transition 方向出现负值，因此不能称为稳定 LoP。平均值为正本身不足以过 gate。
3. `scale=0.5–1.0` 的结果相近，说明在这个极小的合成任务中，主要差异不是“梯度变小一点”，而是是否存在足够强的持续表示损伤；同时 source 初始化和 target 任务结构也会影响 gap。
4. 该实验验证的是检测机制：当人为制造 warm plasticity impairment 时，gate 能识别它。它没有证明 ISRUC 中存在同样的自然机制；完整 ISRUC 仍是 `blocked-inconsistent-direction`。

## 对后续真实 EEG 实验的建议

- 优先测试可解释的、较弱的机制：冻结部分层、降低 encoder 学习率、限制 adapter 容量、过强 L2-SP 或 replay 约束，并保持多 seed、多 transition gate。
- 同时记录 representation rank、梯度范数、参数更新量和 retention；这些是机制诊断，不能替代 fresh-gap gate。
- 不应通过篡改原始 EEG 或标签来“制造”自然 LoP。若需要正控，只使用隔离的 synthetic 数据，并明确标记为 validation control。
