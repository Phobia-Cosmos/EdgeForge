# Synthetic EEG channel-polarity LoP result

本实验只改变派生 synthetic 数据的 target 输入波形，不冻结模型、不缩放梯度、不修改 ISRUC 原始数据。target 的 channel 0 第一半窗口乘以 `-1`，标签保持与翻转前信号一致。这近似模拟导联参考或电极极性发生变化。

## 结果

使用 3 个 seed、8 个连续 target transition 和严格 fresh-gap gate：

| adaptation budget | 平均 fresh-gap | transition 方向 | gate |
|---:|---:|---:|---|
| 5 | +0.521 | 8/8 positive | candidate |
| 10 | +0.545 | 8/8 positive | candidate |
| 25 | +0.497 | 8/8 positive | candidate |
| 50 | +0.211（后段转负） | mixed/negative | blocked-inconsistent-direction |

## 解释

该结果表明，输入端的持续通道极性 shift 足以在有限适应预算内诱导稳定 fresh-better LoP。warm 模型携带 source 的通道极性先验，fresh 模型从无先验开始，因此在 budget 5–25 内更快适应。预算达到 50 后，warm 模型逐渐重新学习，LoP 优势消失。

这是目前比 target label reversal 更接近 EEG 采集过程的数据诱导机制，但仍是 synthetic 正控，不能直接解释为 ISRUC 中存在自然 LoP。真实数据上应优先检查参考电极、导联方向、通道映射和跨设备极性一致性。
