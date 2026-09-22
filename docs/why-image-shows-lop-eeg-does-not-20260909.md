# 为什么图像 smoke 看起来有 LoP，而真实 EEG 没有

短答案：目前不是“图像已经证明 LoP、EEG 没有 LoP”。图像结果是受控 synthetic domain-shift smoke，只有 1 个 seed、2 个 transition，而且同一份报告中 CNN 一个 transition 是负值；它没有通过与真实 EEG 同等级的严格 gate。

## 协议并不等价

| 项目 | 图像 smoke | 完整 ISRUC EEG |
| --- | --- | --- |
| 数据 | synthetic-image，人工生成/旋转/亮度变化 | 真实多通道睡眠 EEG，98 个体 |
| seed | 1 个 | 3 个 |
| transition | 2 个 | 20 个 target transition |
| 每阶段训练 | 24 train / 16 eval，1 epoch | 每个 source/target 文件按固定 epoch 切分，完整 subject stream |
| 适应预算 | 0/1/2 | 0/5/10/25/50 |
| gate | 诊断 smoke，没有正式全 transition gate | 要求所有 transition × seed fresh-gap 同方向且 bootstrap 下界为正 |

图像 Transformer 的 `+0.0625` 只表示在一个 transition、一个 seed、很小的测试集上，fresh 模型比 warm 模型高 1 个样本的准确率；它不是稳定群体结论。图像 CNN 在另一个 transition 是 `-0.375`，已经说明同一图像协议也不必然出现 LoP。

## EEG 为什么更难出现稳定正 fresh-gap

1. **真实 EEG 的个体异质性更复杂。** 完整 ISRUC 中 target RMS 最大/最小约 3.43 倍，subject 65 还是频谱离群点。差异同时来自电极增益、频带功率、通道平衡、波形形态和睡眠阶段比例。warm 模型有时可以复用已有特征，因此 fresh 反而不一定更好。
2. **任务信号和域信号没有被完全分离。** 睡眠阶段标签本身具有个体/阶段先验差异；warm 模型可能学到部分稳定先验，或在某些 subject 上被错误先验拖累。于是 fresh-gap 会随个体变号，而不是整条 stream 同方向。
3. **source 训练和输入尺度是重要混杂因素。** 原始完整 pilot 的 TCN source accuracy 接近多数类基线；增加 source epochs 或 RMS 标定后 source 学习明显改善，但 gate 仍混合。这说明“训练时间不够/输入尺度不合适”会影响结果，却不是唯一原因。
4. **严格 gate 对真实数据要求更高。** 真实实验有 3 seed、20 个体 transition；只要一个 transition 的一个 seed 是零或负，就不能称为稳定 LoP。当前 TCN、BrainUICL、replay、L2-SP 都被 `blocked-inconsistent-direction` 拦截。
5. **图像 smoke 的 domain shift 更容易制造可塑性差异。** 旋转、亮度和人工模式可以让下一阶段的判别边界发生快速、规则的变化；小模型只需少量更新就能观察到 fresh/warm 差异。真实 EEG 的变化更连续、更混杂，且短适应预算可能不足以建立稳定的可塑性优势。

## 已做的同规模对照

在相同的 3 tasks、24 train、16 eval、1 epoch、单 seed 和预算 0/1/2 下，synthetic EEG TCN 的两个 fresh-gap 为 `−0.125` 和 `−0.3125`；图像 Transformer 为 `0` 和 `+0.0625`，图像 CNN 为 `0` 和 `−0.375`。因此差异不能简单归因于“序列模型不如图像模型”：同一图像域中 CNN 也出现负值，而同一 synthetic EEG 协议也可以通过改变任务构造来设计正控。

## 正确结论

当前证据支持的是：真实 ISRUC 没有在现有架构、source 训练、适应预算和严格 gate 下显示稳定自然 LoP；图像结果只证明诊断管线可以在受控小样本条件下观察到一次正 fresh-gap。要公平比较，下一步应使用相同 seed 数、相同 transition 数、相同样本预算，并为 EEG 与图像分别设计已知的 plasticity-lesion positive control；不能把当前图像 smoke 的一个正值与完整 ISRUC 的严格失败直接横向比较。

