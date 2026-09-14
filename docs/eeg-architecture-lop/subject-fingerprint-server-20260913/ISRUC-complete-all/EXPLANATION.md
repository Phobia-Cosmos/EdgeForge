# ISRUC 全量 sequence/epoch 个体特征分析

## 覆盖范围

本目录来自服务器 CPU 的断点式只读分析，覆盖 ISRUC 的全部 98 个被试、4,276 个 sequence 和 85,520 个 30 秒 epoch。每个 sequence 的 20 个 epoch 都被处理；没有按前几个 sequence 截断，也没有修改原始 EEG。

## 每个被试保存的内容

`subjects/subject-XXX.json` 是一个被试的完整画像：

- `median`、`mad`、`mean`、`std`：该被试所有 epoch 的 83 维稳健 profile；
- `sequences`：该被试的每一个 sequence，包含 epoch 数、标签分布，以及该 sequence 的 median/MAD/mean/std；
- `label_counts`：睡眠标签分布；
- `feature_names`：83 个特征的名称和顺序。

对应的 `subject-XXX-features.npy` 保存该被试每一个 epoch 的 83 维特征矩阵，行顺序与 sequence 文件顺序一致。

83 维特征包含每通道 log-RMS、增益不变通道平衡、temporal roughness、差分能量比、delta/theta/alpha/beta/gamma 相对功率、谱熵，以及带符号跨通道连接摘要。

## 全量结果

- 最近中心身份识别准确率：14.70%；
- Logistic identity probe 准确率：73.28%；
- 被试间/被试内 profile 距离比：1.076；
- 随机 98 类基线约为 1.02%。

前半 sequence 形成被试 profile，后半 sequence 做 held-out 测试。Logistic 探针仍显著高于随机基线，说明跨 sequence 可复现的个体信息存在；最近中心准确率较低、距离比接近 1，说明完整数据中的 sequence/session、睡眠阶段和状态变化很强，个体 profile 不是一个紧凑的固定点。

与较小子集相比，全量结果更适合指导后续数据修改：不能把某个被试简单替换成一个固定均值，而应以该被试自己的 profile 和 sequence-level MAD/残差范围为约束，保留个体特征同时允许状态漂移。

## 图如何阅读

- `complete-subject-fingerprint-pca.png`：每个点是一个 epoch，颜色是被试编号。重叠表示个体间共享的睡眠/频谱结构；同色点的扩散表示该被试内部的 sequence 或状态漂移。
- `sequence-drift-by-subject.png`：每个箱线图对应一个被试，纵轴是 sequence profile 到该被试全量 profile 的特征空间距离。箱体高表示 sequence 间变化大，不应使用过强的固定扰动。
- `sequence-drift-by-subject.json`：给出每个被试所有 sequence 的 mean/median/max 漂移，可直接用于选择具有代表性或高漂移的 sequence。

身份可分性仍然不是 LoP 证据。LoP 需要另行检查 warm/fresh gap、retention、rank、谱和梯度等训练指标。
