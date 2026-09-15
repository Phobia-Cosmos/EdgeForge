# BrainUICL 分层个体稳定性图说明

`brainuicl-stage-pca.png` 的每个点是一条 sequence 的 20 个 epoch embedding 的中位数，不是单个 epoch；颜色表示 subject。每个子图对应 BrainUICL 的一个 tap：EEG/EOG 卷积分支、fusion、Transformer 第 1/2/3 次输出、128 维 classifier input 和 5 维 logits。同色点的扩散表示被试内 sequence/state 漂移，不同颜色中心的距离表示被试间表征差异。PCA 只用于二维显示，不是模型训练目标。

`brainuicl-stage-stability.png` 左图比较 epoch token 与 sequence median 的 sequence-disjoint identity probe；右图显示标准化表征空间中的 `between-subject centroid distance / within-subject radius`，大于 1 才表示被试间中心距离超过被试内波动。

本次没有本机可用的训练后 BrainUICL checkpoint，因此是明确的 `random-init architecture audit`。卷积分支的较高 probe 只说明随机卷积仍保留输入中的幅度、频谱和通道结构；Transformer 后接近 chance 的结果也不能代表训练后的 BrainUICL。正式结论需使用相同 sequence manifest 加载 source-pretrained checkpoint 重跑。

数值、embedding 和复现 manifest 位于分析归档 `docs/eeg-architecture-lop-analysis-20260911/brainuicl-subject-stability-audit-20260914/`。
