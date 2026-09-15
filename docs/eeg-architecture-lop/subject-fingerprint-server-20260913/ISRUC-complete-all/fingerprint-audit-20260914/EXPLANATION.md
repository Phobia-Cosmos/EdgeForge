# 2026-09-14 指纹审计图说明

本目录是对 ISRUC 全量个体指纹分析的复核实验。输入不是 CNN 的隐藏层，而是每个 30 秒 epoch 从原始 `(8, 3000)` 波形计算出的 83 维固定特征；共 98 个被试、4,276 个 sequence、85,520 个 epoch。每一行对应一个 epoch，sequence 只用于按被试做完整留出，避免同一 sequence 的相邻 epoch 同时出现在训练和测试中。

`fingerprint-audit.png` 左图显示对 83 列做 `StandardScaler` 后 PCA 的累计解释方差。PCA-1 和 PCA-2 是标准化特征矩阵的前两个左投影坐标，不是两个原始 EEG 通道，也不是 CNN 特征。右图是 sequence-disjoint 的 Logistic identity probe；每个柱子是在去掉一个特征组后重新训练的 98 类被试分类器。虚线是随机猜测基线 `1/98`。

审计结果：PCA-1/2 合计解释约 50.24% 方差；保留全部特征时身份准确率 73.28%。去掉 40 个相对频带功率特征后准确率降至 52.15%，说明频谱结构是最重要的一组；去掉 3 个带符号连接摘要后为 70.95%，去掉差分能量比后为 72.05%。这些是“可复现的统计信息”证据，不是生理身份认证结论，也不是 LoP 证据。完整数值已移到分析归档 `docs/eeg-architecture-lop-analysis-20260911/subject-fingerprint-server-20260913/ISRUC-complete-all/fingerprint-audit-20260914/`。

PCA 轴的正负号没有独立含义：同一轴整体乘以 `-1` 仍是完全相同的投影。应关注解释方差、点间距离、同色点的扩散和跨 sequence 留出准确率。详细数值和前 12 个绝对载荷在 `audit-summary.json` 中。
