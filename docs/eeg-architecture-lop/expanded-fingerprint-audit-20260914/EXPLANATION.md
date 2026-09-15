# 扩展 EEG 波形指纹图说明

`expanded-fingerprint-pca.png` 的每个点是一个 30 秒 epoch，颜色表示 subject。它使用 295 维可解释波形统计向量，不是 CNN embedding；同色点的扩散表示被试内 sequence/state 漂移，颜色中心差异表示个体统计差异。

`expanded-fingerprint-ablation.png` 比较完整 295 维向量和去掉某个特征组后的 sequence-disjoint Logistic identity accuracy。柱子下降越多，表示该组对跨 sequence 个体区分贡献越大，但不表示生理因果重要性。本次去掉 28 个逐对 signed-correlation 特征后准确率由 88.75% 降至 80.83%，是最明显的消融。

扩展特征在原 83 维之外加入均值/中位数/标准差/偏度/峰度/crest factor/过零率、Hjorth activity-mobility-complexity、线长、六频带绝对 log 功率、谱质心/带宽/95% 频谱边缘/平坦度/峰频、PSD 斜率和 28 个通道两两 signed correlation。扩大维度并不自动提高个体不变性，因此还需同时看前后半 profile correlation 和 between/within 距离比。

完整数值和特征矩阵位于分析归档 `docs/eeg-architecture-lop-analysis-20260911/expanded-fingerprint-audit-20260914/`。本次只覆盖 medium ISRUC 的 20 个被试、100 个 sequence，不能替代完整 98 个体复核。
