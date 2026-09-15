# EEG 可视化索引

完整 ISRUC 的文字总结见 [full-isruc-visualization-results-20260909.md](full-isruc-visualization-results-20260909.md)，EEG 与图像特征对照见 [eeg-image-feature-comparison.md](eeg-image-feature-comparison.md)。

图片和机器可读摘要位于仓库内的 [`docs/eeg-architecture-lop/full-isruc-visualizations-v3/`](eeg-architecture-lop/full-isruc-visualizations-v3/)。

- [波形](eeg-architecture-lop/full-isruc-visualizations-v3/eeg-drift-waveforms.png)：共同纵轴下的原始幅度，以及标准化后的形状。
- [增益归一化波形](eeg-architecture-lop/full-isruc-visualizations-v3/eeg-drift-normalized-overlay.png)：去除整体增益后比较个体波形形态。
- [频谱](eeg-architecture-lop/full-isruc-visualizations-v3/eeg-drift-spectra.png)：delta、theta、alpha、sigma、beta 频带组成。
- [RMS 轨迹](eeg-architecture-lop/full-isruc-visualizations-v3/eeg-drift-rms-trajectory.png)：个体内部 epoch 强度变化，虚线区分适应和评估。
- [个体距离矩阵](eeg-architecture-lop/full-isruc-visualizations-v3/eeg-drift-subject-distance.png)：标准化多维信号特征的个体间距离。
- [原始尺度 PCA](eeg-architecture-lop/full-isruc-visualizations-v3/eeg-drift-pca.png)：幅度/功率敏感的个体分离。
- [增益不变 PCA](eeg-architecture-lop/full-isruc-visualizations-v3/eeg-drift-pca-gain-invariant.png)：去除全局 RMS 后的通道平衡、粗糙度和频谱差异。
- [标签先验](eeg-architecture-lop/full-isruc-visualizations-v3/eeg-drift-labels.png)：适应段和评估段的睡眠阶段比例。
- [JSON 摘要](eeg-architecture-lop/full-isruc-visualizations-v3/visualization-summary.json)：所有分组、PCA、RMS、频带和个体距离统计。

## 波形特征与 BrainUICL 网络特征的不变性

- [特征/tap 不变性对比](eeg-architecture-lop/subject-invariance-audit-20260914/subject-invariance-comparison.png)：比较 295 维波形特征和 BrainUICL 各层的 sequence-level ICC、身份 probe，以及按 ICC 选取稳定维度后的变化。
- [波形特征族 ICC](eeg-architecture-lop/subject-invariance-audit-20260914/waveform-feature-family-icc.png)：比较绝对增益、相对频谱、空间连接、形态和谱形状等特征族的跨 sequence 重复性。
- [不变性审计说明](eeg-architecture-lop/subject-invariance-audit-20260914/EXPLANATION.md)：解释 ICC、sequence median、增益校正和图中每个柱子的含义。

这组图的机器可读结果位于 `docs/eeg-architecture-lop-analysis-20260911/subject-invariance-audit-20260914/subject-invariance-summary.json`。绝对功率的 ICC 可能包含电极/增益条件，因此必须与去除绝对尺度后的相对特征结果一起解释。

关键结论：target 个体 RMS 最大/最小约 `3.43x`；原始 PCA 的 PC1 解释 `90.77%` 方差，主要受幅度/功率尺度影响；去除整体 RMS 后 PC1/PC2 解释 `22.22%/16.48%`，仍保留频谱、通道平衡和波形形态差异。subject 65 是频谱离群点，subject 71 与 73 的标准化综合特征距离最大。可视化用于解释数据漂移，不单独构成 LoP 证据。

## 连续 sequence 的 20 个 epoch

针对“一个 sequence 内 20 个连续 30 秒 epoch 如何变化”的查看，结果位于仓库内的 [`docs/eeg-architecture-lop/clean-medium-sequence-waveforms-v1/`](eeg-architecture-lop/clean-medium-sequence-waveforms-v1/)。每个入选 sequence 有三张图：`waveform-stack` 按 E00--E19 依次叠放一个通道的完整波形，`all-channels` 用 8 个热图显示全部 epoch 和通道，`characteristic-epochs` 放大该 sequence 中低/中位/高 RMS、首次标签变化和最大相邻 RMS 跳变对应的完整 30 秒 epoch。图名直接包含 subject 和 sequence 编号，纵轴的 E## 是该 sequence 内的 epoch 索引，不是全数据集的全局编号。

每张图的逐图解释和选择依据见 [clean-medium-sequence-waveforms-v1/EXPLANATION.md](eeg-architecture-lop/clean-medium-sequence-waveforms-v1/EXPLANATION.md)，机器可读的 epoch 指标见 [visualization-summary.json](eeg-architecture-lop/clean-medium-sequence-waveforms-v1/visualization-summary.json)。阅读时先看 `waveform-stack` 的时间顺序和 adaptation/evaluation 分界，再看 `characteristic-epochs` 中各通道是否同步出现振荡、突发或振幅变化，最后用 `all-channels` 判断这种变化是单通道局部现象还是多通道共同变化。这里使用原始 ISRUC medium 数据，class 0--4 只保留标签编码，不把颜色直接解释为具体睡眠阶段；这些图用于定位和解释个体/epoch 差异，不单独证明 LoP。
