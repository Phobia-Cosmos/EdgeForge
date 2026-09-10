# EEG 可视化索引

完整 ISRUC 的文字总结见 [full-isruc-visualization-results-20260909.md](full-isruc-visualization-results-20260909.md)，EEG 与图像特征对照见 [eeg-image-feature-comparison.md](eeg-image-feature-comparison.md)。

图片和机器可读摘要位于：`/home/undefined/UbuntuData/ai-storage/EdgeForge/eeg-architecture-lop/full-isruc-visualizations-v3/`。

- [波形](</home/undefined/UbuntuData/ai-storage/EdgeForge/eeg-architecture-lop/full-isruc-visualizations-v3/eeg-drift-waveforms.png>)：共同纵轴下的原始幅度，以及标准化后的形状。
- [增益归一化波形](</home/undefined/UbuntuData/ai-storage/EdgeForge/eeg-architecture-lop/full-isruc-visualizations-v3/eeg-drift-normalized-overlay.png>)：去除整体增益后比较个体波形形态。
- [频谱](</home/undefined/UbuntuData/ai-storage/EdgeForge/eeg-architecture-lop/full-isruc-visualizations-v3/eeg-drift-spectra.png>)：delta、theta、alpha、sigma、beta 频带组成。
- [RMS 轨迹](</home/undefined/UbuntuData/ai-storage/EdgeForge/eeg-architecture-lop/full-isruc-visualizations-v3/eeg-drift-rms-trajectory.png>)：个体内部 epoch 强度变化，虚线区分适应和评估。
- [个体距离矩阵](</home/undefined/UbuntuData/ai-storage/EdgeForge/eeg-architecture-lop/full-isruc-visualizations-v3/eeg-drift-subject-distance.png>)：标准化多维信号特征的个体间距离。
- [原始尺度 PCA](</home/undefined/UbuntuData/ai-storage/EdgeForge/eeg-architecture-lop/full-isruc-visualizations-v3/eeg-drift-pca.png>)：幅度/功率敏感的个体分离。
- [增益不变 PCA](</home/undefined/UbuntuData/ai-storage/EdgeForge/eeg-architecture-lop/full-isruc-visualizations-v3/eeg-drift-pca-gain-invariant.png>)：去除全局 RMS 后的通道平衡、粗糙度和频谱差异。
- [标签先验](</home/undefined/UbuntuData/ai-storage/EdgeForge/eeg-architecture-lop/full-isruc-visualizations-v3/eeg-drift-labels.png>)：适应段和评估段的睡眠阶段比例。
- [JSON 摘要](</home/undefined/UbuntuData/ai-storage/EdgeForge/eeg-architecture-lop/full-isruc-visualizations-v3/visualization-summary.json>)：所有分组、PCA、RMS、频带和个体距离统计。

关键结论：target 个体 RMS 最大/最小约 `3.43x`；原始 PCA 的 PC1 解释 `90.77%` 方差，主要受幅度/功率尺度影响；去除整体 RMS 后 PC1/PC2 解释 `22.22%/16.48%`，仍保留频谱、通道平衡和波形形态差异。subject 65 是频谱离群点，subject 71 与 73 的标准化综合特征距离最大。可视化用于解释数据漂移，不单独构成 LoP 证据。
