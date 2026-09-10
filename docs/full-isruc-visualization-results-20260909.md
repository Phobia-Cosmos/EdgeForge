# 完整 ISRUC 个体差异可视化结果

本轮在远端 A100 上对完整 ISRUC 分组运行了可视化：source 58 个体（1–60，排除 8、40）、target 20 个体（61–80）、retention 20 个体（81–100），共 98 个体。作业 195161 已完成，退出码为 0。原始数据只读，所有图片和摘要均为派生结果。

结果目录：`/home/undefined/UbuntuData/ai-storage/EdgeForge/eeg-architecture-lop/full-isruc-visualizations-v3/`。

## 如何查看

- [原始/标准化波形](/home/undefined/UbuntuData/ai-storage/EdgeForge/eeg-architecture-lop/full-isruc-visualizations-v3/eeg-drift-waveforms.png)
- [去除增益后的典型波形叠加](/home/undefined/UbuntuData/ai-storage/EdgeForge/eeg-architecture-lop/full-isruc-visualizations-v3/eeg-drift-normalized-overlay.png)
- [目标个体频谱](/home/undefined/UbuntuData/ai-storage/EdgeForge/eeg-architecture-lop/full-isruc-visualizations-v3/eeg-drift-spectra.png)
- [epoch RMS 轨迹热图](/home/undefined/UbuntuData/ai-storage/EdgeForge/eeg-architecture-lop/full-isruc-visualizations-v3/eeg-drift-rms-trajectory.png)
- [目标个体距离矩阵](/home/undefined/UbuntuData/ai-storage/EdgeForge/eeg-architecture-lop/full-isruc-visualizations-v3/eeg-drift-subject-distance.png)
- [原始尺度/功率敏感 PCA](/home/undefined/UbuntuData/ai-storage/EdgeForge/eeg-architecture-lop/full-isruc-visualizations-v3/eeg-drift-pca.png)
- [增益不变 PCA](/home/undefined/UbuntuData/ai-storage/EdgeForge/eeg-architecture-lop/full-isruc-visualizations-v3/eeg-drift-pca-gain-invariant.png)
- [适应/评估标签先验](/home/undefined/UbuntuData/ai-storage/EdgeForge/eeg-architecture-lop/full-isruc-visualizations-v3/eeg-drift-labels.png)
- [机器可读摘要](/home/undefined/UbuntuData/ai-storage/EdgeForge/eeg-architecture-lop/full-isruc-visualizations-v3/visualization-summary.json)

## 定量结果

原始 PCA 使用 13 个特征（8 个通道 log-RMS 和 5 个频带 log-power），共 3977 个抽样 epoch，PC1 解释 90.77% 方差，PC2 解释 3.50% 方差。这说明完整队列的最大可见差异仍主要来自幅度/功率尺度。

增益不变 PCA 使用 21 个特征（逐 epoch 全局 RMS 归一化后的通道 RMS 比例、时间差分 RMS 比例和 0.5–40 Hz 频带比例），同样包含 3977 个抽样 epoch，PC1/PC2 分别解释 22.22%/16.48% 方差。尺度被移除后，方差不再集中到第一主成分，说明剩余差异是多因素的，不能用一个“个体增益”解释完。

target 61–80 的全局 RMS 最大/最小比为 3.43 倍，subject 65 的 RMS 最高（约 5.62×10⁻⁵），subject 76 最低（约 1.64×10⁻⁵）。subject 65 相对 target 中位频谱的 Jensen–Shannon 距离为 0.274，显著高于其他个体，是频谱离群点。subject 63 的 beta 比例约 0.174，subject 74 的 alpha 比例约 0.218，显示频带构成差异。按标准化 log-RMS/频带特征计算的最远个体对为 subject 71 与 73，距离约 11.65。

## 科学解释边界

这些图可以直观呈现三种不同现象：原始波形共同纵轴上的上下移动对应幅度/增益差异；归一化波形和增益不变 PCA 中仍存在的分离对应形态、通道平衡或频谱差异；RMS 热图和标签图中的前后半段变化对应个体内部的时间/标签先验漂移。图像数据的二维空间局部性可以直接在 feature map 或 patch 上定位，而 EEG 的局部性同时沿时间、通道和频率三个轴展开，所以必须联合时域、频域和个体统计空间观察。

可视化本身不构成 LoP 证据。当前完整 ISRUC 的所有架构与 replay/L2-SP 策略仍未通过严格 gate；要称为 LoP，仍需同一预算下 fresh-vs-warm 的正向差异在所有 transition、seed 上稳定，并结合 retention 与表示诊断。当前结果更准确的结论是：完整 ISRUC 存在可见且可量化的个体异质性，其中幅度尺度是首要因素，但去除尺度后仍保留频谱/形态差异。

