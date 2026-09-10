# EEG 与图像数据特征对照

本文把 EdgeForge 当前 EEG 漂移诊断和图像 LoP smoke 放在同一解释框架下。两者都可以使用 warm-vs-fresh、固定适应预算和表示诊断，但不能把图像的视觉分离直接解释为 EEG 的生理差异，也不能仅凭可视化宣布 LoP。

## 当前 EEG 可视化产物

完整 ISRUC 图位于 `/home/undefined/UbuntuData/ai-storage/EdgeForge/eeg-architecture-lop/full-isruc-visualizations-v3/`。主要文件如下：

- `eeg-drift-waveforms.png`：原始通道波形和逐 epoch 标准化波形；前者显示增益/幅度差异，后者显示形状差异。
- `eeg-drift-normalized-overlay.png`：各个体代表性波形去除全局幅度后的叠加。
- `eeg-drift-spectra.png`：0.5–40 Hz 的 Welch 频谱，用于观察 delta/theta/alpha/sigma/beta 组成。
- `eeg-drift-rms-trajectory.png`：epoch RMS 热图，虚线分隔适应和留出评估。
- `eeg-drift-subject-distance.png`：按 log-RMS 和频带功率标准化后的个体距离矩阵。
- `eeg-drift-pca.png`：对幅度/功率敏感的 PCA；当前 PC1 约解释 84.6% 方差。
- `eeg-drift-pca-gain-invariant.png`：逐 epoch RMS 归一化后的 PCA，新增用于区分增益漂移和频谱/形态漂移。
- `eeg-drift-labels.png`：每个个体适应半段与评估半段的睡眠阶段比例。

## 目前能看到的个体差异

完整 ISRUC target 61–80 中，RMS 最大/最小约为 3.43 倍，说明采集增益或总体功率尺度差异明显。频谱方面 subject 65 的 Jensen–Shannon 距离约 0.274，是明显频谱离群点；subject 63 的 beta 比例约 0.174，subject 74 的 alpha 比例约 0.218，也显示出不同的频带组成。标准化特征距离中，subject 71 与 73 最远，说明其多维信号统计特性不同。

原始 PCA 的大部分分离不能直接解释为“脑状态类别分离”，因为第一主成分可能主要由幅度/功率尺度驱动。新增 gain-invariant PCA 后，若个体仍保持分离，更合理的解释是通道间相对增益、时间粗糙度、频带比例或波形形态不同；若分离显著收缩，则原来的差异主要是采集尺度效应。

## EEG 与图像的结构差异

| 维度 | EEG | 图像 |
| --- | --- | --- |
| 输入结构 | 多通道时间序列 `[channels, time]`，ISRUC epoch 为 `(8, 3000)` | 二维规则网格 `[channels, height, width]` |
| 局部性 | 时间邻域、跨通道关系、频谱结构 | 空间邻域、边缘、纹理、形状 |
| 个体/域差异 | 电极增益、RMS、频带功率、睡眠阶段先验、session 漂移 | 光照、颜色、旋转、纹理、视角、空间布局 |
| 适合的可视化 | 波形、Welch 频谱、RMS 热图、频带比例、距离矩阵、PCA/UMAP | 原图网格、CNN feature map、ViT patch attention、saliency |
| 主要风险 | 幅度尺度可能掩盖任务表示；epoch 数和标签先验不完全一致 | 像素尺度和几何变化较直观，但模型可能依赖纹理或背景 |
| LoP 证据 | 固定预算下 fresh accuracy − warm accuracy，并结合多 seed、transition 和 retention | 同样的 fixed-budget fresh-gap 与表示/激活诊断 |

图像中一张 feature map 可以直接对应二维位置；EEG 的“位置”同时包含时间位置和电极通道，频谱又把时间结构变换到频率轴。因此 EEG 可视化通常要把同一数据画成三种视图：时域波形、频域功率、个体级统计空间。

## 结论边界

当前图像 smoke 只验证了 CNN/Transformer 的诊断接口，例如 CNN 存在 near-zero activation，Transformer 的 classifier-input CKA 会随阶段下降；它不是与 ISRUC 等规模和协议匹配的科学对照。当前完整 ISRUC 的图显示真实个体异质性，但所有架构和适应策略仍未通过严格 LoP gate。可视化说明“数据为什么难迁移”，不等于证明“LoP 已发生”。

后续实验优先比较 raw、RMS-equalized、channel-wise standardized 三种输入，并在相同 subject、seed、budget 网格上记录 fresh-gap、retention、频谱和表示变化。这样可以把采集增益漂移、频谱/形态漂移和标签先验漂移分开，而不是用单一 PCA 图或平均准确率作结论。
