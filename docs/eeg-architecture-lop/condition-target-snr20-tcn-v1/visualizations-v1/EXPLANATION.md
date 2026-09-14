# visualizations-v1

本目录只保留图片；机器可读文件已归档到 `eeg-architecture-lop-analysis-20260911/` 下的同一相对路径。
这些图片是描述性 EEG/LoP 诊断，不能单独作为 LoP 结论；LoP 必须查看对应实验的 fresh-gap 严格 gate。

## 图片说明

- `eeg-drift-labels.png`：标签先验图：显示睡眠阶段标签比例，不是模型激活证据。
- `eeg-drift-pca.png`：PCA 图：把 RMS、频带功率等特征投影到低维，用于观察个体/条件分离。
- `eeg-drift-rms-trajectory.png`：RMS 图：显示每个 epoch 的整体信号强度轨迹。
- `eeg-drift-spectra.png`：频谱/PSD 图：比较 0.5–40 Hz 频带能量分布；PSD 是功率谱密度。
- `eeg-drift-waveforms.png`：波形图：比较个体或条件下的时域 EEG 形状与幅度。
