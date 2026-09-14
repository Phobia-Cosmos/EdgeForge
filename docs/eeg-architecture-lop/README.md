# EEG visualization and fingerprint archive

本目录是 EdgeForge 仓库内的可视化与分析归档，换机器后可直接从 GitHub 浏览。这里不包含原始 EEG、模型 checkpoint 或服务器日志；所有 JSON、PNG 和特征矩阵都是从原始数据只读生成的派生结果。

每个含图片目录都有 `EXPLANATION.md`，说明图中坐标、选择规则和可支持的解释。完整 ISRUC 个体画像位于 [`subject-fingerprint-server-20260913/ISRUC-complete-all/`](subject-fingerprint-server-20260913/ISRUC-complete-all/)，FACED 画像位于 [`subject-fingerprint-server-20260913/`](subject-fingerprint-server-20260913/)。

## 推荐入口

- [可视化索引](../eeg-visualization-index.md)：按波形、频谱、漂移、PCA 和连续 sequence 浏览。
- [ISRUC 全量 CPU 报告](subject-fingerprint-server-20260913/ISRUC-complete-all/REPORT.md)：98 个被试、4,276 个 sequence、85,520 个 epoch。
- [ISRUC 全量解释](subject-fingerprint-server-20260913/ISRUC-complete-all/EXPLANATION.md)：每个被试和每个 sequence 的字段说明。
- [ISRUC 全量 PCA](subject-fingerprint-server-20260913/ISRUC-complete-all/complete-subject-fingerprint-pca.png)：每个点是一个 epoch，颜色为被试编号。
- [ISRUC sequence 漂移图](subject-fingerprint-server-20260913/ISRUC-complete-all/sequence-drift-by-subject.png)：比较每个被试内部的 sequence 变化。
- [FACED 报告](subject-fingerprint-server-20260913/FACED-REPORT.md)：123 个被试、3,444 个 trial。

## 可视化目录

- `clean-medium-sequence-waveforms-v1`：clean ISRUC medium 中 4 个目标个体、每个个体 2 个特征 sequence 的完整 20×30 秒波形和 8 通道热图。
- `condition-dose-curve-brainuicl-v1`：查看该目录的 `EXPLANATION.md`。
- `condition-dose-curve-tcn-v1`：查看该目录的 `EXPLANATION.md`。
- `condition-dose-curve-tcn-v2`：查看该目录的 `EXPLANATION.md`。
- `condition-rms-equalized-tcn-v1/visualizations-v1`：查看该目录的 `EXPLANATION.md`。
- `condition-rms-equalized-tcn-v1/visualizations-v2`：查看该目录的 `EXPLANATION.md`。
- `condition-target-channel-polarity-ch0-v1/visualizations-v1`：查看该目录的 `EXPLANATION.md`。
- `condition-target-gain-drift10-tcn-v1/visualizations-v1`：查看该目录的 `EXPLANATION.md`。
- `condition-target-gain-drift10-tcn-v1/visualizations-v2`：查看该目录的 `EXPLANATION.md`。
- `condition-target-gain-drift10-tcn-v1/visualizations-v3`：新版完整 30 秒波形、全部 epoch×8 通道热图和修正后的逐文件 adaptation/evaluation 边界。
- `condition-target-snr10-tcn-v1/visualizations-v1`：查看该目录的 `EXPLANATION.md`。
- `condition-target-snr10-tcn-v1/visualizations-v3`：查看该目录的 `EXPLANATION.md`。
- `condition-target-snr15-tcn-v1/visualizations-v1`：查看该目录的 `EXPLANATION.md`。
- `condition-target-snr15-tcn-v1/visualizations-v3`：查看该目录的 `EXPLANATION.md`。
- `condition-target-snr20-tcn-v1/visualizations-v1`：查看该目录的 `EXPLANATION.md`。
- `continuous-medium-v0.18.0-r2/visualizations-v1`：查看该目录的 `EXPLANATION.md`。
- `continuous-medium-v0.18.0-r2/visualizations-v2`：查看该目录的 `EXPLANATION.md`。
- `continuous-medium-v0.18.0-r2/visualizations-v3`：查看该目录的 `EXPLANATION.md`。
- `continuous-medium-v0.18.0-r2/visualizations-v4`：查看该目录的 `EXPLANATION.md`。
- `full-isruc-visualizations-v2`：查看该目录的 `EXPLANATION.md`。
- `full-isruc-visualizations-v3`：查看该目录的 `EXPLANATION.md`。
- `subject-fingerprint-server-20260913/ISRUC-complete-all`：服务器 CPU 全量画像，包含 98 个被试的完整 profile、每个 sequence 的统计、PCA 和 sequence 漂移图。
- `subject-fingerprint-server-20260913/ISRUC-balanced`：较小的 8-sequence 对照结果。
- `subject-fingerprint-server-20260913`：FACED 全量身份探针报告和 PCA 图。
