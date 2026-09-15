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
- `condition-target-gain-drift10-tcn-v1/visualizations-v3`：新版 30 秒波形、聚合漂移/PCA/频谱图和修正后的逐文件 adaptation/evaluation 边界；逐被试全 epoch 大图不纳入仓库，按需由 `scripts/visualize-eeg-drift.py` 重生成。
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
- `expanded-fingerprint-audit-20260914`：medium ISRUC 的 295 维扩展波形特征 PCA 与特征组消融。
- `subject-invariance-audit-20260914`：按 sequence 聚合的波形特征与 BrainUICL 各层个体不变性、ICC 和增益不变子集对照；大矩阵与 JSON 位于分析归档。
- `brainuicl-subject-stability-audit-20260914`：BrainUICL 随机初始化架构分层审计；对应大矩阵位于分析归档。
- `brainuicl-subject-stability-trained-pilot-20260914`：5 epoch medium source-supervised BrainUICL 分层稳定性 pilot；对应大矩阵位于分析归档。
- `../eeg-architecture-lop-analysis-20260911/subject-id-probe-20260915`：真实 ISRUC/FACED 的冻结表征 subject-ID probe，包含原始输入、CNN branch、fusion、Transformer 各层、classifier input 和 logits。
- `../eeg-architecture-lop-analysis-20260911/subject-id-granularity-20260915`：epoch/sequence 粒度、k-epoch 聚合和 known/unknown open-set 审计。
- `../eeg-architecture-lop-analysis-20260911/subject-id-cl-tracking-20260915`：ISRUC pretrain 与持续学习 checkpoint 的 identity tracking 对照。
