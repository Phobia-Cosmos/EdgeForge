# EEG LoP analysis archive

这是与 `docs/eeg-architecture-lop/` 配套的机器可读分析归档，包含 LoP gate、质量审计、架构/策略对比、diagnostics 和实验设计报告。它已纳入 EdgeForge 仓库，换机器后可直接浏览；不包含原始 EEG 或模型 checkpoint。

当前优先入口是 `subject-id-probe-20260915/`、`subject-id-granularity-20260915/` 和 `subject-id-cl-tracking-20260915/`：它们分别回答编码表征中的身份可解码性、epoch/sequence 粒度与未知被试拒识，以及持续学习 checkpoint 是否改变身份信息。大型 epoch embedding 和逐被试全 epoch 图不是长期归档格式；仓库保留 sequence embedding、摘要、manifest 和可重生成脚本，避免重复占用空间。

`analysis-configs-20260911/` 保存各实验的 JSON/报告快照，`organization-manifest.json` 记录原始分析结果到可视化目录的对应关系。

## 可视化目录

- `condition-dose-curve-brainuicl-v1`：查看该目录的 `EXPLANATION.md`。
- `condition-dose-curve-tcn-v1`：查看该目录的 `EXPLANATION.md`。
- `condition-dose-curve-tcn-v2`：查看该目录的 `EXPLANATION.md`。
- `condition-rms-equalized-tcn-v1/visualizations-v1`：查看该目录的 `EXPLANATION.md`。
- `condition-rms-equalized-tcn-v1/visualizations-v2`：查看该目录的 `EXPLANATION.md`。
- `condition-target-channel-polarity-ch0-v1/visualizations-v1`：查看该目录的 `EXPLANATION.md`。
- `condition-target-gain-drift10-tcn-v1/visualizations-v1`：查看该目录的 `EXPLANATION.md`。
- `condition-target-gain-drift10-tcn-v1/visualizations-v2`：查看该目录的 `EXPLANATION.md`。
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
- `subject-id-probe-20260915`：冻结 BrainUICL 表征的 ISRUC/FACED subject-ID probe 与层级结果。
- `subject-id-granularity-20260915`：epoch、sequence、k-epoch 聚合和 open-set known/unknown 结果。
- `subject-id-cl-tracking-20260915`：持续学习前后身份 probe 的 tracking 报告。
