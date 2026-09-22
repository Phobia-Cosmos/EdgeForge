# EEG 输入归一化 pilot 结果

本地使用 ISRUC medium、TCN、3 个 seed（4321/4322/4323）、8 个 target transition 和预算 0/10/25/50 运行了三种输入协议。归一化只作用于输入张量，标签、通道顺序、epoch 边界和 warm/fresh 对照保持不变。

| 模式 | source accuracy（3 seed） | budget 10 fresh-gap | budget 25 fresh-gap | budget 50 fresh-gap | gate |
| --- | --- | ---: | ---: | ---: | --- |
| `none` | 0.2637 / 0.2013 / 0.1437 | +0.0200 | +0.0208 | -0.0075 | blocked-inconsistent-direction |
| `epoch_rms` | 0.5125 / 0.5850 / 0.6162 | -0.1908 | -0.1742 | -0.0533 | blocked-inconsistent-direction |
| `channel_zscore` | 0.5200 / 0.6513 / 0.4787 | -0.1467 | -0.1467 | -0.0092 | blocked-inconsistent-direction |

结论很明确：归一化显著改善了 source 学习准确率，但没有让 LoP fresh-gap 变成稳定正值。`epoch_rms` 在 budget 10/25 反而使 fresh-gap 更负；`channel_zscore` 在早期 transition 有正向单元，但整体仍混合方向。也就是说，当前问题不只是“输入数值太大导致训练失败”：尺度会影响 source 可学性和迁移难度，但稳定 LoP 还受到个体频谱、形态、标签先验及 warm 状态路径的共同影响。

结果目录：

- `/home/undefined/UbuntuData/ai-storage/EdgeForge/eeg-architecture-lop/normalization-epoch-rms-tcn-pilot-v1/`
- `/home/undefined/UbuntuData/ai-storage/EdgeForge/eeg-architecture-lop/normalization-channel-zscore-tcn-pilot-v1/`
- 原始对照：`/home/undefined/UbuntuData/ai-storage/EdgeForge/eeg-architecture-lop/continuous-medium-v0.18.0-r2/`

每个目录包含 `summary.json`、`audit-v1/tcn.md` 和 `audit-v1/trajectory-catalog.json`。完整 ISRUC 98 个体归一化作业仍在远端排队，完成后用于确认 medium pilot 的趋势是否在全量队列中保持。

