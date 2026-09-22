# Metric-driven EEG LoP induction

该流程根据已有的输入质量、fresh/warm gap 和 checkpoint 诊断选择下一步数据条件。所有变换都在独立目录生成，原始 ISRUC 不修改；任何单个条件的正向平均 gap 都不能直接称为 LoP。

## 当前证据

在 TCN、三 seed、8 个 target transition 的结果中，预算 25 的候选排序如下：

| 条件 | 正/零/负 stage-seed | 平均 fresh-gap | 质量审计 |
| --- | ---: | ---: | --- |
| RMS equalization | 19/3/2 | +0.1058 | 通过 |
| RMS equalization + baseline drift10 | 16/5/3 | +0.1008 | 通过 |
| SNR15 noise | 15/4/5 | +0.0925 | 通过 |
| gain drift10 | 14/4/6 | +0.0708 | 通过 |
| cross-talk10 | 14/4/6 | +0.0442 | 通过 |
| clean | 15/2/7 | +0.0208 | clean 对照 |

这说明 RMS equalization 与轻度低频 drift 是当前最有希望的输入条件，但仍有零值和负值，严格 gate 不能通过。统一 CAR、通道正交旋转、整 epoch 时间反向和 signed montage 置换均没有改善一致性；其中 TCN 的预算 25 平均 fresh-gap 分别为 -0.1575、-0.1792、-0.2242、-0.1442（对应三 seed 的扩展 probe），应停止继续加大这些方向。

候选排序机器结果在 `/home/undefined/UbuntuData/ai-storage/EdgeForge/eeg-architecture-lop-analysis-20260911/induction-probes/metric-plan-b25/PLAN.md`，新增条件的信号质量审计在同目录 `quality-*` 下。

## 指标驱动的调整规则

先用质量 gate 筛选：标签、形状、epoch 边界和有限值必须保持；轻度条件优先要求文件级波形相关系数至少 0.98，所有 RMS/PSD 改变都记录。再按相同 subject 顺序、seed、optimizer、budget 比较 fresh-gap。候选必须在多个 transition 和 seed 同时保持正向，不能只选最佳 budget。

机制 gate 需要在 gap 之前出现可重复的表示变化：TCN/GELU 不称永久 dead，而记录 near-zero、低方差、梯度 nonzero fraction、feature effective/stable rank、奇异值谱和参数相对更新；EEGNet/ReLU 才额外记录 active/dormant/dead coverage。最后做 reset/ReDo 或冻结反事实：如果 reset 不能降低 fresh-gap，输入扰动更可能只是 transfer sensitivity。

## 下一轮

当前推荐顺序是：

1. 保留 RMS equalization，使用 target-only baseline drift5/10 做单因素对照；
2. 固定三个 seed、八个 transition 和 budget 0/10/25/50，补齐 checkpoint diagnostics；
3. 针对出现零 gap 的 subject 11、14 和负向 subject 17，增加 transition 内的 sequence 数，而不是继续增大扰动剂量；
4. 只有在 activation/gradient/rank 指标先于 gap 改变，并且 reset 反事实有效时，才把该条件送入完整 ISRUC/A100 复验。

当前最完整的 RMS+baseline10 诊断位于 `induction-probes/tcn-rms-baseline10-diagnostics/`：rank 在不同 transition 间上下波动、near-zero fraction 为 0、梯度仍可达，尚无 LoP 机制证据。因此目前结论是“已找到高 transfer-sensitivity 候选”，尚未找到稳定 LoP 方法。
