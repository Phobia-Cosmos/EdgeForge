# 图表说明

`subject-invariance-comparison.png`：每个标签对应一个波形特征矩阵或 BrainUICL tap。median ICC 越高，说明同一被试不同 sequence 的 profile 越稳定；identity accuracy 是完整 sequence 留出后的身份可分性。两者必须一起看，单独高准确率可能来自睡眠阶段或采集条件混杂。

`waveform-feature-family-icc.png`：将 295 维按绝对增益、尺度敏感、增益相对比值、空间连接、谱形状和形态学分组，显示每组的中位 sequence-level ICC。绝对功率高不代表真正个体不变，可能只是电极阻抗或放大器增益。

本审计使用 sequence median，而不是把相邻 epoch 当作独立样本；因此图中的稳定性更接近跨 sequence 复现。ICC 仍不是跨 session 的最终结论，后续需使用明确 session 标签、class-balanced probe 和增益/睡眠阶段校正复核。
