# BrainUICL CUDA Compiler 复验（2026-09-15）

## 结论

真实 ISRUC Group I subject 1 sequence 0 与 BrainUICL pretrain seed 4321 checkpoint 已在 RTX 4070 SUPER 上重新完成 `transform → torch.export → compile → run → correctness → benchmark`。eager 和关闭 `pattern_matcher` 的安全 Inductor profile 使用相同输入、checkpoint 与 Graph IR；两份导出图均为 284 个节点，graph digest 为 `7fc77e44d6f5b6bc3094d876c670c423de05939cf376f821eea84cbbc76e5d2b`，输入 shape 为 `[1,20,8,3000]`，输出 shape 为 `[1,5,20]`。

| Backend | correctness | 最大绝对误差 | 平均绝对误差 | compile-stage first call | 100 次均值 | 100 次中位数 | p95 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CUDA eager | PASS | 0 | 0 | 107.435 ms | 1.042 ms | 0.935 ms | 2.088 ms |
| CUDA Inductor safe (`pattern_matcher=false`) | PASS | 0.000041008 | 0.000003447 | 3964.279 ms | 0.768 ms | 0.688 ms | 1.061 ms |

在本次固定 shape、单样本序列和热态重复采样中，安全 Inductor 的平均与中位 steady latency 都约为 eager 的 `1/1.36`，即约 1.36 倍速度提升；代价是显著更高的首次编译/调用延迟。该结论只用于 Compiler pipeline 回归基线，不能推广到训练、多 batch、动态 shape、其它 subject/checkpoint 或模型能力差异。

## 输入与证据

- 数据：`/home/undefined/Disk/datasets/brainuicl/processed/isruc_group1_npy_float32/1/data/0.npy`
- Checkpoint：`/home/undefined/Disk/ai-storage/BrainUICL/model_parameter/ISRUC/Pretrain/*_parameter_4321.pkl`
- 模型源码：`/home/undefined/Desktop/bci/code/tta_security/BrainUICL`
- eager Artifact：`.edgeforge/brainuicl-eager-cuda-20260915/`
- Inductor Artifact：`.edgeforge/brainuicl-inductor-safe-cuda-20260915/`
- PyTorch：`2.9.1+cu130`
- GPU：NVIDIA GeForce RTX 4070 SUPER，12282 MiB
- Driver：`580.173.02`

两个 Artifact 目录分别保存 `brainuicl.pt2`、`ir.json`、reference/compiled output、transform/compile/run/correctness/benchmark JSON。没有修改 BrainUICL 源码、checkpoint 或 EEG 数据。

## 解释边界

安全 profile 延续既有 Gate 决策：默认 Inductor 的自定义 attention pattern 曾出现不可接受的数值偏差，因此当前不能仅为了更高性能重新启用 `pattern_matcher`。本次 safe profile 在 `rtol=0.002`、`atol=0.0002` 下通过，最大误差约 `4.10e-5`；这证明固定输入的 logits parity，不等同于 accuracy/MF1、持续学习能力或多 seed 科研结论。

下一步应从 eager 与 safe Inductor 分别采集 `torch.profiler`/CUDA trace，定位 Conv1d、Attention、LayerNorm/归一化或数据布局中的真实热点，再把被确认的热点映射到 Operator IR、Triton Kernel 和 Auto Tuning。没有 profiler 证据前，不应仅凭算子类型预先实现 Kernel；开发板模型部署不进入这条实验链。
