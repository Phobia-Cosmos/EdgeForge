# EEG decoder registry CPU/CUDA smoke（2026-09-15）

## 结论

本机共享 `brainuicl` 环境可识别 NVIDIA GeForce RTX 4070 SUPER，PyTorch 为 `2.9.1+cu130`，NVIDIA driver 为 `580.173.02`，显存为 12282 MiB。合成 EEG registry smoke 已分别在 CPU 和 CUDA 上完成 13 种 decoder：`atcnet`、`brainuicl`、`cnn_lstm`、`conformer`、`deepconvnet`、`eeg_graph`、`eegnet`、`fbcnet`、`lop_mlp`、`shallowconvnet`、`tcn`、`transformer`、`tsception`。

两份结果都包含 13 个 run、20000 条受上限约束的 EdgeForge metric envelope，递归检查没有 `status=error`。输出继续标记 `scientific_conclusion_allowed=false`：它只验证模型 registry、forward/backward、representation、Jacobian/NTK、gradient、fixed-budget probe 和结果序列化链路，不提供真实 EEG、LoP 或架构优劣结论。

## 执行入口

```sh
cd /home/undefined/Desktop/EdgeForge
PYTHONPATH=src /home/undefined/Disk/python-envs/brainuicl/bin/python \
  scripts/eeg_lop_diagnostics.py \
  --data synthetic-eeg --architectures all \
  --tasks 3 --train-samples 24 --eval-samples 16 \
  --epochs 1 --batch-size 8 --length 64 --channels 4 --classes 3 \
  --probe-steps 0,1,2 --device cuda \
  --output-dir logs/lop-diagnostics-registry-smoke-cuda-20260915
```

CPU 结果位于 `logs/lop-diagnostics-registry-smoke/`，CUDA 结果位于 `logs/lop-diagnostics-registry-smoke-cuda-20260915/`。`logs/` 按仓库规则不进入 Git；本记录保存可复验命令、环境和验收结论。

## CUDA 兼容问题与修复

第一次 CUDA 执行在 `cnn_lstm` 的诊断 backward 阶段失败：诊断器为冻结 dropout/BatchNorm 调用了 `model.eval()`，而 cuDNN RNN 的 eval-mode forward 不保留 backward 所需 reserve space，因而抛出 `cudnn RNN backward can only be called in training mode`。训练本身没有失败，CPU 也没有该约束。

`sampled_parameter_jacobian` 与脚本的 gradient probe 现在只在这类有界诊断 forward/backward 内使用 `torch.backends.cudnn.flags(enabled=False)`，让 LSTM/GRU 走 PyTorch native CUDA 实现；模型的 eval 语义以及调用前 training flag 均保持并恢复。没有通过把整个模型切回 train mode 绕开问题，因此 dropout 和 BatchNorm 诊断策略未改变。

新增两个 CUDA 回归测试：eval-mode CUDA LSTM 的 sampled Jacobian，以及脚本 gradient probe。使用 `unittest` 执行相关测试共 15 项全部通过；共享环境未安装 `pytest`，因此没有修改环境或临时安装依赖。

## 后续实验边界

下一阶段应把这一 plumbing smoke 替换为受版本控制的真实 ISRUC/FACED calibration manifest 与 BrainUICL checkpoint，对同一输入执行 eager/CUDA Inductor correctness、能力指标和性能对照；只有 Profiler 确认热点后才增加 Triton Kernel 或自动调优候选。开发板模型部署和板端推理引擎不属于该实验链。
