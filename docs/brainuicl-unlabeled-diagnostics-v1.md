# BrainUICL 无标签诊断 v1

`scripts/brainuicl-unlabeled-diagnostics.py` 是 EdgeForge 侧的只读 adapter。它只载入 EEG `data/*.npy` 和 checkpoint，不读取 `label/*.npy` 的内容，不执行 optimizer update，因此输出描述的是在线无标签 adapter 在适应前能观察到的信号，而不是 supervised plasticity outcome。

默认输出包括 predictive entropy、归一化 entropy、max-probability/confidence、pseudo-label class coverage、固定相对噪声扰动下的 prediction agreement、概率 KL 和 Transformer representation cosine。每项都带有 `labels_loaded=false`、`optimizer_update=false` 和 protocol 字段；标签路径只作为 provenance 字符串记录，便于复现实验文件身份。

示例：

```bash
/home/undefined/Disk/python-envs/brainuicl/bin/python \
  scripts/brainuicl-unlabeled-diagnostics.py \
  --brainuicl-root /home/undefined/Desktop/bci/code/tta_security/BrainUICL \
  --dataset ISRUC \
  --checkpoint-root /home/undefined/Disk/ai-storage/BrainUICL/model_parameter/ISRUC/Pretrain \
  --data-root /home/undefined/Disk/datasets/brainuicl/processed/isruc_group1_npy_float32 \
  --subject 2 --checkpoint-stage 0 --max-files 2 --max-batches 1 \
  --noise-severity 0.05 --device cuda:0 \
  --output logs/v0.14.0/raeeg/unlabeled-stage0.json
```

2026-08-23 ISRUC subject 2 pretrain smoke 的 normalized predictive entropy 为 `0.50197`，mean max-probability 为 `0.71406`，noise-0.05 prediction agreement 为 `1.0`；FACED subject 1 pretrain smoke 对应为 `0.30097`、`0.75749`、`0.85`。这是无标签可用性基线，不代表模型已经完成 adaptation，也不能替代 fixed-budget fresh-gap outcome。后续正式实验应把这些诊断与同一 checkpoint 的 supervised offline evaluation 配对，并报告 calibration/coverage 随 shift severity 的变化。
