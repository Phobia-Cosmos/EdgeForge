# EEG 架构功效与 LoP 初步实验 v0.17.0

本实验回答一个先决问题：将 BrainUICL 的 EEG 解码器替换为参数量更小的 MLP、1-D CNN 或浅层 Transformer 后，是否仍能保持分类功效，以及这些结构是否表现出与论文一致的 plasticity loss（LoP）信号。它是 EdgeForge 的本地 CPU 复现实验，不把小样本结果写成论文结论。

## 研究设计

同一 ISRUC 文件划分、预处理、标签、优化器、训练步数和 seed 用于所有候选结构。第一轮使用 source subjects `1,3,4`，target subject `2`，retention subject `5`；每个 subject 只取按文件编号排序的第一个配对文件，共 5 个文件、100 个 epoch。数据由 `scripts/create-eeg-mini-split.py` 从共享磁盘只读抽取到 `/home/undefined/Disk/datasets/edgeforge-eeg-mini/v0.17.0`，不进入 Git 工作区。

架构阶梯如下：`lop_mlp`（MLP baseline）、`eegnet`（深度可分离 CNN）、`tcn`（扩张 1-D CNN）、`transformer`（patch-token Transformer）和 `brainuicl`（双分支 CNN + Transformer reference）。Registry 中的其它模型可在第二轮加入。为了公平，第一轮统一使用 raw EEG 输入 `[B,C,S]`；已有 `simple-transformer-lop.py` 的 64 维手工时频特征结果单独标记，不能与 raw EEG 结果混为同一比较。

每个模型固定 `input_length=3000`、`in_channels=8`、`num_classes=5`，在 source 文件上训练固定 epoch/step budget，再在 target 文件上执行同一 supervised-oracle adaptation。每个 target 阶段保留 checkpoint model 与重新初始化的 fresh model，使用相同 target batch 顺序和 adaptation budget。正式 LoP 结论仍需要至少三个独立 seed、多个 checkpoint stage、完整 target stream 和 old-task retention。

## 指标与判定

分类功效报告 accuracy、macro-F1、cross-entropy、参数量、CPU 单 batch latency 和峰值进程内存（可用时）。准确率保持不能只看最后一个数：应报告 source、target fresh、target checkpoint 和 retention 的完整曲线，并使用相同 subject split。

LoP 主 outcome 定义为 `fresh_gap = Acc_fresh - Acc_checkpoint`。辅助指标包括 AULC、old-task retention/BWT、effective rank、stable rank、梯度/激活统计和 attention entropy（仅有 attention 的结构）。effective rank 与 stable rank 是 checkpoint 条件下的表示诊断，不单独证明 LoP 因果机制；eNTK、FTLE、Hessian/Fisher 属于后续扩展。

结果分四级解释：`model-level correctness`（输出和指标契约正确）、`classification utility`（功效达到预设阈值）、`exploratory LoP`（固定设计下观察到正 fresh gap/遗忘趋势）和 `formal scientific conclusion`（跨 seed、跨阶段并完成统计检验）。本版本最多支持前三级中的前两级和探索性信号。

## 可复现命令

```sh
PYTHONPATH=src python3 scripts/create-eeg-mini-split.py \
  --source-root /home/undefined/Disk/datasets/brainuicl/processed/isruc_group1_npy_float32 \
  --output-root /home/undefined/Disk/datasets/edgeforge-eeg-mini/v0.17.0

PYTHONPATH=src /home/undefined/Disk/python-envs/brainuicl/bin/python \
  scripts/benchmark-eeg-architectures.py \
  --data-root /home/undefined/Disk/datasets/edgeforge-eeg-mini/v0.17.0 \
  --output-root /home/undefined/Disk/ai-storage/EdgeForge/eeg-architecture/v0.17.0 \
  --architectures lop_mlp eegnet tcn transformer --seeds 4321 4322 4323
```

输出目录包含 `metadata.json`、每个 architecture/seed 的 `run.json` 和不可覆盖的 `summary.json`。真实 EEG 不上传 GitHub；仓库只保存抽取器、manifest schema、合成 fixture 和本实验说明。

## 与 BrainUICL 的比较边界

BrainUICL 原始结构约 6.6M 参数，使用 EEG/EOG 双分支 1-D CNN 生成 512 维 epoch embedding，再复用 8-head Transformer block 和 MLP 分类头。这里的轻量候选模型主要用于结构和 LoP 机制对照，参数量与输入前端不同，因此“准确率相同”必须在相同 raw EEG 输入和预算下单独验证，不能把简单 Transformer 的手工时频特征结果当成 BrainUICL 等价替代。
