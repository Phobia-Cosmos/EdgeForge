# BrainUICL checkpoint instrumentation v1

`scripts/brainuicl-instrumentation.py` 是 EdgeForge 中的只读 BrainUICL 适配器。它只导入外部 BrainUICL 的网络定义，读取 checkpoint 和 `/home/undefined/Disk/datasets/` 中的处理后 EEG，不会修改 BrainUICL 源码、checkpoint 或数据集。当前支持 ISRUC `(20,8,3000)` 和 FACED `(20,32,2500)` 两种输入契约。

## 快速运行

使用共享的 `brainuicl` 环境（运行前先确认 `/home/undefined/Disk/README.md` 中的环境登记）：

```bash
/home/undefined/Disk/python-envs/brainuicl/bin/python \
  scripts/brainuicl-instrumentation.py \
  --brainuicl-root /home/undefined/Desktop/bci/code/tta_security/BrainUICL \
  --dataset ISRUC \
  --checkpoint-root /home/undefined/Disk/ai-storage/BrainUICL/model_parameter/ISRUC/Pretrain \
  --data-root /home/undefined/Disk/datasets/brainuicl/processed/isruc_group1_npy_float32 \
  --subject 1 --seed 4321 --checkpoint-stage 0 \
  --device cuda:0 --max-files 1 --max-batches 1 --importance-batches 1 \
  --output logs/v0.13.0/brainuicl-instrumentation/stage-0-subject-1.json
```

要测量持续学习 checkpoint 相对于 pretrain 的 representation drift，加入 `--baseline-checkpoint-root`：

```bash
--checkpoint-root experiments/regularization_cl_eeg_runs/clean49_bn_frozen_e10_lr1e6_seed4321/finetune/checkpoints/individual_10 \
--baseline-checkpoint-root /home/undefined/Disk/ai-storage/BrainUICL/model_parameter/ISRUC/Pretrain \
--checkpoint-stage 10
```

`--checkpoint-stage` 是 LoP 分析的 checkpoint stage，写入每个 `task.*` 指标的 `step`；它不是 fixed-budget probe 的 optimizer step。相同的 EEG 文件必须用于 current/reference 两次前向，脚本会检查 representation shape 并记录输入和 checkpoint 的 SHA-256。

脚本还输出 activation near-zero/dormancy 诊断、梯度 Gram/cosine（至少两个 calibration batch 时）、最后线性层 Jacobian feature-factor、有限差分局部线性度、checkpoint 参数更新范数，以及 canonical aliases：`task.attention.layer_1.{entropy,head_entropy,max_probability,offdiag_mass}`、`task.importance.<block>.{mean,max,nonzero_fraction}` 和 `task.weight_norm.<block>.l2`。BrainUICL 代码实际复用同一个 attention module 三次，因此 `layer_1` 是历史兼容的汇总别名，不表示三个独立参数层；主 attention 归一化轴仍记录为 `dim=1`。

## 输出指标

输出同时提供一个 `tasks` 行和 EdgeForge `metrics` envelope。默认 LoP predictor 是唯一的 `task.spectra.transformer_1.effective_rank` 行，因此可以直接交给 `raeeg-metrics-v1` 归一化器。

### Spectrum

`fusion`、`transformer_1` 和 `classifier_input` 都会输出 effective rank、stable rank、spectral entropy、归一化 spectral entropy、rank90/95/99、最大奇异值、Frobenius norm、观测数和特征维度。表示张量按最后一维作为特征维度，其余维度展平为观测，并默认先按特征维度中心化。

对于奇异值 `σᵢ`，脚本使用 `pᵢ = σᵢ / Σⱼσⱼ`，effective rank 为 `exp(−Σᵢ pᵢ log pᵢ)`，stable rank 为 `Σᵢ σᵢ² / σ₁²`。它们是无量纲谱统计，不是样本数量。

### Representation drift

只有提供 `--baseline-checkpoint-root` 时才计算 drift。每个组件会保存 mean L2、RMS、relative Frobenius、mean cosine similarity/distance 和 centroid L2。没有 baseline 时结果明确标记为 `not-computed`，不会把“没有参考”伪装成零漂移。

### Attention entropy

BrainUICL 的 `MultiHeadAttention` 不返回 attention probability，且实现的是 `softmax(dim=1)`。脚本通过 forward pre-hook 读取同一模块的 Q/K 权重重建该概率，不改外部源码；输出的主指标是跨 head 轴的 entropy，并同时保存 conventional key-axis entropy 作为诊断。报告必须保留 `normalization_axis=1`，不能把它解释成标准 key-normalized attention entropy。

### Importance

`task.importance.*` 是在给定 EEG 标签上计算的 empirical Fisher proxy：对交叉熵梯度取平方并按参数平均，另存 block/module 聚合和 top-parameter 列表。它不是精确 Fisher，也不是无标签 BrainUICL adaptation 的因果重要性；跨方法比较时必须固定 subject、label source、batch 数和 checkpoint stage。

### Weight norms

`task.weight_norms.*` 包括 FeatureExtractor、TransformerEncoder、SleepMLP 的 L2 范数、平方 L2 和参数数目，并提供 module 级聚合。它描述参数状态，不等于 representation drift 或重要性。

### Checkpoint parameter update

提供 `--baseline-checkpoint-root` 时，`task.parameter_updates.*` 逐 block 比较当前与参考 checkpoint 的参数状态，记录 global/block `delta_l2`、`reference_l2`、`relative_update`，以及 top-parameter 的 cosine-to-reference。它不是 optimizer 的逐步轨迹；没有保存 optimizer state 时不能把该值解释为学习率、动量或实际更新路径。该诊断用于把“表示变化很大”与“参数确实大幅移动”分开审计。

### Optimizer-state provenance

`task.optimizer_state` 和 `source.optimizer_state_provenance` 只扫描 checkpoint 目录中名称包含 `optimizer`、`adam`、`momentum` 或 `scheduler` 的候选文件，记录 path、size 和 SHA-256，并固定 `loaded=false`。当前 BrainUICL milestone checkpoint 通常只有模型参数和 `regularizer_state.pt`，因此会明确返回 `status=unavailable`；这意味着可以比较参数快照，但不能从它们重建 Adam 一二阶矩或完整学习率轨迹。fixed-budget probe 另外记录 probe-local Adam 的 lr、weight decay、update steps 和 `state_persisted=false`，该信息属于 diagnostic，不是历史 checkpoint optimizer state。

### Fixed-budget outcome

`scripts/brainuicl-fixed-budget-probe.py` 在固定 sequence split、优化器、学习率和步数下，分别从旧 checkpoint 与 fresh initialization 运行 held-out supervised-oracle probe，输出 train/test loss、accuracy、macro-F1、AULC 以及 final/AULC fresh gap。它是测量 LoP gap 的 outcome adapter，不是无标签在线算法；要做正式结论必须在每个 stage、每个 method 使用相同 split/probe budget，并至少运行 3 个真实 seed。

可选的 `--retention-data-root` 与重复的 `--retention-subject` 只增加旧任务评估集。参数更新仍只使用当前目标训练批次，retention labels 仅用于 eval；输出 `task.forgetting.*`，在 matrix bundle 中标记为 `metric_role=retention`，不能替代 LoP outcome 或完整 BWT。

如果需要构造受控数据压力测试，可使用 `scripts/generate-raeeg-shift.py` 生成不覆盖源数据的 amplitude、noise、channel-dropout、time-jitter 或 bandstop 派生数据。输出 manifest 保存 source/output/label SHA-256 和变换参数。数据 shift 只能作为预注册的自变量，不能把“shift 后性能下降”直接叫作 LoP；必须与 clean-data、fresh-init 和 equal-budget controls 配对。

## 当前实测基线

在 PyTorch 2.9.1+cu130、RTX 4070 SUPER、seed 4321、ISRUC subject 1 的单文件 probe 上，pretrain stage 0 的 Transformer effective rank 为约 `7.84498`，stable rank 为约 `2.37688`，实际 `softmax(dim=1)` attention entropy 均值为约 `1.66083`（归一化约 `0.79869`），全局参数 L2 为约 `43.2692`。FACED `sub-001` pretrain smoke 的对应 effective rank 约 `10.14631`、stable rank 约 `2.01167`。这些是单 subject、单文件的 instrumentation smoke 结果，不是 LoP 证据。

将 finetune `individual_10` 与 pretrain 对比时，Transformer mean representation L2 drift 约 `56.2362`、mean cosine distance 约 `0.24097`、relative Frobenius 约 `0.74972`。这说明脚本能够捕获 checkpoint 间变化，但不能单凭一次漂移推断塑性丧失或 LoP。

## LoP 使用边界

要形成 `effective_rank(t−1) → plasticity.acc_gain(t)` 的可审计配对，需要为同一 method、subject order、split、probe budget 和真实 seed 保存连续 checkpoint stage，并在下一 subject 上运行固定预算 fresh-init/held-out probe。当前脚本只负责 checkpoint-level instrumentation；它不会把 spectrum 与 plasticity 自动拼成科学结论，也不会用 tolerance 或缺失数据填充 predictor。
