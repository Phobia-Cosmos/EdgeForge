# EEG 基础网络的 LoP 指标基线

本文档把 LoP 论文中的指标落到常见 EEG 解码结构，并给出一个不依赖真实数据挂载的可复现实验入口。模型组件位于 [`src/edgeforge/eeg_models/`](../src/edgeforge/eeg_models/)，轻量汇总指标位于 [`src/edgeforge/lop_metrics.py`](../src/edgeforge/lop_metrics.py)，曲率与统一报告位于 [`src/edgeforge/lop_diagnostics.py`](../src/edgeforge/lop_diagnostics.py)，envelope 转换位于 [`src/edgeforge/lop_envelope.py`](../src/edgeforge/lop_envelope.py)，实验入口位于 [`scripts/eeg_lop_diagnostics.py`](../scripts/eeg_lop_diagnostics.py)。脚本默认生成合成 EEG/图像流，不下载数据、不读取外部 checkpoint，也不会把 smoke 结果升级成 LoP 科学结论。

## 1. 先固定 LoP 的主结果

LoP 不能由 effective rank、死单元或权重范数单独定义。对第 `t` 个 checkpoint，在同一个新任务、同一个 train/eval split、同一个优化器和更新预算 `B` 下，分别从当前状态（warm）和 fresh initialization 开始适应：

```text
fresh_gap_final(t,B)      = Acc_fresh(t,B) - Acc_warm(t,B)
fresh_loss_gap_final(t,B) = Loss_warm(t,B) - Loss_fresh(t,B)
fresh_auc_gap(t,B)        = AULC_fresh(t,B) - AULC_warm(t,B)
```

对应的 Markdown 数学源为：

$$
\Delta_{\mathrm{acc}}(t,B)=\operatorname{Acc}_{\mathrm{fresh}}(t,B)-\operatorname{Acc}_{\mathrm{warm}}(t,B),
$$

$$
\Delta_{\mathrm{loss}}(t,B)=\operatorname{Loss}_{\mathrm{warm}}(t,B)-\operatorname{Loss}_{\mathrm{fresh}}(t,B).
$$

`fresh_gap_final > 0` 表示 fresh 模型在有限预算后更好，是 LoP 候选信号；`fresh_gap_final < 0` 表示 warm 状态更好，通常是正迁移或 warm-start 优势。它们都必须跨多个阶段和至少三个独立 seed 重复。`acc_gain` 只是 warm 模型适应前后的增益，不能替代 fresh gap。

固定预算 probe 默认冻结 BatchNorm running statistics。否则 warm/fresh 的差异可能来自 batch 统计状态，而不是参数可塑性。

## 2. 当前 BrainUICL 的结构映射

现有 BrainUICL 的自然测量点如下。ISRUC 使用 `[B,20,8,3000]`，其中 20 是 epoch token、8 是 EEG/EOG 通道；FACED 使用 `[B,20,32,2500]`。两者的 sequence length、采样率和通道物理含义不同，不能为了拼表而把它们当作同一个输入空间。

```text
raw EEG sequence
  → Conv1d frontend
  → fusion representation       [B, 20, 512]
  → Transformer representation   [B, 20, 512]
  → classifier input            [B, 20, 128]
  → logits                      [B, 20, classes]
```

ISRUC 的 frontend 是 EEG/EOG 双分支 Conv1d，两个 512 维分支拼接后经 `Linear(1024,512)`；FACED 是单分支 32 通道 Conv1d。Transformer 配置为 `d_model=512`、8 heads、3 次复用同一个自定义 attention block；BrainUICL 实际使用 `softmax(dim=1)` 跨 head 归一化，不是标准 Transformer 的 key-axis softmax。分类头是 `512→256→128→classes`。

在这个结构上建议按以下顺序采集：

| 位置 | 张量布局 | feature axis | 主要指标 | 解释边界 |
| --- | --- | ---: | --- | --- |
| Conv block 输出 | `[B,C,L]` | `1` | effective/stable rank、near-zero、权重范数、梯度 | `C` 是通道特征，时间位置是 observation；不能把 `L` 当 feature 维 |
| fusion | `[B,T,D]` | `2` 或 `-1` | token 谱、token norm、CKA、Procrustes | 需固定 `T=20` 和 calibration 文件数 |
| Transformer 每层 | `[B,T,D]` | `2` 或 `-1` | 谱、attention key entropy、head diversity、gradient | 标准 attention 与 BrainUICL 的归一化轴要分开记录 |
| classifier input | `[B,D]` 或 `[B,T,D]` | `-1` | Jacobian/NTK 代理、谱、漂移 | 最后层 feature-factor 不是完整参数 Hessian |
| logits | `[B,K]` 或 `[B,T,K]` | `-1` | loss/ACC/MF1、校准、fresh gap | 主 LoP outcome 来自 fresh-vs-warm probe |

## 3. 指标的统一含义

### 3.1 表征谱

给定表示 `X`，先按照显式 `feature_axis` 变成 `[N,D]`，可选按 feature 中心化，再计算奇异值 `σ_i`。脚本同时输出 algebraic rank ceiling `min(N,D)`、`observation_count` 和 `feature_dim`，避免不同 calibration 预算下直接比较原始 effective rank。

```text
p_i = σ_i / Σ_j σ_j
effective_rank = exp(-Σ_i p_i log p_i)
stable_rank = Σ_i σ_i² / σ_1²
normalized_effective_rank = effective_rank / min(N,D)
tail_energy = 1 - σ_1² / Σ_i σ_i²
```

effective rank 低说明谱能量更集中，但不自动说明性能低；它必须和 fresh gap、任务难度、观测数、CKA 以及梯度可用性一起解释。对 EEG 至少报告三种层级：epoch-token（`B×20` 个 observation）、sequence-pooled（`B` 个 observation）和 subject/session aggregate。

### 3.2 Jacobian/NTK

脚本支持一个有界的 sampled parameter Jacobian：对每个样本取模型输出的标量均值，计算该标量对所有可训练参数的梯度，得到 `[samples, parameters]` 矩阵，并报告其谱和 `J Jᵀ` 的 NTK 代理（JSON 中 `ntk` 对 kernel 矩阵本身做 SVD，同时保留 `ntk_eigenvalues`）。它不是完整 logits Jacobian，也不是精确 Hessian/GGN；在真实 EEG 上应记录 `sample_count`、`parameter_dim` 和 scalar selector。

如果只分析线性分类器 `z=Wh`，`h` 的 Gram/谱是最后层 feature-factor 的精确部分；这仍然不能代表 Conv、Transformer、BatchNorm 和 optimizer state 的全参数曲率。

`parameter_spectral_summary(model)` 是与之相反的 checkpoint-only probe：Linear/Conv weight 按 `[out, -1]` 展平后做 SVD，报告 `sigma_max`、condition number、effective/stable rank 和截断奇异值；bias、LayerNorm scale 等一维参数标记为 skipped。它不接受 calibration batch，因此不能用来替代表示谱或 NTK。

### 3.3 激活与表示漂移

ReLU 的 zero fraction 可以称为 dead fraction；GELU、ELU、LayerNorm 输出只称 near-zero fraction 或 saturation fraction。标准差使用 `E[x²]-E[x]²`，不能误写成 `E[x²]-E[|x|]²`。raw cosine/Frobenius 漂移会把表示旋转误判成变化，因此脚本额外输出同一 calibration batch 上的 linear CKA 和 Procrustes residual。

### 3.4 梯度、attention 和参数

梯度摘要包含 batch gradient norm、pairwise cosine、负余弦比例和 gradient effective rank；它们必须注明 objective（监督 CE、伪标签、consistency、NT-Xent 或 CPC）。empirical Fisher 只是对应 objective 的平方梯度 proxy。

标准 `nn.MultiheadAttention` 的 attention 权重通常沿最后一个 key/token 轴归一化；BrainUICL 的自定义实现沿 `dim=1` 跨 head 归一化。脚本在结果中保存 `normalization_axis` 和 `normalization_length`，不能跨实现直接比较 entropy 数值。无 attention 的 EEGNet/TCN 不应人为填充 attention 指标。

参数 L2 和 checkpoint delta 描述状态变化，不是 Adam 的真实 moment/update 轨迹。若 checkpoint 没有 optimizer state，应明确记录 unavailable。

序列输入有一个容易漏掉的轴约定：内置 decoder 会把 `[B,T,C,S]` 展平为 epoch batch，再把 bundle 表征恢复为 `[B,T,...]`。因此卷积 tap 的 spec 带 `sequence_axis_policy="prepend"`，诊断器会把单 epoch 布局中的 `feature_axis=1` 自动调整为 `2`；token/embedding 的 `feature_axis=-1` 不变。`MetricConfig.feature_axes` 是最终覆盖值，适合自定义布局或刻意测量序列轴；结果同时记录 `axis_source` 和 `sequence_axis_adjusted`，避免把 `T` 误当成通道特征。

曲率入口是 `hessian_vector_product`（`g = ∇θ L`，`H·v = ∇θ(g·v)`）、`hessian_top_eigenvalue_summary`（power iteration）、`hutchinson_trace_summary`（Rademacher trace estimator）和小模型专用 `exact_hessian_matrix`。它们的结果必须绑定 `objective`、batch、label/pseudo-label source、reduction、train/eval/BN 状态；同一网络在换 calibration 数据或换 CE/NT-Xent/consistency 后得到的是不同 Hessian。完整 Hessian 的存储/计算通常随参数数目平方增长，生产实验优先使用 HVP、top eigenvalue、trace 或最后一层/GGN 近似。

## 4. 常见 EEG 解码器的测量点

### EEG Transformer

先把每个 epoch 切成 temporal patch，得到 `[B,T,D]` token，再通过标准 key-axis attention 或 BrainUICL attention。重点观察：

```text
patch_embed → token_norm → encoder.layer_i → sequence_pool → classifier_input
```

如果后期 `fresh_gap_final` 变大，同时 encoder/classifier-input 的 normalized ER、NTK 小特征值或 gradient rank 下降，才可以提出“谱/梯度退化与可塑性同步”的机制假设。还要检查 attention entropy 是否只是 softmax 轴或温度变化造成。

### EEGNet

建议保留 temporal convolution、depthwise spatial convolution、separable convolution 和 classifier input。对 `[B,C,L]` 输出使用 `feature_axis=1`，时间位置作为 observation；记录每层通道谱、ReLU/ELU near-zero、BN 状态漂移和参数更新。EEGNet 没有 attention，因此 attention 状态应为 `unavailable`。

### DeepConvNet / TCN / CNN-LSTM

DeepConvNet 按 conv block 逐层记录；TCN 要分别记录不同 dilation block 的 `[B,C,L]` 谱和时间梯度冲突；CNN-LSTM 需区分 recurrent hidden token 与最终 pooled decoder。若只保留最后分类向量，会漏掉长时间依赖方向的谱退化。

## 5. 本次可复现实验

### EEG smoke

在共享 `research` 环境中运行（`--architectures all` 会枚举当前 registry 的全部 EEG decoder）：

```bash
cd /home/undefined/Desktop/EdgeForge
PYTHONPATH=src /home/undefined/UbuntuData/python-envs/research/bin/python \
  scripts/eeg_lop_diagnostics.py \
  --data synthetic-eeg --architectures all \
  --tasks 3 --train-samples 24 --eval-samples 16 \
  --epochs 1 --batch-size 8 --length 64 --channels 4 --classes 3 \
  --probe-steps 0,1,2 --max-observations 64 \
  --output-dir logs/lop-diagnostics-smoke-all
```

结果文件写入命令指定的 `--output-dir`，并包含 architecture、representation taps、attention axis、Jacobian/NTK 和（启用时）Hessian provenance。不同 decoder 的参数量和前端归纳偏置不同；这只是验证适配器是否可运行，不是模型规模公平比较。

JSON 同时写出 `metrics[]` 和 `envelope.schema=edgeforge-bundle-v1`。每一行包含 `namespace/name/value/step/context`，其中 `context.metric_role` 明确区分 `predictor`（rank-like spectrum）、`outcome`（fixed-budget fresh-vs-warm）、`retention`（旧任务评估）和 `diagnostic`（NTK/Hessian/Fisher/gradient/attention 等）。因此可以把该 JSON 作为 `edgeforge-bundle-v1` 的结果文件交给 Worker；`runs` 树仍保留完整曲线和诊断细节。`metrics[]` 是存储格式，不会改变“只有 fresh gap 才是 LoP 主 outcome”的科学口径。

若需要在 smoke 中验证曲率接口，可对单一小模型开启有限的 matrix-free probe：

```bash
PYTHONPATH=src /home/undefined/UbuntuData/python-envs/research/bin/python \
  scripts/eeg_lop_diagnostics.py --data synthetic-eeg --architectures brainuicl \
  --tasks 2 --train-samples 8 --eval-samples 8 --epochs 1 --batch-size 4 \
  --length 32 --channels 4 --classes 3 --probe-steps 0,1,2 \
  --hessian-mode power --hessian-iterations 4 --hessian-probes 2 \
  --output-dir logs/lop-diagnostics-hessian-smoke
```

在这个小样本、单 seed 测试中，Transformer 在后一个阶段的 `fresh_gap_final` 约为 `-0.0625`，TCN 约为 `+0.5`，EEGNet 约为 `0`。这说明 fixed-budget gap 对架构、任务阶段和随机样本都很敏感，也说明不能用一次正 gap 宣称 LoP。各层 ER、CKA、梯度和 attention 只提供后续机制分析的观测量。

### 图像对照

```bash
PYTHONPATH=src /home/undefined/UbuntuData/python-envs/research/bin/python \
  scripts/eeg_lop_diagnostics.py \
  --data synthetic-image --architectures all \
  --tasks 3 --train-samples 24 --eval-samples 16 \
  --epochs 1 --batch-size 8 --classes 3 \
  --probe-steps 0,1,2 --max-observations 64 \
  --output-dir logs/lop-diagnostics-image-smoke
```

图像结果在 [`synthetic-image-lop-diagnostics.md`](../logs/lop-diagnostics-image-smoke-v2/synthetic-image-lop-diagnostics.md)。它的作用是验证相同谱/Jacobian/CKA API 可以用于 `[B,C,H,W]` CNN 和 patch Transformer；它不能替代 EEG 的生理信号实验。

如果需要一个真实但无需联网下载的图像数据对照，可以使用 scikit-learn 自带的 8×8 Digits：

```bash
PYTHONPATH=src /home/undefined/UbuntuData/python-envs/research/bin/python \
  scripts/eeg_lop_diagnostics.py \
  --data digits --architectures all \
  --tasks 3 --train-samples 32 --eval-samples 16 \
  --epochs 1 --batch-size 8 --classes 4 --image-size 8 \
  --probe-steps 0,1,2 --max-observations 64 \
  --output-dir logs/lop-diagnostics-digits-smoke
```

该数据只用于验证图像模型和指标接口；任务之间的旋转/亮度变化是受控 domain shift，不等于标准 Class-IL。

## 6. 接入真实 ISRUC/FACED

当前容器没有文档中所写的 `/home/undefined/Disk/datasets/brainuicl/...` 原始 NPY 和 `/home/undefined/Disk/ai-storage/...` checkpoint，因此没有伪造真实 EEG 运行。接入时应写一个 adapter，将：

```text
ISRUC: [B,20,8,3000] → BrainUICL sequence forward → [B,20,classes]
FACED: [B,20,32,2500] → dataset-specific frontend → [B,20,classes]
```

映射到同一结果 schema。若使用本脚本的单 epoch decoder，可以将 `[B,20,C,S]` 展平为 `[B×20,C,S]`，但必须把 `epoch` 标签、sequence index 和 subject/session 写入 context；不能把展平后的 epoch 当成独立 subject。若使用 BrainUICL 的 sequence Transformer，保留 `T=20`，并分别测 epoch-token 与 sequence-pooled 表示。

正式实验建议固定：

```text
datasets       = ISRUC, FACED（分开训练与报告）
architectures  = BrainUICL Transformer, EEGNet, DeepConvNet/TCN
stages         = 0, 10, 25, 49（最后 stage 只做终点谱/retention，不做 next-task probe）
seeds          = 4321, 4322, 4323
probe budgets  = 0, 5, 10, 25, 50
calibration    = 同一 subject/session 文件清单和 digest
```

每个 stage 需要保存 checkpoint digest、architecture/version、数据 manifest digest、subject/session、label source、objective、BN policy、optimizer 配置和 probe budget。每个阶段回访固定 old subjects，才能同时报告 retention/BWT；只看新 subject 的适应曲线不能测遗忘。

## 7. 结论和下一步

这套基线先回答“在基础 EEG 网络上，哪些 LoP 指标能被稳定、逐层、可审计地测出来”：谱指标适用于 Conv/token/decoder，但必须显式声明 feature axis 和观测预算；Jacobian/NTK 先使用有界 sampled 或 last-layer proxy；attention 只在模型真实暴露权重且轴一致时解释；固定预算 fresh gap 才是 LoP 的主要 outcome。

下一步应先在 clean ISRUC/FACED 上生成三个真实 seed 的连续 checkpoint，再加入 replay/EWC/SSL 方法，最后才做 noise、TTA 或投毒压力。任何“同时改善遗忘和 LoP”的说法都必须分别给出 old-task forgetting/BWT、fresh-gap/AULC 和逐层机制指标。
