# ISRUC 全量 EEG 持续学习 LoP 实验方案 v0.19

## 1. 当前证据与本版本目标

远端同步的 `docs/eeg-continuous-lop.md` 记录的是开发性部分结果，不是全量 ISRUC 的 LoP 结论。该运行使用 diversity-selected 的 `v0.18.0` 子集：20 名受试者中 8 名用于 source、8 名用于 target、4 名用于 retention，每名仅取 5 个文件，共 2,000 个 EEG epochs；5 种架构、3 个 seed 形成 15 条 trajectory、120 个 target stage 和 600 个 stage-budget observation。严格 gate 没有被任何架构通过，结果保持 `scientific_conclusion_allowed=false`。

部分结果中，BrainUICL 在 budget 5 和 10 的聚合 accuracy fresh-gap 分别为 `+0.0875` 和 `+0.0408`，TCN 在 budget 10 和 25 分别为 `+0.0200` 和 `+0.0208`。这些均值包含零值或负值的 seed-stage cell，不能证明稳定 LoP；其余架构未出现一致方向也不能证明“不会发生 LoP”。该子集按标签多样性选文件、每名受试者只有 100 个 epochs、source 只训练 3 epochs，而且只有 3 个训练 seed，因此它只能说明数据、连续适应、fresh/warm 对照和审计链路已经跑通。

v0.19 的目标是用 ISRUC-Sleep Group I 的完整本地处理数据回答四个分离的问题：自然 subject stream 中是否出现随持续训练累积的 LoP；该现象是否在 Transformer 系模型和卷积模型间不同；target subject 的数据构成与出现顺序是否改变后续 plasticity；effective rank、stable rank、激活和梯度等状态量能否预测下一阶段的 fresh-gap。分类功效、旧任务 retention 和 LoP 必须分别报告，不能以遗忘或域偏移替代 LoP。

## 2. 研究问题、定义与结论边界

对 architecture `a`、训练 seed `s`、target 顺序 `o`、第 `t` 个 target subject 和固定更新预算 `B`，warm 分支与 fresh 分支在完全相同的 adaptation batch 顺序及 held-out target evaluation set 上比较：

$$
G^{\mathrm{acc}}_{a,s,o,t,B}=\operatorname{Acc}^{\mathrm{fresh}}_{a,s,o,t,B}-\operatorname{Acc}^{\mathrm{warm}}_{a,s,o,t,B}.
$$

$$
G^{\mathrm{loss}}_{a,s,o,t,B}=\operatorname{Loss}^{\mathrm{warm}}_{a,s,o,t,B}-\operatorname{Loss}^{\mathrm{fresh}}_{a,s,o,t,B}.
$$

两个量为正都表示 warm 模型在该固定预算下比 fresh 初始化更难学当前任务，是 LoP 的方向性信号。accuracy AULC 的辅助结果定义为：

$$
G^{\mathrm{AULC}}_{a,s,o,t}=\operatorname{AULC}^{\mathrm{fresh}}_{a,s,o,t}-\operatorname{AULC}^{\mathrm{warm}}_{a,s,o,t}.
$$

单个正 fresh-gap 只表示一个相对 plasticity deficit。要称为持续学习中的“plasticity loss”，还必须看到它随 stage 增长，或后 20% stage 相比前 20% stage 明显增大；若 gap 从第一个 stage 起即为正常数而没有时间趋势，更可能是 fresh/warm 初始化差异、预训练不足或任务难度差异，而不是持续训练造成的累积 LoP。`B=0` 只描述初始化状态，不进入 LoP 显著性检验。

本版本预注册以下问题：

1. `H1`：BrainUICL 在两个随机 target 顺序上都出现正的 late-minus-early fresh-gap，且整体 stage 趋势为正。
2. `H2`：Transformer 与 TCN 的趋势和效应大小是否与 BrainUICL 不同。该比较用于判断现象是否依赖 attention/Transformer 结构，不预设三种架构必须同向。
3. `H3`：在同一 target subject 集合、同一 source checkpoint 和同一 seed 下，按 label-shift 从低到高与从高到低排列会改变 fresh-gap trajectory。该检验采用双侧假设，不根据部分结果预设哪个方向更容易产生 LoP。
4. `H4`：进入当前 target stage 前的表示谱、激活和梯度状态，以及已经经历的累计 data-composition dose，能否预测当前 stage 的 fresh-gap。该部分是机制关联分析；没有干预时不能写成因果结论。

## 3. 全量数据与固定划分

canonical 数据位于 `/home/undefined/Disk/datasets/brainuicl/processed/isruc_group1_npy_float32`，不复制到 Git 仓库。完整处理集有 98 名受试者、4,276 个 data/label pairs 和 85,520 个 EEG epochs，受试者编号 `1..100` 中缺少 `8` 和 `40`。每个 data 文件为 float32 `[20,8,3000]`，配对 label 为 int64 `[20]`；估算 payload 为 8,210,604,160 bytes，约 8.21 GB 或 7.65 GiB。

角色划分由 `split_seed=20260904` 一次性生成，三个角色 subject-disjoint。计划摘要如下：

| Role | Subjects | Files | Epochs | 全量标签计数 `0..4` |
| --- | ---: | ---: | ---: | --- |
| source | 30 | 1,334 | 26,680 | `6188,3494,7909,5262,3827` |
| target | 50 | 2,152 | 43,040 | `9526,5525,14130,8563,5296` |
| retention | 18 | 790 | 15,800 | `3661,1887,5087,3038,2127` |

固定角色列表为：

- source：`3,5,6,11,12,14,15,16,20,21,28,32,39,41,44,45,46,48,50,56,58,59,64,68,70,73,76,78,94,100`
- target：`1,4,7,9,10,13,19,22,23,24,25,26,29,31,34,35,38,42,43,47,51,52,54,55,57,60,61,62,66,72,74,75,77,79,80,81,82,83,84,85,86,87,88,91,92,93,96,97,98,99`
- retention：`2,17,18,27,30,33,36,37,49,53,63,65,67,69,71,89,90,95`

`random-a` 顺序为 `35,10,79,98,22,62,26,80,82,55,54,75,92,38,61,84,24,57,29,81,4,66,72,25,91,87,60,97,88,52,43,96,1,85,86,9,23,31,47,7,34,19,74,51,77,99,93,13,42,83`；`random-b` 顺序为 `25,10,81,79,51,57,61,86,74,42,60,85,99,80,93,43,97,23,35,54,34,26,92,13,82,31,62,75,4,66,52,88,22,24,98,87,29,9,1,91,84,38,77,96,7,19,83,72,47,55`。二者是自然 LoP 的 primary orders。

data-composition 顺序以每个 target subject 的五类标签分布相对 source 聚合标签分布的 Jensen-Shannon divergence 排序。`label-shift-ascending` 为 `82,24,72,61,60,66,75,98,97,86,51,83,85,87,91,96,26,9,52,1,7,62,92,23,47,80,22,55,81,34,57,35,29,43,31,84,77,25,19,38,93,10,79,74,88,99,42,4,54,13`，`label-shift-descending` 是该列表的严格逆序。这个排序使用标签，只用于离线机制实验，不代表可部署的无标签排序策略。

当前 portable plan 的 `plan_digest` 为 `7503fb2fdc896f45a1ab19f2962d27b7227303a43859091d77fa9beafaa130d2`。该 digest 固定 split、顺序和标签 profile，但正式运行还必须另外生成覆盖每个 data 与 label payload 的 SHA-256 manifest；文件名、大小和标签 hash 不能替代 data 内容 hash。

source 的每名受试者按数值文件顺序把最后 `ceil(20%)` 文件作为 held-out source evaluation，其余文件用于预训练。target 的每个 20-epoch 文件固定以前 10 epochs 适应、后 10 epochs held-out evaluation；同一 target 文件的 evaluation 标签不得进入训练、模型选择、early stopping 或 data-composition feature。该设置测量的是“同一新受试者内的在线适应能力”，不是对未知受试者的零样本泛化；邻接 epochs 的时间自相关可能使绝对准确率偏乐观，但对完全配对的 fresh/warm 差值影响较小，仍须写入 limitation。retention 始终来自 18 名独立受试者，confirmatory 阶段用固定 seed 无放回抽取最多 4,096 epochs，并对所有分支复用同一索引；它只报告旧分布稳定性，不能满足 LoP outcome gate。

## 4. 架构与三阶段矩阵

实验由 calibration、confirmatory 和 data-composition 三阶段组成。任何 calibration 决策都必须先写入只增不改的 lock artifact，再开始读取 confirmatory fresh-gap；不能根据 confirmatory 结果换模型、seed、顺序、预算或学习率。

| Phase | Architectures | Seeds | Orders | Target stages | Source epochs | Budgets | 目的 |
| --- | --- | --- | --- | ---: | ---: | --- | --- |
| calibration | `lop_mlp,eegnet,tcn,transformer,brainuicl` | `4321..4323` | `random-a` | 前 12 | 10 | `0,5,10,25,50,100` | 验证训练充分性、显存、吞吐和指标，不产出 LoP 结论 |
| confirmatory | `tcn,transformer,brainuicl` | `4321..4330` | `random-a,random-b` | 全部 50 | 30 | `0,5,10,25,50,100,200` | 两个自然顺序上的正式估计 |
| data-composition | `transformer,brainuicl` | `4321..4330` | `label-shift-ascending,label-shift-descending` | 全部 50 | 30 | `0,5,10,25,50,100,200` | 数据构成和顺序机制实验 |

BrainUICL 是 primary model，因为它最接近项目原始 EEG 网络，也是要解释的主要对象；Transformer 保留相同的 attention/sequence inductive bias，但使用更简化的前端，用于定位 LoP 是否来自 Transformer 类表示动力学；TCN 是不含 attention 的扩张卷积对照，而且部分运行曾出现小的正 gap，值得在完整数据上检验其可重复性。`lop_mlp` 和 EEGNet 只保留在 calibration，分别充当 LoP 文献常用结构对照和轻量 EEG CNN 工程对照，不因 pilot 均值大小进入正式三架构矩阵。

三个正式架构不是通过 confirmatory 数据挑选的。所有架构都使用同一 raw EEG 输入、同一标签、同一 target split 和同一更新预算；参数量不同是研究因素，不应以相同 wall-clock time 伪装成相同 optimizer-step budget。结果同时报告参数量、每 step 时间和峰值显存，性能比较以固定 steps 为主、固定时间为敏感性分析。

## 5. 预训练与 fixed-budget probe

预训练固定使用 Adam、source learning rate `2e-3`、batch size `64`、30 epochs 和 gradient-norm clipping `1.0`；适应固定使用新建 Adam、learning rate `1e-3`、batch size `64`、最大 200 optimizer steps。Adam 的 betas、epsilon、weight decay、dtype、deterministic-algorithm 状态、PyTorch/CUDA/cuDNN 版本都必须落盘。若 source 每个 epoch 复用同一个确定性 permutation，metadata 必须明确记录 `single deterministic permutation repeated each epoch`，不能笼统写成每 epoch shuffle。

每个 architecture × seed 只训练一个 source checkpoint，并通过 SHA-256 在该 seed 的 random-a、random-b 以及适用的 composition orders 间复用。不能为不同顺序重新预训练“同 seed”但参数不同的 source checkpoint。10 个 seed 必须真实改变参数初始化、source batch ordering 和训练随机性；复制同一 checkpoint 或仅改变 probe seed 不构成独立 seed。

在 target stage `t` 开始时：

- warm 是同一 source checkpoint 起步、依次完成前 `t-1` 个 target subjects 后携带来的模型参数和 buffers；每一阶段以最大预算 200 steps 后的 warm 参数进入下一阶段。
- fresh 是相同 architecture 的新随机初始化，不读取 source 或历史 target 权重。fresh seed 必须由 `architecture + model_seed + target_subject` 稳定派生，而不能由 stage index 或 order 派生，这样同一 subject 在 random-a/random-b 中得到相同 fresh 对照。
- warm 和 fresh 使用同一 adaptation 样本、同一 mini-batch 顺序、相同 optimizer 超参数、相同 gradient clipping 及相同 evaluation 样本。二者的 probe optimizer 都在当前 stage 清空重建，因此 v0.19 测量的是 model parameter/buffer state 的 plasticity，不测量历史 Adam moments 造成的 optimizer-state LoP。
- fixed-budget probe 中冻结 BatchNorm running mean/variance，但 affine 参数保持可训练；否则不同历史的 batch statistics 会混入 LoP。是否冻结、受影响 module 数量和 buffer digest 必须写入 metadata。LayerNorm 不使用 running statistics，无需按 BatchNorm 处理。
- budgets 是同一条适应曲线上的 checkpoint，不是七次独立随机训练。`B=200` 的 warm 状态用于进入下一 subject；其余 budget 只做测量。

calibration 必须先验证“模型确实学会 source”和“fresh 分支在给定预算确实能学习”。最低 adequacy gate 使用配置中的两个阈值：held-out source accuracy 相对 source majority-class accuracy 的提升至少 `0.10`；fresh 从 `B=0` 到 calibration 最大预算的 accuracy gain 至少 `0.05`。同时报告 macro-F1、loss、confusion matrix 与每类召回率，防止多数类准确率通过但五分类能力失效。建议以 architecture 在 3 个 calibration seeds × 12 stages 上的中位数和 seed-cluster 95% CI 判定，而不是任取一个成功 cell。若正式三架构之一不通过 gate，只能修正训练/预算并创建新的 protocol/config 版本后重新跑全部 calibration，不能直接进入 confirmatory，也不能用 target confirmatory 数据调参。

## 6. 必须采集的指标

每个 target stage 在适应前、每个 budget 后以及进入下一 stage 前都要记录明确的 `architecture/seed/order/stage/subject/budget/split/metric_role`。原始预测至少保存可重建的 confusion counts、样本数与聚合指标；仅保存四位小数会丢失配对检验所需信息。

| 类别 | 必需指标 | 用途与边界 |
| --- | --- | --- |
| Primary outcome | warm/fresh accuracy、`fresh_gap`、正确数/总数 | 固定预算下的 LoP 主结果 |
| Concordance outcomes | CE loss、macro-F1、`fresh_loss_gap`、`fresh_macro_f1_gap`、accuracy/loss/MF1 AULC | 检查主结果是否只由类别不平衡或单点噪声造成 |
| Classification utility | source held-out、每个 target held-out、retention 的 ACC/MF1/loss、confusion matrix、每类 recall | 确认模型仍有原始 EEG 五分类功效 |
| Representation spectrum | 每个注册 tap 的 effective rank、normalized effective rank、stable rank、normalized stable rank、spectral entropy、tail energy、rank90/95/99、rank ceiling | LoP predictor，不单独定义 LoP；固定 observation 数与 feature axis |
| Activation | mean/std/norm、near-zero fraction、saturation fraction；仅 ReLU 报 dead fraction | GELU/ELU 的 near-zero 不能命名为 dead neuron |
| Gradient/Fisher | supervised CE gradient norm、pairwise cosine、negative-cosine fraction、gradient effective rank、empirical Fisher proxy | 测量更新方向可用性；必须记录 objective、label source 和 sample count |
| Drift/parameters | parameter L2、checkpoint delta、linear CKA、Procrustes residual、weight spectrum | 区分简单旋转、缩放与表示塌缩 |
| Attention | entropy、max probability、head diversity、normalization axis/length | 只用于 Transformer/BrainUICL；BrainUICL 的 `softmax(dim=1)` 不能与标准 key-axis entropy 直接比较 |
| Optional expensive | sampled parameter Jacobian/NTK proxy、HVP top eigenvalue/Hutchinson trace | 固定小 calibration reservoir，在预注册 stages 采样；不是完整 Hessian |
| Data composition | 每 subject 类别计数/entropy/majority fraction、adapt/eval label JS、相对 source label JS、signal RMS、per-channel RMS、temporal-difference RMS、距 source feature center | 解释 task difficulty 和累计 shift dose，不使用 held-out label 调参 |
| Operations | elapsed time、data loading time、samples/s、GPU peak memory、CPU RSS、device UUID、异常/重试次数 | 资源核算和编译/运行时后续接入 |

谱和激活至少在 Conv/frontend、fusion/token、最后一个 sequence block 和 classifier input 四个语义位置采集；无对应层的架构记录 `unavailable + reason`，不得填 0。`[B,C,L]` Conv 输出以 `C` 为 feature axis，`[B,T,D]` token 以 `D` 为 feature axis。所有谱比较使用同一固定 calibration reservoir、相同中心化设置、相同 `max_observations`，并保存 reservoir manifest digest。

机制分析只允许用 stage 开始前的 warm 指标预测当前或下一阶段 outcome，例如 `ER(t-1) -> fresh_gap(t,B)`；不能把适应完当前 task 后测得的 ER 与同一适应过程的 gap 拼成“预测”。target adaptation labels 可用于 CE gradient/Fisher，但应写成 `label_source=true`；held-out target labels只用于 outcome evaluation。完整 Hessian 不在主矩阵逐 cell 计算，建议在 stages `0,10,20,30,40,49`、每个 seed 的固定小样本上运行 HVP/Fisher，以免昂贵诊断改变主实验吞吐和失败率。

## 7. 数据构成变量与干预边界

自然数据构成分析同时保留当前 subject shift 与历史累计 dose。令 `d_t` 为当前 target subject 相对 source 的 label/feature distance，累计 dose 可写为：

$$
C_t=\frac{1}{t-1}\sum_{j=1}^{t-1}d_j,\qquad t>1.
$$

正式表中至少报告 `d_t`、`C_t`、最近 5 个 stage 的 rolling mean、类别熵和 train/eval label JS。ascending/descending 顺序使用相同的 50 名 target subjects，因此 order 主效应和 `order × stage` 交互反映历史构成差异，而不是样本集合差异。两种 composition orders 必须复用相同 architecture-seed source checkpoint、subject-keyed fresh 初始化、retention reservoir 和 batch construction。

v0.19 不把人工 noise、channel dropout、重采样或标签重加权混入主矩阵。先确定自然全量流中是否有 LoP 和哪些 composition covariates 与其相关，再在新版本中对候选变量做单因素、label-preserving 干预。这样可以避免把直接降低当前输入质量的 domain-shift degradation 错写成模型 plasticity loss。

## 8. 统计分析与多重比较

primary analysis 使用所有 10 个独立训练 seed 和两个 random orders，不把 50 个 stage 当作 500 个独立随机重复。以 budget 为 categorical variable、stage 归一化到 `[0,1]`，对 fresh-gap 建立 crossed mixed-effects model：

$$
G_{a,s,o,t,B}=\beta_0+\beta_a+\beta_B+\beta_\tau\tau_t+\beta_o+\beta_{a\times\tau}+\beta_{B\times\tau}+\beta_{o\times\tau}+u_s+v_{q(t)}+w_{a,s,o}+\epsilon_{a,s,o,t,B}.
$$

其中 `u_s` 是 seed random intercept，`v_{q(t)}` 是 target-subject random intercept，`w_{a,s,o}` 是 trajectory random intercept；同一 trajectory 内的 stage residual 使用 AR(1) 或 cluster-robust covariance。模型必须报告估计值、standard error、95% CI、标准化 effect size 和收敛诊断，不只报告 p-value。若 mixed model 不收敛，预注册 fallback 为按 seed-order 汇总 late-minus-early contrast，再做 seed-paired bootstrap，而不是不断更换模型直到显著。

LoP 的主要时间对比是最后 10 个 target stages 与最前 10 个 target stages 的平均 fresh-gap 差。BrainUICL 为 primary architecture；Transformer 和 TCN 是预注册的结构对照。每个正预算 `5,10,25,50,100,200` 的 architecture × budget planned contrasts 构成同一 family，使用 Holm correction，`alpha=0.05`；`B=0`、loss/MF1/AULC、retention、单层诊断和未注册 covariate 都是 supporting 或 exploratory，不与主 outcome 混成新的“阳性”机会。

bootstrap 固定 `10,000` repeats。主 bootstrap 以 seed 为 cluster，每次抽取完整 seed bundle，保留该 seed 的两个随机顺序、全部 stages、budgets 和架构，以维护序列依赖与配对关系。报告 percentile 95% CI，并以 BCa 或 wild-cluster bootstrap 作为只有 10 个 seed 时的敏感性分析；禁止把 stage cell 独立重采样来虚增样本量。subject random effect 支持对当前 50-subject cohort 的异质性描述，但由于顺序状态依赖，不宣称简单 subject bootstrap 给出了任意 ISRUC 人群的因果效应。

data-composition analysis 对 ascending/descending 使用相同 mixed model，并重点检验 `order × stage`、late-minus-early 的 paired difference 和整条 gap trajectory 的 area difference。机制指标使用 lagged regression：先加入 architecture、budget、stage、subject difficulty 和 data-composition covariates，再加入前一 stage 的 normalized effective/stable rank、near-zero/dead fraction 与 gradient effective rank；报告增量解释度和 seed-cluster CI。层、指标和 lag 很多，因此机制结果采用 Benjamini-Hochberg FDR，并统一标为 exploratory association。

结论分为四种，不允许把“不显著”写成“不存在”：

- `supported natural LoP`：BrainUICL 的 late-minus-early gap 为正、Holm-adjusted CI 排除 0、两个 random orders 同向，且 source/fresh adequacy gates 通过；loss 或 AULC 至少一个同向支持。
- `order-sensitive plasticity`：order interaction 明确，或两个 random orders 方向相反。此时只能说 LoP 依赖数据顺序，不能给出无条件总体结论。
- `no detectable LoP at tested budgets`：效应 CI 包含 0 且实验完整。该表述不等于证明 LoP 不存在。
- `positive transfer / warm advantage`：fresh-gap 稳定为负且 CI 排除 0。仍需排除 fresh 学习不足后才能解释为 warm advantage。

现有 EdgeForge strict gate 要求每个 seed、每个 transition 都为正，适合作为极保守 robustness check，但不作为唯一科学检验；生物信号中少数负 cell 不应自动否定总体趋势。strict gate、mixed model 和 bootstrap 三套输出都要保留，若结论不同应并列解释，不能只选择有利的一套。

## 9. 运行规模与资源预算

| Phase | Trajectories | Target stages | Stage-budget cells | Warm+fresh adaptation steps |
| --- | ---: | ---: | ---: | ---: |
| calibration | `5×3×1=15` | 180 | 1,080 | 36,000 |
| confirmatory | `3×10×2=60` | 3,000 | 21,000 | 1,200,000 |
| data-composition | `2×10×2=40` | 2,000 | 14,000 | 800,000 |
| 合计 | 115 | 5,180 | 36,080 | 2,036,000 |

每个 stage-budget cell 含 warm 和 fresh 两个 evaluation branch，因此 36,080 不是独立样本数。source checkpoint 按 architecture × seed 缓存；confirmatory 只需 30 个 source pretrains，composition 中的 BrainUICL/Transformer 复用其中 20 个，不重复训练。正式统计的独立训练随机重复仍为 10 个 seed。

本机 RTX 4070 SUPER 有 12 GiB 显存。当前 runner 会在 host 侧读取约 8.21 GB payload，并构造 tensor、文件清单和 retention reservoir，建议至少 24 GiB 可用 RAM，32 GiB 更稳妥；GPU batch `64` 必须先由 calibration 实测，OOM 时只能在 lock 前统一降低 batch 并重新 calibration。结果、环境、日志、source/stage checkpoints 和分析 artifact 建议预留至少 50 GiB，共享输出放在 `/home/undefined/Disk/ai-storage/EdgeForge/eeg-lop-full/v0.19.0/`，不能放进 Git 或 dataset 目录。

在没有完整 calibration timing 前不承诺具体小时数。完成 calibration 后按架构记录 `source_seconds`、`warm_step_seconds`、`fresh_step_seconds` 和每个 budget 的 evaluation seconds，并用下式生成带 20% 余量的 ETA：

$$
T_{\mathrm{phase}}=\sum_{r\in\mathrm{trajectories}}\left(T^{\mathrm{source}}_r+\sum_{t=1}^{N_r}\left(200T^{\mathrm{warm-step}}_{r,t}+200T^{\mathrm{fresh-step}}_{r,t}+T^{\mathrm{eval}}_{r,t}\right)\right).
$$

单卡执行顺序固定为 calibration → confirmatory random-a → confirmatory random-b → composition ascending → composition descending。优先完成一个 order 的全部 10 seeds 会造成时间批次与 order 混杂，因此实际调度应在 architecture-seed 粒度交错两个 orders，并记录开始时间、GPU temperature/power 与软件版本。正式运行至少需要 stage 级原子 checkpoint/resume；只在 50-stage trajectory 末尾写 `run.json` 会使中断后整条 seed 重算，不能作为长实验验收状态。

## 10. 执行流程

先生成只读 plan，不复制全量数据：

```sh
PYTHONPATH=src /home/undefined/Disk/python-envs/brainuicl/bin/python \
  scripts/plan-isruc-full-lop.py \
  --source-root /home/undefined/Disk/datasets/brainuicl/processed/isruc_group1_npy_float32 \
  --output logs/archive/v0.19.0/isruc-full-plan/plan.json \
  --split-seed 20260904 --source-count 30 --target-count 50
```

任何训练前先 dry-run calibration，检查展开命令和外部输出目录：

```sh
PYTHONPATH=src /home/undefined/Disk/python-envs/brainuicl/bin/python \
  scripts/run-eeg-lop-full-experiment.py \
  --plan logs/archive/v0.19.0/isruc-full-plan/plan.json \
  --config config/eeg-lop-full-isruc-v0.19.0.json \
  --data-root /home/undefined/Disk/datasets/brainuicl/processed/isruc_group1_npy_float32 \
  --output-root /home/undefined/Disk/ai-storage/EdgeForge/eeg-lop-full/v0.19.0 \
  --phase calibration --device cuda
```

calibration gate 和 protocol lock 通过后才加 `--execute`。confirmatory 与 data-composition 分开执行并可恢复：

```sh
PYTHONPATH=src /home/undefined/Disk/python-envs/brainuicl/bin/python \
  scripts/run-eeg-lop-full-experiment.py \
  --plan logs/archive/v0.19.0/isruc-full-plan/plan.json \
  --config config/eeg-lop-full-isruc-v0.19.0.json \
  --data-root /home/undefined/Disk/datasets/brainuicl/processed/isruc_group1_npy_float32 \
  --output-root /home/undefined/Disk/ai-storage/EdgeForge/eeg-lop-full/v0.19.0 \
  --phase confirmatory --device cuda --execute

PYTHONPATH=src /home/undefined/Disk/python-envs/brainuicl/bin/python \
  scripts/run-eeg-lop-full-experiment.py \
  --plan logs/archive/v0.19.0/isruc-full-plan/plan.json \
  --config config/eeg-lop-full-isruc-v0.19.0.json \
  --data-root /home/undefined/Disk/datasets/brainuicl/processed/isruc_group1_npy_float32 \
  --output-root /home/undefined/Disk/ai-storage/EdgeForge/eeg-lop-full/v0.19.0 \
  --phase data-composition --device cuda --execute
```

当前 orchestrator 是否已满足 BN freeze、subject-keyed fresh seed、共享 source checkpoint 和 stage-level resume，必须由测试和 dry-run manifest 确认；缺一项时只允许 calibration，不允许启动正式矩阵。命令示例表达目标协议，不能用 CLI 能退出 0 代替科学协议验收。

## 11. 日志、版本与不可覆盖产物

外部输出建议使用以下结构，每个 phase/run 只增不改：

```text
v0.19.0/
├── plan/
│   ├── isruc-full-plan.json
│   ├── dataset-sha256-manifest.json
│   └── validation-report.json
├── protocol/
│   ├── config.json
│   ├── calibration-lock.json
│   ├── git-state.json
│   └── environment.json
├── source-checkpoints/<architecture>/seed-<seed>/
├── runs/<phase>/<order>/<architecture>/seed-<seed>/
│   ├── metadata.json
│   ├── stages/<stage>-<subject>.json
│   ├── checkpoints/
│   ├── stdout.log
│   └── stderr.log
├── analysis/
│   ├── completeness.json
│   ├── mixed-effects.json
│   ├── bootstrap.json
│   ├── multiple-comparison.json
│   └── report.md
└── SHA256SUMS
```

`environment.json` 至少保存 git commit、dirty diff digest、Python/PyTorch/NumPy/CUDA/cuDNN 版本、GPU UUID、driver、hostname、deterministic flags 和完整命令；`metadata.json` 保存 plan/config/dataset/source-checkpoint/calibration-reservoir digest、seed 派生规则、subject order、split、budget、optimizer 和 BatchNorm policy。每个 stage 完成后先原子写临时文件再 rename，checkpoint 与 stage metric 均带 run signature。失败、超时、OOM 和重试日志不得删除；重试创建新的 attempt ID，并通过 parent run ID 关联。

Git 只归档小型 plan、config、命令、测试、聚合结果、分析报告和 `SHA256SUMS`，不提交 8 GB dataset、模型 checkpoint 或逐样本预测。每次改变 split、模型配置、BN policy、fresh 定义、预算或统计模型都创建新实验版本，不能覆盖 v0.19.0；部分运行和 null result 同样保留。raw run artifact 永远保持 `scientific_conclusion_allowed=false`，只有独立 completeness/statistical audit 通过后才能生成带结论资格的派生 report。

## 12. 停止条件与最终验收

以下任一情况立即停止当前 phase 并保留失败证据：data/label 缺配对、shape 或标签越界、98-subject/4,276-file/85,520-epoch 总量不符、角色交叉、plan 或 payload digest 改变；出现 NaN/Inf、同 signature 重跑不一致、BN policy 或 fresh seed 规则未记录；磁盘不足以原子落盘、GPU OOM、source/fresh adequacy gate 未通过；任一 planned seed/order 缺 stage 且无法由相同 signature 恢复。protocol failure 只能修复后创建新 attempt 或新版本，不能跳过失败 cell 后继续做显著性检验。

实验采用固定样本量，不因中间 gap 显著、方向不利或“看起来已经稳定”提前停止。至少完成 10 个真实 independent seeds、每个 random order 的 50 stages 和三种正式架构后才执行 primary analysis。composition phase 可在 confirmatory 完整后单独报告，但不能反向改变 confirmatory 协议。对于 paired order comparison，任一 seed 的一侧不完整时，该 seed 的两侧都不进入 paired test，并必须先尝试恢复；不得用新 seed 局部补齐。

最终验收清单如下：

- 数据验收：98 subjects、4,276 pairs、85,520 epochs，source/target/retention 为 `30/50/18` 且互斥，四个 order 与 plan digest 一致，完整 payload SHA-256 通过。
- 训练验收：source checkpoint 对跨 order 的 digest 一致，10 个 seed 参数确实不同，held-out source 与 fresh-learning adequacy gate 通过，没有使用 target evaluation 标签调参。
- 对照验收：warm/fresh batches、eval set、optimizer 和预算一致，fresh seed 对 subject 跨 order 一致，BN running stats 冻结，optimizer state scope 明确。
- 完整性验收：confirmatory 为 60/60 trajectories、3,000/3,000 stages、21,000/21,000 stage-budget cells；composition 为 40/40、2,000/2,000、14,000/14,000；每项均有 signature、日志和 hash。
- 指标验收：primary ACC gap、loss/MF1/AULC、source/target/retention utility、注册层谱、激活、梯度以及 data-composition covariates齐全；unavailable 指标有原因而不是静默填零。
- 统计验收：mixed model 收敛或使用预注册 fallback，10,000 次 seed-cluster bootstrap 完成，Holm/FDR family 明确，effect size、CI、调整后 p-value、missingness 和 sensitivity analyses 全部保存。
- 结论验收：报告严格区分 LoP、forgetting、domain shift 与分类功效；没有通过上述 gate 时只能发布 `incomplete`、`blocked` 或 `no detectable evidence`，不能发布“已证明/已否定 LoP”。

完成本版本后，下一步不是立即制造更强的数据扰动，而是先根据自然流结果选出稳定的 data-composition predictor。如果 label JS、class entropy、signal distance 或累计 shift dose 能在两个随机顺序和多个 seed 上预测 late-stage fresh-gap，再创建 v0.20 的单因素干预矩阵；如果自然 LoP 不明显，则优先延长 task stream、增加每阶段更新总量或采用更深的原始 BrainUICL 配置，并保持 fresh control 和分类 adequacy gate，而不是通过减少数据或削弱 fresh baseline 人为制造阳性结果。
