# EEG 上的 LoP 统一解释与 FACED/ISRUC 实验计划

本文把 `LoP论文统一数学原理与推导.md` 的问题转换成 EdgeForge 可以执行的测量协议。这里的 LoP（Loss of Plasticity）指模型处于长期训练状态后，在相同的新任务、相同的数据划分、相同的优化器和相同的更新预算内，相比 fresh initialization 更难学习，而不是一次普通的准确率下降。

## 1. 统一数学对象的直观解释

对一批 EEG 输入 `X`，模型输出 logits `f_θ(X)`。参数 Jacobian `J = ∂f_θ(X)/∂θ` 是一个“参数方向 → 输出变化”的线性地图：第 `j` 列描述只改变第 `j` 个参数一点点时，所有样本 logits 如何变化。它不是输入数据本身，也不是 Transformer 专有对象。

“谱”是矩阵的奇异值或对称核的特征值集合。若 `K = J Jᵀ = Q Λ Qᵀ`，`qᵢ` 是第 `i` 个输出空间方向，`λᵢ` 是该方向的可学习强度。对新任务的初始残差 `e₀ = f_θ(X_new) − y_new`，系数 `cᵢ = qᵢᵀe₀` 就是残差在该方向上的投影。`rank` 只数非零方向；`effective rank`/`stable rank` 还描述谱质量是否集中，因此高 rank 仍可能有很慢的谱尾。

普通 SGD 的局部预条件器取 `P_t = I`，意思是每个参数方向使用同一欧氏度量；不是说输入是单位矩阵。Adam、momentum、weight decay 等会让实际更新依赖 optimizer state，此时必须把状态和参数一起记录，不能把完整 Adam 永远简化成一个固定核。

在固定 Jacobian 的平方损失近似下，一步更新给出 `e_{k+1} ≈ (I − ηK)e_k`，所以 `λᵢ` 很小的模态在有限 `B` 步内几乎不动，`λ_max` 过大则要求更小学习率，`κ = λ_max / λ_min⁺` 变大时一个全局学习率难以同时照顾快慢方向。LoP 的操作化现象是固定预算适应变差；“完全不收敛”只是极端情形，不是必要条件。训练曲线仍下降但 AUC、最终 loss 或 fresh gap 变差，也属于预算相关 LoP。

### EEG 使用的损失不是平方损失

BrainUICL 的监督 probe 使用多类交叉熵，输出是 logits，通常形状为 `[batch, classes, sequence]`。平方损失推导是便于说明谱动力学的代理模型。交叉熵在 logits 局部二阶展开时有曲率 `C = Diag(p) − p pᵀ`，其中 `p = softmax(logits)`；`C` 是 logits 损失的 Hessian，不是 EEG 数据矩阵。`C` 的核（kernel）是线性代数中的零空间：`ker(C) = {v : Cv = 0}`，不是一个 Python 函数。实际网络还存在模型非线性 Hessian 项，因此 `Jᵀ C J` 是 GGN/加权 NTK 代理，而非无条件等于真实 Hessian。

经验 Fisher 使用观测标签的平方梯度平均；model Fisher 则对模型分布中的标签取期望。二者、GGN 和真实 Hessian 在一般非线性 EEG 网络中不相等。EdgeForge 当前记录的是明确标注为 `empirical-fisher-cross-entropy` 的 proxy，避免把它误称为精确曲率。

局部线性近似只在同一激活区域、参数扰动足够小且 Jacobian 变化不大时可信。对 EEG 不能先验宣称成立；每个 checkpoint 应用小的参数扰动做有限差分，报告二阶差分与一阶差分的比值，并把它作为 validity diagnostic。当前 instrumentation 的 effective rank、drift 和 Fisher 是可测代理，不是已经验证局部定理的证明。

## 2. RL 结论能否迁移到无标签 EEG

可以迁移的是与监督/自监督更新共同的数学结构：固定预算 fresh probe、表示谱、梯度 Gram、参数 Jacobian、churn、optimizer state 和 replay/context 对比。不能直接迁移的是依赖策略分布、TD error、奖励和 on-policy 探索的结论。无标签 EEG 的在线适应还需要单独定义伪标签或一致性目标，并报告伪标签置信度；用真实标签做的 probe 只能作为受控 oracle outcome。

2024 年同时处理遗忘与 LoP 的 UPGD 类方法在“同一任务、同一预算、同一模型和明确旧任务 utility”条件下是合理的工程干预，但不能据此声称普遍同时解决两者。EdgeForge 会把 `fresh gap`、`plasticity` 和 `forgetting/BWT` 分开记录，只有三者的反事实实验都通过才讨论 trade-off。

## 3. FACED 与 ISRUC 的统一实验矩阵

| 维度 | ISRUC | FACED | 统一要求 |
| --- | --- | --- | --- |
| 输入契约 | `(20, 8, 3000)`，6 EEG + 2 EOG | `(20, 32, 2500)`，31 EEG + 1 辅助通道 | 不重采样到同一物理含义；只统一 stage/metric schema |
| 类别 | 5 类睡眠阶段 | 9 类情绪 | 每个数据集独立训练/比较，不混合标签 |
| 当前 checkpoint | seed 4321 有 pretrain 与 ISRUC CL stages | 当前有 seed 4321 pretrain；连续 CL checkpoint 仍需补齐 | seed 4322/4323 必须真实重跑并保存 checkpoint |
| 方法组 | Finetune、EWC、Online-EWC、SI、MAS、Plain ER、SPR/PuriDivER、BrainUICL | 同一方法组，先完成 clean smoke 再扩展 | 相同任务顺序、split、batch、学习率和预算 |
| stage | pretrain `0`、CL checkpoints（例如 `10/25/49`） | 先定义同样的 stage 语义 | 用 `effective_rank(t−1) → plasticity(t)` 配对 |

第一阶段不改数据，先证明 clean protocol 能区分 fresh-init、长期 checkpoint 和普通训练退化。每个 stage 至少保存：checkpoint digest、数据 manifest digest、subject/split、method、seed、optimizer 超参数和 probe budget。

## 4. 必须采集的指标

### Outcome（LoP 的直接证据）

- held-out `loss`、`accuracy`、`macro-F1` 曲线；
- fixed-budget AULC、final gain、fresh loss/accuracy gap；
- train/test gap，用于区分优化型 LoP 与泛化型退化；
- old-task accuracy、BWT、forgetting，用于与 LoP 分离。

### Predictor 与机制诊断

- `task.spectra.fusion/transformer_1/classifier_input.{effective_rank,stable_rank,rank90,rank95,sigma_max,condition_number,tail_energy,epsilon_rank}`；
- `task.jacobian.last_layer.*`：最后线性层 feature-factor 的 exact proxy，以及 scalar softmax-curvature weighted proxy；
- `task.representation.<component>.{drift_from_previous,cosine_to_previous,frobenius_relative_change}`；
- `task.attention.layer_1.{entropy,head_entropy,max_probability,offdiag_mass}`，并保留实际 `softmax(dim=1)` 轴；
- `task.activation.*` 的 ReLU dead fraction、其他激活 near-zero fraction；
- `task.gradient.{norm_mean,cosine_mean,negative_fraction,effective_rank}`；
- `task.importance.<block>.{mean,max,nonzero_fraction}`（经验 Fisher proxy）；
- `task.weight_norm.<block>.l2`、参数更新范数和 relative update（有前后 checkpoint 时）。

effective rank 只是 predictor 候选，不是 LoP 定义。正式检验仍然以 fixed-budget fresh gap 为主，谱、attention、Fisher 和 activation 只用于解释可能路径。

## 5. 数据修改如何让 LoP 更容易出现

数据修改必须先做剂量扫描和 clean 对照，不能用“性能下降”直接命名 LoP。推荐按以下顺序逐步增加压力：

1. **标签保持的传感器/域 shift**：幅值缩放、相对通道噪声、band-stop、有限时间 jitter、随机 channel dropout。每种变换记录 severity、频带、随机种子和输入/输出 digest；`generate-raeeg-shift.py` 已实现这些受控派生数据。
2. **任务切换压力**：扩大 subject/domain 顺序差异、提高切换频率、减少每任务样本量。必须另外做“同等总样本、不切换”的难度控制，区分任务本身难与长期状态 LoP。
3. **primacy/过拟合压力**：对早期 task 做过量更新或重复采样，再把相同后续任务交给旧状态与 fresh 状态。该设计对应 primacy bias，需固定总梯度更新数和数据暴露量。
4. **replay/context 消融**：replay ratio 设为 `0/0.1/0.5`，分别比较无 replay、短期 replay 和固定旧样本 replay；不要把 replay 与 Transformer 结构变化同时改变。
5. **组合 shift**：只在单因素结果稳定后做 noise × task-switch 或 channel-dropout × replay factorial design，预先写明交互项，不在事后挑选最差组合。

最先推荐的 ISRUC/FACED 剂量网格是 `severity ∈ {0, 0.25, 0.5, 1.0}`，每种变换 3 个真实 seed、相同 subject order 和 `B ∈ {0, 5, 10, 25, 50}` 的 probe。FACED 与 ISRUC 的物理采样率不同，band-stop 的频率必须按各自 Nyquist 频率定义，不能复用 Hz 数值而不记录采样率。

## 6. LoP 判定门槛与当前状态

对每个 method/dataset，先计算

`Δ_acc(t;B) = Acc_fresh(B) − Acc_old(B)`，`Δ_loss(t;B) = Loss_old(B) − Loss_fresh(B)`。

只有当 gap 在相同新任务、相同预算、相同 split 下跨 stage 重复出现，并且至少 3 个独立 seed 的 seed-cluster bootstrap 区间支持方向，才把它标记为“LoP candidate”。EdgeForge 的 exact context 会比较稳定任务身份（dataset、subject、split、method），而把 predictor 与 outcome 各自的 measurement protocol/probe budget 留在审计上下文中。即使统计上通过，EdgeForge 的 `scientific_conclusion_allowed` 仍为 `false`，因为这只是描述性证据；机制结论需要预注册干预和反事实。

截至 2026-08-22：ISRUC seed 4321 的 instrumentation 和 fixed-budget probe 已跑通；FACED seed 4321 的 pretrain instrumentation smoke 已跑通。历史 819 个结果中绝大多数没有 checkpoint-level predictor，不能回填 effective rank；seed 4322/4323 的连续 checkpoint 仍缺失，所以两个数据集都还没有正式多 seed LoP 结论。下一阶段应优先生成 FACED/ISRUC 各 3 个 seed 的 clean CL checkpoint，再做 shift 剂量实验，而不是先扩大数据破坏强度。
