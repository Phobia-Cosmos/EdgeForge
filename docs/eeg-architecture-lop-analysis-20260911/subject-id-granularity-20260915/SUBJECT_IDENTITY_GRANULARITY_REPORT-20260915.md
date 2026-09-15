# EEG subject-specific information：epoch、sequence 与开放集身份实验（2026-09-15）

## 先给出直接结论

1. ISRUC 和 FACED 都含有可重复的 subject-specific information，但“能否识别”取决于粒度、表征、采集条件和是否有 enrollment。闭集条件下，已 enrollment subject 的新 sequence 可以得到具体 subject id；完全未见过的 subject 不能被一个只训练过旧 id 的分类器直接命名。
2. 单个 epoch 的身份线索明显弱于 5–20 个 epoch 的聚合。当前严格独立的 raw/frontend epoch probe 在 ISRUC EEG branch 上为 36.48%，FACED 为 13.52%；用 20 epoch median 聚合后分别升至 75.41% 和 86.22%。这支持“跨 epoch 稳定成分通过聚合提高信噪比”的解释，但不表示单个 epoch 完全没有身份信息。
3. Transformer token 的 epoch probe 不能简单称为“只看一个 epoch”：BrainUICL 先把 20 个 epoch 编成 20 个 token，attention token 会访问同一 sequence 的其他 19 个 token。因此 transformer layer 的 epoch 结果是 sequence-contextual probe；真正的单 epoch 结论应优先看 raw 和 frontend tap，或者另行实现长度为 1/带 mask 的模型输入。
4. 当前结果是身份可分性审计，不是 LoP 证据，也不是医学身份认证。前端身份信息很强可能来自生理差异，也可能来自电极增益、参考、通道布局、记录批次和任务状态；需要 session-disjoint、通道/增益消融和多 seed 才能作生理解释。

## 哪些特征可能携带身份信息，能否解释

### 显式、可解释的输入特征

EdgeForge 先前的 83 维 fingerprint 审计明确计算了每通道 log-RMS、跨通道中心化 RMS、差分 RMS/roughness、delta/theta/alpha/beta/gamma 相对带功率、谱熵，以及通道间相关系数的均值/标准差/绝对值均值。这些特征对应可解释的候选身份因素：稳定的头皮阻抗和电极接触幅值、个体化的频带能量比例、通道间空间相关结构、眼电/肌电混入模式、参考电极和布线造成的极性/相关性变化。它们不是“人类唯一生物码”，而是跨记录较稳定的统计量。

其中增益归一化是一个很有信息量的消融：ISRUC 的 raw sequence probe 从 3.83% 降到 2.32%，说明幅值尺度贡献明显；FACED 的 raw probe 约 4.85% 在归一化后没有下降，说明其身份/批次线索不只来自整体增益，还可能来自通道比例、频谱和空间关系。这个消融不能单独区分生理差异和采集差异。

### 学习到的 EEG branch 特征

BrainUICL 的 ISRUC `eeg_branch` 不是一个手工频带特征，而是每个 epoch 经过一串共享卷积和池化：`Conv1d(6→64, kernel=50, stride=6)`、BatchNorm、GELU、`MaxPool1d(8)`、Dropout，再经过 `64→128→256→512` 的 kernel=8 卷积、BatchNorm/GELU，最后 `MaxPool1d(4)` 和 `AdaptiveAvgPool1d(1)` 得到 512 维向量。它以局部时间模式为输入，通过卷积核学习波形边缘、短时节律和跨通道混合，再用池化获得对相位小偏移更稳定的统计表征；ISRUC 的 EOG branch 对前 2 个 EOG 通道重复同样结构，最后把 EEG/EOG 两个 512 维向量拼成 1024 维并用 `Linear(1024→512)` fusion。

FACED 当前实现是 32 通道共享单分支：`eog` 和 `eeg` 在进入 `features` 前重新拼回 32 通道，所以本轮 FACED 表中的 `eeg_branch`、`eog_branch` 和 `fusion` 是同一个 shared frontend tap，不能解释成三个独立的生理分支。

### 当前还没有完成的因果解释

从“某一层 subject probe 较高”只能知道该层保留了可线性解码的身份信息，不能知道到底是哪一个频带、哪一个通道或哪一种波形形态造成了高分。下一轮应做 channel occlusion、频带 notch/band masking、幅值/参考归一化、时间反转/相位随机化、通道置换和 integrated-gradients/遮挡敏感度；同时报告任务准确率，避免把身份相关的采集伪影误认为生理身份特征。

作为假设生成而不是最终因果结论，我们对已有 83/323 维 handcrafted fingerprint 做了全矩阵标准化 Logistic 系数的绝对值排序。ISRUC 排名前列是多个通道的 `diff_ratio`（差分 RMS / 全局 RMS）、跨通道 `corr_mean`、beta/gamma/alpha 相对带功率；FACED 排名前列主要是若干通道的 `log_rms` 和 `log_rms_centered`。这与“波形粗糙度、通道增益/比例、频谱和空间参考结构携带信息”的解释一致，但该排序没有使用独立 attribution split，不能写成某个通道的生理因果重要性。

## 为什么网络越深，身份 probe 通常下降

这不是“深度天然删除身份”的定理，而是当前模型和 probe 的联合结果，主要有四个原因。

第一，BrainUICL 的训练目标是睡眠/情绪任务和自监督一致性，不是 subject classification；网络会保留有助于任务的稳定成分，并尝试压低与任务无关的个体/记录变化。第二，Transformer 对 20 个 token 做 attention、LayerNorm、残差和 ReLU feed-forward，会把每个 epoch 的局部特征混合成任务需要的时序表示，单纯线性 probe 可能不再容易恢复原来的身份坐标。第三，SleepMLP 明确把 512 维压到 256 再到 128，最后 logits 只有 5（ISRUC）或 9（FACED）个任务类别，低维瓶颈必然降低可线性解码的身份容量。第四，logits 只表达任务决策边界，两个 subject 可以得到相同的睡眠/情绪类别，所以 logits 不需要区分他们。

还要强调：probe 分数下降只说明“线性可分性下降”，不等于身份互信息为零；非线性 probe、最近邻或原始数据侧信道仍可能恢复身份。FACED 的 epoch layer1 比 frontend 高、而 sequence layer1 比 frontend 低，正说明 token context 和聚合方式会改变结果，不能把“越深越低”当成普遍规律。

## 本轮具体实验协议

每个 `.npy` 文件都是一个不可拆分的完整 group：ISRUC 为 20 个 30 秒 epoch，FACED 为当前 processed 副本中的 20 个 trial epoch。先随机固定 80% subject 为 enrolled/known，剩余 20% subject 完全不进入 probe 训练。对每个 known subject，完整 group 的一半用于 enrollment/train，另一半用于 known-test；unknown subject 的所有 group 只进入 unknown-test。训练、测试和未知评估都在 group 层级隔离，因此没有把同一 sequence 的相邻 epoch 分到 train/test。

probe 是 `StandardScaler → PCA(128) → multinomial LogisticRegression`。PCA 只用于让高维 representation 的线性 probe 在全量数据上可重复运行，指标仍按真实 representation 维度记录。known-test 报具体 subject id 的 top-1/top-5；unknown-test 不报告“预测正确的 subject id”，而报告基于 enrollment confidence 的拒识率和 AUROC。

当前全量运行：ISRUC 为 4,276 个 group、85,520 个 epoch、78 known subject、20 unknown subject；FACED 为 984 个 group、19,680 个 epoch、98 known subject、25 unknown subject。具体数值保存在 `ISRUC/summary.json`、`FACED/summary.json` 和自动报告中。

## RQ1：EEG 中是否存在可识别的 subject-specific information？

答案是“存在，但必须限定为跨记录可重复的统计身份线索”。在 80% subject enrollment 的 open-set protocol 中，ISRUC sequence-level EEG branch 的 known-test top-1 为 75.41%，机会水平为 1/78 = 1.28%；FACED 为 86.22%，机会水平同样为 1/98 = 1.02%。原始下采样波形只有 ISRUC 3.83%、FACED 4.85%，但仍高于机会水平，说明原始输入并非完全没有身份信息，只是当前低维对齐 probe 很难直接使用它。

对 unknown subject，分类器没有这些新 subject 的 class，因此不能给出真实的新 id。当前 EEG branch unknown AUROC 约为 ISRUC 0.6633、FACED 0.8251，表示 FACED 更容易用 enrollment confidence 检测“这不是已知 subject”；这不是“识别出 unknown 的具体姓名”。

## RQ2：subject-specific information 在单 epoch 和 sequence 分别有多强？

在严格 group-disjoint 的 epoch probe 中，已知 subject 的 top-1 如下：

| dataset | raw epoch | frontend epoch | Transformer layer1 epoch（有 20-token context） | frontend sequence（20 epoch median） |
| --- | ---: | ---: | ---: | ---: |
| ISRUC | 3.28% | 51.98% | 58.57% | 75.41% |
| FACED | 1.65% | 44.27% | 66.82% | 86.22% |

为直接观察 epoch 数量的影响，我们对每个 group 随机抽取 k 个 epoch 做 median，并保持 group-disjoint：

| k | ISRUC EEG branch | FACED EEG branch |
| ---: | ---: | ---: |
| 1 | 36.48% | 13.52% |
| 2 | 52.49% | 30.10% |
| 5 | 58.58% | 59.44% |
| 10 | 70.82% | 75.26% |
| 20 | 75.41% | 86.22% |

这个曲线支持“更多 epoch 聚合通常提高身份稳定性”，但不是严格单调定律：随机抽样、睡眠状态变化和 median 对异常 epoch 的抑制会造成波动。更小粒度当然可以继续做：一个 epoch 内可切成 1–5 秒 window 或固定采样点 patch；但 window 越短，身份线索越弱、状态/噪声比例越大，需要 nested split 和足够多的独立 recording 才能避免伪重复。

当前 epoch probe 的 raw/frontend 表示是独立 epoch；Transformer layer 的 epoch 表示仍从完整 20-token sequence 得到，因此只能回答“序列上下文中的 token 身份可分性”。若要回答严格的“只给一个 epoch，Transformer 能否识别谁”，应改成长度 1 或 attention mask 的推理路径，并单独训练/校准该输入契约。

## RQ3：持续学习模型是否会学习/记忆 subject characteristics？

本轮 frozen pretrain probe 证明的是“checkpoint 中已有的 representation 含有 subject 信息”，不是“持续学习一定记忆了 subject”。要把 RQ3 做成 LoP 研究，需要在相同 task、优化器、更新步数和 seed 下建立 subject-wise continual stream，在每个 checkpoint 冻结模型并重复上述 epoch/sequence probe：比较新 subject、旧 subject、未见 subject 的 identity accuracy、within-subject radius、between-subject distance、effective rank、Jacobian/NTK 和任务准确率。若在持续学习更新后 task accuracy 稳定但 identity probe、旧 subject retention 或表征距离发生系统变化，才能讨论模型吸收、遗忘或重编码 subject characteristics。

已有的 ISRUC `individual_1` 和 `individual_38` checkpoint 已做了第一轮固定 split 对照，结果见 [subject-id-cl-tracking-20260915/REPORT-20260915.md](../subject-id-cl-tracking-20260915/REPORT-20260915.md)：EEG branch sequence top-1 从 Pretrain 的 0.7541 上升到 0.7645/0.7715，Transformer layer1 从 0.5284 上升到 0.5638/0.5615。这个趋势只说明持续学习 checkpoint 的可解码身份信息发生变化，不能单独证明“记住了某个 subject”；仍需 task accuracy、stream step、三种 seed 和 replay/随机更新对照。

推荐的三条对照流是：普通 fine-tuning；带 replay 的 continual learning；subject-adversarial/domain-invariant regularization。最后一条的目标不是让 identity probe 越高越好，而是检验在任务准确率可接受的前提下是否能降低非必要 subject leakage。LoP 的 fresh-vs-warm、AULC gap、旧任务 retention 和 plasticity 指标仍必须单独报告。

## “不看睡眠阶段，仅凭一个 EEG epoch，到底能不能知道它属于谁？”

对当前数据和模型，最准确的回答是：一个 epoch 含有可超过机会水平的 subject 线索，但不能稳定、普适、无条件地给出人的真实身份。ISRUC 单 epoch frontend probe 约 52.0%，FACED 约 44.3%（严格按每个测试 epoch 计）；当只取每个 group 的一个随机 epoch 时，ISRUC 36.5%、FACED 13.5%。这两个数都不是“认证成功率”：它们依赖已知 subject enrollment、设备、预处理、task distribution 和线性 probe。完全未见过的 subject 只能被判定为“可能未知”，不能凭空输出一个新的正确 id。

## 为什么原始波形几乎分不出个体

本实验的 `raw_waveform` 不是把 3,000/2,500 个采样点原样送入一个可学习的 raw CNN，而是每个 epoch 沿时间平均池化到 128 点/通道；sequence 结果还对 20 个 epoch 取 median。这种固定坐标的线性 probe 对时间相位、睡眠状态、瞬态事件、参考极性、跨通道延迟和幅值变化很敏感；同一个人在不同 group 的波形不一定在每个采样点对齐，跨人的同一节律也可能发生相位平移。PCA(128) 和 LogisticRegression 进一步只检测线性分离，无法自动学习频带或局部形状。

因此“raw 低分”不等于原始 EEG 没有身份信息。卷积 frontend 通过可学习滤波器、非线性、池化和通道组合获得相位/局部形状鲁棒性，前端高分正说明信息可能在 raw 中但需要合适的编码器才能提取；同时也可能放大了设备/参考伪影。要区分二者，必须做频带、增益、参考、通道和 phase-randomization 消融。

## 指纹为什么能用于身份识别，人脸如何类似

这里的“指纹”不是一个数学上绝对唯一的字段，而是高维向量 `z`：同一 subject 的多次记录希望距离小，不同 subject 的向量希望距离大。enrollment 阶段保存 subject profile 或 embedding，测试时用分类器/最近邻/距离阈值判断“最像谁”并估计是否未知。噪声、年龄、状态、设备变化、同卵双生和攻击都可能造成碰撞或误拒，因此只能给概率和置信区间。

人脸系统也是相同范式：摄像头图像经过 CNN/ViT 得到 embedding，训练目标可能是分类、contrastive、triplet 或 ArcFace，使同一人的不同姿态靠近、不同人的向量分开；部署时做 verification（是否同一人）或 identification（在已 enrollment 人群中是哪一人），并用阈值控制 FAR/FRR。EEG 的困难更大，因为脑电受睡眠/情绪/电极接触/参考和 session 漂移影响更强，所以需要 session-disjoint 和 unknown rejection，不能把高 closed-set accuracy 当作绝对唯一身份。

## 代码、结果与下一步

- 粒度和开放集脚本：[subject_identity_granularity.py](../../../scripts/subject_identity_granularity.py)
- ISRUC 全量结果：[ISRUC/REPORT.md](ISRUC/REPORT.md) 和 [ISRUC/summary.json](ISRUC/summary.json)
- FACED 全量结果：[FACED/REPORT.md](FACED/REPORT.md) 和 [FACED/summary.json](FACED/summary.json)
- 已有 sequence-level 全部 tap 结果：[subject-id-probe-20260915/](../subject-id-probe-20260915/)

下一阶段应按优先级完成：

1. 从原始元数据建立 recording/session-disjoint split，并重复 RQ1/RQ2；
2. 对每个 subject 做 task-class-balanced split，排除睡眠/情绪分布作为身份代理；
3. 做 channel、band、gain、reference、phase 和 temporal-window 消融，生成可解释性贡献表；
4. 将 identity probe 插入 continual-learning checkpoint 时间线，联合 LoP 指标判断“学习了身份”与“塑性丧失”是否相关；
5. 增加 verification/open-set calibration（EER、FAR、FRR、unknown rejection）和至少 3 个独立 seed。
