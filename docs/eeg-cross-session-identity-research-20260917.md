# RA-EEG / EdgeForge 跨 session EEG 身份特征研究

更新日期：2026-09-18

## 结论摘要

> 2026-09-18 实验更新：本地实际没有 BED payload；BED 仍是 Zenodo 受限访问候选。本机已有且已完成实验的是 SEED、ISRUC-II、BCICIV-2a 和 BCICIV-2b。完整新结果、特征分层、verification EER 和解耦设计见 [EEG 独立实验报告](../../EEG/docs/cross-session-identity-foundation.md)。

1. EdgeForge 当前使用的 ISRUC 数据来自 Subgroup I；原始协议明确为每人一次整夜 PSG，因此项目里的多个 epoch/chunk 不能解释为多个 session、trial 或日期。ISRUC 只有 Subgroup II 的 8 人各有两次记录，而且公开资料只说明记录位于不同日期，没有给出统一间隔。
2. FACED 是每名参与者一次连续佩戴下完成 7 个 block、28 个视频 trial 的情绪诱发实验；block、trial 和穿插的算术题都是同一次记录内部结构，不是跨 session 或跨天数据。
3. OpenNeuro ds004148 是当前最贴合“跨时间、跨状态身份稳定性”的公开数据：60 人、3 个 session、每个 session 均含 EO、EC、减法、音乐回忆和记忆五种状态；S1 与 S2 同日相隔 90 分钟，S3 约 30 天后。它同时支持 short-term、long-term、within-task 和 cross-task 设计。
4. Sensors 2023 的 29 人 × 20 个不同日期 session 是强纵向验证基准，采集跨度平均约 70 天，但原始数据仅可向作者申请。BMT_EEG 的同行评议论文是 23 人 × 3 session、相邻 session 至少 15 天，支持随机与 skilled forgery；完整数据同样需要申请，当前 GitHub 只公开代码和极少样例。
5. 已下载并验证两个完整公开跨 session 数据集：BCI Competition IV 2a（9 人 × 2 个不同日期 session）和 2b（9 人 × 5 个不同日期 session）。2a 是无反馈四类 MI 的跨天基准；2b 更适合研究同一任务中多日 session/反馈条件漂移，但官方资料只给出前两个 screening 日期位于两周内，没有给出全部五天的精确跨度。
6. 对稳定身份识别，优先级应从“任务响应本身”转向“跨任务仍保留的个体残差”：个体化谱形、IAF/相对谱功率、1/f 参数、归一化空间协方差或连接、以及显式去除 task/session 信息的表示。SSVEP 谐波、ERP 波形、MI 的局灶 ERD/ERS、睡眠分期谱和高频肌电都很强，但首先编码的是任务或状态；如果不做 task-conditioned 分析，它们很容易产生伪身份结果。

## 2026-09-18 本地跨 session 实验结果

| Dataset | 主协议 | 幅值不变 logistic identification | Cosine-centroid control | Logistic verification EER | 主要发现 |
|---|---|---:|---:|---:|---|
| SEED | 两个完整 session 训练、第三个测试，三折 | 73.50% | 55.48% | 8.84% | 跨未见 emotion 仍为 66.17%–69.00%；情绪左右不对称未优于通用谱形/形态 |
| BCICIV-2a | session 1↔2 | 89.77% | 59.33% | 4.09% | 任务期谱形强于 pre-cue baseline；跨 MI 类别仍约 78%–80% |
| BCICIV-2b | 五次 leave-one-session-out | 87.40% | 73.34% | 含协议迁移主汇总为 5.48% | screening→feedback 为 78.53%，反向为 82.65%；绝对功率有明显 shortcut 风险 |
| ISRUC-II | recording 1↔2 | 69.65% | 40.59% | 16.68% | N1/N2 最强、W 最弱；N1↔N2 可迁移，W→N2 明显下降 |

四套数据的 chance 为 6.67%–12.50%，两类分类器均高于 chance。结果证明跨 session identity 存在，但分类器容量显著影响数值；尤其 ISRUC-II 的旧五频带 prototype 36.80% 不能再解释为“睡眠身份弱”，相同五频带换 logistic 已达 63.77%，七细频带为 69.65%。绝对功率只作为 gain/阻抗 shortcut 诊断，不进入稳定身份主结论。

当前解耦阶段采用数据集专属前端与共同审计接口：SEED 使用 62 导空间图和左右同源分支；BCICIV 使用 sensorimotor mu/beta ERD 与 C3/C4；ISRUC 使用 30 秒宏观节律加 2–5 秒 spindle/slow-wave 微事件。identity、state/task、session embedding 分开，依次加入 supervised contrastive、训练 session adversary 和 identity/state independence；所有结论必须同时报告 subject probe、state/session leakage、跨状态矩阵和 EER。

## ISRUC 与 FACED 的真实采集协议

| 数据集 | 参与者与记录 | 记录层级 | 通道与采样 | 对 EdgeForge 现有结果的含义 |
|---|---|---|---|---|
| ISRUC-Sleep | Subgroup I：100 人，每人 1 次记录；Subgroup II：8 人，每人 2 次记录；Subgroup III：10 人，每人 1 次记录 | 每次记录为一次整夜 PSG；Subgroup II 的两次记录位于不同日期，但未找到权威的固定间隔 | 原始数据 200 Hz；常用 EEG 为 F3-A2、C3-A2、O1-A2、F4-A1、C4-A1、O2-A1，另有双 EOG 和其他 PSG 通道 | EdgeForge catalog 指向 `ISRUC-Group-I`。所以当前 20 epoch 文件只是同一夜内部切块，不能作为跨 session、跨 trial 或跨天身份稳定性证据。若要利用 ISRUC 做重复记录，只能另建 Subgroup II 管线，并保留其“间隔未知”的限制 |
| FACED | 123 人；每人 28 个情绪视频 | 每人一次连续记录，7 个 block，每 block 4 个 trial；3 个正性、3 个负性、1 个中性 block，block 间至少休息 30 秒并完成 20 道算术题；trial 内含 5 秒注视、34–129 秒视频和评分 | 32 通道；92 人原始 1000 Hz、31 人原始 250 Hz，发布的预处理数据统一为 250 Hz | 7 block 和 28 trial 都是同一次佩戴/访问内部结构。公开 BIDS/NEMAR 为 123 subjects、123 scans，未提供 repeat-session 字段；因此 FACED 可验证跨情绪/跨 trial 的任务稳健性，但不是纵向身份验证集 |

ISRUC 的 session 数来自原始数据论文 [Khalighi et al., 2016](https://pubmed.ncbi.nlm.nih.gov/26589468/)；通道与分组可由 [ISRUC 项目页](https://sleeptight.isr.uc.pt/?page_id=76) 和 [NEMAR nm000111](https://nemar.org/dataexplorer/detail?dataset_id=nm000111) 交叉核验。FACED 的流程、采样率和 block/trial 结构来自 [Scientific Data 数据论文](https://doi.org/10.1038/s41597-023-02650-w)；“单次连续记录”是根据论文流程与 [NEMAR nm000112](https://nemar.org/dataexplorer/detail?dataset_id=nm000112) 的每人单一 scan 文件结构所作的证据化判断，而不是把 block 自行重命名成 session。

ISRUC 原始 Subgroup I 的协议人数是 100；EdgeForge 既有完整运行使用 98 人，是因为本地工作流排除了 subject 8 和 40。这个本地可用数差异不改变“一人只有一次整夜记录”的协议判断。

## 跨 session 数据集比较

| 数据集 | 人数 | session 与时间间隔 | 任务/状态 | EEG 通道；采样率 | 可用性与本次状态 | 适合的问题 |
|---|---:|---|---|---|---|---|
| [OpenNeuro ds004148](https://doi.org/10.18112/openneuro.ds004148.v1.0.1) | 60 | 3；S1→S2 同日 90 min，S2→S3 约 30 d | EO、EC、减法、音乐回忆、情景记忆；各约 5 min | 64 电极系统，分析 EEG 61 通道；500 Hz | CC0；[Scientific Data 数据论文](https://doi.org/10.1038/s41597-022-01607-9) 与 [NEMAR 镜像](https://nemar.org/dataexplorer/detail?dataset_id=on004148) 已核验。NEMAR 当前显示原始约 93.2 GB、zip 约 44.4 GB；本地剩余空间不足，未假装下载子集为完整数据 | 当前首选：短期/长期 test–retest、跨状态身份、1/f/IAF/连接稳定性 |
| [M3CV](https://doi.org/10.1016/j.neuroimage.2022.119666) | 原始 106；95 人完成两次并进入当前 BIDS | 2；6–139 d，均值约 20 d | 6 paradigms/14 类信号：EO/EC、运动执行、VEP/AEP/SEP、SSVEP、P300、选择性 SSVEP、SSAEP、SSSEP | 原始 64 通道、1000 Hz；发布版 250 Hz | [NEMAR nm000166](https://nemar.org/dataexplorer/detail?dataset_id=nm000166) 约 21.1 GB；未下载，以免占满当前共享数据分区 | 大规模跨天、跨任务身份；比本次下载集更接近最终 EdgeForge benchmark |
| Sensors 2023 多 session 静息数据 | 29 enrolled + 23 单 session impostors | 20 个不同日期 session；跨度均值约 70 d，范围 43–129 d | 睁眼静息，每次分析 3 min | 19 湿电极；500 Hz | [论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC9963573/) 已核验；Data Availability 为向作者申请，未下载 | 训练 session 数量、长期 verification、外部 impostor |
| BMT_EEG | 同行评议论文 23；GitHub README 仍写 20 | 3；相邻 session 至少 15 d | 2 resting、7 motor movement/imagery、2 VEP，共 11 protocol | 32 通道采集、处理后 20 通道；256 Hz | [IEEE Access 论文](https://doi.org/10.1109/ACCESS.2026.3678803)、[GitHub](https://github.com/AnamSuri/BMT_EEG) 和 [IEEE DataPort](https://doi.org/10.21227/ayfc-hc17) 已核验；完整数据需申请，仓库当前仅有单个样例 subject 的部分任务 | 多任务 authentication、随机/同任务 skilled forgery；需要先解决版本与人数 provenance |
| [BED](https://doi.org/10.5281/zenodo.4309472) | 21 | 3；相邻约 1 周 | EO、EC、算术、情绪图片及多频 VEP，共 12 类刺激 | 14；256 Hz | [Zenodo 记录](https://zenodo.org/records/4309472) 受限访问，未下载 | 低通道跨周 verification；TIFS 2024 使用它做 session 1–2 train、session 3 test |
| BCI Competition IV 2a | 9 | 2；明确为不同日期 | 左手、右手、双脚、舌 MI；每 session 288 trials | 22 EEG + 3 EOG；250 Hz | **已完整下载并校验** | 小样本跨天同任务 sanity check；可直接验证 session leakage 与 MI 状态残差 |
| BCI Competition IV 2b | 9 | 5 个 session、5 个不同日期；前 2 个 screening 日期在两周内、无反馈，后 3 个日期有反馈；完整跨度未报告 | 左/右手 MI | 3 双极 EEG（C3/Cz/C4 周围）+ 3 EOG；250 Hz | **已完整下载并校验** | 多日 session 与反馈条件漂移；可作跨日证据，但不能补造为月级 permanence |

ds004148 的 60 × 3 × 5 = 900 条 task recordings 与 75 小时总量和五种状态相互一致。原始论文明确把设计描述为短期 90 分钟和长期 1 个月 test–retest；[1/f test–retest 复核研究](https://pmc.ncbi.nlm.nih.gov/articles/PMC10793580/) 还明确说明 S3 在约 30 天后并尽量匹配 S1 的星期和时刻。OpenNeuro 当前数据版本与 NEMAR 镜像容量可能不同，容量数字必须随版本记录，不能把旧页面的压缩大小当作原始大小。

## 任务无关与纵向 EEG 身份论文

### IEEE TIFS 主线

| 论文 | 数据与时间跨度 | 任务；通道；采样率 | 网络/分类器与特征 | 评估协议与本研究可用结论 |
|---|---|---|---|---|
| Maiorana, La Rocca & Campisi, [On the Permanence of EEG Signals for Biometric Recognition](https://doi.org/10.1109/TIFS.2015.2481870), TIFS 2016 | 50 人，3 session，约 1.5 个月 | EC/EO；19 通道；256 Hz | AR、PSD、spectral coherence；L1/L2/cosine 等距离与 score fusion | session 之间比较，不把随机 epoch 划分当 permanence；说明谱与连接特征在月级可保留身份，但仍受 session 明显影响 |
| Maiorana & Campisi, [Longitudinal Evaluation of EEG-Based Biometric Recognition](https://doi.org/10.1109/TIFS.2017.2778010), TIFS 2018 | 45 人有 5 session，30 人另有第 6 session；S1 后均值约 1 周、1 月、7 月、16 月、36 月 | EC、EO、算术、语音想象；19 通道；原始 256 Hz，8–30 Hz 后降采样 64 Hz | 每通道 HMM；AR reflection coefficients、MFCC、Morlet-wavelet bump 表示；决策融合 | enrolment 与 probe 按真实时间距离分离，覆盖 3 年；是“长期 permanence”最强的早期 TIFS 证据，同时显示跨年退化不可忽略 |
| Nakamura, Goverdovsky & Mandic, [In-Ear EEG Biometrics](https://doi.org/10.1109/TIFS.2017.2763124), TIFS 2018 | 15 clients × 2 天，间隔 5–15 d；另 5 人 impostor 数据 | EC；2 个耳内通道；1200 Hz | 0.5–30 Hz；AR(10)、Welch PSD 与 alpha 比值；cosine、LDA、SVM | 明确比较 rigorous day-disjoint 与混合日期的 biased split，同时报告 identification/verification；证明少通道可用，但 60 s probe 的严格跨日 rank-1 约 67.8%，远低于混合日期结果 |
| Wang et al., [CNN Using Dynamic Functional Connectivity](https://doi.org/10.1109/TIFS.2019.2916403), TIFS 2019 | PhysioNet EEGMMIDB，109 人；14 个 runs 属于一次访问，**不是纵向 session** | EO/EC 与 4 类真实/想象运动状态；64 通道；160 Hz | within-frequency 与 cross-frequency 动态功能连接图；GCNN | 强项是跨状态而不是跨日期；可支持 task/state robustness，但不能独立支持 permanence 声称 |
| Kumar et al., [Evidence of Task-Independent Person-Specific Signatures in EEG Using Subspace Techniques](https://doi.org/10.1109/TIFS.2021.3067998), TIFS 2021 | 数据集 1：30 人，平均 3.1 session，1–193 d；TUH 子集：920 人，平均 3.14 session，最长 126 月 | 12 种多模态/想象任务及临床 EEG 状态；共同使用 9 通道；数据集 1 为 250 Hz | 3–30 Hz PSD spectrogram；modified i-vector、x-vector、组合 ix-vector，cosine backend；比较 EEGNet/CNN-RNN | 按时间取前 60% session 训练、后续 20%/20% 验证测试；同时报 rank-1 与 closed-set EER。ix-vector rank-1 为 86.4%（30 人）和 35.9%（920 人），显示规模扩大和临床异质性会显著压低看似漂亮的实验室结果 |
| Wang, Yin & Hu, [Cancellable Deep Learning Framework for EEG Biometrics](https://doi.org/10.1109/TIFS.2024.3369405), TIFS 2024 | BED：21 人 × 3；SEED：15 人 × 3 | BED EO/EC，14 通道、256 Hz；SEED 中性电影，62 通道、200 Hz | 生成器式不可逆变换 + CNN verification，兼顾 cancellability 与 model-inversion 防护；比较 AR、PSD、fuzzy entropy、graph features 等基线 | within-session 用 S1；cross-session 用 S1+S2 train、S3 test。重点提醒：隐私保护和跨 session 泛化是两个独立维度，保护模板并不会自动消除 session drift |

### 直接相关的非 TIFS 论文

| 论文 | 数据、任务与硬件 | 方法 | 评估与结果 |
|---|---|---|---|
| Maiorana, [Learning Deep Features for Task-Independent EEG-Based Biometric Verification](https://doi.org/10.1016/j.patrec.2021.01.004), PRL 2021 | 45 人 × 5 session；前三次在 1 个月内，S4 平均约 6 个月，S5 平均约 15 个月；EC/EO、MI、语音想象、视觉形状、算术；19 通道、256 Hz | 8–30 Hz、64 Hz、5 s window；每通道 Siamese CNN + contrastive loss，one-class SVM/score fusion | 表示学习的 30 人与 open-set 测试的 15 人不重叠；跨 session、跨任务、单/多 session enrolment。多任务 Siamese 学习优于手工 AR/MFCC，但 long-term 和 cross-task EER 仍显著高于 within-task |
| DelPozo-Banos et al., [Evidence of a Task-Independent Neural Signature in the Spectral Shape](https://pubmed.ncbi.nlm.nih.gov/28835183/), IJNS 2018 | 6 个异质数据集，涵盖静息、认知、运动、视觉/听觉诱发；其中 Keirn 与 Yeom 数据有 2 session | real cepstrum 与 LPC/AR 谱形；任务本身不输入分类器 | Task-CV 用未见任务测试，Sess-CV 再跨 session；4/6 数据库 verification accuracy >89%。它支持“谱形中有任务无关成分”，但不同数据库协议不统一，不能用聚合准确率替代单一严格纵向 benchmark |
| Plucińska et al., [Leveraging Multiple Distinct EEG Training Sessions](https://pmc.ncbi.nlm.nih.gov/articles/PMC9963573/), Sensors 2023 | 29 人 × 20 个不同日期 session；23 个额外 impostor；REO；19 通道、500 Hz | 0.2–70 Hz、50 Hz notch、CAR；7.5 s segment；1–45 Hz Welch PSD/raw 与 dB；单隐藏层、1 hidden neuron 的浅 FFNN，LM backprop | 最后 5 session 测试时逐步增加前 1–15 session；另做前 1–8 train、最后 12 test。15 train accuracy 96.7±4.2%，8 train/12 test 94.9±4.6%；超过 8 session 没有显著增益；15 train 的成功 impostor attack 3.1±2.2% |
| Suri et al., [BMT_EEG](https://doi.org/10.1109/ACCESS.2026.3678803), IEEE Access 2026 | 23 人 × 3 session，相邻至少 15 d；11 protocol；32 通道采集/20 通道处理；256 Hz | 1 Hz high-pass、average reference、ICA、Lmax normalization；100-sample non-overlap window；EEGNet 与 1D CNN-GRU | S1+S2 train、S3 test；random 与 skilled forgery。EEGNet 最佳 EER：random MI2 0.06±0.04%，skilled MI1 2.14±4.58%；CNN-GRU 对应 1.29±0.92% 和 6.15±3.78%。数值很强，但须等完整 23 人数据可审计后再作为基准 |
| Huang et al., [M3CV](https://doi.org/10.1016/j.neuroimage.2022.119666), NeuroImage 2022 | 当前发布版 95 人 × 2 个不同日期 session；6 paradigms/14 signal types；64 通道、发布版 250 Hz | 数据集/competition，不限定模型；提供多任务 enrollment、calibration、testing | 天然适合 task-disjoint 与 session-disjoint identification/verification，是下一阶段最值得获取的中型公开基准 |

这些论文共同说明三件事：第一，同一 session 随机切 epoch 几乎总会高估身份性能；第二，跨任务不等于跨时间，PhysioNet 多 run 之类数据只能证明 state robustness；第三，训练中加入更多真实 session 通常有帮助，但收益会饱和，不能用同一 session 的更多窗口代替更多日期。

## 频段、脑区与身份稳定性分析

| 任务/状态 | 主要频段与脑区 | 最容易学到的状态线索 | 对稳定身份建模的处理建议 |
|---|---|---|---|
| SSVEP/VEP | 刺激基频及谐波；枕叶视觉皮层与枕外视觉区，典型 O1/Oz/O2/POz | 刺激频率、相位锁定、屏幕时序和视觉注意 | 不把精确基频/谐波峰直接当身份；在相同刺激下比较谐波间相对形状、空间传播或去除 stimulus template 后的残差。跨刺激频率测试才是真正 task-independent |
| 运动执行/MI | sensorimotor μ 约 8–12 Hz、β 约 13–30 Hz 的 ERD/ERS；C3/C4/Cz 及对侧中央区 | 动作类别、优势手、运动强度、想象策略 | 用左右对称或归一化协方差保留稳定拓扑；任务标签分层，跨动作训练/测试；避免把单侧 ERD 强度直接解释为身份 |
| 睡眠 | NREM 慢波 δ 约 0.5–4 Hz，N2 spindle/sigma 约 10–15 Hz；慢 spindle 偏额区、快 spindle 偏中央/顶区；REM 与觉醒混合特征 | sleep stage、睡眠压力、药物和整夜时相 | 先按 stage 条件化，再研究同 stage 内个体残差；可使用 spindle 频率/拓扑等相对稳定参数，但必须跨夜验证。混合 sleep stage 的身份准确率首先可能是个人睡眠结构差异 |
| 算术/工作记忆 | frontal-midline θ 约 4–8 Hz（Fz/内侧额区）；随负荷常见 posterior α 抑制和 frontoparietal 调制 | 难度、正确率、疲劳、策略 | 用难度/行为分数作协变量；优先个体化峰频、谱形和连接，而不是 θ 绝对功率；在 ds004148 中做 Math↔rest/music/memory 交叉验证 |
| 音乐回忆/聆听 | 双侧颞听觉区并涉及额、顶、运动和边缘网络；θ/α/β 变化依熟悉度、情绪与是否实际聆听而异 | 曲目、熟悉度、情绪、内隐节拍和回忆策略 | 把音乐条件视为强 domain；优先跨状态保持的谱包络/连接骨架，不能预设单一“音乐频段” |
| EO/EC 静息 | EC 后部 α 通常增强；EO 有视觉输入和 alpha blocking；额/中央/顶/枕全局网络均参与 | 眼状态、困倦、眨眼/EOG、当天警觉性 | IAF、相对谱形和归一化连接可能比绝对 alpha amplitude 稳定；EO↔EC 应作为跨状态压力测试，不应混合后随机划分 |
| 情绪视频 | frontal alpha asymmetry、θ/β/γ 与广泛 frontotemporal/limbic 网络，但电影内容与唤醒度影响很大 | 视频、情绪类别、面部/眼肌、刺激时序 | FACED 可做跨 trial/情绪 task robustness；必须按视频或情绪 group split，并将高 γ 结果与 EMG 污染分开解释 |

频段与区域依据包括：[SSVEP 的枕叶/谐波特性](https://pmc.ncbi.nlm.nih.gov/articles/PMC6871301/)、[MI 的 μ/β ERD/ERS 综述](https://doi.org/10.3389/fninf.2018.00078)、[frontal-midline theta 与认知控制/记忆综述](https://pmc.ncbi.nlm.nih.gov/articles/PMC3859771/)、[算术策略 EEG 综述](https://pubmed.ncbi.nlm.nih.gov/27220781/)、[音乐对 EEG 谱与区域效应的综述](https://pmc.ncbi.nlm.nih.gov/articles/PMC6130927/)、[睡眠 spindle 的额区与中央顶区差异](https://pmc.ncbi.nlm.nih.gov/articles/PMC4958487/)；这些生理先验用于构造混杂检验，不应被当作身份稳定性的先验结论。

建议按以下顺序评估候选身份特征：

1. **个体化谱形而不是绝对幅值**：log/relative PSD、band ratio、IAF、real cepstrum、LPC/AR。绝对幅值对阻抗、参考、增益和警觉性敏感；谱形特征已有跨任务证据，但仍需跨日期验证。
2. **periodic 与 aperiodic 分离**：用 SpecParam 类方法提取 1/f slope/offset、峰频与峰宽。ds004148 上已有 [90 分钟与 30 天的 1/f test–retest 研究](https://pmc.ncbi.nlm.nih.gov/articles/PMC10793580/)，因此可直接把 ICC/稳定性与 identity discriminability 分开评估；稳定并不自动等于可辨识。
3. **归一化空间结构**：shrinkage covariance、correlation/coherence/phase-based connectivity、Riemannian tangent features。它们可减弱全局幅值漂移，但也可能学习固定 montage、参考或 volume-conduction 伪迹，所以必须跨重戴/跨日并做通道扰动测试。
4. **显式解耦表示**：identity encoder 配合 task/session adversary，或 multi-task/multi-session contrastive sampling；正样本必须来自同一人不同日期/不同任务，hard negatives 应匹配任务和 session 条件。只在同一 session 内组成正对不会学到 session invariance。
5. **task-conditioned residual**：先预测或回归任务、stage、负荷、刺激频率和行为分数，再对残差做 identity probe；同时报告原始表示与 residual 表示，确认“去状态”没有顺带移除身份信号。

应降权或单独审计的特征包括：未经归一化的通道幅值、精确 stimulus-locked 谐波/ERP、30 Hz 以上易受面部和颈部 EMG 影响的功率、固定坏导/阻抗模式、文件边界/滤波 padding、以及 session 特定参考与设备元数据。这些特征可以产生很高的身份准确率，却不一定来自脑的稳定个体特性。

## 推荐的 EdgeForge 后续实验协议

不重复已完成的 ISRUC/FACED class-conditioned probe 和 continual-learning checkpoint tracking。下一步直接转向真正的 session/date split：

1. **BCICIV-2a 管线冒烟测试**：S1 只作 train/enrolment，S2 只作 test/probe；任务匹配评估四类 MI，并另外做 train-three-tasks/test-held-out-task。先用 log-relative PSD、IAF/peak、covariance-Riemannian、cepstrum 四组轻量特征，不从随机 epoch split 开始。
2. **BCICIV-2b 条件漂移测试**：用 session chronology 划分，分别报告 no-feedback→feedback、early-session→late-session 和逐日结果。3 个双极 EEG 通道让它适合检验“极低通道身份信号是否仍存在”；精确时间跨度未知时只报告 session/day 序号，不自行换算成天数。
3. **ds004148 正式 benchmark**：空间允许后使用完整下载，或先用 NEMAR Zarr 流式读取少量 subject 做 loader；主矩阵应为 S1→S2（90 min）、S1→S3（30 d）、S1+S2→S3，并对 EC/EO/Math/Music/Memory 做 5×5 task transfer。报告 feature ICC、subject rank-1/balanced accuracy、verification EER/FAR/FRR，三者不能相互替代。
4. **已登记个体 closed-set 控制**：当前目标不是把表示迁移到训练阶段未见的人，而是识别训练时已登记个体的后续 EEG。representation-training 和 identity-evaluation 使用同一 subject 集合，但 recording/session 必须严格分开；开放集 subject-disjoint 只保留为未来可选研究，不作为当前完成条件。
5. **混杂消融**：统一参考、重采样和带宽；移除 EOG/非 EEG；做 channel permutation、短时相邻窗口去重、文件名/metadata blind、task-balanced negative sampling；将 day/session 预测准确率作为不变性诊断。

最小结果表应同时包含：数据集版本、subject 数、日期/session 划分、任务组合、有效通道、有效时长、窗口重叠、特征、模型、超参数选择所用 session、rank-1/balanced accuracy、EER/FAR/FRR、置信区间，以及 task/session probe。禁止以随机窗口 k-fold 的数字代表 cross-session 性能。

## 下载与数据完整性

下载前已阅读 `/home/undefined/Disk/README.md`，并复用已登记的 `/home/undefined/Disk/python-envs/brainuicl`（Python 3.12.3、MNE 1.12.1、SciPy 1.18.0）进行只读验证；没有在仓库内创建数据或 Python 环境。完成后已把 `BCICIV-2a/` 和 `BCICIV-2b/` 加入该共享数据集注册表。

### BCICIV-2a

- 路径：`/home/undefined/Disk/datasets/BCICIV-2a`
- 大小：995 MB；18 GDF、18 label MAT、官方说明 PDF、原始 zip 与 `manifest.json`
- ZIP integrity：通过；全部 18 GDF 可由 MNE 读取，均为 25 总通道、250 Hz；从 header 汇总 13.3852 h
- 全部 label MAT 可由 SciPy 读取，每文件 288 labels，类别为 1–4
- data zip SHA-256：`65fe93cb766e4b00ece69a200312d81f54bba17c642406bd922d913a8aedc024`
- labels zip SHA-256：`a5e21dbd7e92d2959a3579d18de0defa9ca1a85eec17931118f81dbe141c6302`
- description PDF SHA-256：`7914fdf5c8e04468c97d70486372405da4ed66fb1469160bec7cc27bd1676110`
- 官方来源：[competition download](https://www.bbci.de/competition/iv/download/index.html)、[protocol PDF](https://www.bbci.de/competition/iv/desc_2a.pdf)

### BCICIV-2b

- 路径：`/home/undefined/Disk/datasets/BCICIV-2b`
- 大小：488 MB；45 GDF、45 label MAT、官方说明 PDF、原始 zip 与 `manifest.json`
- ZIP integrity：通过；全部 45 GDF 可由 MNE 读取，均为 6 总通道、250 Hz；从 header 汇总 26.2751 h
- 全部 label MAT 可由 SciPy 读取，每文件 120–160 labels，类别为 1–2
- data zip SHA-256：`1af1625d8a2c7c793d23236000f49cef2a064a71337194585b4fa353eff62e15`
- labels zip SHA-256：`e0b7a96b0bef76cd83bba9d5a0662a3caaa8979b69b5707b72cc723bf958cd8a`
- description PDF SHA-256：`cf520e61276cb693505b1f0bf4c0bd098ae76c57b8a1d191165599aa932f1af2`
- 官方来源：[competition download](https://www.bbci.de/competition/iv/download/index.html)、[protocol PDF](https://www.bbci.de/competition/iv/desc_2b.pdf)、[BCI Competition IV 综述](https://doi.org/10.3389/fnins.2012.00055)

两个目录内的 `manifest.json` 保存来源 URL、协议摘要、哈希和逐文件验证状态。ds004148 没有落盘：NEMAR 当前完整压缩包约 44.4 GB，而下载前共享分区可用空间约 26 GB；在空间不够时中止是完整性要求，不是任务遗漏。BMT_EEG、Sensors 2023 和 BED 均因访问控制而未伪造“已下载”状态。

## 引用与证据边界

- 数字优先级：同行评议数据论文/方法论文 > 官方数据仓库和协议 PDF > 代码仓库 README > 二次综述。BMT_EEG 的 23/20 人冲突依此采用 23，并保留 README 落后这一 provenance 警告。
- “没有找到 session 间隔”不等于“同日”：ISRUC Subgroup II 只记为不同日期、间隔未报告。
- “多个 run/trial/block”不等于“多个 session”：FACED、PhysioNet EEGMMIDB 和单夜 ISRUC 都按真实佩戴/访问边界解释。
- 2a 官方资料只说明两个不同日期，未给出精确间隔；2b 的竞赛综述明确五个 session 来自五个不同日期，协议 PDF 只进一步说明前两个 screening 日期在两周内。报告不补造完整跨度。
- 本报告记录的是截至 2026-09-17 可核验的版本与访问状态；外部仓库容量、权限和 README 可能继续变化，后续下载必须把版本和哈希写入 manifest。
- 引用审计共检查 35 个唯一外链，全部返回成功的 HTTP 2xx；本地两个数据集的四个 zip 再次通过完整性测试，六个文件哈希、全部 GDF header 和全部 MAT label 也与 manifest 一致。
