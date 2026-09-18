# EEG 身份、任务、EER 与特征解耦结果汇总

更新日期：2026-09-18

## 结果究竟表示什么

当前目标是已登记个体的 closed-set identification：每个 subject 都在训练数据中出现，但训练和测试 recording/session/domain 严格分开。测试时随机给出留出域中的一个 EEG 窗口，IdentityNet 在该数据集已经登记的全部 subject 中输出一个身份。它不表示模型可以识别训练阶段完全没见过的新用户，也尚未实现 unknown-user rejection。FACED 是例外中的弱协议：所有 subject 仍然已登记，但训练和测试只是同一次 recording 内不同 block，不能称为 cross-session permanence。

EER 不是用 `1-accuracy` 代替。对每个 checkpoint，使用训练域全部 identity embedding 的均值建立一个 subject enrollment prototype；测试 embedding 与全部 prototype 做 cosine similarity。正确 subject prototype 是 genuine claim，其余已登记 subject prototype 是 impostor claims；在 empirical ROC 相邻点间线性求 FAR=FRR 交点。该 EER 是 registered-impostor、closed-set verification，不包含外部未知 impostor。10 窗 EER 只聚合同一 subject、同一 domain 中连续且不重叠的 10 个窗口。

## 统一结果表

表中的多 seed 项报告均值 ± 样本标准差；单 seed/单方向结果不附标准差。Identity 是 N-way rank-1 accuracy；Task 是同步训练的独立 TaskNet 在完全相同测试 split 上的 accuracy/Macro-F1。两个网络没有共享参数，也没有把 task 标签输入 IdentityNet。

| 数据集与测试协议 | 登记人数 | 训练/测试窗口 | Identity 单窗 | Identity 10窗 | EER 单窗 | EER 10窗 | 同期 Task accuracy / Macro-F1 | Task 基线 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SEED S1+S2→S3，2 seeds | 15 | 3,600 / 1,800 | 66.28% ± 4.87% | 78.33% ± 3.14% | 13.46% ± 0.70% | 8.59% ± 0.36% | emotion 55.36% ± 1.37% / 55.05% ± 2.02% | 3 类均衡 33.33% |
| FACED block 0–5→6–7，3 seeds | 123 | 14,760 / 4,920 | 99.53% ± 0.31% | 99.80% ± 0.00% | 0.325% ± 0.250% | 0.140% ± 0.110% | emotion 23.59% ± 1.90% / 19.79% ± 1.56% | 9 类均匀 11.11%；多数类 15.61% |
| ISRUC-II recording 1→2 | 8 | 9,069 / 8,739 | 55.21% | 62.11% | 25.09% | 23.54% | sleep stage 64.90% / 62.81% | 5 类均匀 20%；多数类 21.97% |
| ISRUC-II recording 2→1 | 8 | 8,739 / 9,069 | 66.96% | 75.91% | 16.50% | 11.41% | sleep stage 61.27% / 59.74% | 5 类均匀 20%；多数类 21.17% |
| BCICIV-2a session 1→2 | 9 | 2,328 / 2,368 | 64.48% | 64.81% | 35.26% | 35.19% | 4-class MI 46.66% / 44.88% | 25%；多数类 25.34% |
| BCICIV-2b screening→feedback，3 seeds | 9 | 1,844 / 3,423 | 86.14% ± 2.97% | 94.59% ± 1.19% | 6.26% ± 1.08% | 3.22% ± 0.78% | 2-class MI 72.22% ± 1.10% / 71.88% ± 1.03% | 50%；多数类 50.16% |
| BCICIV-2b feedback→screening | 9 | 3,423 / 1,844 | 87.47% | 97.22% | 5.64% | 1.48% | 2-class MI 67.35% / 67.15% | 50%；多数类 50.22% |

论文比较必须匹配 split、窗口聚合、registered/external impostor 和 EER 定义。FACED 的极低 EER 只说明同 recording 的 block 保持了极强身份指纹，不能与 BMT_EEG 或跨天数据的 EER 直接排名。BCICIV-2a 的 10 窗没有改善，说明错误主要不是独立窗口噪声；BCICIV-2b、SEED 和 ISRUC-II 的聚合改善则说明连续观察提供了互补身份证据。

## 训练数据如何组成

- SEED：62 通道、200 Hz、4 秒窗口；每个电影 trial 最多均匀取 8 个不重叠/分散窗口。每人每 session 120 窗；S1+S2 的 240 窗/人训练，S3 的 120 窗/人测试。15 人都出现在两侧，emotion 为 positive/neutral/negative。
- FACED processed：32 通道、250 Hz、10 秒窗口；每人 8 个 processed block、每 block 20 窗。block 0–5 的 120 窗/人训练，block 6–7 的 40 窗/人测试，共 123 人；九类 emotion 在训练和测试都存在。它没有第二次佩戴。
- ISRUC-II：6 通道、200 Hz、10 秒窗口；每个 30 秒 sleep epoch 分为三个不重叠窗口，每个 recording、每个 stage 最多均匀选择 80 个 epoch。8 人都具有两个 recording；一个完整 recording 训练，另一个完整 recording 测试，并做双向实验。任务为 W/N1/N2/N3/REM。
- BCICIV-2a：22 通道、250 Hz；每个未拒绝 MI trial 截取 cue 后 2.5–6.0 秒的 3.5 秒窗口。9 人的 session 1 全部有效 trial 训练，session 2 全部有效 trial 测试；任务为左手、右手、双脚、舌运动想象。
- BCICIV-2b：C3/Cz/C4 三通道、250 Hz；每个未拒绝 trial 截取 cue 后 3.5–7.0 秒。screening session 1–2 与 feedback session 3–5 严格互换训练/测试；任务为左右手运动想象。

所有网络输入在每个窗口、每个通道上做 z-score，因此固定通道乘常数的 `gain ×0.5` 会被归一化消除。它是管线负对照，不代表真实 EEG 幅值在生理上没有信息。训练没有把同一 recording 的随机窗口打散到两侧；FACED 虽按 block 切分，仍属于同 recording 范围。

任务网络的 loss 还有一个版本差异需在论文对比中披露：SEED 和 FACED 使用按训练类频率加权的 cross-entropy；首批 ISRUC-II 和 BCICIV-2a/2b pilot 使用普通 cross-entropy。后三者的测试类别分布接近均衡，因此当前数字可用作 pilot，但正式与论文对比前应统一 loss 并补齐多 seed。

## 当前“解耦”做到什么程度

当前完成的是双网络可控干预归因，而不是已经训练完成的 disentangled representation。IdentityNet 只用 subject cross-entropy，TaskNet 只用 task cross-entropy，网络彼此独立；尚未加入 identity SupCon、task/session adversary、HSIC/cross-covariance 或 cross-session prototype stability。因此下面的“身份特异、任务特异、共享、弱”是根据同一原始 EEG 干预对两个网络输出和 latent 的影响划分，不应描述成数学上已经正交的两个子空间。

跨数据集最强共性是：身份不只来自 PSD。保留逐通道 PSD 的相位随机化和逐通道独立循环时移，会在所有数据集显著移动 identity/task fusion 与 embedding，并明显降低输出性能。identity embedding 对相位随机化的平均 cosine change 分别为 SEED 0.436、ISRUC-II 0.980、BCICIV-2a 0.773、BCICIV-2b 0.644、FACED 0.650；循环时移结果非常接近。这支持“多通道相位关系、跨通道时间对齐和 waveform morphology 是身份与任务共享骨架”，而不是“仅频带功率足以解释身份”。

| 数据集 | 稳定身份保护候选 | 稳定任务强、身份弱候选 | 稳定共享特征 | 当前弱/冗余或未通过项 |
| --- | --- | --- | --- | --- |
| SEED | alpha、beta；中央—顶叶区域 | 暂无跨 seed 稳定项 | frontal；相位随机化；跨通道时移 | gamma、time reversal 弱；delta/theta/颞枕区域不稳定 |
| FACED | 暂无满足严格阈值的独占项 | 暂无 | frontal、parietal-occipital；相位随机化；跨通道时移 | beta/gamma 对两输出均弱；delta/theta/time reversal 接近但未通过三 seed 硬阈值 |
| ISRUC-II | alpha | slow 0.5–2 Hz | theta；额/中央/枕区；相位随机化；跨通道时移 | beta/delta/gamma/sigma 与 time reversal 跨方向不稳定 |
| BCICIV-2a | 只能列单 seed 候选，不能称稳定 | time reversal 为单 seed 候选 | alpha/mu、motor/C3、相位随机化、跨通道时移在单 seed 同时影响两者 | 只有一个 seed、一个方向，所有结论待复核 |
| BCICIV-2b | beta-low 13–20 Hz | 暂无 | alpha/mu；C3/Cz/C4 与整体 motor 区；相位随机化；跨通道时移 | delta/theta/gamma/time reversal 不稳定；beta-high 较弱 |

差异与任务生理一致：睡眠的 slow oscillation 对 sleep-stage 很重要但对身份影响很小，theta 和全脑六通道结构则为共享信息；运动想象的 mu/alpha 与 C3/Cz/C4 同时承担任务和身份，beta-low 更偏身份；SEED 的 alpha/beta 与中央—顶叶更偏身份，frontal 更像 identity/emotion 共享结构；FACED 的任务网络泛化仅约 24%，所以目前不能从很小的 task drop 推导“任务无关”。

早期 attribution audit 曾把 ISRUC-II slow 0.5–2 Hz 视为最可信的可修改候选：audit subset 上两个 session 方向的 identity drop 为 −2.73% 到 0.76%，sleep-stage task drop 为 6.37% 到 16.19%。随后 full held-session 主动干预发现 R1→R2 完全衰减时 identity 下降 2.46 pp，超过 2 pp 容差，因此该候选已降级为方向相关，不再作为跨方向稳定结论。当前稳定通过的是 FACED processed 的 1–8 Hz 衰减，但它仅是 within-recording 且任务 baseline 较弱；详情见 `eeg-identity-protected-task-intervention-20260918.md`。身份保护候选仍包括 ISRUC-II alpha、BCICIV-2b beta-low，以及 SEED 的 alpha/beta 和中央—顶叶信息。相位结构、跨通道同步和任务主脑区应默认视为共享敏感特征，不能直接修改。

机器可读结果位于 `/home/undefined/Desktop/EEG/outputs/server_a100_pilots/aggregate_summary.json`、`verification_eer_aggregate.csv`、`intervention_aggregate.csv` 和 `latent_aggregate.csv`；各 run 目录内保留独立 `summary.json` 与 `verification_eer.json`。
