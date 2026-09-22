# 身份保护下的任务特征干预实验

更新日期：2026-09-18

## 实验目标

本实验直接检验：在已训练的 closed-set 身份网络与同期任务网络上，能否修改原始 EEG 的任务相关成分，使任务准确率明显下降，同时让身份预测和身份 latent embedding 基本不变。它不是跨人泛化认证，也不是重新训练后的鲁棒性实验；身份类别均为训练阶段已登记的受试者。

输入干预发生在 channel-wise normalization 之前的原始 EEG 窗口。每个候选同时由三类身份保护条件约束：单窗口身份准确率最多下降 2 个百分点；干预前后身份 embedding 的平均 cosine distance 不超过 0.07；此前判为身份重要的显式频带，其 log-power MAE 不超过 0.02。只有在上述条件全部满足、且同期任务准确率至少下降 5 个百分点时，才记为通过。

不同数据集使用与任务和既有归因结果相符的候选，而不是假设同一频带对所有范式作用相同：

| 数据集 | 任务 | 本轮主要候选 | 显式身份保护频带 |
| --- | --- | --- | --- |
| SEED | 三分类情绪 | 1–4 Hz、1–8 Hz 衰减；右颞区向全通道均值混合；时间反转混合 | alpha 8–13 Hz、beta 13–30 Hz |
| FACED processed | 九分类情绪 | 1–4 Hz、1–8 Hz 衰减；时间反转混合 | 暂无经过跨 seed 验证的 identity-only 频带 |
| ISRUC-II | 五分类睡眠阶段 | slow 0.5–2 Hz、beta 13–30 Hz 衰减；时间反转混合 | alpha 8–13 Hz |
| BCICIV-2a | 四分类 motor imagery | 1–4 Hz、1–8 Hz、alpha/mu 8–13 Hz 衰减；时间反转混合 | alpha/mu 8–13 Hz |
| BCICIV-2b | 二分类 motor imagery | 1–4 Hz、1–8 Hz、alpha/mu 8–13 Hz 衰减；时间反转混合 | beta-low 13–20 Hz |

频带干预通过 FFT 将选定频率 bin 的复谱乘以 `1-strength` 后逆变换实现，strength 为 0.25、0.5、0.75 或 1.0。时间反转候选在原窗口与反转窗口之间线性混合；因此 strength 1.0 的完全反转保持每通道 PSD，但会改变时间方向、瞬态形态和网络响应。

## 主要结果

五个数据集/协议组共运行 12 个既有 checkpoint，得到 72 个“候选×强度”聚合行。只有 FACED processed 的 1–8 Hz 衰减在三个 seed 上稳定通过全部条件。

| 数据/干预 | 重复数 | 身份准确率（前→后） | 任务准确率（前→后） | 身份下降均值/最坏 | 任务下降均值/最小 | 身份 embedding 变化均值/最大 | 结论 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| FACED，1–8 Hz 衰减 75% | 3 seeds | 99.53%→98.99% | 23.59%→15.12% | 0.54/0.89 pp | 8.48/6.48 pp | 0.042/0.044 | 稳定通过，首选 Pareto 点 |
| FACED，1–8 Hz 完全去除 | 3 seeds | 99.53%→98.48% | 23.59%→14.29% | 1.05/1.57 pp | 9.30/7.30 pp | 0.053/0.055 | 稳定通过，但身份扰动更大 |
| BCICIV-2a，完全时间反转 | 1 run | 64.48%→63.01% | 46.66%→39.65% | 1.48 pp | 7.01 pp | 0.045 | 单次候选，尚非稳定结论 |


FACED 的 75% 衰减是当前更合适的操作点：与完全去除相比，它少牺牲约 0.52 个百分点身份准确率，而仍使 emotion task 平均下降 8.48 个百分点。三个 seed 的 task drop 分别为 6.48、7.89 和 11.06 个百分点，identity drop 分别为 0.30、0.41 和 0.89 个百分点，不是由单一随机 seed 驱动。

这项 FACED 结果只能称为同一次 recording 内、held-out video block 上的稳定 proof-of-concept。FACED 没有第二 session；其九分类情绪 baseline 也只有 23.59%（chance 为 11.11%，测试多数类为 15.61%）。此外，FACED 尚无可登记为 identity-only 的显式保护频带，所以表中的显式频带误差为零只是“没有该项约束”，不是证明某个 FACED 身份频带被保护。真正有效的保护证据是 identity accuracy 和 identity embedding 在三个 seed 上均保持稳定。

BCICIV-2a 的完全时间反转几乎不改变 alpha/mu PSD（log-power MAE 约 `5.0e-8`），同时满足一次身份/任务阈值。这说明任务网络可能比身份网络更依赖时间方向或瞬态形态，但目前只有一个 seed、一个 session 方向，必须增加 seed 并做反向 session 才能升级为结论。

## 未通过与修正结论

- SEED：所有候选都未在两个 seed 上达到“任务至少下降 5 pp、身份最多下降 2 pp”。例如 1–8 Hz 衰减 75% 时，身份平均下降 5.39 pp，而情绪仅下降 2.56 pp，说明当前候选没有隔离出安全可改的情绪特征。
- ISRUC-II：slow 0.5–2 Hz 衰减在 recording 2→1 方向表现很好；完全去除使身份仅下降 0.52 pp、sleep-stage 下降 18.86 pp。但 recording 1→2 的身份下降为 2.46 pp，超过预设 2 pp 容差。因此此前基于较小 audit subset 得出的“跨两个方向稳定”结论被本次 full held-session sweep 降级为方向相关候选。
- BCICIV-2b：没有候选在三个 screening→feedback seeds 和一个 feedback→screening 方向上共同通过。1–8 Hz 衰减 75% 使 identity 平均下降 12.71 pp，只使 task 下降 1.81 pp，明显不是安全任务特征。

这些失败不是“没有任务信息”的证据，而是说明当前简单频带/时间变换没有找到满足严格身份保护条件的输入空间方向。SEED 与 BCICIV-2b 的身份和任务表征共享程度更高；ISRUC-II 还存在明显 acquisition-direction shift。下一阶段应在新的 validation split 上学习受约束的输入扰动，并把 identity logit、identity embedding 与显式身份频带联合设为不可越过的约束，而不是放松阈值来制造通过结果。

## 解释边界与复现

本轮是对已用过的 held domain 做 exploratory Pareto sweep，阈值和强度选择都看到了该 held domain；因此结果不能再把同一测试域当作完全未见的最终检验。下一轮需要固定 FACED 的 75% 方案后换新 video folds；对真正的长期身份结论，则应优先在 SEED、ISRUC-II、BCICIV-2a/2b 的新 session 方向或新 fold 上复核。

本轮没有直接篡改 latent 坐标。实际操作是修改可回映射到原始 EEG 的显式信号成分，并用身份 latent embedding 作为保护约束；这样结果具备“修改哪段原始 EEG、身份和任务各下降多少”的因果可操作性。直接 latent editing 可作为机制定位工具，但若无法可靠解码回 EEG，就不应作为最终可实施的保护方案。

实现与机器可读结果位于 EEG 工作目录：

- `/home/undefined/Desktop/EEG/scripts/evaluate_identity_protected_interventions.py`
- `/home/undefined/Desktop/EEG/scripts/slurm_identity_protected_interventions_all.sh`
- `/home/undefined/Desktop/EEG/scripts/summarize_identity_protected_interventions.py`
- `/home/undefined/Desktop/EEG/outputs/server_a100_pilots/identity_protected_intervention_aggregate.json`
- `/home/undefined/Desktop/EEG/outputs/server_a100_pilots/identity_protected_intervention_aggregate.csv`

Slurm jobs 207934–207938 均以 exit code 0 完成；每个 run 的原始结果保存在对应输出目录的 `identity_protected_interventions.json`。聚合器要求多次运行全部通过才标记 `stable_pass`；单次通过只能标记 `single_run_candidate`，不会与稳定结果混合。
