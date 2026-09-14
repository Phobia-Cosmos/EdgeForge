# 服务器 ISRUC/FACED 个体特征分析（2026-09-13）

本目录保存服务器上只读分析得到的 FACED 个体指纹图和报告。分析没有改写原始 EEG，也没有把身份可分性当作 LoP 结论。

## 结果

| 数据集 | 被试 | 样本 | 特征维度 | 最近中心准确率 | Logistic 身份探针 | 被试间/被试内距离比 |
|---|---:|---:|---:|---:|---:|---:|
| ISRUC（4 sequence pilot） | 98 | 7,840 | 83 | 26.12% | 55.59% | 1.651 |
| ISRUC（8 sequence CPU balanced） | 98 | 15,680 | 83 | 22.45% | 60.69% | 1.492 |
| FACED | 123 | 3,444 trials | 323 | 94.13% | 98.61% | 2.593 |

三项实验均采用被试内部的 sequence/trial-disjoint 划分：前半部分形成被试 profile，后半部分只作 held-out 测试。因此 FACED 的高准确率表示跨 trial 可复现的个体模式，而不是单个 epoch 的绝对生物识别保证。ISRUC 的平衡 CPU 结果在扩大到每人 8 个 sequence 后，Logistic 探针从 55.59% 上升到 60.69%，但最近中心从 26.12% 降到 22.45%，距离比从 1.651 降到 1.492；这说明个体信息可复现，但 profile 受睡眠阶段、sequence/session 漂移和采集条件影响，不能简化成固定模板。

平衡 CPU 结果位于 `ISRUC-balanced/`，采用全部 98 个被试、每人 8 个 sequence、每个 sequence 的 20 个 epoch，共 15,680 个 epoch。此前尝试处理全部 4,276 个 sequence 文件时计算量过大，已停止；原始数据未修改。

## 保留哪些 EEG 特征

后续定向修改应同时约束通道 log-RMS、增益不变的通道平衡、频带相对功率、谱熵、temporal roughness 以及带符号的跨通道连接摘要。只保持 RMS 或 PSD 会丢失参考/极性和通道关系。建议以每个被试的 median/MAD profile 作为 envelope，从该被试残差分布采样，并在每次修改后拒绝越界样本。

## GPU 调度状态

已提交 Slurm 作业 `196614`（1×A100、8 CPU、64 GB、12 h），用于完整 ISRUC+FACED 特征分析。当前唯一 GPU 节点 `node193` 被 Slurm 标记为 `DOWN* / Not responding`，作业状态为 `PENDING`，因此尚未消耗 GPU，也不能宣称 GPU 实验已经运行。节点恢复后该作业会按修正后的脚本继续执行。

## 解释边界

个体可分性只证明存在可复现的 subject-specific pattern，不等于 LoP。LoP 仍需单独报告 warm/fresh gap、retention、rank、谱和梯度等训练指标；本目录的结果用于为“保留个体特征、改变任务/漂移”的数据变换提供约束。
