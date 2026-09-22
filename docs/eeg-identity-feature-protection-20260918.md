# EEG 身份特征解耦与保护设计

更新日期：2026-09-18

当前目标明确为已登记个体的 closed-set identification：训练阶段包含全部待识别 subject，测试阶段输入这些 subject 在新 session、不同状态或后续时间采集的 EEG。未见 subject 泛化和开放集拒识不是本阶段要求；完整 session 留出仍必须保留，用于排除一次佩戴和连续窗口泄漏。

特征分析从现在起同时覆盖 CNN 前显式特征和网络内部 latent，而不是先后割裂。显式特征包括相对谱、aperiodic、归一化时域形态、空间相关/连接、左右不对称及数据集专属事件；latent 包括多尺度 temporal、脑区汇聚、spatial、fusion 和最终 identity embedding。每一项同时评估 held-session subject probe、跨 session 同人稳定、emotion/stage/task 与 session 泄漏、输入或表示干预后的 accuracy/margin 下降。

最终清单分为四类：稳定身份核心、状态调制身份、session/acquisition shortcut、弱或冗余。只有第一类进入持续学习保护集合；第二类进入条件化分支；第三类只作混杂诊断；第四类先进入 plastic 候选集合。单维置零影响小只表示“单独非关键”，多个冗余维度仍可能共同承载身份，因此正式保护单位应是跨 fold/seed 稳定的子空间或参数重要性 mask，而不是某次训练的固定坐标号。

已有 SEED `S1+S2 → S3` checkpoint 已完成一次 5,400 窗口的联合 pilot。原 CNN 为 67.44%；固定表示的线性 identity probe 中，区域 temporal latent 为 68.72%、spatial latent 为 65.72%、fusion 为 65.89%、最终 256 维 embedding 为 66.28%。幅值不变显式联合模型为 72.11%；移除相对谱下降 12.11 个百分点，移除归一化时域形态下降 5.33 个百分点，而移除谱熵或同源不对称没有下降。该结果把相对谱和时域形态列为首批强候选，把谱熵/同源不对称列为当前弱或冗余候选，但尚不能跨 seed 定案。

最终 embedding 仍能以 49.33% 预测三类 emotion（chance 33.33%），说明当前纯 subject cross-entropy 网络尚未完成状态解耦。第一版损失确实只有 subject CE；后续将在三 fold × 三 seed baseline 后逐项加入跨 session supervised contrastive、identity 分支的 state/session adversary、独立 state 分支和 cross-covariance/HSIC 约束。保护阶段再比较冻结、EWC/SI 和梯度投影，并以跨 session identity accuracy、EER 和正确类别 margin 是否保持作为通过条件。

完整方法、表格、限制和产物见 `/home/undefined/Desktop/EEG/docs/explicit-latent-identity-protection-plan.md`。机器可读结果位于 `/home/undefined/Desktop/EEG/outputs/seed_explicit_latent_audit_s3_seed4321/`；其中 `feature_inventory.csv` 汇总全部显式块和 latent tap，`embedding_dimension_audit.csv` 保留全部 256 个维度的强/弱证据。复现脚本为 `/home/undefined/Desktop/EEG/scripts/analyze_seed_explicit_latent_identity.py`。这次没有重复既有 ISRUC/FACED class-conditioned probe 或 continual-learning checkpoint tracking。
