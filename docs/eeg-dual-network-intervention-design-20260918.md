# EEG 身份—任务双网络与原始信号干预设计

更新日期：2026-09-18

本轮明确采用两个独立网络作为解耦前基线：IdentityNet 只预测已登记 subject，TaskNet 只预测 emotion、sleep stage 或 motor-imagery task。此前其他数据集并非没有实验：SEED、ISRUC-II、BCICIV-2a/2b 已有严格 cross-session 显式特征结果，FACED 有 within-recording control；只有显式—latent 联合审计先在已有深度 checkpoint 的 SEED 上完成。下一轮补齐 ISRUC-II、BCICIV 和 FACED 的波形网络与 latent，不重复已有线性 probe。

新管线对原始 EEG 施加频带抑制、脑区遮挡、区域 gain、保 PSD 相位随机化、逐通道循环移位和时间反转，同时测量身份与任务准确率下降、显式谱/形态/相关变化以及 temporal、spatial、fusion、embedding latent 位移。身份下降小而任务下降大的干预是“身份弱相关、任务强相关”的可修改候选；身份下降大而任务下降小的是保护候选；二者都大的是共享特征；二者都小的仍需组合干预排除冗余。

联合解耦只在独立 IdentityNet 达到 cross-session 可用门槛后开始。候选损失依次加入 subject CE、task CE、跨训练 session 的 identity SupCon、identity latent 上的 task/session GRL、identity/task HSIC 或 cross-covariance，以及同人跨 session stability。权重按单项消融与 Pareto 曲线选择，身份准确率/EER 是硬门槛，不通过任意加权和换取表面上的低 task leakage。

完整实验契约、公式、数据集专属干预、保护标准和服务器配置见 `/home/undefined/Desktop/EEG/docs/dual-network-intervention-and-loss-design.md`。实现入口为 `/home/undefined/Desktop/EEG/scripts/prepare_waveform_cache.py`、`train_dual_networks.py` 和 `slurm_dual_pipeline.sh`。

A100 首批单 seed pilot 已完成：BCICIV-2a S1→S2 identity 为 64.48%、task 为 46.66%；ISRUC-II R1→R2 identity 为 55.21%、10 窗聚合 62.11%、task 为 64.90%；BCICIV-2b screening→feedback identity 为 89.16%、10 窗聚合 95.28%、task 为 72.19%。只有 2b 当前达到可用 pilot 水平，已继续提交两个 seed 和反向 protocol；2a/ISRUC 先改数据集专属前端，不提前加入 adversarial 解耦。

早期 attribution audit 中，BCICIV-2b screening→feedback 三 seed 的单窗口 identity 为 86.14% ± 2.97%，10 窗为 94.59% ± 1.19%，task 为 72.22% ± 1.10%；反向 protocol 的单窗口/10 窗为 87.47%/97.22%。首个 seed 的 delta 任务选择性没有跨 seed 复现，不能定为可修改特征。ISRUC-II 反向 R2→R1 的单窗口/10 窗为 66.96%/75.91%；较小 audit subset 曾提示 slow 0.5–2 Hz 在两个方向都是“身份弱、任务强”候选，但随后 full held-session sweep 发现 R1→R2 完全衰减时 identity 下降 2.46 pp，超过 2 pp 容差，因此已降级为方向相关候选。alpha 仍是两个方向一致的身份保护候选。最新主动干预结果见 `eeg-identity-protected-task-intervention-20260918.md`。

SEED S1+S2→S3 两个 seed 的单窗口 identity 为 69.72%/62.83%，10 窗为 80.56%/76.11%，emotion task 为 54.39%/56.33%。当前模型已学到跨 session 身份证据，但尚未达到可用门槛。两个 seed 一致支持 alpha、beta 和中央—顶叶为身份保护候选，相位与跨通道时序为 identity/emotion 共享结构；单 seed 的右颞任务选择性未复现，不列为已确认可修改特征。

FACED processed blocks 0–5→6–7 三 seed 的单窗 identity 为 99.53% ± 0.31%，10 窗为 99.80% ± 0.00%，emotion 为 23.59% ± 1.90%。这证明 123 人的同 recording 未见 block 身份区分非常强，但不能宣称为 cross-session permanence。三 seed 没有任何干预同时通过 identity drop ≤2% 与 task drop ≥5% 的可修改门槛；相位随机化、跨通道时移、额区与顶枕区遮挡是稳定共享特征。身份 embedding 对前两种干预的平均 cosine change 为 0.650/0.642；beta/gamma 当前对两输出都弱，但不应解释为生理上不存在信息。

远端 `~/lzh/datasets/FACED` 和 `~/lzh/datasets/ISRUC` 已分别核验为与本地相同的 raw FACED 和 Group I processed ISRUC，不再重复传输。新补数据归档到同一 `~/lzh/datasets/` 下，实验目录只保留软链接。FACED processed 已核验为 1,969 文件、6,298,039,326 bytes，manifest 和抽样 NPY SHA-256 与本地相同；FACED raw 仍使用既有 32 GB 副本。
