# EdgeForge 当前内容索引

更新时间：2026-09-18

本索引区分当前开发入口与历史证据。历史版本、发布说明和日志不因本次整理删除；它们用于复现版本演进和审计。原始 EEG、模型 checkpoint 和服务器工作目录不纳入仓库，派生结果必须带日期、manifest 和复现脚本。

## 当前开发入口

- `src/edgeforge/`：compiler、IR、runtime、backend registry、worker/artifact 协议、LoP metrics 和实验 API。
- `scripts/eeg_lop_diagnostics.py`：统一 EEG LoP 诊断和指标 plumbing。
- `scripts/subject_identity_probe.py`：固定表征的跨 sequence subject-ID probe。
- `scripts/subject_identity_granularity.py`：epoch/sequence 粒度、k-epoch 聚合、known/unknown open-set 审计。
- `scripts/brainuicl-subject-stability-audit.py`：BrainUICL 各 tap 的 embedding 提取和可视化，可按需重生成已裁剪的大型中间矩阵。
- `scripts/analyze-subject-invariance.py`：waveform/BrainUICL 的 sequence-level ICC 与稳定性分析。
- `scripts/brainuicl-instrumentation.py`、`scripts/brainuicl-model-pipeline.py`：模型导出、层级诊断和前后端验证。

## 当前结果入口

- `docs/eeg-cross-session-identity-research-20260917.md`：SEED、ISRUC-II、BCICIV-2a/2b 的真实跨 session 身份研究、任务特异特征、verification EER、BED/ds004148/BMT_EEG 可用性和解耦路线；2026-09-18 已补实际实验结果。
- `docs/eeg-identity-feature-protection-20260918.md`：已登记个体 closed-set 目标、显式—latent 联合审计、strong/weak 特征定义和持续学习保护路线。
- `docs/eeg-dual-network-intervention-design-20260918.md`：身份/任务双网络、原始 EEG 可控干预、数据集专属归因和解耦损失选择。
- `docs/eeg-identity-task-eer-summary-20260918.md`：各数据集闭集身份准确率、训练/测试组成、prototype verification EER、同期任务准确率以及显式—latent 解耦共性与差异。
- `docs/eeg-identity-protected-task-intervention-20260918.md`：在身份准确率、身份 latent 与显式身份频带联合约束下修改原始 EEG；记录 FACED 跨三 seed 稳定结果、BCICIV-2a 单次候选以及 SEED/ISRUC-II/BCICIV-2b 的失败边界。
- `/home/undefined/Desktop/EEG/docs/server-a100-execution-20260918.md`：school-gpu 数据去重、Slurm 作业、A100 结果及跨运行干预稳定性记录。
- `docs/eeg-architecture-lop-analysis-20260911/subject-id-probe-20260915/`：ISRUC/FACED 的输入和编码后身份可解码性。
- `docs/eeg-architecture-lop-analysis-20260911/subject-id-granularity-20260915/`：粒度/open-set 结果。
- `docs/eeg-architecture-lop-analysis-20260911/subject-id-cl-tracking-20260915/`：持续学习身份追踪。
- `docs/eeg-architecture-lop-analysis-20260911/identity-fingerprint-audit-20260915/`：可解释波形指纹特征、消融和跨数据集报告。
- `docs/eeg-architecture-lop/`：面向阅读的图表和解释；逐被试全 epoch 大图按需重生成，不作为长期归档。
- `.edgeforge/*20260915*`、`docs/*20260915*`：本轮 CUDA/compiler、RK3588 和 EEG 验证产物；保留日期和运行配置。

## 仍需保留的历史内容

- `releases/`、`CHANGELOG.md`、`logs/` 和 `logs/archive/`：每个 version 的发布说明与原始运行日志。
- `docs/design-v*.md`、`docs/roadmap-v7-plus.md`、`docs/system-direction-2026-08-16.md`：设计演进和方向决策，不能用当前 README 替代。
- `config/`：实验契约和可复现配置；过期配置只作为历史版本输入，不应当被当前默认流程自动选用。

## 清理边界

本次删除的是可由脚本重生成且与现有 sequence-level 结果重复的 v11 编译中间二进制、旧 identity-audit epoch 矩阵和重复逐被试大图。没有删除源代码、测试、版本日志、sequence embedding、summary、manifest 或实验报告。若需要旧二进制/大图，应从 Git 历史或脚本重新生成。
