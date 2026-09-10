# EdgeForge V7：RA-EEG Experiment Contract

## 目标

V7 将本机已经完成的 EEG 持续学习实验和新的 LoP Probe 接入 EdgeForge，使研究结果具备统一身份、可重放规格、结构化指标、内容寻址 Bundle、Worker/Runtime 版本和不可覆盖日志。它不重写 BrainUICL、SPR、PuriDivER 或正则化算法。

```text
Existing or new EEG experiment
          │
          ▼
ExperimentSpec
          │
          ▼
EdgeForge experiment_run
          │
          ├── command: execute trusted existing runner
          └── import: read an existing metrics.json
          │
          ▼
RA-EEG metrics adapter
          │
          ├── summary metrics
          ├── per-task plasticity/current before/after
          ├── forgetting/BWT/stability series
          └── effective/stable rank and weight norm
          │
          ▼
ExperimentBundle + Artifact digest + MetricSeries
```

## ExperimentSpec

必需字段包括 `experiment_id`、`workload`、dataset、model、protocol、method、seed 和 runner。dataset/model 至少包含 `name`；本地 Catalog 还记录 split manifest digest 和 checkpoint component digest。

runner 有两种模式：

- `command`：Worker 通过既有命令白名单执行真实实验，成功后读取 `result_path`。
- `import`：不重新训练，读取 Worker 工作根目录内已有 `metrics.json`，将历史实验纳入同一证据系统。

`result_path` 和 `cwd` 都受 Worker `work_root` 边界保护。命令仍使用 `shell=False`，不扩大现有执行权限。第一版 adapter 支持 `raeeg-metrics-v1` 和已经规范化的 `edgeforge-bundle-v1`。

## ExperimentBundle 与 MetricSeries

Bundle 保存完整 spec、环境/硬件快照、源结果 SHA-256、源结果大小、摘要和归一化指标。Bundle 自身作为 `experiment-bundle` 写入内容寻址 Artifact Store；数据库保存 Artifact digest，而不是复制 checkpoint 或原始 EEG。

`experiment_runs` 保存 workload、dataset、model、protocol、method、seed、控制面版本、Worker Runtime 版本、摘要和 Artifact；`experiment_metrics` 保存 namespace、name、value、step、unit 和 context。RA-EEG context 当前可包含 subject，后续可以扩展 layer、split、checkpoint stage 与 probe budget。

RA-EEG adapter 会从 summary、stability、plasticity、forgetting、spectrum、weight norm 和逐 task 结构提取数值；如果已有结果只有 current-before/current-after，系统显式派生 `plasticity.acc_gain` 和 `plasticity.mf1_gain`。派生值是一次适配前后变化，不自动等价于固定预算 LoP Probe。

## 本地实验 Catalog

`config/raeeg-local-catalog.json` 当前登记八个真实结果：

| Comparison group | 方法 | 是否可直接同表比较 |
|---|---|---|
| `aligned-full49` | BrainUICL、Finetune、EWC、Online EWC、SI、MAS | 是，但仍需同时报告 replay/资源差异 |
| `method-transfer` | SPR-EEG、PuriDivER-EEG | 否；协议、backbone 或数据划分不同 |

Catalog 记录的结果文件 SHA-256 由 ExperimentBundle 在导入时重新计算。批量脚本只提交 import 任务，所有解析和持久化仍经过 Worker 与控制面标准协议。

## LoP Probe

`experiments/raeeg_lop_probe.py` 位于现有 BrainUICL 工作区。它加载真实 pretrain 或 milestone checkpoint，对 checkpoint 之后的下一个 ISRUC subject 做确定性 train/eval sequence split，在模型副本上执行固定预算 supervised oracle probe，并记录：

- step-wise held-out ACC/MF1、learning gain 与 AULC；
- Fusion、Transformer 和 classifier-input 的 Effective Rank、Stable Rank、Rank@90/95 和最大奇异值；
- Feature Extractor、Transformer、Classifier 和全局 Weight Norm；
- checkpoint stage、next subject、训练/评估 sequence 数量和 probe 配置。

该 Probe 明确标为 `supervised-oracle-fixed-budget-heldout`。它诊断“当前表示和参数是否仍能在固定预算下学习”，但不代表 BrainUICL 无监督 target protocol 的部署性能。正式 LoP 结论至少需要多个 checkpoint stage、完整预算、多个 seed 和预注册统计分析；V7 release smoke 只验证链路可运行。

## 版本化 LoP 分析

控制面现在可以从已持久化的 Experiment Bundle 创建 `lop-lagged-correlation-v1` 分析。默认关系是 `ER(t-1) → Plasticity(t)`，其中 `lag=1` 表示 checkpoint 排序中的前一个 stage，而不是把 stage 整数减一；因此 stages `[0, 10, 25]` 会产生 `0→10` 和 `10→25` 两个 transition。

分析只读取每个 `experiment_id` 最新一次 task 的指标，并把 task、控制面/Runtime 版本、source digest 和 Artifact digest 写入分析身份。输入实验必须共享 workload、protocol、comparison group、method、checkpoint transition 和 lagged context pairing grid，每个实验必须提供不同 seed；用户选中的任何实验缺少可用 pair 时都会阻断完整分析。`context_policy=exact` 只在完全相同的 metric context 内配对，用于防止不同 subject 串配；`aggregate-step` 则先在同一 stage 内取均值。

输出包含 Pearson、Spearman、确定性 seed-cluster percentile bootstrap 95% CI、实际 pair、阻断原因和 SHA-256 `analysis_digest`。最低 seed 门槛固定为 3，即使调用者传入更小的 `minimum_seeds` 也不会放宽。`status=ok` 只说明版本化描述性统计具备足够输入，不代表显著性、因果性或 LoP 机制成立；`scientific_conclusion_allowed` 永远为 `false`。

## 数据与安全边界

- 原始 ISRUC/FACED payload 保留在共享数据盘，控制面只看到数据身份、manifest digest 和 Worker 能力。
- checkpoint 不因 V7 自动上传；Catalog 只保存组件 digest。后续 Model Registry 再定义受控 checkpoint Artifact 策略。
- Worker 只允许显式列入白名单的 Python 环境，result/cwd 不能越过工作根目录。
- 现有实验中的 evaluator 标签只用于诊断指标；Catalog metadata 显式记录 target label 是否参与训练。

## 当前边界

V7 还没有 Model promotion/reject/rollback policy，也没有按 LoP 阈值自动发布；这些属于 V8 Capability Gate。V7 的目标是先保证身份、输入、环境、指标和证据可追溯。旧结果的 summary 字段仍可能因实验协议不同而语义不同，因此 comparison group 和 protocol 是查询与比较的硬前提。
