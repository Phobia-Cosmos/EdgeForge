# RA-EEG LoP Versioned Analysis Contract

## 目标与边界

`lop-lagged-correlation-v1` 将已持久化的 RA-EEG Experiment Bundle 转换为可审计、可重复、内容寻址的 LoP 描述性分析。默认检验前一 checkpoint 的 Transformer effective rank 是否与后一 checkpoint 的 accuracy plasticity gain 相关，即 `task.spectra.transformer_1.effective_rank(t-1) → plasticity.acc_gain(t)`。

该模块不运行 BrainUICL、不读取原始 EEG、不复制 checkpoint，也不把历史导入或单 seed smoke 升级为科学结论。所有输出固定包含 `scientific_conclusion_allowed=false` 和 `interpretation="descriptive association only; not a causal LoP conclusion"`。

## 配对与统计契约

- 默认 `lag=1` 按每个实验实际存在的有序 checkpoint stage 配对。stages `[0, 10, 25]` 产生 `0→10` 与 `10→25`，而不是查找不存在的 `9` 或 `24`。
- `context_policy=aggregate-step` 对同一 stage 的同名指标取均值；`context_policy=exact` 只在完全相同的 context JSON 内配对，防止 subject 或其他上下文串配。不同 seed 还必须具有相同的 `(source stage, outcome stage, context)` 配对网格，避免 subject 集合不一致造成加权偏差。
- Pearson 与 Spearman 使用全部有效 lagged pair 计算。predictor 或 outcome 无变化时相关系数不可定义，状态为 `insufficient-variation`。
- 95% CI 使用确定性的 seed-cluster percentile bootstrap。每次重采样以 seed/experiment cluster 为单位，避免把同一 seed 的多个 checkpoint 当成独立样本。
- `minimum_seeds` 的系统下限是 3；调用者可以提高但不能降低。相同 seed 的重复实验会返回 `blocked-duplicate-seeds`，不能充当独立证据。
- 分析只读取每个 `experiment_id` 最新一次 task 的 MetricSeries，并在结果中绑定 task ID、控制面/Runtime 版本、source digest 与 Experiment Bundle Artifact digest。

## 可比性状态

| Status | 含义 |
| --- | --- |
| `ok` | 输入满足当前描述性分析契约；仍不允许科学或因果结论 |
| `blocked-incomparable-scope` | workload、protocol、comparison group 或 method 不一致/缺失 |
| `blocked-incomparable-stages` | 实验的有序 checkpoint transitions 不一致 |
| `blocked-incomparable-contexts` | 实验的 lagged stage/context 配对网格不一致 |
| `blocked-duplicate-seeds` | 多个实验使用相同 seed，不能视为独立重复 |
| `blocked-incomplete-evidence` | 用户选择的至少一个实验没有可用 lagged pair |
| `insufficient-pairs` | 有效 pair 数低于配置门槛 |
| `insufficient-seeds` | 唯一有效 seed 少于系统门槛 |
| `insufficient-variation` | predictor 或 outcome 无变化，相关系数不可定义 |

阻断优先级先检查 scope、stage、seed 独立性和完整证据，再检查样本数与统计可定义性。`reasons` 会同时保留所有已发现的问题，避免只修复第一个错误后才发现下一项。

## API 与 CLI

创建分析：

```sh
python3 -m edgeforge lop-analyze --token "$EDGEFORGE_TOKEN" \
  --experiment-id lop-seed-4321 \
  --experiment-id lop-seed-4322 \
  --experiment-id lop-seed-4323 \
  --predictor task.spectra.transformer_1.effective_rank \
  --outcome plasticity.acc_gain \
  --lag 1 \
  --context-policy exact \
  --bootstrap-repeats 2000 \
  --bootstrap-seed 20260821 \
  --minimum-pairs 3 \
  --minimum-seeds 3
```

等价 API 是 `POST /api/v1/lop-analyses`，body 使用相同字段，其中 `experiment_ids` 是数组。查询接口为 `GET /api/v1/lop-analyses` 和 `GET /api/v1/lop-analyses/{analysis_id}`；CLI 对应 `lop-analyses` 与 `lop-analysis`。

对于还没有导入 SQLite 的本地结果，`lop-audit --catalog <catalog.json>` 提供只读的一键审计。它读取 catalog 指向的 `metrics.json`/`RESULTS.json`，运行同一套归一化器，按方法/协议/comparison group 检查 source 是否存在、predictor/outcome 是否存在、stage 是否可配对、seed 是否足够，并在证据可用时调用本分析器。`--summary` 输出方法级状态，包括 `replay_values`、predictor/outcome 覆盖率和分析的 pair/seed 数量；该命令不会上传原始结果，也不会把 `status=ok` 变成科学结论。

审计状态的含义是：`missing-source` 表示 catalog 路径失效；`missing-predictor` 表示有 plasticity 但没有 ER 指标；`missing-outcome` 表示缺少 plasticity 指标；`candidate` 表示两类指标都存在，之后仍需通过 stage、context、seed 和统计门槛。正则化或 replay 方法只有在补齐同一协议下的 ER trajectory、future plasticity、checkpoint stage 和至少 3 个真实 seed 后，才能进入正式 LoP 分析。

`analysis_digest` 是规范化完整结果的 SHA-256，`analysis_id` 是其前 32 个十六进制字符。相同实验运行证据与相同规范化配置会得到同一 ID，重复创建不会新增第二条分析或第二个 `lop.analysis.created` 事件；实验重跑、指标、来源 digest 或配置改变都会产生新的分析身份。

## 当前研究状态

2026-08-20 的扩大预算验证仍只有真实 seed 4321，缺少 seed 4322/4323 对应 checkpoint。因此系统现在具备正式多 seed 分析合同，但没有足够本地证据执行具有群体解释力的 LoP 研究；当前状态继续是 `blocked-by-checkpoint`，不是 PASS。
