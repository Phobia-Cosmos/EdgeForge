# EdgeForge V6：Compiler-aware Scheduler

## 目标

V6 把 V5 已积累的 Kernel、Auto Tuning、Benchmark 和 Worker 状态连接成执行规划器。调用方只描述 Operator 与约束，不再必须提前指定 Kernel 或 Worker；控制面会生成候选执行路径、估算成本、选择最优路径，并保存完整决策证据。

```text
Operator + Requirements + Policy
                 │
                 ▼
      Kernel compatibility filter
                 │
                 ▼
      Online Worker constraint filter
                 │
                 ▼
 Performance DB estimate + live load
                 │
                 ▼
  Select Kernel + Backend + Worker
                 │
                 ▼
       Persist decision and run
```

## 成本模型

当前目标函数为：`objective_ms = estimated_latency_ms + compile_weight × estimated_compile_ms + load_weight_ms × (load_1m / cpu_count + active_tasks)`。

默认策略为 `compile_weight=0.05`、`load_weight_ms=1.0`、`unknown_latency_ms=1000.0`。compile weight 表示把一次编译成本按预期复用次数摊销；调用方可以通过 Plan/Run CLI 修改策略。

历史估计按三个层级选择：

1. `exact-worker`：相同 Worker、Kernel、Operator、shape 和 dtype 的 correctness PASS 样本。
2. `same-architecture`：没有 exact 样本时，使用相同架构的同 Kernel 样本。
3. `unseen`：没有历史样本时使用策略中的 unknown latency，再由实时负载打破候选之间的平局。

同一层级存在多个样本时使用 latency median 和 compile-time median。候选最终按 objective、Worker ID 和 Kernel ID 确定性排序。

## 硬约束与接口

Planner 先应用 Worker ID、architecture、accelerator、label 和 memory 约束，再应用 Kernel 的 operator、dtype、architecture、accelerator 约束。额外支持 `kernel_ids` 和 `backends` 白名单。

- `POST /api/v1/plans` 或 `compiler-plan`：只生成解释，不创建任务。
- `compiler-run`：生成计划并创建 `compiler_run` 任务。
- `GET /api/v1/schedule-decisions` 或 `schedule-decisions`：查询已执行任务的不可变决策记录。

## 决策快照

创建 `compiler_run` 时，控制面把完整候选列表、策略、选择原因、Kernel snapshot 和目标 Worker 写入任务。`schedule_decisions` 表与 `scheduler.decision` 事件保存同一份证据。Worker 执行现有 Compiler Pipeline，结果继续写入 Benchmark 数据库，使后续 Plan 能吸收新的实测样本。

这形成第一版反馈闭环：`历史 Benchmark → Plan → Execute → 新 Benchmark → 更新后的 Plan`。

## 当前边界

当前规划发生在任务创建时，并绑定当时选中的 Worker；Worker 在排队期间离线时仍依赖现有 lease timeout/retry 机制，尚未自动改选不同 Kernel。成本模型只使用 latency、compile time 和负载，暂未包含功耗、显存、数据传输、置信区间与动态 shape 插值，这些属于后续 V7/V8 的 Runtime Cost Model。
