# EdgeForge V5：Auto Tuning

## 目标

V5 在 V4 `compile → correctness → benchmark` Pipeline 之上增加参数搜索层。第一阶段选择 Triton MatMul，使用确定性的 Grid Search 搜索 `block_m`、`block_n`、`block_k`、`num_warps` 和 `num_stages`，让真实硬件测量结果决定最佳配置。

```text
Operator + Kernel + Search Space
              │
              ▼
       Candidate Validation
              │
              ▼
Compile → Correctness → Benchmark
              │
              ▼
      Persist Every Candidate
              │
              ▼
 Select Lowest Valid Median Latency
              │
              ▼
   Update Kernel Registry Metadata
```

## 搜索空间

控制面在创建 `kernel_autotune` 任务时规范化并冻结候选列表。当前 block size 允许 16、32、64、128，warp 数允许 1、2、4、8，stage 数允许 1–5；单次任务最多 64 个候选。没有显式传入候选时使用四个保守的默认配置。

无效候选、非 Triton backend 或非 MatMul Operator 会在控制面被拒绝，不进入 Worker 队列。候选去重后写入任务快照，因此 Registry 后续更新不会改变历史搜索空间。

## 选择规则

只有同时满足 `status=succeeded`、correctness PASS 且具有正数 median latency 的候选才可参与选择。排序键依次为 median latency、p95 latency、compile time 和原始候选顺序，使同一组记录可以确定性重放。

最佳候选写入 `kernels.metadata.tuning_config`，并标记 `tuning_version`。后续普通 `kernel_pipeline` 创建任务时会携带最新 Kernel 快照，Triton backend 自动复用该配置。

## 持久化与事件

每次搜索写入一条 `tuning_runs` 记录，保存硬件指纹、Operator、shape、dtype、完整 search space、所有候选结果、最佳配置和 autotune manifest digest。最佳候选同时写入原有 Benchmark 表，使 V6 Compiler-aware Scheduler 可以直接复用 Performance Database。

事件账本增加 `autotune.candidate`、`autotune.completed` 和 `kernel.tuned`。Artifact Store 保存包含工具链版本、源码 digest、搜索空间、候选结果和最佳配置的 `autotune-manifest`。

## 当前边界

当前实现是单 Worker、单 shape 的顺序 Grid Search，没有并行搜索、提前停止、统计置信区间或跨 shape cost model。首次 launch 时间包含 Triton JIT 和缓存状态影响，不能直接当作纯编译器前端耗时。V6 将使用 tuning/benchmark 数据进行 backend、Kernel 和硬件选择。
