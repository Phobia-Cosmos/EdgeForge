# EdgeForge V3 Kernel Registry

## 为什么需要 Kernel identity

V2 只能回答“某个 Operator 在某种架构上跑得多快”。V3 增加 `kernel_id` 和 `kernel_version` 后，可以区分 `cuBLAS`、Triton 候选、RKNN 图和 RVV 实现，并在同一个 shape/dtype 上做实现级别的性能回归。

## Registry 数据模型

```text
KernelSpec
├── id
├── operator
├── backend
├── version
├── architectures
├── dtypes
├── shape_constraints
└── metadata
```

Registry 不保存可执行代码本身。artifact、编译参数、源码 digest 和容器镜像 digest 放在 `metadata`，真正的代码由后续 Backend/Artifact Store 管理。这样控制面不会意外变成代码执行仓库。

## 调度流程

提交 `operator_benchmark` 时，控制面检查 Kernel 是否支持 Operator 和 dtype，并把 Kernel 描述嵌入任务快照。Worker 领取任务时，控制面先依据 Kernel 的 `architectures` 过滤在线 Worker，再执行原有负载/内存/加速器评分。任务结果会保留提交时的 Kernel snapshot，即使 Registry 后续更新也不会改变历史结果。

```text
Task(kernel_id)
      |
      v
Registry compatibility check
      |
      v
Kernel architecture filter
      |
      v
Load / memory / accelerator score
      |
      v
Worker(runtime_version, hardware_fingerprint)
```

## 回归检测

性能比较键为 `operator + shape + dtype + backend + kernel_id + architecture`。同一键按创建时间排序，最新记录与上一条记录比较；超过阈值的结果通过 `/api/v1/regressions` 暴露。V4 将把“上一条”替换为显式 baseline，并增加置信区间、warmup、样本数和编译时间。

## V4 入口

下一阶段为 Artifact/Compiler Backend：Kernel 注册时增加 artifact digest、编译器版本和编译参数；Worker 任务增加 `compile -> correctness -> benchmark` 三段状态，并把编译失败也写入同一版本事件账本。

