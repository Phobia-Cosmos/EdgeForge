# EdgeForge V4：Artifact Store 与 Compiler Pipeline

## 目标

V4 把 V3 的 Kernel Registry 从“可查询的实现元数据”推进为“可执行、可复现、可审计的 Kernel 流水线”。控制面负责调度和记录，Worker 负责在本机编译、正确性检查与性能测量，编译产物以内容寻址方式保存。

核心流水线固定为：`compile → correctness → benchmark`。任一阶段失败都会保留任务结果和阶段事件，后续阶段不会伪造为成功。

## 组件边界

```text
Client / CLI
     │  task + Kernel ID
     ▼
Control Plane ── SQLite Registry / Event Ledger
     │  lease by architecture, accelerator and Kernel constraints
     ▼
Worker Runtime ── Backend Adapter
     │
     ├─ python-reference：跨 x86_64 / aarch64 / riscv64 的 correctness 基线
     └─ triton：当前 RTX 4070 SUPER 的 FP16/BF16 MatMul 实验 backend
     │
     └─ Artifact Store：SHA-256 manifest/blob，重复内容自动去重
```

控制面不执行 Kernel 代码，也不把二进制直接放入任务消息。Worker 回传小型 compiler manifest 的 base64 内容，控制面写入 Artifact Store 后，仅把 digest 绑定到 Kernel 和 Benchmark。

## 数据与可追溯性

- `kernels.artifact_digest` 指向实现或编译 manifest；`kernels.compiler` 保存工具链元数据。
- `benchmarks.artifact_digest`、`compile_ms` 与 `runtime_version` 保存实际执行版本。
- `events` 为每个 pipeline 阶段写入 `pipeline.compile`、`pipeline.correctness` 和 `pipeline.benchmark` 事件。
- Artifact 使用 SHA-256 内容寻址，路径为 `sha256/<前两位>/<完整 digest>/blob`，历史 blob 不覆盖。
- SQLite schema 使用启动时的幂等迁移逻辑，V3 的 release、task、event、benchmark 和 kernel 记录可由 V4 继续读取。

同一编译 manifest 被多个 Worker 使用时只保留一个 Artifact 行；这是内容去重，不会丢失 Benchmark 或事件记录。

## 任务协议

`kernel_pipeline` 任务 payload 包含 `operator.name`、`operator.shape`、`operator.dtype`、`kernel_id`、`repeats` 和 `warmup`。控制面先按 Kernel 的 operator、dtype、架构和 accelerator 约束筛选 Worker，再由 Worker 选择注册的 backend。

成功结果至少包含 `correctness`、`timings_ms`、`summary`、`compile_ms`、`pipeline` 和 `artifact_upload`。控制面完成上传后返回 `artifact` 元数据，并自动更新 Kernel 的 digest 和 Benchmark 记录。

## 当前验证边界

Python reference backend 用于跨架构功能闭环，不模拟设备特定的 FP16/BF16 舍入。Triton backend 当前只覆盖 CUDA RTX 4070 SUPER 上的 MatMul；RKNN、RVV 和其他 NPU/CUDA 实现应在同一 Kernel/Artifact 协议下继续接入。

后续优先级是：显式 baseline 与置信区间、编译缓存命中率、Artifact 下载/签名策略、RKNN/RVV backend，以及 Agent 层的任务分解和失败重试策略。
