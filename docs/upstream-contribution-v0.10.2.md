# V0.10.2 Backend/Target Contract 与上游贡献准备

## 为什么先做这一层

EdgeForge 的真实 RA-EEG/LoP 运行暴露出一个跨项目共性问题：Backend 名称不能隐式决定目标设备。`torch-compile`、ONNX Runtime、IREE 和 RKNN 都可能覆盖多个架构或加速器；如果没有显式 target，编译器可能选择错误的默认设备，Benchmark 也会把不同实现混在一起。

V0.10.2 在 EdgeForge 内先把该约束固定为可测试的 contract：`compiler.backend`、`target.architecture`、`target.device`、`target.accelerator` 和 Worker 的 advertised backends 必须共同满足 Registry。未知 custom backend 也必须声明 target architecture。

## 当前 Registry

| Backend | 约束 |
| --- | --- |
| `python-reference` | 任意架构，允许无显式 target，用于 correctness baseline |
| `torch-eager` / `torch-compile` | 必须显式 architecture |
| `onnx-runtime` | 支持 x86_64/aarch64/riscv64，但必须显式 architecture |
| `triton` | x86_64 + nvidia-gpu |
| `iree` | architecture + device，避免默认 target |
| `rknn` | aarch64 + rk3588-npu |

API `GET /api/v1/backend-capabilities` 和 CLI `backend-capabilities` 用于把该 contract 提供给前端、Agent 和调度器。Worker 默认只广告 `python-reference`，其他后端必须通过真实验证后的 `EDGEFORGE_BACKENDS` 显式声明。

## 对上游 PR 的最小贡献边界

这层可以形成两个独立的小型上游贡献方向：

1. IREE compiler/config：要求 Vulkan/CPU target device 显式传入，在缺失时返回可读错误，并补一个 unit test；EdgeForge 的 `iree` Registry 和失败任务可提供 reproducer。该方向对应“移除/禁用隐式 Vulkan 默认 target”类 issue，但不实现 IREE #24760 的 dilation lowering。
2. Runtime adapter：在 ONNX Runtime/IREE adapter 中把 architecture/device/provider 写入 manifest，禁止把 host fallback 当成目标 backend；EdgeForge 的 `model_runs` 和 Artifact digest 可作为回归证据。

EdgeForge 不复制 IREE 源码，也不把上游项目改动当作本仓库主线。只有当真实 BrainUICL 模型在某个 target 触发可复现问题时，才从 EdgeForge Artifact 导出最小 reproducer，再在对应上游仓库建立独立分支、测试和 PR。

## 验证

本版本覆盖：Registry 列举、IREE 缺少 architecture/device 的拒绝、RKNN 缺少 ARM64/NPU 的拒绝、custom backend target 要求、Worker capability advertisement、模型任务的 backend/target 持久化。所有测试和日志仍归档在 EdgeForge 版本目录中。
