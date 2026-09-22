# V12 Target Probe（2026-08-20）

本阶段先建立非 x86 目标的真实能力证据，不预设 Orange Pi 的 NPU、Vulkan 或 IREE 可用。`scripts/probe-target.py` 只读执行 `uname`、`lscpu`、`/proc/meminfo`、Runtime version 和设备节点检查；SSH 使用 BatchMode 和连接超时，不修改板端文件，也不读取 EEG、checkpoint 或模型 Artifact。

## 已探测目标

| target | architecture | CPU | memory | probe status | Python Runtime | Vulkan/IREE | RKNN/NPU | 结论 |
|---|---|---|---:|---|---|---|---|---|
| Orange Pi | `aarch64` | Cortex-A55/A76, 8 cores | 15964 MiB | online | numpy only；torch/onnxruntime absent | `vulkaninfo`/IREE unavailable | unavailable | 可作为 ARM64 CPU/DRM 探索目标，暂不注册 NPU/Vulkan backend |
| P550 | `riscv64` | 未由 `lscpu` 暴露型号, 4 cores | 25981 MiB | online | numpy/torch/onnxruntime absent | `vulkaninfo`/IREE unavailable | unavailable | 仅作为控制面、CLI/API、Artifact 与可移植性目标 |
| Meles | `riscv64` | 未探测 | 未探测 | offline-by-operator | 未探测 | 未探测 | 未探测 | 保留目标定义，不纳入本轮执行 |
| 4070S workstation | `x86_64` | AMD Ryzen 7 9700X, 16 threads | 31121 MiB | online | system Python 无 torch；编译环境另行管理 | `vulkaninfo`/IREE unavailable | unavailable | 继续作为 BrainUICL compiler/reference 主节点 |

原始探测 JSON 保存在 `.edgeforge/v12-target-probes/`。这轮结果说明 Orange Pi 已可通过 SSH 访问并确认 ARM64、RK3588 Linux、DRM 节点和内存；但没有证据证明 Vulkan loader/`vulkaninfo`、IREE、RKNN 或 `/dev/rknpu` 可用，因此不能宣称 NPU 推理或 Vulkan 性能。

针对真实 BrainUICL ARM64 preflight manifest 的部署前置检查返回 `BLOCKED`，原因是缺少 `torch_python`；`execution_performed=false`，没有向 Orange Pi 发送模型执行命令。该结果是能力门禁证据，不是模型失败结果。

## 下一步门禁

1. 在 Orange Pi 上安装或确认一个真实可运行的 CPU Runtime，先执行 BrainUICL 离线 EEG inference correctness。
2. 若 Vulkan loader、`vulkaninfo` 和 IREE Vulkan runtime 同时可用，再建立条件性 IREE manifest；否则保持 CPU 路径。
3. 只有 RKNN toolkit、NPU device、算子覆盖和量化前后 correctness 全部存在时，才注册 `rknn` backend。
4. P550/Meles 不承载 PyTorch 模型编译；继续验证控制面、Worker、Artifact 和 RISC-V 可移植性。
