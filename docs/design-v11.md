# V11 Orange Pi Target Probe

V11 的第一步是把“板子型号”与“可执行 Backend”分开。`python3 -m edgeforge target-probe --output <path>` 在目标节点生成 schema v1 JSON，记录架构、CPU features、板型/SoC、内存、设备节点、内核 GPU/NPU 驱动、Vulkan ICD/loader，以及 Python、ONNX Runtime、IREE、Vulkan 和 RKNN 工具的可执行文件证据。对具有稳定只读参数的工具会执行最长五秒的 version/summary probe，并保存 exit code、stdout 和 stderr；没有稳定安全 version 参数的工具只记录路径。

本机 PyTorch 路径沿用同一边界：Worker 必须显式设置 `EDGEFORGE_BACKENDS=python-reference,torch-eager,torch-compile`，控制面不会因为发现 PyTorch 文件就自动调度任务。`torch-eager` 和 `torch-compile` manifest 当前只使用 CPU synthetic 输入，真实 CUDA/BrainUICL 结果需要另外的模型与设备证据。

Probe 不是 Backend 验收。`backend_claims.inferred` 始终为空，Worker 仍只默认广告 `python-reference`。`EDGEFORGE_BACKENDS` 只能在管理员确认安装并完成对应 Runtime correctness 后显式设置。尤其是 RK3588 compatible string 只证明 SoC 身份，不证明 NPU 驱动、RKNN Runtime、算子覆盖或量化正确性；只有真实 `/dev/rknpu*` 节点才进入 accelerator advertisement，仍不能单独解锁 RKNN 模型任务。

Orange Pi 的后续顺序固定为：保存 target probe → 选择实际存在的 CPU/ONNX/IREE Vulkan 候选 → 传输绑定 digest 的模型 Artifact → Runtime 加载 → numerical correctness → 离线 EEG inference → 冷启动、steady latency、内存和 Artifact size。任一步缺少真实证据时保持 blocked，不使用 Python reference 结果替代部署成功。

Vulkan ICD manifest、loader library 或 DRM device 只作为调查证据。只有 `vulkaninfo --summary` 成功以及目标 Runtime correctness 通过后，才允许把 Vulkan 路径登记为可用 Backend。IREE 仍使用现有 source identity、patch digest、binary allow-list 和 blocked status 边界；V11 Target Probe 不改变这些条件，也不要求下载 LLVM/IREE 整个源码树。
