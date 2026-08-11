# EdgeForge V7+ 场景优先路线

## 调整原因

V1–V6 已完成多节点控制、Operator IR、Kernel Registry、Artifact/Compiler Pipeline、Triton Auto Tuning 和 Compiler-aware Scheduler，证明了“测量 → 选择 → 执行 → 数据回流”的基础闭环。当前主要缺口不再是更多抽象 Compiler Pass，而是真实模型、Orange Pi NPU、推理引擎、端到端应用、安全与故障演练。

因此暂停原定直接进入 Dynamic Shape/Fusion 的顺序，先建立一个真实业务闭环，再让实际 workload 数据决定后续 Compiler 优化优先级。

## 主验证场景

主场景为离线工业现场的视觉告警、解释与多架构软件发布：Orange Pi 使用 RK3588 NPU 做持续感知；RTX 4070 SUPER 处理异常事件的重模型/VLM/LLM；P550 运行低功耗常开控制面、任务账本和离线队列；Meles 执行规则校验、Watchdog、备用任务和第二种 RISC-V 兼容验证。

系统不要求每个推理请求强制经过所有节点。四节点共同价值来自角色分工、容错和发布门禁；任何单节点离线时都应有明确降级行为。

## V7：场景契约与 Model/Service Registry

- 定义图像/事件输入、告警输出、延迟目标、断网行为、数据保留和失败降级标准。
- 增加 Model Registry、Service Registry 和结构化 `inference` 任务。
- 记录模型格式、量化方式、Runtime、目标设备、Artifact digest、部署与加载状态。
- 完成四节点软件能力探测，并让控制面可部署到 P550。
- 验收：推理请求能按模型和 Runtime 能力被验证、调度、执行与审计。

## V8：Orange Pi RKNN/NPU Backend

- 安装验证 RKNN Toolkit/Runtime，选择小型视觉模型完成转换与量化。
- 实现 `rknn` Backend、模型 Artifact、板端缓存、correctness 和 benchmark。
- 对照 Orange Pi CPU 与 NPU 的延迟、内存、功耗和算子覆盖。
- 验收：真实图像在 RK3588 NPU 推理，结果进入统一 Performance Database。

## V9：端到端异构推理应用

- Orange Pi 做本地检测，仅上传结构化事件或必要裁剪图。
- 4070S 对异常事件执行重模型分析和自然语言解释。
- P550 编排持久任务与离线队列；Meles 做规则确认、Watchdog 或备用执行。
- 增加 Inference Gateway、预热、缓存、超时、重试和 fallback。
- 验收：完成“输入 → NPU 感知 → GPU 分析 → 规则确认 → 报告”，并演练节点离线降级。

## V10：安全、可靠性与可观测性

- 使用 Tailscale/WireGuard 或 mTLS 保护节点网络，轮换凭证并限制 Worker 权限。
- 增加 trace ID、队列深度、阶段延迟、模型加载、温度和 GPU/NPU 指标。
- 故障注入覆盖 Worker 断电、控制面重启、网络中断、任务超时、Artifact 缺失和模型加载失败。
- 通过压测确认瓶颈后再选择 PostgreSQL、etcd、Redis 或消息队列，不预先堆叠组件。

## V11：Multi-Architecture CI / Debug Agent

- Worker/Runtime 发布必须在 x86、ARM、P550 和 Meles 完成构建、测试和兼容性报告。
- Agent 关联编译错误、测试失败、硬件指纹和历史版本，生成解释与候选 Patch。
- 所有 Patch 必须重新通过四节点门禁，不能绕过失败测试。

## V12+：真实数据驱动的 Compiler 研究

- 使用真实推理 trace 决定 Dynamic Shape Kernel Variant、Operator Fusion、量化和 Cost Model 的顺序。
- RKNN 转换/算子覆盖是瓶颈时优先做 Lowering；GPU shape 回退是瓶颈时再做 Dynamic Shape；数据传输占主导时再研究图切分与 placement。
- 保留 V1–V6 的版本和验证数据，不为了新场景重写已经稳定的控制面协议。
