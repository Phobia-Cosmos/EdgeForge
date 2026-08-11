# EdgeForge V7+：AI Infra、Compiler 与推理优化路线

## 方向调整

V1–V6 已完成多节点控制、Operator IR、Kernel Registry、Artifact/Compiler Pipeline、Triton Auto Tuning 和 Compiler-aware Scheduler，证明了“测量 → 选择 → 执行 → 数据回流”的基础闭环。系统当前没有摄像头，长期目标也更偏向 AI Infra、AI Compiler 与推理优化，因此不再以工业视觉或嵌入式边缘部署作为主场景。

新的主场景是“异构 AI Runtime 发布验证与推理优化平台”：一次模型、Kernel、Compiler 或 Runtime 变更，需要在 x86_64、ARM64 和两种 RISC-V 设备上自动构建与测试，在 RTX 4070 SUPER 上完成 GPU 推理与 Kernel 优化，通过差分正确性和性能回归检查形成可追溯的发布结论。

## 四节点组合方式与边界

| 节点 | 主角色 | 不应强行承担的角色 |
|---|---|---|
| RTX 4070 SUPER 主机 | Triton/CUDA Kernel 优化、真实 GPU 推理、Profiler、主要性能门禁 | 低功耗常开控制面 |
| Orange Pi | ARM64 构建与 Runtime 验证、CPU Reference、小模型推理、可选 RKNN/NPU 探索 | 主线必须依赖的摄像头/NPU 节点 |
| P550 | 可迁移控制面、第一种 RISC-V 编译与 Runtime 验证、CI Worker | 与 GPU 竞争大模型吞吐 |
| Meles | 第二种独立 RISC-V 兼容性与性能验证、CI Worker、故障演练 | 为了“多机”而参与每次在线请求 |

四节点的主要交互链是 `build → correctness → benchmark → regression → release gate`，而不是要求一次在线推理依次经过四台设备。不同 ISA 和两种独立 RISC-V 平台能暴露工具链、ABI、依赖、数值结果和性能退化问题；4070S 则提供真正的 GPU 优化目标。若未来研究范围完全收缩为单一 NVIDIA GPU Kernel，三块板将成为辅助 CI 资源而不是核心算力，这一边界需要保留。

## V7：AI Workload、Model 与 Runtime Registry

- 建立 Workload Registry，首批覆盖 MatMul、RMSNorm、RoPE、Softmax/Attention 和 KV Cache 等推理核心算子，并记录 shape、dtype、输入生成、正确性容差与性能目标。
- 扩展 Model/Runtime/Toolchain Registry，记录模型或子图、Runtime、编译器、目标 ISA、依赖、Artifact digest 和复现参数。
- 定义结构化 Experiment Spec，固定代码版本、输入、设备、Backend、编译参数、运行参数与随机种子。
- 完成 x86_64、ARM64、P550 RISC-V 和 Meles RISC-V 的能力及工具链盘点。
- 验收：同一个实验规格可以被提交、校验、调度、重放和审计，结果可精确关联软件与 Artifact 版本。

## V8：Multi-Architecture CI 与 Compiler/Runtime Validation Farm

- 建立 x86_64、ARM64 和两种 RISC-V 的原生或交叉构建任务，保存 toolchain manifest、sysroot、产物和完整日志。
- 对 Runtime、算子库和 Worker 变更执行差分正确性、ABI/依赖检查、基础性能测试与兼容矩阵生成。
- 让 P550 承担可迁移控制面和任务账本角色，验证控制面与计算节点分离。
- 组成 Release Gate：各架构通过自己的正确性与兼容标准，GPU 节点额外通过性能标准，不做无意义的 CPU/GPU 绝对速度竞争。
- 验收：一次代码变更自动完成多架构构建、测试和报告，任一必需目标失败都会阻止版本被标记为可发布。

## V9：GPU 推理 Runtime 与 Kernel 优化

- 在 4070S 接入一条真实推理路径，优先从 PyTorch/Triton 或 llama.cpp CUDA 中选择与当前环境最容易复现的一条，再按模型需求评估 vLLM。
- 将 V5 Auto Tuning 扩展到真实推理算子，记录 TTFT、TPOT、吞吐、显存、Kernel 延迟、首次编译与缓存命中。
- 使用 V6 Scheduler 在 Reference、候选 Kernel 和不同 Runtime 配置间按能力与性能数据选择实验路径。
- 验收：至少一个公开小模型或可复现子图完成端到端推理，优化前后具有正确性对照、稳定 Benchmark 与可定位 Artifact。

## V10：性能回归与评测平台

- 建立固定算子、子图和模型 Benchmark Suite，区分冷启动、热运行、批量、序列长度、动态 shape 与并发场景。
- 为 Compiler、Runtime、Kernel 和模型版本保存 Baseline/Canary，以重复采样和明确阈值判定回归。
- 增加按设备、版本、workload、artifact 和编译参数查询的对比报告，定位回归首次出现的版本。
- 验收：给定两个版本，系统能回答结果是否一致、性能变化发生在哪个设备/算子/阶段，以及是否允许发布。

## V11：Inference Gateway 与实验编排可靠性

- 增加模型/Runtime 感知的实验和推理入口，根据设备能力、模型驻留、队列、历史延迟与实验目标选择节点。
- 增加任务优先级、并发限制、超时、重试、缓存、预热、断点恢复和 Worker 离线重调度。
- 4070S 作为主要 GPU Serving/Benchmark 节点；板卡主要承担控制、CI、Reference 和兼容性任务，仅在适合的小模型 CPU/NPU 场景参与推理。
- 将凭证轮换、最小权限、trace、温度/内存/队列指标和故障注入作为跨版本要求持续实施。
- 验收：长时间实验队列与交互式推理共存，节点离线不会丢失任务账本，并能生成完整执行链路。

## V12：Compiler / Inference Optimization Agent

- Agent 读取 IR、编译日志、正确性失败、Profiler、历史 Benchmark 与硬件指纹，提出 Kernel 参数、布局、融合、量化或 Runtime 配置候选。
- 每个候选都必须经过 `compile → correctness → benchmark → regression gate`，Agent 不能绕过失败测试或覆盖已发布 Artifact。
- Dynamic Shape、Operator Fusion、量化和完整 Cost Model 的顺序由 V9–V11 的真实瓶颈决定。
- 验收：Agent 至少对一个真实回归或热点生成可解释候选，并通过自动验证证明收益或明确拒绝候选。

## 可选并行探索：Orange Pi RKNN/NPU

- 不依赖摄像头，使用公开模型、离线输入或合成张量验证模型转换、量化、算子覆盖与板端 Runtime。
- 只有稳定跑通真实 RKNN 环境后，才将其注册为正式 Backend 并纳入 Release Gate；此探索不阻塞 V7–V12 主线。

V1–V6 的历史版本、验证数据库和日志继续保留。后续每个版本仍按既有版本规则归档，路线调整不覆盖历史事实。
