# EdgeForge V7+：RA-EEG 驱动的可靠性、Compiler 与 Infra 路线

详细的方向审计、代码现状和系统边界见 [system-direction-2026-08-16.md](system-direction-2026-08-16.md)。路线从通用 LLM/边缘推理平台调整为“使用 RA-EEG 真实 workload 建设持续适应模型的评测、编译验证与发布基础设施”。

## 路线约束

- EdgeForge 和 RA-EEG 保持独立版本：EdgeForge 管执行与证据，RA-EEG 管数据、模型、持续学习协议和研究指标。
- `/home/undefined/Desktop/bci/code/tta_security/BrainUICL/` 是当前已用真实数据运行的实现来源；`raeeg-v0.2.zip` 是待整合的 SDK 骨架，不直接替代现有实验代码。
- 不上传原始 EEG 到控制面；只保存数据 manifest digest、版本和 Worker 数据访问能力。
- 不要求每次实验使用所有设备。4070S 是主线计算节点，Orange Pi 和 RISC-V 板只在相应部署、Runtime 或兼容目标中参与。
- 不先做攻击和自动优化 Agent。自然 LoP、可塑性指标和可复现实验链未稳定前，不扩大研究面。

## V7：Experiment Contract 与 RA-EEG Adapter（已完成）

- 增加版本化 `ExperimentSpec`、`ExperimentBundle`、Metric envelope 和环境快照 schema。
- 新增受约束的 `experiment_run` 任务，不依赖通用 command 的 stdout 解析来表达实验结果。
- 保存 dataset/manifest、model、checkpoint、CL protocol/method、seed、code revision、Runner 版本、设备需求和父实验关系。
- 允许 Worker 上传配置、指标、日志、checkpoint/manifest 等 Artifact；原始 EEG 不进入 Artifact Store。
- RA-EEG 侧先完成真实 BrainUICL Adapter、可移植 fixture、嵌套 ISRUC manifest 和正式 experiment CLI。
- 验收：4070S 根据同一份 ExperimentSpec 重放一个真实 ISRUC smoke/probe，EdgeForge 能查询任务、Bundle、Artifact、Metric 和完整版本日志。

## V8：Model Registry 与 Capability Gate（已完成）

- 增加 Model/Checkpoint Registry，状态包括 candidate、accepted、rejected、production 和 rollback。
- 保存 accuracy、MF1、forgetting、BWT、plasticity curve、learning gain、AULC、layer-wise rank/spectrum 和 norm 等研究指标时序。
- 增加版本化 Gate Policy 与逐规则 GateEvaluation，输入和结果都绑定 digest，禁止覆盖历史判断。
- 区分 supervised FineTune baseline 与 BrainUICL 无监督持续适应协议，不能在同一 baseline 中混合比较。
- 验收：至少一个 candidate 能依据固定 policy 产生可解释 PASS/FAIL；失败 candidate 不进入 Compiler 发布阶段，已接受版本可以 rollback。

## V9：4070S 真实模型 Compiler Pipeline（进行中；0.9.0 完成 IREE 算子路径）

- 第一条路径只比较 PyTorch eager 与 `torch.compile`，先验证可复现性、数值正确性和模型能力等价，再扩大 Backend。
- 0.9.0 先完成独立的 IREE `conv_nchwc` dilation runtime-only Pipeline：以真实注册的预构建 binary 完成 correctness/benchmark/manifest 闭环；这不是模型级 `torch.compile` 验收。
- 保存 graph break、编译日志、compile time、first-call latency、steady latency、峰值显存、accuracy/MF1 差异和编译 Artifact。
- 使用 Profiler 确认实际热点后，再决定是否为 Conv1d、Attention、LayerNorm 或其他算子增加 Triton Kernel；不以现有 MatMul Demo 代替模型级结论。
- 将研究 Gate 和 Compiler Gate 串联，任何更快但能力回退超阈值的 Artifact 都不能 promote。
- 验收：一个真实 RA-EEG checkpoint 完成 `export/compile → numerical correctness → capability regression → performance benchmark`，并可从 Artifact 与版本记录重放。

## V10：模型级 Pipeline 基座（已完成；0.10.0–0.10.2）

- 统一 `Model/Dataset/Transform/Frontend/Compiler/Runtime` manifest，固定 transform digest 与完整 provenance。
- 新增 `model_pipeline` 任务，按 `export → transform → compile → run → correctness → benchmark` 执行，所有外部命令都受 Worker allow-list、work-root 和 timeout 约束。
- 结果写入 `model_runs`、版本化事件和 `model-compiler-manifest` Artifact；提供 Python reference baseline，真实 PyTorch/ONNX/Triton/IREE 通过同一 argv contract 接入。
- 验收：标准库 reference adapter 在本机完成合成 EEG normalize/window、编译描述、推理、数值正确性和 benchmark；0.10.1 完成 BrainUICL 817 个历史结果的 catalog 扫描和 LoP 验证；0.10.2 增加 explicit Backend/Target Registry；不把 reference 或历史导入结果当作真实硬件性能。

## V11：Orange Pi 部署目标与条件性 IREE（进行中；0.11.0 完成 Target Probe 基座）

- 0.11.0 增加 `target-probe`，记录 ARM64 CPU、板型/SoC、驱动、Vulkan、内存、设备节点和可用 Runtime 的可审计证据；不从 RK3588 型号或 Vulkan 文件推断 NPU/Backend 已就绪。
- 在 Orange Pi 上执行并保存 probe 后，基于真实结果选取可运行路径，不预设 NPU 已可用。
- 4070S 上真实 BrainUICL seed 4321 已完成 eager 与 Inductor 模型级 correctness/benchmark；默认 Inductor attention 候选被 Gate 拦截，safe profile 仅作为临时安全路径。该 GPU 证据不等价于 Orange Pi 部署成功。
- 在模型可稳定导出后，比较 LLVM CPU、条件性 IREE Vulkan 和 PyTorch/ONNX 可用路径的正确性、冷启动、steady latency、内存与 Artifact size。
- RKNN/NPU 是并行实验：只有工具链、算子覆盖和量化正确性真实通过后才注册为正式 Backend。
- P550/Meles 在这一阶段运行 Agent、CLI/API、Artifact 校验和 RISC-V 构建 smoke；仅在 Runtime 真实可构建时增加模型执行门禁。
- 验收：至少一个非 x86 目标加载真实模型 Artifact 并完成离线 EEG inference；不能用 Python Reference Operator 代替模型部署成功。

## V12：模型级回归、调度与多架构 CI（本机前置已开始）

- 0.11.0 在本机共享 PyTorch 环境完成 `torch-eager` 与 CPU `torch-compile` adapter contract、correctness failure gate 和重复 benchmark；这不代表 CUDA 或真实 BrainUICL 模型结论。
- V12 已实际探测 Orange Pi/P550，并对 Orange Pi BrainUICL ARM64 manifest 执行 deployment preflight；因缺少 `torch_python` 返回 `BLOCKED`，没有执行板端模型命令。
- 当前开发线暂不考虑 4 卡或开发板并行；所有 LoP 结论先在本机单 GPU/CPU 复现，未来若恢复并行也只能作为不改变 cell 语义的调度层。
- 将 V6 的算子 Cost Model 扩展为 workload/model/runtime 级候选，不把不同能力或不同精度的路径当作可互换候选。
- 对 checkpoint、Compiler、Runtime 和目标设备建立 Baseline/Canary，使用重复采样与明确阈值判断回归。
- 建立适配不同目标的发布矩阵：4070S 执行能力与性能门禁，Orange Pi 执行部署门禁，P550/Meles 执行协议与可移植性门禁。
- 增加长实验优先级、资源配额、断点恢复、Worker 离线重调度和 checkpoint 安全恢复。
- 验收：给定两个候选版本，系统能回答研究能力、编译正确性和性能在哪个阶段发生变化，并自动阻止不满足目标 profile 的发布。

## V13：可靠性与 Compiler Optimization Agent

- Agent 读取 ExperimentSpec、IR/graph、Profiler、编译日志、研究指标、Gate 失败和历史 Benchmark，提出实验诊断或优化候选。
- 所有候选必须进入隔离分支和不可绕过的 `experiment → capability gate → compile → correctness → performance gate`。
- Dynamic Shape、Operator Fusion、布局、量化和 Kernel 参数优化由真实热点与回归数据决定。
- 验收：Agent 对至少一个真实失败或热点生成可解释候选，并由完整 Gate 链证明接受或拒绝；Agent 无权直接修改 production 状态。

## V14：本地 LoP Matrix 与证据轨迹（进行中）

- `scripts/run-raeeg-lop-matrix.py` 将 dataset、method、condition、subject 和 checkpoint stage 展开为可恢复的本地 cell，调用只读 BrainUICL instrumentation 与 fixed-budget probe。
- 每个 cell 的 bundle、命令、stdout/stderr、状态和 digest 单独保存；同一 subject/seed/method 的多 stage 另生成 trajectory catalog，避免连续 checkpoint 被错误当成重复 seed。
- `matrix-report` 同时列出 effective/stable rank、fresh gap、parameter relative update 和 representation drift；报告只做描述性汇总，audit 仍要求至少三个真实 seed。
- 0.14.0 已在单机完成 ISRUC 五方法 × 四 stage 和 FACED pretrain/mini-CL smoke；下一里程碑是为两个数据集补齐真实多 seed 连续 checkpoint，再进入 controlled shift 矩阵。

## V15：LoP 与遗忘的分离证据（进行中；0.15.0 完成 retention 基座）

- fixed-budget probe 支持显式 old-task retention 集；参数只在新任务训练批次上更新，旧任务集只做 step-0/step-B 评估。
- retention ACC/MF1 drop 与 loss delta 以独立 `metric_role=retention` 保存，和 `plasticity.acc_gain`、effective-rank predictor、无标签 diagnostic 分开；不把 retention drop 自动解释为 BWT 或 LoP。
- matrix dry-run 检查 retention data/label 文件；trajectory identity 同时绑定 split 与 retention design，避免不兼容的旧任务集合被拼接进同一 seed trajectory。
- 0.15.0 已完成 ISRUC 单 cell 真实 retention smoke 和角色分离验证；下一步仍是补齐真实 seed 4322/4323 的连续 checkpoint，然后在相同 protocol 下同时运行 plasticity、retention、clean/shift 对照。
- checkpoint instrumentation 同时记录 optimizer/moment/scheduler provenance；当前参数快照没有 Adam state 时明确为 unavailable，避免把 parameter delta 误称为 optimizer trajectory。

## V16：ARM/RISC-V Target Setup 与 Backend 边界（进行中；0.16.1 完成 RK3588 API smoke）

- 用只读 inventory 分离 ARM/RISC-V 架构、Runtime 文件、设备节点和显式 Backend 广告；厂商 RKNN 文件存在不等于 RK3588 NPU 可执行。
- `scripts/configure-arm-target.py --apply --sync-source` 为 Orange Pi/P550 创建用户目录、源码、work/log/config 和 Worker 启动模板，不安装未知包、不写凭证、不启动远端任务。
- 明确 `OperatorSpec`、`torch.export` Graph IR 与 MLIR/LLVM/backend lowering 的层次；后端只能通过独立 adapter 消费自己支持的输入格式。
- Orange Pi 已完成 Mali OpenCL vector-add 与用户目录 RKNPU2 v1.5.2 C API MobileNet smoke；Vulkan loader 可加载但 Mali ICD 缺失，系统 demo 的 `libjpeg.so.62`/Runtime 版本问题单独留存。只有 RKNN conversion、量化、EEG 子图 correctness 和 benchmark 通过后才解除 `rknn` backend blocked。
- `brainuicl.pt2` 的统一 lowering 暂不扩展为无证据的多后端广播；先绑定 Graph IR digest、layout/dtype contract 和一个真实转换后端。RKNN Toolkit2 只在主机侧负责转换，板端只接收受控 `.rknn` Artifact。

## 独立研究里程碑

RA-EEG 的科研节奏不与 EdgeForge 版本号绑定：先对齐 BrainUICL 协议与已有 CL 方法，再运行 ISRUC/FACED 自然 LoP 多 seed 基线，随后分析 `ER(t-1) → Plasticity(t)`，之后才决定攻击、归因、防御和 controlled shift。0.14.0 已完成版本化 matrix/trajectory/report 基座；真实多 seed checkpoint 证据仍是下一里程碑，当前单 seed smoke 不能升级为科研结论。科研结论失败或不显著也必须被完整记录，不能为了基础设施演示预设 LoP 必然存在。

## 暂缓项

- 摄像头或工业视觉主场景。
- 让 P550/Meles 承担 PyTorch 训练或大模型推理。
- 没有真实可用性验证前承诺 Orange Pi NPU。
- 与当前 EEG workload 无关的 RoPE/KV Cache、llama.cpp/vLLM Serving 和 HAMi 多租户 GPU 方向。
- 为了匹配上游 Issue 而提前实现没有内部需求的 Compiler 功能。

EdgeForge v0.1.0–v0.15.0 的发布说明、验证数据库和日志按版本保持独立。任何新版本仍必须完成 Changelog、`releases/vX.Y.Z.md`、自动化测试、真实 workload 验证、日志归档和 Git tag。
