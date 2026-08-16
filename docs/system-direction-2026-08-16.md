# EdgeForge × RA-EEG 系统方向决策（2026-08-16）

## 结论

系统主线调整为“面向持续适应模型的可靠性评测、编译验证与发布基础设施”。RA-EEG 提供真实研究 workload 和模型能力指标，EdgeForge 提供异构执行、Artifact、版本证据、性能回归与发布准入。两者保持独立版本和清晰接口，不把 EdgeForge 变成 EEG 训练框架，也不让 RA-EEG 重写调度、Artifact Store 和多节点控制面。

四块现有设备是可利用的验证资源，不是项目选题约束。RTX 4070 SUPER 是主训练、Profiler 和 Compiler 节点；Orange Pi 是 ARM64/可选 Vulkan、LLVM CPU 与 RKNN 验证节点；P550 和 Meles 是 RISC-V 可移植性、Agent/Runtime 兼容与故障验证节点。任何研究结论都不能因为“必须同时使用四台设备”而引入没有实际必要的跨节点推理路径。

## 本次审计依据

本次方向调整同时核对了 `START_HERE.md`、`MASTER_SUMMARY.md`、`CURRENT_LOCAL_ACTIONS.md`、`UPSTREAM_PR_ROADMAP_20260816.md`、`raeeg-v0.2.zip`、EdgeForge `v0.6.0` 源码，以及 `/home/undefined/Desktop/bci/code/tta_security/BrainUICL/` 中已经存在的完整实验代码和真实运行记录。

审计后需要区分三种状态：

| 层次 | 已经真实存在 | 仍是设计目标 |
|---|---|---|
| RA-EEG v0.2 | ISRUC 单样本格式检查、简化 BrainUICL-style 模型、层级特征、Effective/Stable Rank、Weight Norm、Plasticity Probe、最小 FineTune/Replay、信号可用性指标、逐任务 JSON | 完整真实数据训练入口、已有 CL 方法适配、Model Registry、可靠性策略、Compiler/EdgeForge 集成 |
| BCI/BrainUICL 工作区 | 完整 ISRUC 数据与 checkpoint、原 BrainUICL 持续适应流程、多种攻击/防御迁移、较完整测试和实验报告 | 统一 Experiment Spec、跨实验指标契约、LoP/ER 的正式多 seed 基线、受控发布接口 |
| EdgeForge v0.6.0 | 四节点 Worker、任务租约、Operator IR、Kernel/Artifact Registry、Pipeline、Triton MatMul Auto Tuning、Compiler-aware Scheduler、版本日志 | 模型/实验级任务、研究指标时序、Model Registry、Capability Gate、真实模型 Compiler Backend |

因此，`MASTER_SUMMARY.md` 中的 Reliability Engine、Model Registry、IREE/torch.compile 和 Promote/Reject/Rollback 是目标架构，不能表述成 v0.2 已经完成的功能。

## 代码层发现与修正

1. RA-EEG v0.2 的 `ContinualExperimentRunner` 可以运行单个 subject 并输出 JSON，但 CLI 只有 pair 检查、forward、manifest 和标签扫描，没有正式 experiment/train 命令。
2. v0.2 测试直接依赖 `/mnt/data/41(1).npy` 与 `/mnt/data/41.npy`，所谓“4 passed”只代表原上传样本环境，不是可移植测试基线。
3. manifest builder 只支持两个平铺目录的同名文件，而本机真实 ISRUC 数据是 `<subject>/data/*.npy` 与 `<subject>/label/*.npy` 的嵌套结构。
4. v0.2 的 `FineTuneMethod` 使用真实标签做监督训练，而 BrainUICL 主场景是无监督 individual continual adaptation。它可以作为 supervised oracle/baseline，但不能直接代表 BrainUICL 协议。
5. RA-EEG v0.2 重新实现了一个简化模型；本机 BCI 仓库已经有经过真实数据运行的 BrainUICL 完整实现。因此后续应增加 Adapter 和对齐测试，不应维护两个彼此漂移的“主实现”。
6. EdgeForge 当前只接受 command、benchmark、operator benchmark、kernel pipeline、kernel autotune 和 compiler run 六类任务；Benchmark schema 也是算子级，尚不能保存 accuracy、MF1、BWT、plasticity curve 或 layer-wise spectrum。
7. EdgeForge 的 Triton 路径目前只覆盖 MatMul。RA-EEG 模型包含 Conv1d、BatchNorm、GELU、Attention、LayerNorm 和分类头，不能用单一 MatMul 优化结果代表模型编译能力。

## 新系统边界

```text
Research plane
BCI/BrainUICL source of truth
        │
        ├── RA-EEG dataset/model/CL adapters
        ├── Plasticity / Forgetting / Spectrum probes
        └── Experiment runner
                  │
                  │ ExperimentSpec / ExperimentBundle
                  ▼
Control and evidence plane
EdgeForge Experiment Registry
        ├── task scheduling and worker capability
        ├── content-addressed artifacts
        ├── metric series and regression baselines
        ├── gate policy and immutable decision evidence
        └── release/version logs
                  │
                  ▼
Execution and compiler plane
4070S: eager / torch.compile / profiling / training
Orange Pi: ARM64 / LLVM CPU / optional Vulkan or RKNN
P550 + Meles: RISC-V build, runtime smoke and portability
Optional A100: only after access and workload need are verified
```

RA-EEG 与 EdgeForge 通过版本化数据契约集成，不通过 Python 内部对象或共享 SQLite 表耦合。第一版契约包含：

- `ExperimentSpec`：实验 ID、数据集与 manifest digest、模型/协议、CL 方法、seed、阶段、Runner 版本、代码 revision、设备要求和资源限制。
- `ExperimentBundle`：环境快照、配置、checkpoint digest、指标文件、日志、阶段状态和父实验关系。
- `MetricSeries`：研究指标与系统指标的统一 envelope，但保留各自 namespace。
- `GateEvaluation`：版本化 policy、输入 Artifact/Metric digest、逐规则结果、最终 PASS/FAIL 和解释。

中央控制面不上传原始 EEG。数据集 payload 留在共享数据盘或授权 Worker，本系统只保存数据集标识、manifest digest、处理版本和访问能力标签。模型 checkpoint、编译产物、环境清单、指标与日志可以进入内容寻址 Artifact Store。

## 两类指标不能混为一谈

研究能力指标包括 accuracy、MF1、forgetting、BWT、learning gain、AULC、layer-wise effective/stable rank、spectrum 和 weight norm。系统指标包括 compile time、first-call/cold-start latency、steady latency、吞吐、峰值内存、Artifact size 和数值误差。

发布准入必须按顺序组合两类指标：

```text
Candidate checkpoint
  → protocol/data validation
  → continual-learning functional test
  → old-capability regression
  → plasticity gate
  → compiler numerical correctness
  → inference performance regression
  → promote / reject / rollback
```

阈值必须由版本化 policy 显式给出，不能把相关性分析直接当作因果结论，也不能因为某个编译 Backend 延迟更低就忽略模型 accuracy 或持续学习能力退化。

## 硬件角色调整

| 设备 | 主线任务 | 条件性任务 | 暂不承担 |
|---|---|---|---|
| RTX 4070 SUPER 12 GB | ISRUC/FACED 实验、LoP/ER instrumentation、PyTorch eager/compile 对照、Profiler、GPU correctness/performance gate | Triton 热点 Kernel、IREE CUDA | 多用户 GPU 资源隔离研究 |
| Orange Pi 5 Ultra | ARM64 构建、CPU Runtime、Artifact 可加载性、冷启动/内存测试 | 驱动与 Runtime 真实可用后接 IREE Vulkan 或 RKNN | 摄像头采集主线、未经验证的 NPU 承诺 |
| P550 | EdgeForge Worker/控制面兼容、RISC-V 构建与 CLI/API smoke test | 可获得稳定工具链后做 LLVM/IREE CPU Runtime 验证 | PyTorch 训练、GPU 级性能竞争 |
| Meles | 第二种 RISC-V 系统/工具链差分、故障与版本混跑测试 | 小型原生 Runtime benchmark | 强制参与每次模型推理 |
| 可选 A100 配额 | 访问、CUDA/MIG 配额和 workload 均确认后做跨 GPU 与并发验证 | vLLM/HAMi 相关研究 | 当前发布基线 |

没有摄像头不影响这个方向：EEG 输入来自离线真实数据集，Orange Pi 的价值是部署与 Runtime 验证，而不是传感器采集。RISC-V 板也不再决定算法选题；只有 Runtime、Agent、编译产物或发布协议需要跨 ISA 时才进入门禁。

## 研究、产品与上游贡献的优先级

RA-EEG 研究主线先完成自然 LoP：对齐真实 BrainUICL 协议，接现有 CL 方法，至少 3 seeds 记录 plasticity、forgetting、spectrum 和 norm，再检验前一阶段 ER 是否预测后一阶段 plasticity。自然机制未建立前，不进入大规模攻击、归因或防御搜索。

EdgeForge 产品主线先支持真实 ExperimentBundle 与 Capability Gate，再进入模型编译和多架构部署。原计划中的 LLM 专用 RoPE/KV Cache、llama.cpp/vLLM Serving、HAMi 并发与安全问题不再作为近期版本目标；除非未来增加独立 LLM workload，否则它们与当前 EEG 模型没有直接产品依赖。

上游 Issue/PR 是能力学习和贡献通道，不是系统路线的驱动器。2026-08-16 的 Issue 状态快照只用于候选筛选：IREE 工作必须由真实模型 export/target 问题触发，Braindecode hook 必须先由内部 instrumentation 验证稳定接口，vLLM/HAMi 必须等待实际服务器和对应 workload。

## 立即执行顺序

1. 为 RA-EEG 确定受版本控制的正式位置和代码来源，保留 `raeeg-v0.2.zip` 作为输入快照，不直接复制进 EdgeForge。
2. 以本机完整 BrainUICL 实现为 source of truth，给 RA-EEG 增加 Adapter；简化模型只保留为 reference/smoke 实现，并建立输出与 checkpoint 对齐测试。
3. 修复 v0.2 测试数据依赖，生成小型确定性 fixture；新增真实 ISRUC 嵌套目录 manifest 和正式 experiment CLI。
4. 先在 4070S 跑通一个受版本控制的 ISRUC FineTune/BrainUICL probe，生成完整 ExperimentBundle，不立即启动长时间多 seed 实验。
5. 在 EdgeForge V7 增加模型实验任务和 Artifact/Metric 契约，把该 Bundle 纳入版本日志与可重放任务。
6. Bundle 与 Gate 跑通以后再做 `eager → torch.compile` correctness/performance 对照；Orange Pi、IREE 和上游 PR 排在可导出模型之后。

本决策不修改或重写 EdgeForge v0.1.0–v0.6.0 的历史。后续功能只有在自动化测试、真实 workload 验证、发布说明、版本日志与 Git tag 全部完成后才冻结为新版本。
