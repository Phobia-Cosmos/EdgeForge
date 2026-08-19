# Changelog

本文件记录 EdgeForge 每个公开版本的用户可见变更。不可变的发布验证详情保存在 `releases/vX.Y.Z.md`，运行期结构化日志保存在配置的 `EDGEFORGE_LOG_DIR/vX.Y.Z/`，控制面事件、任务和 Benchmark 则保存在 SQLite。

## 0.10.1 - 2026-08-19

### Added

- BrainUICL/RA-EEG 历史结果迁移模块和 `build-raeeg-catalog.py`。
- 支持扫描 `metrics.json` 与 `RESULTS.json`，保存源文件 SHA-256、相对路径、seed、方法、协议和 comparison group。
- 为 817 个本地结果提供可重放的 catalog 生成能力，不复制 EEG、checkpoint 或研究源码。
- 增加 Backend × Target 说明和 RA-EEG 分批迁移文档。

### Safety

- 自动迁移结果统一标记为 historical import，默认 `scientific_conclusion_allowed=false`。
- `aligned-full49`、`method-transfer` 和 `historical-unclassified` 分组隔离，未完成 protocol review 的结果不能直接进入 Capability Gate。

### Validation

- 66/66 EdgeForge 自动化测试通过。
- 全量 catalog 扫描发现 817 个结果文件，生成约 1.2 MiB catalog。
- 8 组 curated ISRUC 结果迁移成功，并完成 1 个 EdgeForge 三阶段 LoP smoke；任务、8396 条指标、9 个 Artifact 和 WAL checkpoint 数据库已归档。

## 0.10.0 - 2026-08-19

### Added

- Model/Dataset/Transform manifest 与 transform digest。
- `model_pipeline` 任务，覆盖 frontend export、dataset transform、compile、runtime、correctness 和 model benchmark stage。
- 受控外部 Backend 接口，可接入 Python reference、PyTorch eager/compile、ONNX Runtime、Triton 或 IREE。
- `model_runs` 持久化表、model compiler manifest Artifact、Model Pipeline API/CLI 和版本化完成事件。

### Safety

- Stage command 使用 argv、Worker allow-list、work-root containment、`shell=False`、超时和输出上限。
- 未提供的 stage 显式记录为 skipped；没有真实 Runtime 或编译器时不会伪造性能证据。
- IREE 仍是可插拔 backend，#24760 不属于本版本的必做实现。

### Validation

- 63/63 EdgeForge 自动化测试通过（含模型流水线成功、失败、API 和持久化测试）。
- 验证记录见 `releases/v0.10.0.md`。

## 0.9.0 - 2026-08-18

### Added

- packed `conv_nchwc` Operator IR，支持 stride、dilation 和 accumulate 属性。
- 受信任的 IREE `iree-ukernel` runtime-only `compile → correctness → benchmark` Pipeline。
- IREE compiler manifest Artifact，记录 IREE repository、commit、patch digest、packed shape 和 benchmark 输出。
- x86_64 与 aarch64 的 kernel registration 配置和 Orange Pi 运行说明。

### Safety

- IREE test/benchmark command 必须来自 Worker allow-list，并受 work-root/workdir 边界约束；所有执行使用 `shell=False`。
- `operator_benchmark` 不再接受非 reference backend，避免把 Python fallback 误记为 IREE 证据；IREE 必须使用 `kernel_pipeline`。
- `blocked-*` source 状态禁止执行；真实状态必须绑定 repository、完整 commit 与 patch SHA-256；fake 合同测试必须显式标记为 `contract-only-not-real-iree`。
- IREE #24760 仍是开放 issue；当前双节点只完成 adapter contract validation，不宣称已构建 IREE compiler、验证真实 IREE binary 或取得硬件性能结果。

### Validation

- 见 `releases/v0.9.0.md`；V8 的 Model Registry 与 Capability Gate 记录保持不变。

## 0.8.0 - 2026-08-16

### Added

- Model Registry，保存 candidate、accepted、rejected、production 与 rolled_back 状态及来源实验身份。
- 不可变 Capability Gate Policy/Evaluation，保存 policy、metric 和逐规则结果快照。
- Model/Gate API 与 CLI，以及显式 promote、reject、production switch 和 rollback 操作。
- `raeeg / eeg-cl-v1-aligned / aligned-full49` 的首个固定验收策略。

### Safety

- Model、Policy 与 Experiment 必须具有完全相同的 workload、protocol 和 comparison group，禁止 SPR/PuriDivER 与 aligned-full49 跨协议门禁。
- Gate PASS 只进入 accepted；Worker、实验事件与 Agent 均不会自动 promote，生产状态动作必须带 Operator reason。

### Validation

- 自动化、V7 数据库迁移和真实 EEG 指标 PASS/FAIL、生产切换、rollback 验证见 `releases/v0.8.0.md`。

## 0.7.0 - 2026-08-16

### Added

- 版本化 `ExperimentSpec` 与 `ExperimentBundle`，支持执行新实验或导入已有结果。
- `experiment_run` 任务、RA-EEG 指标归一化、`experiment_runs` 和 `experiment_metrics` 数据库。
- `experiments`、`experiment-metrics`、`experiment-run` API/CLI 和 `experiment.completed` 事件。
- 本地 RA-EEG 八实验 Catalog，覆盖 aligned BrainUICL、Finetune、EWC、Online EWC、SI、MAS、SPR-EEG 和 PuriDivER-EEG。
- 基于真实 BrainUICL checkpoint 的固定预算 Plasticity、Effective Rank、Stable Rank 和 Weight Norm 后验 Probe。

### Changed

- V7 主 workload 从通用 LLM 算子扩展调整为 RA-EEG 持续适应模型；既有 Operator/Kernel/Compiler 路径保持兼容。
- 数据集只通过名称、路径能力和 manifest digest 引用，原始 EEG 不上传中央 Artifact Store。

### Validation

- 46/46 EdgeForge 自动化测试与 2/2 BrainUICL LoP Probe 测试通过。
- 8 组历史实验和 2 组真实 checkpoint LoP smoke 共写入 10 个 Experiment Bundle、8421 条指标与 10 个 Artifact。
- V6 验证数据库迁移、虚拟环境入口回归和版本日志归档通过；详情见 `releases/v0.7.0.md`。

## 0.6.0 - 2026-08-11

### Added

- 基于历史 latency、compile time 和 Worker load 的 Compiler-aware Cost Model。
- 自动选择 Kernel、Backend 与 Worker 的 `compiler_run` 任务。
- 可解释 Plan API/CLI、调度决策账本和 `scheduler.decision` 事件。

### Validation

- 40/40 自动化测试通过，V5 数据迁移验证通过。
- 四节点建立 5 条同 shape 候选路径，Compiler-aware Scheduler 自动选择 tuned Triton + RTX 4070 SUPER。
- `compiler_run` correctness 通过，执行结果回流后下一次 Plan 使用 2 个 exact-worker 样本重新估计成本。

## 0.5.0 - 2026-08-10

### Added

- Triton MatMul 参数网格搜索与最佳候选选择。
- `kernel_autotune` 任务、tuning run 数据库、候选事件和 Kernel 最佳配置回写。
- `kernel-autotune` 与 `tuning-runs` CLI/API。

### Validation

- 36/36 自动化测试通过，V4 数据迁移验证通过。
- RTX 4070 SUPER 上 4/4 Triton MatMul 候选 correctness 通过，最佳配置已回写并被后续 Pipeline 复用。

## 0.4.0 - 2026-08-10

### Added

- SHA-256 内容寻址 Artifact Store 与 artifact 查询 API/CLI。
- `compile -> correctness -> benchmark` Kernel Pipeline。
- Reference Compiler Backend 与 RTX 4070S Triton MatMul Backend。
- Kernel/Benchmark artifact digest、compile time 和 pipeline stage 事件。

### Validation

- 32/32 自动化测试通过，V3 数据迁移验证通过。
- 四节点 Reference Pipeline 与 RTX 4070 SUPER Triton FP16 MatMul Pipeline correctness 均通过。
- 中央验证数据库保存 5 个任务、5 条 Benchmark 和 2 个唯一 Artifact。

## 0.3.0 - 2026-08-10

### Added

- Kernel Registry、KernelSpec、架构/dtype 兼容性过滤。
- Benchmark 结果中的 kernel identity 和性能回归查询。
- `kernels`、`kernel-register` 和 `regressions` CLI/API。
- 稳定硬件指纹和从 V2 SQLite schema 到 V3 的向后兼容迁移。

## 0.2.0 - 2026-08-10

### Added

- 增加按版本、组件和进程运行实例分区的 JSONL 文件日志。
- 增加 SQLite `events`、`releases` 和 `benchmarks` 表。
- 任务同时记录控制面版本和 Worker Runtime 版本。
- 增加轻量 Operator IR，首批支持 MatMul、Softmax、RMSNorm 和 SiLU。
- 增加无第三方依赖的 Python reference correctness/benchmark backend。
- 增加 `operator-benchmark`、`events`、`benchmarks`、`releases` 和 `release` CLI。
- 增加版本事件、发布清单与 Benchmark 查询 API。
- 增加跨节点版本日志归档脚本和发布验证文档。

### Changed

- 支持 `operator_benchmark` 任务类型并将结果写入 Performance Database 雏形。
- systemd 模板增加 `/var/log/edgeforge` 版本日志目录。

## 0.1.0 - 2026-08-10

### Added

- 首个可运行的控制面、Worker Runtime、SQLite Registry 和约束调度器。
- Worker 注册、心跳、能力发现、系统指标、远程命令与重复 Benchmark。
- Bearer Token 鉴权、命令白名单、工作目录边界和任务租约。
- 在 RTX 4070S 主机、Orange Pi、P550 和 Meles 上完成四节点真实验证。
