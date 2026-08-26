# Changelog

本文件记录 EdgeForge 每个公开版本的用户可见变更。不可变的发布验证详情保存在 `releases/vX.Y.Z.md`，运行期结构化日志保存在配置的 `EDGEFORGE_LOG_DIR/vX.Y.Z/`，控制面事件、任务和 Benchmark 则保存在 SQLite。

## 0.16.4 - 2026-08-26 (development snapshot)

### Added

- 新增 `scripts/brainuicl-multiseed-pretrain.py`，为 ISRUC 使用固定 split-seed 和独立真实训练 seed 生成隔离的三组件 checkpoint、训练日志、验证历史、环境快照与 SHA-256 manifest。
- 新增 `scripts/brainuicl-faced-multiseed-pretrain.py`，复用 FACED aligned BrainUICL frontend，生成同契约的三 seed 预训练产物。
- `scripts/brainuicl-model-pipeline.py` 支持 `--dataset FACED`，可对 FACED `(20,32,2500)` 图执行 export、CPU eager、CUDA Inductor、correctness 与 benchmark。

### Validation

- ISRUC seed 4321/4322/4323 均完成 100 epoch 正式预训练，最佳验证准确率分别为 0.6439、0.6511、0.6466；FACED 分别为 0.3047、0.3008、0.3000。
- 三个 ISRUC seed 的 torch.export graph digest 相同；CPU eager correctness 全部通过，CUDA Inductor correctness 全部通过，最大绝对误差不超过 `2.94e-5`。
- FACED seed 4321 的 CPU eager 与 CUDA Inductor correctness 通过，输出形状为 `(1,9,20)`；CUDA Inductor 最大绝对误差约 `6.78e-4`，仍在当前容差内。
- Orange Pi RK3588 重新完成 OpenCL vector-add、RKNN C API/MobileNet runtime 和显式 Vulkan ICD loader/physical-device smoke；这些是 runtime/API 证据，不是完整 EEG 模型部署结论。

### Safety

- 每个 seed 使用独立初始化和独立 checkpoint 路径，未复制或覆盖 4321；固定 split-seed 只用于保持 subject partition 一致。
- 预训练完成不等于 LoP 结论；仍需在每个 seed 上重新运行相同 CL trajectory、fresh control、retention set 和 seed-cluster bootstrap。
- RKNN/OpenCL/Vulkan 仍未宣称 BrainUICL EEG 模型级 correctness；主机 RKNN Toolkit2 conversion 与板端 EEG 子图验证是后续工作。

## 0.16.3 - 2026-08-25 (development snapshot)

### Added

- 新增 `config/raeeg-lop-matrix-v16.3-dose-local.json`：ISRUC subject 2、finetune、seed 4321、checkpoint stages `0/10/25/49` 的 clean、相对噪声 1.0 和 50% channel-dropout 剂量矩阵。
- 新增 `config/raeeg-lop-matrix-v16.3-retention-dose-local.json`：在相同矩阵上加入 subject 1 retention set 与无标签 confidence/entropy/consistency 诊断，明确区分 outcome、predictor、retention 和 diagnostic metric roles。
- 保存两套 12-cell 矩阵的 catalog、trajectory、audit、cell bundle、命令日志、派生 shift manifest 和 SHA-256 清单；原始 ISRUC 数据与 checkpoint 未被修改。
- fixed-budget probe 现在记录 seed、CUBLAS workspace、cuDNN deterministic 和 `torch.use_deterministic_algorithms` 状态；CUDA 上的重复 probe 已验证输出一致。

### Validation

- 两套矩阵均为 `12/12 succeeded`；每套包含 3 conditions × 4 stages，`scientific_conclusion_allowed=false`。
- 可复现重跑的 clean 条件 `fresh_gap` 为 `-0.025/-0.075/-0.075/-0.200`（stage `0/10/25/49`）；50% channel-dropout 为 `+0.050/0.000/-0.225/-0.250`；noise 1.0 为 `0.000/0.000/-0.150/-0.200`。正 fresh gap 表示 fresh control 在固定预算后高于 checkpoint，属于 LoP candidate 方向，但本版本只有 seed 4321。
- retention/无标签诊断已运行：retention accuracy drop、loss delta、normalized entropy、mean confidence、prediction agreement 和 representation cosine 均写入 bundle；这些指标不替代 fixed-budget plasticity outcome。
- 可复现重跑的三个条件 lagged ER→plasticity audit 都为 `insufficient-seeds`（每组 3 个 stage pairs、1 个 seed；Pearson 分别为 clean `-0.9276`、dropout `+0.5957`、noise `-0.0871`），不构成 LoP 结论。

### Safety

- seed 4322/4323 的 8 个多 seed cells 仍因缺少真实 seed-specific checkpoint 被 preflight 阻断；没有复制 seed 4321，也没有用 shift-induced degradation 冒充 LoP。
- shift 数据写入 `/home/undefined/Disk/datasets/edgeforge-raeeg-shifts/v0.16.3/`，采用 label symlink 和 source/output digest；原始数据目录保持只读。
- 0.16.3 结果是单 seed、短预算、subject-2 的描述性验证，不能支持跨数据集、跨方法或因果机制结论。

## 0.16.2 - 2026-08-25 (development snapshot)

### Added

- `accelerator_probe.probe_vulkan` 和 `rk3588-accelerator-smoke.py` 支持显式的用户目录 Vulkan ICD manifest/loader；探测会记录 manifest、库文件及 SHA-256，并在进程结束后恢复 `VK_ICD_FILENAMES`，不会修改板端系统配置。
- `target_probe`/`scripts/probe-target.py` 支持 `--vulkan-icd`，将用户目录候选 manifest 与系统 loader、DRM 和 OpenCL 证据分开保存；候选文件仍必须通过 API smoke 才能满足部署门禁。
- Backend registry 现在显式列出 `opencl` 与 `vulkan` target contracts；它们只声明架构边界，实际 loader/ICD、API smoke、模型 correctness 和 benchmark 仍由 deployment preflight 逐项门禁。
- 新增 Vulkan 用户态验证证据：Orange Pi RK3588 使用 Rockchip `g610-g24p0` Mali GBM library 在用户目录注册临时 ICD 后，Vulkan loader 创建 instance 并枚举到 Mali-G610 physical device；同时 OpenCL vector-add 与 RKNN C API smoke 均通过。
- 修复 NumPy 2.5 移除 `np.trapz` 后 fixed-budget LoP curve summary 的兼容性问题，避免在 `np.trapezoid` 已存在时仍提前求值旧别名。

### Safety

- 没有向 Orange Pi `/usr/lib`、`/etc/vulkan` 或系统包数据库写入文件；`g610-g13p0`（当前系统 X11/GBM 包）被明确记录为缺少 Vulkan ICD entry points，未用它冒充成功驱动。
- Vulkan 结果的正确性范围仍是 loader → instance → physical-device enumeration；没有把它外推为 EEG 模型算子覆盖、模型数值 correctness 或性能结论。RKNN smoke 仍只使用零输入 MobileNet，并标记 `model_correctness=not-evaluated`。

### Validation

- Orange Pi `g610-g24p0` candidate library SHA-256：`4d7cb76a1d073c39a4fee34692e0422b1421ff258045a6cef40e9f91492c89a6`；ICD manifest SHA-256：`b35de60dd3478f8193a1aad4ff522ddec464edbd2cc2b5fd99367851d234b12f`。
- Vulkan API：loader `1.3.204`、instance result `0`、physical device count `1`、Mali-G610 / `DRIVER_ID_ARM_PROPRIETARY`；OpenCL max absolute error `0.0`；RKNN API `1.4.0` / driver `0.9.6`，三次零输入输出 digest 一致。
- `g610-g13p0` control candidate 返回 `VK_ERROR_INCOMPATIBLE_DRIVER (-9)` 且没有 `vk_icdGetInstanceProcAddr`，验证了探测逻辑不会把任意 Mali ELF 当作 Vulkan ICD。
- 以同名 `orangepi` target probe + g24p0 full smoke 驱动 `vulkan` deployment preflight，结果为 `PASS`（execution 未执行，仍需模型 artifact/correctness gate）。
- 完整证据与命令、测试和清单归档于 `logs/archive/v0.16.2/rk3588-vulkan-userspace-audit/`。

## 0.16.1 - 2026-08-24 (development snapshot)

### Added

- 新增 `edgeforge.accelerator_probe` 与 `scripts/rk3588-accelerator-smoke.py`：在无摄像头、离线、确定性输入下实际执行 Mali OpenCL vector-add、Vulkan loader/device instance 和 RKNN C API `init → query → run → output` smoke。
- Target Probe 增加 DRM RKNPU driver、platform NPU、OpenCL 用户态、Vulkan loader/ICD 和 OpenCL device evidence；RK3588 的 `card1/renderD129` 不再因为没有 `/dev/rknpu*` 被误报为缺失。
- RKNN deployment preflight 改为要求板端 `librknnrt.so`/`librknn_api.so` 与任一 RKNPU DRM/character/platform 证据，不要求板端安装 PC 侧 RKNN-Toolkit2 Python 包；新增 `opencl`/`vulkan` capability contracts。
- 加入显式 `--runtime-validation` gate：仅有文件/设备证据的 RKNN/OpenCL/Vulkan 目标仍为 BLOCKED，必须提供已保存的 API smoke 结果。
- 新增 [RK3588 accelerator validation](docs/rk3588-accelerator-validation-v1.md) 手册对照、结果边界和后续 RKNN/EEG adapter 计划。

### Safety

- smoke 只读取显式离线 `.rknn` 文件并使用零输入，不接摄像头、不启动或替换系统 Runtime、不把 Vulkan loader 版本冒充 Mali device；所有结果保留 `scientific_conclusion_allowed=false`。
- 验证采用用户目录缓存的 RKNPU2 v1.5.2 library/model；系统 1.4.0 demo 的 `Invalid RKNN format` 被保留为版本兼容性证据，不通过软链接伪造 `libjpeg.so.62`。
- Worker 可识别 DRM RKNPU 作为 accelerator evidence，但默认仍只广告 `python-reference`；实际 `rknn` backend 需模型 conversion、算子覆盖、correctness 和 benchmark 门禁。

### Validation

- Orange Pi RK3588：OpenCL Mali-G610 vector-add PASS（max abs error `0.0`）；系统 RKNN Runtime 1.4.0 与用户缓存 v1.5.2 fallback 对 RK3588 MobileNet 模型均 PASS，3 次零输入输出 deterministic，run 中位数约 `2.20 ms`/`1.94 ms`（diagnostic only）。
- Orange Pi Vulkan loader 可加载但无 Mali ICD manifest，`vkCreateInstance=-9`，因此 accelerator smoke 总状态为 `partial`，不是硬件不支持结论。
- Target Probe 记录 `RKNPU` DRM 节点 `/dev/dri/card1`、`/dev/dri/renderD129` 与两个 platform device；完整 JSON、命令输出、测试和 SHA-256 清单归档于 `logs/archive/v0.16.1/rk3588-accelerator-validation/`。

## 0.16.0 - 2026-08-24 (development snapshot)

### Added

- 新增 `scripts/configure-arm-target.py`，支持 Orange Pi `aarch64` 与 P550 `riscv64` 的只读 inventory、用户目录级配置、源码同步、远端 import 检查和内容摘要。
- ARM/RISC-V Worker 配置模板默认只广告 `python-reference`，不写入真实凭证，也不把 RKNN 共享库或 DRM/RKNPU 驱动文件推断为可用 NPU。
- Target Probe 增加 RKNN runtime 文件证据，并将 `/dev/rknpu*` 设备节点检查扩展为 glob，区分“厂商文件存在”和“NPU 可执行”。
- 新增 AI 编译器问题回答与 ARM target 配置说明，明确 `OperatorSpec`、`torch.export` Graph IR、MLIR/LLVM lowering、Triton/IREE/RKNN 边界及 `brainuicl.pt2` 当前未统一下发的原因。

### Safety

- 配置器默认只读；`--apply` 只创建远端用户目录和模板，`--sync-source` 排除数据、checkpoint、日志、`.git` 和缓存，不安装系统包、不复制凭证、不启动 Worker。
- Orange Pi 的 RKNN 广告保持 blocked，直到真实 `/dev/rknpu*`、Runtime、模型转换、算子覆盖和 numerical correctness 全部通过。

### Validation

- Orange Pi 与 P550 SSH inventory 均通过，架构分别为 `aarch64` 与 `riscv64`。
- 两个目标均完成用户目录配置、EdgeForge 源码同步和远端 `import edgeforge` 检查；Worker 模板只含 `python-reference`。
- Orange Pi 实测存在 `librknnrt.so`、`librknn_api.so` 和 demo 模型，但没有 `/dev/rknpu*`；RKNN NPU 仍未通过部署门禁。
- 新增 ARM 配置器单元测试；完整测试结果记录在本版本发布记录和日志归档中。

## 0.15.0 - 2026-08-23（development snapshot）

### Added

- fixed-budget BrainUICL probe 支持显式 old-task retention 集：参数更新只发生在当前新任务训练批次，保留集只做 eval，输出 ACC/MF1/loss 的 initial/final/AULC 与 drop/delta。
- retention 指标以 `metric_role=retention` 写入 matrix bundle，并在 `predictor`、`outcome`、`diagnostic` 之外形成独立证据角色；LoP audit 不会把 retention 指标当作 `plasticity.acc_gain` outcome。
- matrix manifest 支持 `retention: {data_root, subjects, max_files, batch_size}`，dry-run 会检查每个旧任务 subject 的 data/label 文件；trajectory grouping 同时绑定 split 与 retention design，避免跨 protocol 合并。
- matrix descriptive report 增加 retention ACC/MF1 drop、loss delta 与角色表；缺失 retention 时保留空值，不做插值。
- instrumentation 增加 `checkpoint-optimizer-provenance-v1`：只记录 colocated optimizer/moment/scheduler 文件的身份和 digest，不加载未知序列化对象；参数快照没有 Adam state 时明确标为 unavailable。
- fixed-budget probe 保存 Adam 的 lr、weight decay、update steps 和 `state_persisted=false`，matrix 中归入 `metric_role=diagnostic`，避免把 probe-local optimizer 当成历史 optimizer trajectory。

### Safety

- retention 评估不读取 optimizer state、不执行 optimizer update，也不修改 canonical EEG、checkpoint 或 BrainUICL 源码；结果明确标注为 old-task stability diagnostic，不是 BWT 或 LoP 结论。
- 只有 manifest 显式提供 retention 集时才运行；缺少旧任务文件会阻断 cell，不会复制或猜测旧任务结果。

### Validation

- EdgeForge 自动化测试：`136/136 passed`。
- ISRUC subject 2 / seed 4321 的真实 retention smoke 成功：2 个新任务序列、1 个旧任务 subject-1 序列、固定预算 0→2；checkpoint retention ACC drop `0.1`、MF1 drop `0.02545`，仅为接口/证据链验证。
- 真实 matrix retention smoke 成功生成 `predictor/outcome/retention/diagnostic` 四种 metric role；单 stage、单 seed audit 仍为阻断状态，未生成 LoP 科学结论。
- 当前 parameter-only checkpoint 的 optimizer provenance 为 `status=unavailable, loaded=false`；probe-local Adam 与 retention 的 `optimizer_update=false` 均被记录为 diagnostic/provenance，而不是历史状态。
- 新增 `config/raeeg-lop-matrix-v15-isruc-multiseed-retention-preflight.json`：12 个 cell 只读预检得到 seed 4321 的 4 个 stage `ready`、seed 4322/4323 的 8 个 stage `blocked`，并明确列出缺失的三份模型参数文件。
- 运行日志、bundle、catalog、report、audit 和 SHA-256 清单归档于 `logs/archive/v0.15.0/raeeg-lop-retention-local-smoke/`。

## 0.14.0 - 2026-08-23（development snapshot）

### Added

- 新增 `scripts/run-raeeg-lop-matrix.py`，把 FACED/ISRUC 的 dataset、method、condition、subject 和 checkpoint stage 展开为可恢复的本地实验 cell；同一 subject/seed 的 stage 会另外汇总为 trajectory catalog，避免把连续 checkpoint 误判为重复 seed。
- 每个 cell 同时运行 checkpoint instrumentation 与 supervised-oracle fixed-budget probe，生成 `edgeforge-bundle-v1`、cell catalog、trajectory catalog、描述性 report 和只读 LoP audit，并保存 command/流程状态/stdout/stderr 与输入配置摘要。
- instrumentation 增加当前 checkpoint 相对 baseline 的参数更新 L2、relative update 与 top-parameter cosine 诊断，用来和 representation drift 分开审计。
- 新增矩阵 manifest 文档和 `config/raeeg-lop-matrix-v14-local-smoke.json` 本地复现实例；支持 clean/controlled-shift 条件和 checkpoint path template，但本版本不调度多卡、不提交远端任务。

### Safety

- runner 强制 output root 与 BrainUICL、checkpoint、canonical/shift 数据目录 disjoint，拒绝把不同 manifest 混入已有 run；失败 cell 和 audit 阻断状态都会留存。
- 结果继续标记 `scientific_conclusion_allowed=false`；单 seed、少 stage 的本地 smoke 只验证接口和证据链，不构成 LoP 结论。

### Validation

- EdgeForge 自动化测试：`128/128 passed`。
- 单机 RTX 4070 SUPER、共享 `brainuicl` 环境上，ISRUC subject 1 / seed 4321 的 stage 0 与 stage 10 两个 cell 均成功，分别生成 486 与 525 条合并指标；audit 正确保持单 seed 阻断。
- 同一单机上完成 ISRUC subject 2 / seed 4321 的 5 方法 × 4 stage（finetune、EWC、online-EWC、SI、MAS；共 20 cells）；全部成功，trajectory audit 为每种方法形成 3 个 lagged transitions，但因仅 1 个真实 seed 保持 `insufficient-seeds`。
- FACED subject 1 / seed 4321 的 pretrain 与已有 mini-CL stage 1 两个 cell 均成功；effective rank、parameter update、drift 和 probe 链路可用，但只有 1 个 transition，audit 保持 `insufficient-pairs`。
- 生成并运行 ISRUC subject 2 的 clean vs label-preserving noise severity 0.5 对照（finetune、4 stages、8 cells）；shift manifest 保存 source/output/label digest，两个条件均成功但各自仅 1 seed，audit 保持 `insufficient-seeds`。
- 新增 `docs/raeeg-lop-local-results-20260823.md`，集中记录本机 ISRUC/FACED/shift 结果、fresh-gap 符号约定和“数据 shift ≠ LoP”证据边界。
- 新增 `brainuicl-unlabeled-diagnostics.py`，记录不读取标签、不更新参数的 pseudo-label confidence/entropy/扰动一致性和表示 cosine，明确区分无标签可观测性与 supervised-oracle plasticity。
- 无标签诊断同时在 ISRUC 和 FACED pretrain smoke 上通过；FACED subject 1 的 noise-0.05 prediction agreement 为 `0.85`，仍只作为接口/可观测性基线。
- LoP matrix runner 增加 `--with-unlabeled-diagnostics`，可把上述无标签指标按 `metric_role=diagnostic` 合并到同一 cell bundle，但不会改变 predictor/outcome 选择。
- matrix `--dry-run` 增加只读 input preflight，逐 cell 报告缺失的数据目录、baseline/fresh/checkpoint 文件和 ready/blocked 计数。
- compact matrix manifest 支持 `seeds` 网格；缺少 seed checkpoint 的 cell 会在 preflight 中阻断，不会复制或冒充已有 seed。
- `config/raeeg-lop-matrix-v14-isruc-multiseed-preflight.json` 的只读预检生成 12 cells：seed 4321 的 4 个 stage `ready`，seed 4322/4323 的 8 个 stage 明确 `blocked`（checkpoint 不存在）。
- `matrix-report` 增加 clean 对照与每个 shift condition 的 paired `shift_minus_clean` 描述性 delta，明确标注 domain-shift sensitivity 不等于 LoP outcome。
- 完成 ISRUC clean/noise 0.5 的 8-cell `--with-unlabeled-diagnostics` 矩阵；bundle 同时保存 predictor、supervised outcome 和 label-free diagnostic，paired report 可比较 entropy/confidence/consistency 的 shift delta。
- 本次矩阵事件、cell 日志、catalog、audit 和 SHA-256 清单归档在 `logs/archive/v0.14.0/raeeg-lop-matrix-local-smoke/`。

## 0.13.0 - 2026-08-22（development snapshot）

### Added

- BrainUICL/RA-EEG 只读 instrumentation，覆盖 ISRUC/FACED 表示谱、attention、activation、经验 Fisher、梯度干扰、最后层 Jacobian proxy、checkpoint drift 和局部线性诊断。
- 固定预算 held-out probe 与 AULC/fresh-gap outcome envelope，以及 amplitude/noise/channel-dropout/time-jitter/bandstop 的 provenance-preserving 数据 shift 生成器。
- LoP metric alias、learning-curve normalizer 和稳定 context pairing，兼容历史 `transformer`/`transformer_1` 与 `task.plasticity`/`plasticity` 命名。

### Safety

- 所有 instrumentation 和 shift 工具都标记为 read-only/controlled derivative，不覆盖外部 BrainUICL、checkpoint 或 canonical EEG；单 seed smoke 不得升级为科学结论。
- `effective_rank`、Fisher、attention entropy 等保持诊断/代理语义；正式 LoP 仍必须使用 fixed-budget fresh gap、至少 3 个独立 seed 和预注册反事实。

### Validation

- 112/112 EdgeForge 自动化测试通过。
- ISRUC seed 4321 多阶段 instrumentation/probe pipeline 与 FACED seed 4321 pretrain smoke 通过；多阶段 audit 明确为 `insufficient-seeds`。
- FACED 5-subject symlink mini-split 的 finetune 1-task checkpoint smoke 通过，并完成 EdgeForge stage-1 指标回收；该结果仅是接口验证。
- v0.13.0 运行产物及 SHA-256 清单归档于 `logs/archive/v0.13.0/lop-instrumentation-20260822/`。

## 0.11.0 - 2026-08-21

### Added

- `target-probe` CLI 与 schema v1 设备证据清单，覆盖架构、CPU features、板型/SoC、内存、设备节点、内核 GPU/NPU 驱动、Vulkan ICD/loader 和 Runtime executable probe。
- 打包后的 `edgeforge.reference_model_pipeline` adapter，使 synthetic 六阶段流水线可在隔离 Worker `work_root` 中运行。
- 可选 `edgeforge.torch_model_pipeline` adapter 与本机 eager/`torch.compile` manifest；PyTorch 只在 Worker 外部环境中提供。
- 模型级 regression API/CLI，按模型、数据集、transform、Backend/Compiler identity 和完整 Target 隔离比较 correctness、steady latency 和 compile time。
- 版本化 `lop-lagged-correlation-v1` 分析、SQLite/API/CLI 持久化与内容寻址 digest，支持 Pearson、Spearman、exact context 和确定性 seed-cluster bootstrap CI。
- 新增只读 `lop-audit` CLI，可对本地 RA-EEG catalog 按方法检查 LoP predictor/outcome、stage、seed、source 和 replay 元数据，并输出方法级摘要。
- `lop-audit` 支持 method filter、完整 JSON 报告输出、自定义 predictor 的探索性标记，以及对非标准/无效历史结果的逐文件容错。
- 新增只读 `target-audit` CLI，将模型 manifest、Target Probe 和模型级 correctness 运行绑定为部署证据门禁；缺少显式 Backend 广告、目标架构匹配、accelerator 或成功 correctness 运行时保持 `blocked`。
- 新增 `POST /api/v1/deployment-audits`，让发布自动化可以复用同一部署证据门禁；报告是即时只读结果，不持久化原始 Probe 或 EEG。

### Safety

- RK3588 compatible string 不再自动广告 `rk3588-npu`；必须存在真实 `/dev/rknpu*` 设备节点。
- Target Probe 的 `backend_claims.inferred` 固定为空；文件、驱动或可执行程序存在都不等价于 Backend correctness 已验证。
- IREE 保持 blocked contract，不下载 LLVM/IREE 源码，也不生成推测的 Orange Pi 性能数据。
- LoP 分析拒绝混合 workload、protocol、comparison group、method、checkpoint transition 或重复 seed；最低有效 seed 门槛固定为 3，且所有结果都保持 `scientific_conclusion_allowed=false`。

### Validation

- 97/97 EdgeForge 自动化测试通过，包含独立临时 `work_root` 的六阶段 reference pipeline、LoP 分析和本地 catalog 审计回归。
- 对 BrainUICL 工作区 672 个现有结果完成只读全量审计；全部缺少正式 ER predictor，未生成伪造 LoP 结论。
- 共享 `research` 环境 PyTorch 2.11.0 下，CPU `torch-eager` 与 `torch-compile` 两条六阶段 pipeline 均通过；CUDA 未启用，不生成 GPU 性能结论。
- 模型失败任务会保留为 `model_runs.task_status=failed`，但不会被用作后续 baseline。
- 模型流水线的每个 stage 现在写入 `model_pipeline.<stage>` 版本化事件，保存状态、exit code、耗时和结构化输出摘要。
- LoP 回归覆盖非连续 checkpoint 配对、exact subject context、scope/method/stage/context-grid/duplicate-seed 阻断、常量指标、缺失证据、最新 task 指标隔离以及 API 幂等持久化。
- 本机 `target-probe` 和 synthetic pipeline 通过；Orange Pi probe、Runtime correctness 和离线 EEG inference 尚待真实串口/板端验证，因此本版本当前为候选状态。

本地真实 BrainUICL 证据补充：4070S 上 seed 4321 已完成真实六阶段 eager/Inductor 验证；默认 Inductor attention correctness 失败，`inductor-no-pattern` 安全 profile 通过（max error `4.148e-5`，steady latency `0.994 ms`）。Orange Pi/P550 的 V12 探测与 ARM64 preflight 仍保持 blocked，不把板端能力或单 seed 结果升级为部署/LoP 结论。

## 0.10.2 - 2026-08-20

### Added

- Explicit Backend/Target Registry，统一描述 architecture、device、accelerator 和 Worker advertised backends。
- `GET /api/v1/backend-capabilities` 与 `backend-capabilities` CLI。
- IREE explicit architecture/device 校验、RKNN ARM64/NPU 校验、custom backend target 校验。
- 上游贡献准备文档，明确 IREE explicit target 和 runtime manifest 两个小型 PR 边界。

### Safety

- 模型任务不再允许非 reference Backend 隐式继承 host target；缺少 target 时在控制面拒绝。
- Worker 默认只广告 `python-reference`，其余 Backend 必须由真实环境显式声明。

### Validation

- 72/72 EdgeForge 自动化测试通过。
- 该版本不修改 IREE 源码，也不声称实现 IREE #24760；上游工作仍需由真实模型 reproducer 驱动。
- IREE 源码未在本版本 clone 进 EdgeForge；当前磁盘余量约 8.8 GiB，避免无必要的大型源码副本。

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
- 扩大预算的单 seed LoP 验证完成；非法 stage 配置被正确拒绝并留存，多 seed 状态明确为 `blocked-by-checkpoint`。

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
