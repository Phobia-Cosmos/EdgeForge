# EdgeForge

EdgeForge 是面向 x86_64、ARM64、RISC-V64、GPU 与 NPU 的异构 AI Compiler / Runtime 实验基础设施。当前 `0.16.5` development snapshot 在 ISRUC/FACED 多 seed 预训练、BrainUICL 的 torch.export/CPU eager/CUDA Inductor 验证和 RK3588 runtime smoke 之上增加 EEG 对齐的简单 Transformer LoP 诊断；模型框架通过受控外部命令接入，IREE 仍是可插拔后端而非系统依赖。

完整的 V1 取舍见 [docs/design-v1.md](docs/design-v1.md)，V4 Compiler Pipeline 见 [docs/design-v4.md](docs/design-v4.md)，V5 Auto Tuning 见 [docs/design-v5.md](docs/design-v5.md)，V6 Compiler-aware Scheduler 见 [docs/design-v6.md](docs/design-v6.md)，V7 RA-EEG Experiment Contract 见 [docs/design-v7.md](docs/design-v7.md)，V8 Model Registry/Capability Gate 见 [docs/design-v8.md](docs/design-v8.md)，V9 IREE runtime-only Pipeline 见 [docs/design-v9.md](docs/design-v9.md)，V10 Model Pipeline 见 [docs/design-v10.md](docs/design-v10.md)，V11 Target Probe 见 [docs/design-v11.md](docs/design-v11.md)，BrainUICL/RA-EEG 迁移见 [docs/raeeg-migration.md](docs/raeeg-migration.md)。2026-08-16 的方向决策见 [docs/system-direction-2026-08-16.md](docs/system-direction-2026-08-16.md)，V7+ 路线见 [docs/roadmap-v7-plus.md](docs/roadmap-v7-plus.md)，双机进度与 LoP 计划见 [docs/sync-progress-and-lop-plan-20260822.md](docs/sync-progress-and-lop-plan-20260822.md)，LoP 统一数学与 FACED/ISRUC 实验计划见 [docs/lop-eeg-unified-plan-20260822.md](docs/lop-eeg-unified-plan-20260822.md)，BrainUICL 指标采集见 [docs/brainuicl-instrumentation-v1.md](docs/brainuicl-instrumentation-v1.md)，LoP 矩阵运行器见 [docs/raeeg-lop-matrix-v1.md](docs/raeeg-lop-matrix-v1.md)，连续 EEG 架构 LoP 适配与一键审计见 [docs/eeg-continuous-lop.md](docs/eeg-continuous-lop.md)，0.14.0 本地结果见 [docs/raeeg-lop-local-results-20260823.md](docs/raeeg-lop-local-results-20260823.md)，0.15.0 retention 结果见 [docs/raeeg-lop-retention-local-results-20260823.md](docs/raeeg-lop-retention-local-results-20260823.md)，0.16.3 ISRUC 剂量/retention 验证见 [releases/v0.16.3.md](releases/v0.16.3.md)，RK3588 Vulkan 用户态验证见 [docs/rk3588-vulkan-userspace-validation-v1.md](docs/rk3588-vulkan-userspace-validation-v1.md)，版本与日志规则见 [docs/versioning-and-logs.md](docs/versioning-and-logs.md)，历史变更见 [CHANGELOG.md](CHANGELOG.md)。

## 当前硬件基线

| Worker | 架构 | 资源 | 第一阶段角色 |
| --- | --- | --- | --- |
| RTX 4070S 主机 | x86_64 | 16 CPU，32 GB，RTX 4070 SUPER 12 GB | RA-EEG 实验、GPU Compiler/Profiler、控制面 |
| Orange Pi 5 Ultra | aarch64 | 8 CPU，16 GB，RK3588 | ARM64 部署验证；Vulkan/RKNN 仅在真实可用后启用 |
| P550 | riscv64 | 4 CPU，26 GB | RISC-V Agent/Runtime 兼容、构建与协议 smoke |
| Meles | riscv64 | 当前下线 | 暂不纳入调度和验证；恢复后再作为第二种 RISC-V 差分节点 |

## 本机快速启动

项目没有第三方运行依赖。开发模式下直接设置源码路径：

```sh
cd /home/undefined/Desktop/EdgeForge
export PYTHONPATH="$PWD/src"
export EDGEFORGE_TOKEN="$(openssl rand -hex 32)"
python3 -m edgeforge control --bind 0.0.0.0 --database ./edgeforge.db
```

在任何目标节点启动 Worker 前，先生成不带推测结论的能力证据清单：

```sh
python3 -m edgeforge target-probe --output ./edgeforge-target-probe.json
```

清单记录架构、CPU features、板型/SoC、内存、设备节点、内核 GPU/NPU 驱动、Vulkan ICD/loader 和已发现 Runtime 的实际探测结果。`backend_claims.inferred` 固定为空；只有显式配置且通过 Runtime correctness 后，Backend 才能加入 Worker 广告。

ARM/RISC-V 目标的用户目录级准备使用 [docs/ai-compiler-arm-target-setup-v1.md](docs/ai-compiler-arm-target-setup-v1.md) 和 `scripts/configure-arm-target.py`。默认只读；`--apply --sync-source` 只创建 work/log/config、同步排除数据和 checkpoint 的源码，并让 Orange Pi/P550 只广告 `python-reference`。RKNN 文件或 `/dev/dri` 节点不会自动解锁 NPU Backend。

```sh
PYTHONPATH=src python3 scripts/configure-arm-target.py \
  --apply --sync-source \
  --output-root .edgeforge/arm-target-setup/v0.16.3
```

模型发布前可用 `target-audit` 将 manifest、Target Probe 和模型级 correctness 证据绑定检查。它要求目标架构匹配、Backend 被显式广告、所需 accelerator 出现在真实 probe 中，并且存在同一 manifest digest 的成功模型运行；缺少任一证据都会返回 `blocked` 并以退出码 1 结束：

```sh
python3 -m edgeforge target-audit \
  --manifest config/model-pipeline-synthetic.json \
  --probe ./edgeforge-target-probe.json \
  --model-runs ./model-runs.json \
  --output ./deployment-audit.json
```

该审计只判断部署证据是否完整，不把 RK3588 字符串、驱动文件或 `backend_claims.inferred` 当成 NPU/Runtime correctness，也不产生性能或科研结论。完整契约见 [docs/deployment-target-audit.md](docs/deployment-target-audit.md)。

RKNN/OpenCL/Vulkan 的 capability preflight 还要求已保存的 API smoke 证据；例如 RKNN 使用 `scripts/model-deploy-preflight.py --runtime-validation .edgeforge/rk3588-accelerator-smoke-v0.16.3.json`。Orange Pi 的 Mali Vulkan 用户态包应以 `--vulkan-icd` 显式传入用户目录 manifest，再把同一份 smoke JSON 交给 preflight；仅发现 `librknnrt.so`、DRM 节点或 Vulkan loader 不会自动解锁 backend。

另开终端启动本机 Worker：

```sh
cd /home/undefined/Desktop/EdgeForge
export PYTHONPATH="$PWD/src"
export EDGEFORGE_TOKEN='<与控制面相同的令牌>'
python3 -m edgeforge worker \
  --control-url http://127.0.0.1:8080 \
  --worker-id worker-4070s \
  --label role=gpu
```

查看节点并提交任务：

```sh
python3 -m edgeforge workers --token "$EDGEFORGE_TOKEN"

python3 -m edgeforge submit \
  --token "$EDGEFORGE_TOKEN" \
  --arch x86_64 \
  --wait \
  -- uname -a

python3 -m edgeforge submit \
  --token "$EDGEFORGE_TOKEN" \
  --kind benchmark \
  --arch riscv64 \
  --repeats 10 \
  --wait \
  -- python3 -c 'sum(range(1000000))'

python3 -m edgeforge operator-benchmark \
  --token "$EDGEFORGE_TOKEN" \
  --operator matmul \
  --shape 32,64,32 \
  --dtype fp32 \
  --arch riscv64 \
  --repeats 5 \
  --wait
```

`submit` 的 `--` 之后是原样参数数组，不经过 shell。Worker 默认只允许 `uname`、`python3`、`true` 和 `false`。CI 节点可重复使用 `--allow-command git --allow-command cmake --allow-command ctest` 扩展白名单；`--allow-any-command` 只应用于完全可信的隔离 Worker。

## API

| 方法与路径 | 用途 |
| --- | --- |
| `GET /healthz` | 无鉴权存活检查 |
| `POST /api/v1/workers/register` | 注册或刷新 Worker 能力 |
| `POST /api/v1/workers/{id}/heartbeat` | 上报动态指标 |
| `POST /api/v1/workers/{id}/lease` | 领取一个匹配任务 |
| `GET /api/v1/workers` | 查询节点注册表 |
| `GET /api/v1/backend-capabilities` | 查询显式 Backend/Target 能力契约 |
| `POST /api/v1/tasks` | 创建命令或 Benchmark 任务 |
| `GET /api/v1/tasks/{id}` | 查询任务与结果 |
| `POST /api/v1/tasks/{id}/complete` | Worker 回传结果 |
| `GET /api/v1/events` | 查询版本化生命周期事件 |
| `GET /api/v1/benchmarks` | 查询 Operator 性能记录 |
| `GET/POST /api/v1/releases` | 查询或登记版本发布 |
| `GET/POST /api/v1/kernels` | 查询或登记 Kernel 候选 |
| `GET /api/v1/regressions` | 查询相邻 Benchmark 性能回归 |
| `GET/POST /api/v1/artifacts` | 查询或写入内容寻址 Artifact |
| `GET /api/v1/tuning-runs` | 查询 Auto Tuning 搜索历史和最佳配置 |
| `POST /api/v1/plans` | 生成可解释的 Kernel/Worker 执行计划 |
| `GET /api/v1/schedule-decisions` | 查询已执行任务的调度决策快照 |
| `GET /api/v1/experiments` | 查询模型实验、协议、方法、seed、摘要与 Bundle Artifact |
| `POST /api/v1/model-pipelines` | 创建模型 frontend/transform/compile/runtime/correctness/benchmark 任务 |
| `GET /api/v1/model-runs` | 查询模型级编译、正确性和性能结果 |
| `GET /api/v1/model-regressions` | 按模型/数据集/Backend/架构查询模型级 correctness、compile 和 steady latency 回归 |
| `POST /api/v1/deployment-audits` | 在控制面上审计 manifest、Target Probe 与模型 correctness 证据（不持久化原始输入） |
| `GET /api/v1/experiment-metrics` | 查询 Plasticity、Forgetting、BWT、Spectrum 等结构化指标 |
| `GET/POST /api/v1/lop-analyses` | 查询或创建版本化的 `ER(t-1) → Plasticity(t)` 描述性分析 |
| `GET /api/v1/lop-analyses/{id}` | 查询一个内容寻址的 LoP 分析结果与证据快照 |
| `GET/POST /api/v1/models` | 查询或注册绑定来源实验的模型 candidate |
| `GET/POST /api/v1/gate-policies` | 查询或创建不可变 Capability Gate Policy |
| `GET/POST /api/v1/gate-evaluations` | 查询或执行不可变 Gate Evaluation |
| `POST /api/v1/models/{id}/promote\|reject\|rollback` | 执行带原因的显式模型状态动作 |

除 `/healthz` 外，所有接口都要求 `Authorization: Bearer <token>`。

## 版本化日志与性能数据

每个进程运行实例会生成独立且不覆盖旧文件的日志：

```text
logs/
└── v0.8.0/
    ├── control/<UTC-run-id>.jsonl
    ├── worker/<UTC-run-id>.jsonl
    └── cli/<UTC-run-id>.jsonl
```

正式环境通过 `EDGEFORGE_LOG_DIR=/var/log/edgeforge` 固定日志根目录。SQLite 同时保存 release、task、event 和 operator benchmark 结构化记录：

```sh
python3 -m edgeforge releases --token "$EDGEFORGE_TOKEN"
python3 -m edgeforge events --token "$EDGEFORGE_TOKEN" --version 0.6.0
python3 -m edgeforge benchmarks --token "$EDGEFORGE_TOKEN" --operator matmul
python3 -m edgeforge artifacts --token "$EDGEFORGE_TOKEN" --kind compiler-manifest
python3 -m edgeforge tuning-runs --token "$EDGEFORGE_TOKEN" --operator matmul
python3 -m edgeforge schedule-decisions --token "$EDGEFORGE_TOKEN"
python3 -m edgeforge experiments --token "$EDGEFORGE_TOKEN" --workload raeeg
python3 -m edgeforge experiment-metrics --token "$EDGEFORGE_TOKEN" --name plasticity.acc_gain
python3 -m edgeforge lop-analyses --token "$EDGEFORGE_TOKEN" --workload raeeg-lop
python3 -m edgeforge models --token "$EDGEFORGE_TOKEN" --workload raeeg
python3 -m edgeforge gate-evaluations --token "$EDGEFORGE_TOKEN"
```

## RA-EEG 实验

### EEG decoder registry 与 LoP 诊断

可替换的 EEG 前端/上下文/分类头位于 [`src/edgeforge/eeg_models`](src/edgeforge/eeg_models)，registry 名称包括 `eegnet`、`tcn`、`transformer`、`conformer`、`shallowconvnet`、`deepconvnet`、`fbcnet`、`tsception`、`atcnet`、`cnn_lstm`、`eeg_graph`、`lop_mlp` 和 `brainuicl`。每个 decoder 都实现 `model(x) -> logits`、`forward_bundle(x)` 和 `representation_specs()`；`[B,C,S]` 与 `[B,T,C,S]` 输入可以共用同一诊断适配器。LoP/NTK/Hessian 诊断器见 [`src/edgeforge/lop_diagnostics.py`](src/edgeforge/lop_diagnostics.py)，统一 envelope 转换见 [`src/edgeforge/lop_envelope.py`](src/edgeforge/lop_envelope.py)，其中 `MetricConfig.freeze_batch_norm` 会在诊断期间冻结 BatchNorm running statistics 并恢复调用方模式；参数谱被标为 checkpoint-only，表示谱/Jacobian/NTK/Hessian 被标为 calibration/objective-dependent。

运行不依赖外部 EEG 数据的 registry smoke：

```sh
cd /home/undefined/Desktop/EdgeForge
PYTHONPATH=src /home/undefined/UbuntuData/python-envs/research/bin/python \
  scripts/eeg_lop_diagnostics.py --data synthetic-eeg --architectures all \
  --tasks 3 --train-samples 24 --eval-samples 16 --epochs 1 --batch-size 8 \
  --length 64 --channels 4 --classes 3 --probe-steps 0,1,2 \
  --output-dir logs/lop-diagnostics-registry-smoke
```

该输出只验证模型/指标 plumbing；正式 LoP 结论必须使用固定 calibration manifest、至少三个独立 seed、多个 checkpoint stage、相同 fresh-vs-warm 预算和 old-task retention。完整指标边界见 [`docs/eeg-lop-metrics-baseline-20260827.md`](docs/eeg-lop-metrics-baseline-20260827.md) 与模板 [`config/eeg-lop-diagnostics.example.json`](config/eeg-lop-diagnostics.example.json)。

诊断目标可通过 `--objective` 选择 `cross_entropy`、`mse`、`mse_to_zero`、`output_mean` 或 `output_norm`，并通过 `--label-source true|pseudo|none` 明确标签权限；`pseudo` 会从当前 logits 生成伪标签，`none` 只适用于不需要标签的目标。训练仍使用 stream 自带标签，诊断 objective 只决定 gradient/Hessian/Fisher 的测量，不会改变训练过程。

真实 BrainUICL checkpoint 缺失时，先用 `experiments/raeeg_lop_probe.py --preflight-only`；适配器会返回结构化 `blocked-by-checkpoint`/`blocked-by-data` 报告，不会将 synthetic smoke 冒充真实 EEG 证据。

模型级流水线使用一个 JSON manifest 描述模型、数据集、变换和后端命令。[config/model-pipeline-synthetic.json](config/model-pipeline-synthetic.json) 是可运行的标准库 reference baseline，覆盖六个 stage；adapter 通过已安装的 `edgeforge.reference_model_pipeline` 模块启动，因此可以在独立 Worker `work_root` 中运行。真实 BrainUICL 接入时只需替换对应 argv，并将 Worker 的 `--work-root` 指向受信任工作区：

```sh
python3 -m edgeforge model-pipeline --token "$EDGEFORGE_TOKEN" \
  --spec config/model-pipeline-synthetic.json --worker-id worker-4070s --wait
python3 -m edgeforge model-runs --token "$EDGEFORGE_TOKEN" --model brainuicl
python3 -m edgeforge model-regressions --token "$EDGEFORGE_TOKEN" --model brainuicl --threshold 0.2
```

本机 PyTorch contract 使用共享 `research` 环境（PyTorch 2.11.0），不改变 EdgeForge 的无第三方依赖控制面。先在 Worker 进程中激活环境并把 EdgeForge `src` 放进绝对 `PYTHONPATH`，再运行 eager 或 compile manifest：

```sh
source /home/undefined/UbuntuData/python-envs/activate-research.sh
export PYTHONPATH=/home/undefined/Desktop/EdgeForge/src${PYTHONPATH:+:$PYTHONPATH}
export EDGEFORGE_BACKENDS=python-reference,torch-eager,torch-compile
python3 -m edgeforge model-pipeline --token "$EDGEFORGE_TOKEN" \
  --spec config/model-pipeline-torch-eager.json --worker-id worker-local-torch --wait
python3 -m edgeforge model-pipeline --token "$EDGEFORGE_TOKEN" \
  --spec config/model-pipeline-torch-compile.json --worker-id worker-local-torch --wait
```

这两个 manifest 当前只验证 CPU eager 与 `torch.compile` 的模型接口、编译/首调用、数值 correctness 和重复 benchmark；没有 CUDA 时不会声称 GPU 性能，真实 BrainUICL checkpoint 仍需独立接入。

BrainUICL 历史结果迁移和 LoP smoke 验证记录见 [docs/raeeg-migration.md](docs/raeeg-migration.md) 与 [docs/raeeg-lop-validation-20260819.md](docs/raeeg-lop-validation-20260819.md)。

扩大 sequence/eval 预算后的单 seed LoP 验证见 [docs/raeeg-lop-expanded-validation-20260820.md](docs/raeeg-lop-expanded-validation-20260820.md)；多 seed 当前明确标记为 `blocked-by-checkpoint`。

版本化 LoP 分析器用于检验前一 checkpoint 的 ER 与后一 checkpoint plasticity 之间的描述性关联。默认 `lag=1` 按每个实验实际存在的有序 stage 配对，例如 `[0, 10, 25]` 产生 `0→10` 与 `10→25`，不会解释成整数 `stage - 1`。分析要求 workload、protocol、comparison group、method 和 checkpoint transition 一致，至少 3 个不重复的真实 seed；bootstrap 按 seed cluster 重采样。结果始终保存 `scientific_conclusion_allowed=false`，不能自动变成因果或群体 LoP 结论。完整契约见 [docs/raeeg-lop-analysis.md](docs/raeeg-lop-analysis.md)。

```sh
python3 -m edgeforge lop-analyze --token "$EDGEFORGE_TOKEN" \
  --experiment-id lop-seed-4321 \
  --experiment-id lop-seed-4322 \
  --experiment-id lop-seed-4323 \
  --context-policy exact \
  --bootstrap-repeats 2000

python3 -m edgeforge lop-analysis --token "$EDGEFORGE_TOKEN" <analysis-id>
```

对尚未导入控制面的本地 catalog，可一键做只读 LoP 证据审计。该命令会按方法报告 source、ER predictor、plasticity outcome、stage、seed 和 replay 元数据；`--summary` 只输出方法级摘要，不上传或修改原始结果：

```sh
PYTHONPATH=src python3 -m edgeforge lop-audit \
  --catalog config/raeeg-local-catalog.json \
  --summary
```

如果要检查是否达到 LoP 的固定预算要求，使用本地只读 requirement gate；它只把 `fresh_gap > 0` 作为主 outcome，要求相同设计与 stage grid、至少三个真实 seed，并按 seed cluster 输出 95% CI：

```sh
PYTHONPATH=src python3 scripts/evaluate-raeeg-lop-gate.py \
  --catalog logs/<run>/trajectory-catalog.json \
  --required-stage 0 --required-stage 10 --required-stage 25 \
  --output logs/<run>/lop-gate.json \
  --markdown-output logs/<run>/lop-gate.md
```

返回 `candidate` 也不等于科学结论；`scientific_conclusion_allowed` 始终为 `false`。replay、forgetting、BWT 和 old-task retention 会单独列在 `retention_inventory`，不能替代 fresh-gap。

审计发现 `plasticity.acc_gain` 不等于 LoP 证据；必须同时存在可配对的 ER predictor 和至少 3 个不重复 seed。当前 catalog 中 EWC、Online-EWC、SI、MAS、Finetune 有 plasticity，但缺少 `task.spectra.transformer_1.effective_rank`；BrainUICL、SPR、PuriDivER 的登记 source 路径也需要先修复，因此它们目前都不能报告 LoP 结果。

如果想检查已有正则化轨迹中的其它关联，可显式指定自定义 predictor；系统会将结果标成探索性关联而不是 LoP：

```sh
PYTHONPATH=src python3 -m edgeforge lop-audit \
  --catalog config/raeeg-local-catalog.json \
  --predictor task.importance.mean \
  --outcome plasticity.acc_gain \
  --summary
```

当前 seed 4321 单次结果的探索性 Pearson 值为：EWC `-0.0414`、Online-EWC `0.0336`、SI `-0.0796`、MAS `0.0031`；Finetune 的 predictor 无变化，相关系数不可定义。它们全部因只有一个 seed 而为 `insufficient-seeds`，不能解释为 LoP 或方法差异结论。

对 BrainUICL 工作区全部 672 个历史结果的验证记录见 [docs/raeeg-lop-method-validation-20260821.md](docs/raeeg-lop-method-validation-20260821.md)：所有结果均缺 ER predictor，尚不能回答正则化或 replay 是否出现 LoP。

查询模型 Backend 与 Target 约束：

```sh
python3 -m edgeforge backend-capabilities --token "$EDGEFORGE_TOKEN"
```

4070S Worker 需要把工作根目录设为现有 BrainUICL 仓库，并只允许受信任的 BrainUICL Python 环境：

```sh
python3 -m edgeforge worker \
  --control-url http://127.0.0.1:8080 \
  --worker-id worker-4070s \
  --work-root /home/undefined/Desktop/bci/code/tta_security/BrainUICL \
  --allow-command /home/undefined/Disk/python-envs/brainuicl/bin/python \
  --label workload=raeeg
```

运行真实 ISRUC LoP smoke：

```sh
python3 -m edgeforge experiment-run \
  --token "$EDGEFORGE_TOKEN" \
  --spec config/raeeg-lop-smoke.json \
  --worker-id worker-4070s \
  --accelerator nvidia-gpu \
  --wait
```

当前本机 LoP 矩阵入口（只使用本地单机/单 GPU，不提交开发板或多卡任务）见 [docs/raeeg-lop-matrix-v1.md](docs/raeeg-lop-matrix-v1.md)。它会为每个 dataset × method × condition × subject × checkpoint stage 保存 cell bundle，并把同一 subject/seed 的连续 stage 汇总到 trajectory catalog：

```sh
PYTHONPATH=src /home/undefined/Disk/python-envs/brainuicl/bin/python \
  scripts/run-raeeg-lop-matrix.py \
  --manifest config/raeeg-lop-matrix-v14-isruc-local.json \
  --log-root logs
```

本地 0.14.0 已验证 ISRUC 五种方法 × 四个 stage 共 20 个 cell，以及 FACED pretrain/mini-CL 两个接口 smoke；所有结果和 SHA-256 清单在 `logs/archive/v0.14.0/raeeg-lop-matrix-local-smoke/`。0.15.0 新增可选 retention 集：probe 只在新任务上更新参数，同时在固定旧任务集上记录 ACC/MF1/loss 变化；这些指标以 `metric_role=retention` 保存，不能替代 LoP outcome 或 BWT。0.15.0 的真实 retention smoke 归档在 `logs/archive/v0.15.0/raeeg-lop-retention-local-smoke/`，bundle 还显式区分 optimizer provenance diagnostic。所有运行只有 seed 4321，audit 会保持 `insufficient-seeds`/`insufficient-pairs`，因此不能据此声称 LoP 已发生。

无标签在线适应前的可观测量可用 [docs/brainuicl-unlabeled-diagnostics-v1.md](docs/brainuicl-unlabeled-diagnostics-v1.md) 中的 `brainuicl-unlabeled-diagnostics.py` 采集；它与 supervised-oracle probe 分开记录，不读取标签值也不更新参数。

在补齐真实多 seed checkpoint 前，可先运行只读多 seed preflight：

```sh
PYTHONPATH=src /home/undefined/Disk/python-envs/brainuicl/bin/python \
  scripts/run-raeeg-lop-matrix.py \
  --manifest config/raeeg-lop-matrix-v14-isruc-multiseed-preflight.json \
  --dry-run --log-root logs
```

当前预检结果为 `12 cells = 4 ready + 8 blocked`；4322/4323 的缺失 checkpoint 会被明确列出，不会复用 4321。

连续 EEG 架构实验可直接使用 [docs/eeg-continuous-lop.md](docs/eeg-continuous-lop.md) 中的一键命令生成标准 trajectory catalog 并运行 LoP gate。适配器脚本为 `scripts/build-eeg-continuous-trajectory-catalog.py`，一键审计入口为 `scripts/audit-eeg-continuous-lop.py`，数据画像入口为 `scripts/analyze-eeg-continuous-lop.py`；这些工具均只读原始实验结果，不上传数据。当前 r2 结果的 gate 明确显示五个架构都未通过严格的全 transition 正 fresh-gap 要求。若要验证 gate 能否识别已知的可塑性受损条件，可使用 [docs/eeg-synthetic-lop-positive-control.md](docs/eeg-synthetic-lop-positive-control.md) 中的合成正控；它不会被解释为真实 EEG 的科学结论。

批量导入本机已有八组 EEG 实验结果：

```sh
PYTHONPATH=src python3 scripts/import-raeeg-catalog.py \
  --token "$EDGEFORGE_TOKEN" \
  --worker-id worker-4070s \
  --wait
```

Catalog 将 `aligned-full49` 和 `method-transfer` 标记为不同 comparison group；SPR/PuriDivER 不会被错误地混入 BrainUICL/正则化六方法公平对比。原始 EEG 不上传，Experiment Bundle 只保存配置、环境、结果 digest、摘要和归一化指标。

用固定 aligned 策略评估已注册 candidate；PASS 只进入 accepted，随后仍需显式 Operator promote：

```sh
python3 -m edgeforge gate-policy-put --token "$EDGEFORGE_TOKEN" \
  --policy config/raeeg-aligned-gate-v1.json
python3 -m edgeforge gate-evaluate --token "$EDGEFORGE_TOKEN" \
  --model-id model-brainuicl-aligned --policy-id raeeg-aligned-capability-v1
python3 -m edgeforge model-promote --token "$EDGEFORGE_TOKEN" \
  model-brainuicl-aligned --reason 'V8 release approval'
```

Operator IR 当前支持 `matmul`、`softmax`、`rmsnorm`、`silu` 和 packed `conv_nchwc`。`python-reference` Backend 用于跨架构 correctness 基线，不宣称模拟 FP16/BF16 舍入，也不用于替代后续 CUDA、Triton、RKNN 或 RVV Kernel。IREE `conv_nchwc` dilation 的 runtime-only 接入、信任边界和 Orange Pi 运行步骤见 [docs/integration-iree-conv-nchwc.md](docs/integration-iree-conv-nchwc.md)。

注册一个只允许 ARM64 的 Kernel，并用它提交任务：

```sh
python3 -m edgeforge kernel-register \
  --token "$EDGEFORGE_TOKEN" \
  --id kernel-softmax-arm-v1 \
  --operator softmax \
  --backend python-reference \
  --version 1 \
  --arch aarch64

python3 -m edgeforge operator-benchmark \
  --token "$EDGEFORGE_TOKEN" \
  --operator softmax --shape 1024 \
  --kernel-id kernel-softmax-arm-v1 \
  --worker-id worker-orangepi --wait
```

运行 V4 Kernel Pipeline：

```sh
python3 -m edgeforge kernel-pipeline \
  --token "$EDGEFORGE_TOKEN" \
  --kernel-id kernel-softmax-arm-v1 \
  --operator softmax \
  --shape 1024 \
  --dtype fp32 \
  --worker-id worker-orangepi \
  --wait
```

上传一个本地 manifest 或其他小型 Artifact：

```sh
python3 -m edgeforge artifact-put \
  --token "$EDGEFORGE_TOKEN" \
  --kind compiler-manifest \
  --media-type application/json \
  ./manifest.json
```

在 NVIDIA GPU Worker 上使用默认 Grid Search 调优 Triton MatMul：

```sh
python3 -m edgeforge kernel-autotune \
  --token "$EDGEFORGE_TOKEN" \
  --kernel-id kernel-triton-matmul-autotune-v1 \
  --operator matmul \
  --shape 256,256,256 \
  --dtype fp16 \
  --worker-id worker-4070s \
  --repeats 10 \
  --warmup 5 \
  --wait
```

可重复传入 `--candidate '{"block_m":64,"block_n":64,"block_k":32,"num_warps":4,"num_stages":3}'` 覆盖默认搜索空间。只有 correctness 通过的候选会参与最佳配置选择。

让 V6 Cost Model 解释并执行最佳路径：

```sh
python3 -m edgeforge compiler-plan \
  --token "$EDGEFORGE_TOKEN" \
  --operator matmul \
  --shape 64,64,64 \
  --dtype fp16

python3 -m edgeforge compiler-run \
  --token "$EDGEFORGE_TOKEN" \
  --operator matmul \
  --shape 64,64,64 \
  --dtype fp16 \
  --repeats 5 \
  --warmup 5 \
  --wait
```

可以使用 `--backend`、`--kernel-id`、`--arch` 和 `--worker-id` 限制候选，并通过 `--compile-weight`、`--load-weight-ms` 与 `--unknown-latency-ms` 调整成本策略。

## 测试

```sh
cd /home/undefined/Desktop/EdgeForge
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

`deploy/systemd/` 提供生产化服务模板。部署时应为每台机器创建独立 `edgeforge` 系统用户，把源码安装到系统 Python 可发现的位置，并在 `/etc/edgeforge/edgeforge.env` 中配置控制面地址、令牌、稳定 Worker ID 和日志目录。
