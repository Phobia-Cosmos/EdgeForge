# EdgeForge 远端同步、双机进度与 LoP 后续计划（2026-08-22）

## 同步状态

当前分支 `integrate-iree-conv-nchwc` 已快进到远端 `origin/integrate-iree-conv-nchwc` 的 `ce35756`，没有执行 destructive reset，也没有覆盖本地 V11/V12 实验目录。同步前的本地未提交工作已保存在 Git stash `local-v11-v12-work-before-remote-sync-20260822`，并恢复到工作树；远端新增逻辑全部保留。

远端 2026-08-21 的 8 个提交依次完成：目标探测与 portable reference pipeline、PyTorch eager/compile contract、model regression、stage provenance events、版本化 LoP lagged-correlation 分析、一键 LoP audit、跨方法 LoP evidence validation、deployment target evidence audit。同步后自动化测试为 `105/105` 通过（远端基线 104，加上本地 V12 preflight/probe 测试）。

## 两台电脑/工作区的可确认进度

### 当前 4070S 工作站

- 已在真实 BrainUICL ISRUC seed 4321 checkpoint 上完成 `torch.export → transform → compile → run → correctness → benchmark`。
- eager correctness 最大/平均误差为 `0`；默认 Inductor 在自定义 attention 上失败；关闭 `pattern_matcher` 的安全 profile 通过，最大误差约 `4.148e-5`，steady latency 约 `0.994 ms`。
- 已建立固定 `[1,20,512]` attention reproducer，默认 Inductor 最大误差约 `0.356`，安全 profile 约 `1.23e-7`。
- 已通过 SSH 探测 Orange Pi（aarch64/RK3588、8 cores、15964 MiB、DRM）和 P550（riscv64、4 cores、25981 MiB、DRM）。Orange Pi 只有 numpy，没有 torch/ONNX Runtime/IREE/RKNN；P550 没有 numpy/torch/ONNX Runtime/IREE/RKNN。
- ARM64 BrainUICL deployment preflight 已正确返回 `BLOCKED`（缺少 `torch_python`），没有执行模型命令。

### 远端同步工作区/第二台电脑可确认进度

- 已把模型 pipeline 从 synthetic/reference contract 扩展为可隔离的 PyTorch adapter、回归查询和每 stage provenance event。
- 已把 LoP 分析从单次脚本提升为持久化 API/CLI：lag=1 按实际 checkpoint stage 配对，支持 exact context、Pearson/Spearman、seed-cluster bootstrap、重复 seed 和 scope/stage/context gate。
- 已新增 `lop-audit --catalog`，可只读扫描 BrainUICL 结果，按方法输出 predictor/outcome/source/stage/seed 状态。
- 已新增 `target-audit`，要求 manifest target、probe digest、显式 Backend advertisement、accelerator 和成功 correctness run 同时匹配；缺一项即 `blocked`。
- 远端提交没有证明 Orange Pi 已部署模型，也没有宣称 IREE/Vulkan/RKNN 可用。

## LoP 当前实验状态

2026-08-22 对 BrainUICL 工作区重新扫描得到 `819` 个 `metrics.json`/`RESULTS.json`。正式 predictor `task.spectra.transformer_1.effective_rank` 在 `819/819` 个结果中均缺失，正式 `ER(t-1) → plasticity.acc_gain(t)` 没有可配对证据。因此当前 LoP 状态仍是 `blocked-by-checkpoint/missing-predictor`，不是“没有 LoP”，也不是“已经发现 LoP”。

已有结果中不少方法有 plasticity outcome：EWC、Plain ER、MAS、SI、Online EWC、Finetune、SPR/PuriDivER adapted 等；但普通 accuracy/plasticity 曲线不能替代 representation predictor。将 `task.importance.mean` 临时作为自定义 predictor 时，部分分组可形成相邻 task pair，但会因 stage 不一致、重复 seed 或有效 seed 少于 3 而被标记 `blocked-incomparable-stages`、`blocked-duplicate-seeds` 或 `insufficient-seeds`。这些只能说明审计器工作正常，不能用于方法排名或 LoP 机制结论。

## 可以借鉴并迁移到 EEG 的方法点

1. **EWC / Online-EWC**：把旧任务 Fisher 重要性作为稳定性约束；迁移到 EEG 时同时记录每个 stage 的 layer-wise effective rank、Fisher norm、plasticity gain 和 forgetting。重要性本身不是 ER predictor，但可以作为探索性机制变量。
2. **SI / MAS**：记录参数路径积分或梯度敏感度，比较重要性集中是否预测下一 subject 的适应收益；需要统一 task order 和 probe budget。
3. **Plain ER / replay**：固定 replay capacity、new/replay ratio 和采样策略，研究 replay 是否改变表示秩退化与 plasticity；不能只比较最终 accuracy。
4. **SPR / PuriDivER 类方法**：可借鉴 replay 保护、teacher/student consistency、置信度过滤和稳定性约束，但必须把 EEG 无标签/伪标签协议与原论文监督假设分开；当前已有实现是 adapted/防护组合，不能直接称为原方法复现。
5. **BrainUICL**：作为 EEG 自监督持续适应主模型，保留 source replay、teacher pseudo-label、CEA 等组件，同时增加统一 checkpoint instrumentation。

## 4 卡 GPU 的使用方式

当前本机 `nvidia-smi` 只看到 1 张 RTX 4070 SUPER；4 卡需要申请后在 Worker 上显式登记。建议把 4 卡用于四个独立可审计任务：

- GPU0：BrainUICL seed 4321 baseline/eager 与 frozen-reference；
- GPU1：同一 seed 的 safe Inductor/profile 或 attention instrumentation；
- GPU2：方法 A（例如 EWC/Online-EWC）固定 subject/task split；
- GPU3：方法 B（例如 replay/SI/MAS）固定 subject/task split。

真正的多 seed LoP 需要真实 seed-specific checkpoint（至少 4321、4322、4323），不能用 4 张卡复制 seed 4321 冒充独立重复。申请到 4 卡后，先运行同一 checkpoint 的四路 profile/subject shard 并保存 GPU UUID、CUDA/PyTorch 版本、manifest digest；等 seed 4322/4323 checkpoint 到位，再切换为多 seed 矩阵。

## 后续优先级

1. 在 4070S 上完成 checkpoint instrumentation：每个 stage 输出 ER、rank/spectrum、importance、norm、plasticity 和 forgetting，并写入统一 Metric envelope。
2. 先跑 seed 4321 的四种方法对齐实验，验证指标链和 stage/context pairing，不宣称 LoP。
3. 获取真实 seed 4322/4323 checkpoint，固定相同 task order、subject split、probe budget 和 context grid。
4. 通过 `lop-audit` readiness 后再执行 `lop-analyze`；结果无论显著与否都保留 `scientific_conclusion_allowed=false`，交由独立科研审阅。
5. 只有在 CPU/ONNX/IREE/RKNN Runtime 真实安装并通过 correctness 后，才把 Orange Pi 加入模型部署矩阵；P550 继续做控制面、Artifact、CLI/API 和 RISC-V portability smoke。
