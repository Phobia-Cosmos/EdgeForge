# EdgeForge V9：IREE `conv_nchwc` Runtime-only Pipeline

## 目标与边界

0.9.0 为 IREE `conv_nchwc` dilation ukernel 准备独立 Compiler/Runtime 适配层。IREE #24760 当前是开放的功能 issue，不是已经存在的 upstream 实现；因此本版本交付的是 adapter scaffold 与 contract validation，而不宣称真实 IREE binary 已可用。IREE 源码、LLVM、模型和数据仍留在外部仓库或 Worker 本地；EdgeForge 只保存注册 Kernel、执行证据、Benchmark 和 compiler manifest Artifact。

该版本的 compile stage 明确标记为 `prebuilt`。在真实 IREE binary 与 source identity 可用后，它用于证明该 binary 可以通过 EdgeForge 的任务租约、Worker allow-list、correctness 和 benchmark 合同运行；当前 contract-only 验证只证明适配器和数据链可运行。两种情况都不证明 EdgeForge 构建了 IREE compiler，也不替代 MLIR lit 测试或模型级 `torch.compile` 验收。

## Operator 与参数

`conv_nchwc` 使用 packed rank-9 shape：`[N, OC_outer, OH, OW, IC_outer, FH, FW, k0, c0]`。首版验证 stride/dilation 为 1～64 的整数，`accumulate` 必须是布尔值；IREE Kernel metadata 的 `k0/c0` 必须和 shape 相等。benchmark adapter 将 shape、stride、dilation 和 accumulate 转换为显式 `--name=value` 参数，不拼接 shell 字符串。

## Trust Boundary

Kernel metadata 是受控注册配置，而不是来自不可信实验结果的自动发现。IREE test/benchmark command 必须是 argv 数组；Worker 只允许显式 allow-list 中的 executable，relative command/workdir 不能越过 work root，执行始终使用 `shell=False`。IREE backend 未被允许通过 `operator_benchmark` 的 Python fallback 冒充；非 reference backend 必须走 `kernel_pipeline`。`blocked-*` 状态会在启动 executable 前被拒绝；除显式的 `contract-only-not-real-iree` 状态外，执行必须提供 repository、40 位 commit 和 64 位 patch SHA-256。

## Evidence

Pipeline 返回 compile、correctness、benchmark 三个 stage，保留有限长度 stdout/stderr、timings、throughput、exit code 和 source identity。manifest Artifact 记录 IREE repository、commit、patch SHA-256、validation status、Kernel snapshot 和 OperatorSpec。x86_64 与 aarch64 使用独立 Kernel ID，避免不同架构结果互相覆盖。当前两台机器完成的是 fake executable contract validation；其中固定 timing 只用于验证数据链，不能解释为 IREE 性能。

## 验收顺序

1. 用 fake executable 单测及双节点 contract run 验证 flags、allow-list、trusted metadata、任务调度、Artifact/Benchmark 持久化和 throughput parsing；
2. 跟踪 IREE #24760，取得可审查的实现 commit/patch 后再解除 blocked registration；
3. 在具备实际预构建 IREE runtime binary 的 x86_64 Worker 上完成 correctness/benchmark；
4. 在 Orange Pi 上使用 ARM64-native 或正确交叉编译 binary 重新注册并验证，不能复用 x86 结果；
5. 后续再把 accepted RA-EEG checkpoint 接入 eager/`torch.compile` 模型级 Pipeline，并串联 V8 Capability Gate。

本 PR 是 CPU ukernel scaffold，与 Vulkan、RKNN、Meles 或 P550 验证无关。
