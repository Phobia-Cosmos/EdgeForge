# EdgeForge V9：IREE `conv_nchwc` Runtime-only Pipeline

## 目标与边界

0.9.0 将 IREE `conv_nchwc` dilation ukernel 作为独立 Compiler/Runtime 集成接入 EdgeForge。IREE 源码、LLVM、模型和数据仍留在外部仓库或 Worker 本地；EdgeForge 只保存注册 Kernel、执行证据、Benchmark 和 compiler manifest Artifact。

该版本的 compile stage 明确标记为 `prebuilt`。它证明了一个已注册的 IREE runtime/ukernel binary 可以通过 EdgeForge 的任务租约、Worker allow-list、correctness 和 benchmark 合同运行，不证明 EdgeForge 构建了 IREE compiler，也不替代 MLIR lit 测试或模型级 `torch.compile` 验收。

## Operator 与参数

`conv_nchwc` 使用 packed rank-9 shape：`[N, OC_outer, OH, OW, IC_outer, FH, FW, k0, c0]`。首版验证 stride/dilation 为 1～64 的整数，`accumulate` 必须是布尔值；IREE Kernel metadata 的 `k0/c0` 必须和 shape 相等。benchmark adapter 将 shape、stride、dilation 和 accumulate 转换为显式 `--name=value` 参数，不拼接 shell 字符串。

## Trust Boundary

Kernel metadata 是受控注册配置，而不是来自不可信实验结果的自动发现。IREE test/benchmark command 必须是 argv 数组；Worker 只允许显式 allow-list 中的 executable，relative command/workdir 不能越过 work root，执行始终使用 `shell=False`。IREE backend 未被允许通过 `operator_benchmark` 的 Python fallback 冒充；非 reference backend 必须走 `kernel_pipeline`。

## Evidence

Pipeline 返回 compile、correctness、benchmark 三个 stage，保留有限长度 stdout/stderr、timings、throughput、exit code 和 source identity。manifest Artifact 记录 IREE repository、commit、patch SHA-256、Kernel snapshot 和 OperatorSpec。x86_64 与 aarch64 使用独立 Kernel ID，避免不同架构结果互相覆盖。

## 验收顺序

1. 用 fake executable 单测验证 flags、allow-list、trusted metadata 和 throughput parsing；
2. 在具备实际预构建 IREE runtime binary 的 x86_64 Worker 上完成 correctness/benchmark；
3. 在 Orange Pi 上使用 ARM64-native 或正确交叉编译 binary 重新注册并验证，不能复用 x86 结果；
4. 后续再把 accepted RA-EEG checkpoint 接入 eager/`torch.compile` 模型级 Pipeline，并串联 V8 Capability Gate。
