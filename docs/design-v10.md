# V10 Model Pipeline

V10 把 EdgeForge 的边界从 Operator/Kernel 扩展到模型级实验，但不把 PyTorch、ONNX、IREE 或任何单一框架作为 Python 运行依赖。控制面接收一个 JSON manifest，Worker 在隔离的 `work_root` 内按固定顺序执行六个 stage：`export`、`transform`、`compile`、`run`、`correctness`、`benchmark`。`dataset.transform_command` 可连接 EEG normalize、window、channel selection 或特征缓存脚本；没有提供时 transform 会显式记录为 skipped，但 transform digest 仍然固定写入 provenance。

manifest 包含 `model`、`dataset`、`transforms`、`frontend`、`compiler`、`runtime`、`correctness`、`benchmark` 和 `target`。数据集 manifest digest 和规范化后的 transform digest 会写入任务 payload、model compiler manifest Artifact、SQLite `model_runs` 表与版本化事件，保证同一模型输入可以复现和比较。

Stage command 是 argv 数组，不经过 shell。Worker 对首个 executable 执行 allow-list 检查，限制 cwd 必须位于 `work_root`，限制超时和输出长度；未提供 command 的 stage 会显式记录为 `skipped`，不会伪造编译或性能结果。每个 stage 的最后一行若为 JSON object，会被保存为结构化输出；因此外部 PyTorch eager、`torch.compile`、ONNX Runtime、Triton 或 IREE adapter 可以共享同一接口。

模型 pipeline 完成后会上传 `model-compiler-manifest` Artifact，并写入 `model_runs`：模型/数据集、frontend、compiler backend/identity、目标架构、correctness、compile_ms、first_call_ms、steady_latency_ms、peak_memory_mb、output digest 和 Artifact digest。Orange Pi 上没有对应 Runtime 时，应保持任务失败或 capability blocked，不填入推测性能。

首要落地顺序是 `python-reference` correctness baseline、`torch-eager`、`torch-compile`、ONNX Runtime ARM64 验证，再按 profiler 结果增加 Triton/IREE。IREE 只作为可插拔后端；IREE issue #24760 不属于 V10 必做范围。
