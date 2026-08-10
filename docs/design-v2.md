# EdgeForge V2 Operator Benchmark Framework

## 数据路径

V2 把 AI workload 从不透明的远程命令提升为结构化 Operator IR。CLI 提交 operator、shape、dtype、backend 与调度约束；控制面验证 IR 并创建带控制面版本的任务；Worker 用自己的 runtime version 执行 correctness 和重复 Benchmark；控制面把完整任务结果、生命周期事件和可查询性能行分别持久化。

```text
Operator IR
    |
    v
Task(version, constraints)
    |
    v
Scheduler -> Worker(runtime_version)
    |
    v
Correctness + timings + checksum
    |
    +--> tasks: 完整原始结果
    +--> events: created / leased / completed
    +--> benchmarks: 可供筛选与 Cost Model 使用的结构化行
```

## Operator IR

当前 IR 字段为 `name`、`shape`、`dtype`、`backend` 和 `attrs`。MatMul shape 使用 `[M, K, N]`，其他首批一维算子使用 `[N]`。V2 对 rank、正维度、最大 workload 和 dtype 做入口验证，避免错误任务进入集群。

首批算子是 MatMul、Softmax、RMSNorm 与 SiLU，原因是它们覆盖矩阵乘、归一化、归约与逐元素计算四种基本模式，并且与 LLM workload 直接相关。

## Reference Backend

`python-reference` 无第三方依赖，目的是在 x86_64、aarch64 和 riscv64 上建立一致的 correctness 与协议基线。输入由确定性序列生成，输出保存 checksum，任务重复记录全部耗时以及 min、median 和 P95。

Reference Backend 不代表硬件原生性能，也不模拟 FP16/BF16 舍入。后续 CUDA、Triton、RKNN、ARM 和 RISC-V Backend 必须对同一个 OperatorSpec 实现独立执行器，并将输出与 reference checksum 或误差策略比较。

## Performance Database

每条 Benchmark 至少包含 task id、控制面 version、runtime version、worker、architecture、operator、shape、dtype、backend、correctness、原始 timings 和摘要。V3 Kernel Registry 将在此基础上增加 kernel identity、compiler flags、hardware fingerprint 和 artifact digest，避免把不同实现的结果合并。

## 版本与日志

文件 JSONL 负责保存每个进程运行期间的可读诊断，SQLite event ledger 负责生命周期检索，task result 负责完整 stdout/stderr 或 Operator 结果，benchmark 表负责性能查询。四种记录互相补充，任何一个都不能单独替代其他记录。

