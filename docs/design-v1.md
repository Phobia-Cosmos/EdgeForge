# EdgeForge V1 系统设计

## 目标与边界

V1 只解决一个问题：让本机、Orange Pi、P550 和 Meles 形成可注册、可调度、可执行、可观测的异构计算集群。V1 不做跨节点 Tensor/Layer 切分，不自研编译器 IR，不把 Agent 放进关键控制路径；这些能力都依赖可靠的任务与性能数据底座，应在基础闭环稳定后增加。

V1 的最小闭环是：Worker 自动发现硬件能力并注册，控制面持久化节点状态，用户提交带硬约束的任务，调度器选择 Worker，Worker 主动领取并执行，最后回传退出码、日志、每次运行耗时和聚合结果。

```text
CLI / future Agent
        |
        | authenticated HTTP/JSON
        v
Control Plane (x86 first, P550 later)
  - Worker Registry
  - Task State Machine
  - Constraint Scheduler
  - SQLite Performance Records
        ^
        | outbound register / heartbeat / lease / result
        |
  +-----+------------+-------------+
  |                  |             |
RTX 4070S       Orange Pi       P550 / Meles
x86_64 + CUDA   ARM64 + RK3588  RISC-V64 CPU
```

## 核心决策

Worker 使用 pull 模型。开发板只需访问控制面的一个端口，不需要各自开放 RPC 端口；临时掉线时 Worker 可重新注册，控制面也能在租约过期后重排任务。

V1 使用 Python 3.10+ 标准库。本机和三块板子已分别具备 Python 3.10、3.11 或 3.12，因而不需要建立跨架构虚拟环境或下载依赖。性能敏感的 Backend 是独立进程，未来可以是 CUDA、Triton、RKNN、llama.cpp 或 RVV 程序，Worker 只负责生命周期和数据采集。

SQLite 是 V1 的权威状态存储。当前单控制面规模下，它比引入 etcd、Redis 和消息队列更容易验证；当任务吞吐或控制面高可用成为真实瓶颈时，再按观测数据拆分。

远程任务只接受参数数组并以 `shell=False` 执行。Worker 默认只允许少量探测命令，CI Worker 必须显式扩展白名单。控制 API 使用共享 Bearer Token；正式跨不可信网络时应在前面增加 TLS 或 WireGuard/Tailscale，而不是裸露端口。

## 任务状态机

```text
queued -> running -> succeeded
                  -> failed
       -> running --worker timeout--> queued (attempts remain)
                                    -> failed (attempts exhausted)
```

调度先应用硬约束：`worker_ids`、`architectures`、`accelerators`、`labels`、`min_memory_mb`。候选节点再按可用内存、归一化负载、正在执行的任务数和首选加速器评分。该策略是确定性的，后续 Performance Database 将替换启发式分数，成为 compiler-aware cost model。

## 后续演进

V2 增加 Operator IR 与固定的 MatMul、Softmax、RMSNorm、SiLU correctness/benchmark task，所有结果写入独立的 benchmark 表。

V3 增加 Kernel Registry，键由 operator、shape、dtype、backend、kernel version、hardware fingerprint 构成，并建立性能回归阈值。

V4 接入 CUDA/Triton、RKNN/RKLLM、ARM CPU 和 RISC-V CPU Backend。Backend 仍是 Worker 调用的受控适配器，不侵入控制面。

V5 才引入 Agent，让 Agent 读取结构化状态、日志和 Benchmark 数据，提出调优候选并通过既有 correctness/benchmark 门禁验证，Agent 不直接绕过任务状态机操作节点。

