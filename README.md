# EdgeForge

EdgeForge 是面向 x86_64、ARM64、RISC-V64、GPU 与 NPU 的异构 AI Compiler / Runtime 实验基础设施。当前 `0.2.0` 在 V1 分布式闭环之上增加了版本化日志、发布账本、Operator IR 和跨架构 Performance Database 雏形。

完整的 V1 取舍见 [docs/design-v1.md](docs/design-v1.md)，版本与日志规则见 [docs/versioning-and-logs.md](docs/versioning-and-logs.md)，历史变更见 [CHANGELOG.md](CHANGELOG.md)。

## 当前硬件基线

| Worker | 架构 | 资源 | 第一阶段角色 |
| --- | --- | --- | --- |
| RTX 4070S 主机 | x86_64 | 16 CPU，32 GB，RTX 4070 SUPER 12 GB | 控制面、GPU Worker |
| Orange Pi 5 Ultra | aarch64 | 8 CPU，16 GB，RK3588 | ARM / NPU Worker |
| P550 | riscv64 | 4 CPU，26 GB | RISC-V Worker，后续控制面候选 |
| Meles | riscv64 | 4 CPU，16 GB，TH1520/C910 | RISC-V 兼容性与性能对照 |

## 本机快速启动

项目没有第三方运行依赖。开发模式下直接设置源码路径：

```sh
cd /home/undefined/Desktop/EdgeForge
export PYTHONPATH="$PWD/src"
export EDGEFORGE_TOKEN="$(openssl rand -hex 32)"
python3 -m edgeforge control --bind 0.0.0.0 --database ./edgeforge.db
```

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
| `POST /api/v1/tasks` | 创建命令或 Benchmark 任务 |
| `GET /api/v1/tasks/{id}` | 查询任务与结果 |
| `POST /api/v1/tasks/{id}/complete` | Worker 回传结果 |
| `GET /api/v1/events` | 查询版本化生命周期事件 |
| `GET /api/v1/benchmarks` | 查询 Operator 性能记录 |
| `GET/POST /api/v1/releases` | 查询或登记版本发布 |

除 `/healthz` 外，所有接口都要求 `Authorization: Bearer <token>`。

## 版本化日志与性能数据

每个进程运行实例会生成独立且不覆盖旧文件的日志：

```text
logs/
└── v0.2.0/
    ├── control/<UTC-run-id>.jsonl
    ├── worker/<UTC-run-id>.jsonl
    └── cli/<UTC-run-id>.jsonl
```

正式环境通过 `EDGEFORGE_LOG_DIR=/var/log/edgeforge` 固定日志根目录。SQLite 同时保存 release、task、event 和 operator benchmark 结构化记录：

```sh
python3 -m edgeforge releases --token "$EDGEFORGE_TOKEN"
python3 -m edgeforge events --token "$EDGEFORGE_TOKEN" --version 0.2.0
python3 -m edgeforge benchmarks --token "$EDGEFORGE_TOKEN" --operator matmul
```

Operator IR 当前支持 `matmul`、`softmax`、`rmsnorm` 和 `silu`。`python-reference` Backend 用于跨架构 correctness 基线，不宣称模拟 FP16/BF16 舍入，也不用于替代后续 CUDA、Triton、RKNN 或 RVV Kernel。

## 测试

```sh
cd /home/undefined/Desktop/EdgeForge
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

`deploy/systemd/` 提供生产化服务模板。部署时应为每台机器创建独立 `edgeforge` 系统用户，把源码安装到系统 Python 可发现的位置，并在 `/etc/edgeforge/edgeforge.env` 中配置控制面地址、令牌、稳定 Worker ID 和日志目录。
