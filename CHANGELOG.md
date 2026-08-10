# Changelog

本文件记录 EdgeForge 每个公开版本的用户可见变更。不可变的发布验证详情保存在 `releases/vX.Y.Z.md`，运行期结构化日志保存在配置的 `EDGEFORGE_LOG_DIR/vX.Y.Z/`，控制面事件、任务和 Benchmark 则保存在 SQLite。

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
