# 版本与日志策略

EdgeForge 使用语义化版本 `MAJOR.MINOR.PATCH`。协议不兼容、任务或 IR 语义不兼容时增加 MAJOR；增加向后兼容能力时增加 MINOR；仅修复兼容问题时增加 PATCH。当前开发线为 `0.16.2` RK3588 Vulkan userspace validation snapshot，V2–V16 的契约保持兼容；Orange Pi/P550 的真实 EEG 模型 Runtime 证据仍需单独完成后才能标记部署稳定。

每个版本必须同时留下四类记录：`CHANGELOG.md` 的用户可见变更、`releases/vX.Y.Z.md` 的发布验证证据、`EDGEFORGE_LOG_DIR/vX.Y.Z/` 的进程 JSONL 日志，以及控制面 SQLite 中的 release/event/task/benchmark 结构化数据。只写其中一种不能视为完整发布。

JSONL 文件名使用 UTC 启动时间、PID 和随机 run id，因此同版本重启不会覆盖旧文件。控制面任务写入创建它的 `version`，Worker 完成任务时写入自己的 `runtime_version`。Benchmark 查询必须同时保留这两个字段，不能假设集群会原子升级。

发布流程固定为：更新源码版本和 Changelog，建立发布记录草稿，执行自动化测试，部署到所有目标架构，执行 correctness/benchmark，核对版本混跑情况，更新发布记录为稳定，然后再创建 Git tag。测试失败的版本记录也必须保留并标记失败，不能删除以制造连续成功的历史。

运行日志可能含命令输出和路径，不应提交 Git。正式环境应备份 SQLite 与日志目录，并使用系统日志轮转或对象存储归档；轮转只能压缩或迁移旧文件，不能按版本覆盖。`scripts/archive-version-logs.sh VERSION [DESTINATION]` 会先归档本机 `EDGEFORGE_LOG_DIR/vVERSION/`，再按可选的 `ARCHIVE_REMOTE_HOSTS` 拉取远端 Worker 日志，最后生成 `SHA256SUMS`；远端节点离线时只写入 `REMOTE_GAPS`，不会丢失本机记录或中断归档。

当前四节点实验环境在版本冻结后运行 `scripts/archive-version-logs.sh 0.2.0`，将三块开发板的对应版本 Worker 日志集中保存到 `logs/archive/v0.2.0/`。0.13.0 的 BrainUICL/LoP 指标产物归档到 `logs/archive/v0.13.0/lop-instrumentation-20260822/`，0.14.0 的本地矩阵 runner 产物归档到 `logs/archive/v0.14.0/raeeg-lop-matrix-local-smoke/`，0.15.0 的 retention smoke 归档到 `logs/archive/v0.15.0/raeeg-lop-retention-local-smoke/`，0.16.0 的 ARM/RISC-V inventory、用户目录配置、远端 probe、测试日志和 SHA-256 清单归档到 `logs/archive/v0.16.0/arm-target-setup/`，0.16.1 的 RK3588 GPU/NPU API smoke、DRM/RKNN probe、手册说明和测试日志归档到 `logs/archive/v0.16.1/rk3588-accelerator-validation/`，0.16.2 的 Vulkan ICD candidate、目标板 probe、OpenCL/Vulkan/RKNN API smoke、测试日志和 SHA-256 清单归档到 `logs/archive/v0.16.2/rk3588-vulkan-userspace-audit/`；每个版本分别保存 manifest、trajectory/report、运行命令日志、测试日志和 SHA-256 清单，不能用新版本覆盖旧版本。中央控制面的 SQLite 应在停止写入或完成 SQLite backup/checkpoint 后复制到 `data/`；不能在 WAL 尚未同步时只复制主数据库文件。

版本归档索引由 `scripts/index-version-archives.py` 写入 `logs/archive/ARCHIVE_INDEX.json`。当前 0.10.1、0.10.2、0.11.0 和 0.12.0 均有独立归档与 SHA-256 清单；V11 BrainUICL 编译验证和 V12 目标探测/preflight 记录不会覆盖旧版本。
