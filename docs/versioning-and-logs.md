# 版本与日志策略

EdgeForge 使用语义化版本 `MAJOR.MINOR.PATCH`。协议不兼容、任务或 IR 语义不兼容时增加 MAJOR；增加向后兼容能力时增加 MINOR；仅修复兼容问题时增加 PATCH。当前开发线为 `0.9.0`，V2–V8 已通过验证并冻结。

每个版本必须同时留下四类记录：`CHANGELOG.md` 的用户可见变更、`releases/vX.Y.Z.md` 的发布验证证据、`EDGEFORGE_LOG_DIR/vX.Y.Z/` 的进程 JSONL 日志，以及控制面 SQLite 中的 release/event/task/benchmark 结构化数据。只写其中一种不能视为完整发布。

JSONL 文件名使用 UTC 启动时间、PID 和随机 run id，因此同版本重启不会覆盖旧文件。控制面任务写入创建它的 `version`，Worker 完成任务时写入自己的 `runtime_version`。Benchmark 查询必须同时保留这两个字段，不能假设集群会原子升级。

发布流程固定为：更新源码版本和 Changelog，建立发布记录草稿，执行自动化测试，部署到所有目标架构，执行 correctness/benchmark，核对版本混跑情况，更新发布记录为稳定，然后再创建 Git tag。测试失败的版本记录也必须保留并标记失败，不能删除以制造连续成功的历史。

运行日志可能含命令输出和路径，不应提交 Git。正式环境应备份 SQLite 与日志目录，并使用系统日志轮转或对象存储归档；轮转只能压缩或迁移旧文件，不能按版本覆盖。

当前四节点实验环境在版本冻结后运行 `scripts/archive-version-logs.sh 0.2.0`，将三块开发板的对应版本 Worker 日志集中保存到 `logs/archive/v0.2.0/`。中央控制面的 SQLite 应在停止写入或完成 SQLite backup/checkpoint 后复制到 `data/`；不能在 WAL 尚未同步时只复制主数据库文件。
