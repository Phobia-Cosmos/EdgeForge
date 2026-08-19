# BrainUICL/RA-EEG 迁移到 EdgeForge

EdgeForge 不复制 `/home/undefined/Desktop/bci/code` 的源码、数据集或 checkpoint。BrainUICL 继续作为研究 source of truth，EdgeForge 只保存实验契约、结果 digest、归一化指标、Artifact 和版本事件。这样既能重放原实验，也不会让控制面持有原始 EEG。

## Backend 与开发板

Backend 描述模型的导出、编译和执行实现；Target/Worker 描述实际运行位置。一个 Backend 可以对应多个架构，一个架构也可以有多个 Backend：

| Backend | 主要用途 | 可选 Target |
| --- | --- | --- |
| `torch-eager` | 研究基线和能力正确性 | 4070S/x86_64，取决于 PyTorch 环境 |
| `torch-compile` | graph break、编译耗时和 GPU 性能 | 首先 4070S/x86_64 |
| `onnx-runtime` | 中立模型格式与 ARM64 可加载性 | 4070S、Orange Pi（需真实安装） |
| `python-reference` | 无依赖 correctness 基线 | 所有能运行 Python 的 Worker |
| `triton` | 经过 profiler 确认的 GPU 热点 | 4070S/GPU |
| `iree` | 可插拔编译/Runtime 实验 | 由真实 target、驱动和 artifact 支持决定 |
| `rknn` | RK3588 NPU 专用路径 | Orange Pi/RK3588（需工具链和算子覆盖） |

因此调度键应理解为 `model × backend × target × capability`，而不是“每块板一个后端”。P550/Meles 默认承担协议、Artifact、构建和 Runtime smoke，不强制承担 PyTorch 训练。

Worker 默认只广告 `python-reference`。当节点经过真实安装和验证后，可在启动 Worker 前显式设置 `EDGEFORGE_BACKENDS=python-reference,torch-eager,torch-compile`（或对应 ONNX/IREE 名称）；EdgeForge 不会因为发现 `python3` 就自动声称这些后端存在。

## 历史结果迁移

当前 BrainUICL 工作区中发现至少 817 个 `metrics.json/RESULTS.json`。迁移工具会为每个文件生成稳定的 `experiment_id`、源文件 SHA-256、数据集名称、方法、协议、seed 和相对结果路径：

```sh
cd /home/undefined/Desktop/EdgeForge
export PYTHONPATH="$PWD/src"
python3 scripts/build-raeeg-catalog.py \
  --root /home/undefined/Desktop/bci/code/tta_security/BrainUICL \
  --output /tmp/raeeg-brainuicl-catalog.json \
  --dataset-manifest-digest 5344651092da22c1fa3dc068064e6c3c8f5ef9e87178fe8f77f2fb89d67d5346
```

先查看或抽样生成 catalog，再提交到控制面。提交前 Worker 必须以 BrainUICL 仓库作为 `--work-root`，并只允许受信任的 BrainUICL Python：

```sh
python3 -m edgeforge worker \
  --control-url http://127.0.0.1:8080 \
  --worker-id worker-4070s \
  --work-root /home/undefined/Desktop/bci/code/tta_security/BrainUICL \
  --allow-command /home/undefined/Disk/python-envs/brainuicl/bin/python \
  --label workload=raeeg

python3 scripts/import-raeeg-catalog.py \
  --catalog /tmp/raeeg-brainuicl-catalog.json \
  --worker-id worker-4070s \
  --wait
```

批量迁移不等于批量宣称公平结论。工具按路径标记 `aligned-full49`、`method-transfer` 和 `historical-unclassified`；只有同一 workload、protocol、comparison group 的实验才能进入 Capability Gate。首次建议只迁移已有的 8 组 curated ISRUC catalog，再按目录逐批导入：

```sh
python3 scripts/build-raeeg-catalog.py --root /home/undefined/Desktop/bci/code/tta_security/BrainUICL --contains rttdp_brainuicl_runs --limit 20 --output /tmp/raeeg-rttdp-20.json
```

迁移后的 `experiment_run` 负责历史结果归档和指标规范化；后续需要模型编译时，再从同一个实验的 checkpoint/config 生成 `model_pipeline` manifest。历史结果本身不能伪造 compile、runtime 或硬件 benchmark。
