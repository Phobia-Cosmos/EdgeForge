# Deployment Target Audit

`target-audit` 是一个本地只读的发布矩阵门禁。它把三个证据面分开检查：模型 pipeline manifest 声明的 Backend/Target、Worker `target-probe` 显式广告的能力，以及 `model_runs` 中同一 manifest 在目标上的 correctness 结果。

```sh
PYTHONPATH=src python3 -m edgeforge target-probe --output target-probe.json
PYTHONPATH=src python3 -m edgeforge target-audit \
  --manifest config/model-pipeline-synthetic.json \
  --probe target-probe.json \
  --model-runs model-runs.json \
  --output deployment-audit.json
```

`--model-runs` 可以是控制面 `GET /api/v1/model-runs` 返回的对象，也可以是直接的数组。没有该文件或没有匹配的成功 correctness 运行时，审计保持 `status=blocked`；这是部署证据尚未完成，而不是模型失败的科研结论。

自动化发布系统也可以调用 `POST /api/v1/deployment-audits`，请求体使用 `{"manifest": {...}, "probe": {...}, "model_runs": [...]}`。省略 `model_runs` 时，控制面只读取同名模型的已记录运行；审计报告本身不写入数据库，也不会上传原始 EEG。

审计要求：

- manifest 的 `target.architecture` 必须存在并且与 probe 的 `capabilities.architecture` 一致；
- `compiler.backend` 必须出现在 `backend_claims.advertised`，只出现在 `inferred` 时仍为 blocked；
- Backend 所需 accelerator 必须出现在 probe 的真实 `capabilities.accelerators`；
- probe 必须带有与内容一致的 `probe_digest`；
- 至少一个模型运行必须同时匹配 manifest digest、模型/数据集、Backend、目标架构，且 `task_status=succeeded`、`correctness=true`。

SoC compatible string、驱动模块、Vulkan ICD、可执行文件存在性和 `backend_claims.inferred` 都不会单独解锁部署。Backend readiness 仍需管理员显式配置并由真实 Runtime correctness 产生广告；本审计不下载 LLVM/IREE，不连接开发板，也不生成性能、因果或 LoP 结论。
