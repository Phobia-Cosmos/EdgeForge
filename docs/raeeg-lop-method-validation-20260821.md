# RA-EEG Regularization/Replay LoP Evidence Validation — 2026-08-21

## 结论

EdgeForge 已完成正则化与 replay 历史结果的 LoP 证据审计，但当前数据不能证明或否定这些方法中存在 LoP。原因不是 plasticity 缺失，而是 672 个已发现结果文件全部缺少正式 predictor `task.spectra.transformer_1.effective_rank`。因此 `ER(t-1) → Plasticity(t)` 无法配对，所有结果必须保持 `missing-predictor`，不能根据普通 accuracy/plasticity 曲线反推 LoP。

## 验证范围

只读扫描根目录：`/home/undefined/Desktop/bci/code/tta_security/BrainUICL`。发现 `metrics.json`/`RESULTS.json` 共 672 个，临时 catalog 位于 `/tmp/edgeforge-raeeg-all-results-catalog.json`，未写入 EdgeForge 仓库，也未修改 BrainUICL、checkpoint 或原始 EEG。

执行命令：

```sh
PYTHONPATH=src python3 scripts/build-raeeg-catalog.py \
  --root /home/undefined/Desktop/bci/code/tta_security/BrainUICL \
  --output /tmp/edgeforge-raeeg-all-results-catalog.json

PYTHONPATH=src python3 -m edgeforge lop-audit \
  --catalog /tmp/edgeforge-raeeg-all-results-catalog.json \
  --summary
```

全量审计结果：`record_count=672`、`status_counts={"missing-predictor": 672}`。审计器对非 EdgeForge 顶层 `metrics` 列表采用 RA-EEG normalizer，并对无效 JSON/过大文件逐条记录，不会因一个历史文件中止全量验证。

## 方法覆盖

| 方法 | 结果数 | 有 plasticity outcome | 有 ER predictor | 当前状态 |
| --- | ---: | ---: | ---: | --- |
| EWC | 210 | 209 | 0 | `missing-predictor` |
| Plain ER | 95 | 95 | 0 | `missing-predictor` |
| MAS | 68 | 68 | 0 | `missing-predictor` |
| SI | 68 | 68 | 0 | `missing-predictor` |
| Online EWC | 65 | 65 | 0 | `missing-predictor` |
| Finetune | 61 | 61 | 0 | `missing-predictor` |
| Full SPR adapted | 31 | 31 | 0 | `missing-predictor` |
| Full PuriDivER adapted | 25 | 25 | 0 | `missing-predictor` |
| BrainUICL | 7 | 0 | 0 | `missing-predictor` |
| SPR-EEG | 7 | 0 | 0 | `missing-predictor` |
| PuriDivER-EEG | 7 | 0 | 0 | `missing-predictor` |
| SPR-ER | 4 | 4 | 0 | `missing-predictor` |
| PuriDivER-CRU | 4 | 4 | 0 | `missing-predictor` |
| PuriDivER memory-CE | 4 | 4 | 0 | `missing-predictor` |

另有 16 个 `historical_import` 汇总结果，没有可用于正式 LoP 的 predictor/outcome envelope。不同 protocol、comparison group 和 method 仍被分别分组，不能因都属于 replay 或正则化而混池计算。

## 探索性覆盖检查

Curated aligned-full49 catalog 的 EWC、Online-EWC、SI、MAS、Finetune 都记录了 `task.importance.mean`。显式把它设为自定义 predictor 后，每种方法形成 48 个相邻 task pair，但每种方法都只有 seed 4321，所以状态统一为 `insufficient-seeds`，输出的 `analysis_scope` 固定为 `exploratory-custom-association`，不是 LoP。

| 方法 | Pearson | Spearman | Pair | Seed | 状态 |
| --- | ---: | ---: | ---: | ---: | --- |
| EWC | -0.0414 | -0.0620 | 48 | 1 | `insufficient-seeds` |
| Online EWC | 0.0336 | -0.1096 | 48 | 1 | `insufficient-seeds` |
| SI | -0.0796 | -0.1457 | 48 | 1 | `insufficient-seeds` |
| MAS | 0.0031 | -0.0424 | 48 | 1 | `insufficient-seeds` |
| Finetune | 不可定义 | 不可定义 | 48 | 1 | `insufficient-seeds` |

这些数值只验证自定义 predictor 路径能够运行。它们不是 ER、没有 cluster bootstrap CI、没有多 seed，不能用于方法排名、LoP 判断或因果解释。

## 继续形成正式证据所需内容

每种目标方法需要在同一 dataset、protocol、comparison group、task order 与固定 probe budget 下重新运行至少 3 个真实 seed，并在每个 checkpoint stage 同时记录：

- `task.spectra.transformer_1.effective_rank`；
- `plasticity.acc_gain`，最好同时记录 MF1 gain/AULC；
- checkpoint stage、next subject 与完全一致的 context pairing grid；
- source/checkpoint digest、method hyperparameters、replay capacity/ratio 或正则化强度；
- 非自适应 baseline 和预注册的 stage/subject selection。

补齐后可以继续使用 `lop-audit` 做 readiness 检查，再通过 `lop-analyze` 生成持久化、内容寻址的正式描述性分析。无论结果是否显著，`scientific_conclusion_allowed` 仍保持 `false`，直至独立科研审阅。
