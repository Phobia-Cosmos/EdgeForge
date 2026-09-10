# 0.14.0 本地 RA-EEG LoP 结果记录

日期：2026-08-23。本文只记录本机单 GPU 的可复现实验，不把单 seed 结果升级为 LoP 科学结论。原始数据和 BrainUICL checkpoint 均留在共享存储，EdgeForge 只写派生 bundle、catalog、audit、report 和日志。

## 已完成的运行

| 矩阵 | cell 数 | 数据/方法 | 证据状态 |
| --- | ---: | --- | --- |
| `isruc-lop-v014-methods-local` | 20 | ISRUC subject 2，finetune/EWC/online-EWC/SI/MAS，stage 0/10/25/49，clean | 每种方法 3 个 lagged pairs；`insufficient-seeds`（只有 seed 4321） |
| `faced-lop-v014-local-smoke` | 2 | FACED subject 1，pretrain 与 mini-CL stage 1 | `insufficient-pairs`（只有 1 个 transition） |
| `isruc-lop-v014-noise-s0p5-local` | 8 | ISRUC subject 2，finetune，clean/noise-0.5，各 stage 0/10/25/49 | 每个 condition 3 个 lagged pairs；`insufficient-seeds` |
| `isruc-lop-v014-noise-s0p5-local` + `--with-unlabeled-diagnostics` | 8 | 同一 clean/noise matrix，同时采集 entropy/confidence/consistency | predictor/outcome/diagnostic 三类 role 分离；仍为单 seed |

## ISRUC 五方法的描述性轨迹

以下数值来自 `methods-local-v2/matrix-report.md`；`fresh_gap = fresh_accuracy − checkpoint_accuracy`，所以正值表示 fresh control 在固定预算后更高，负值表示 checkpoint 起点更有利。`relative parameter update` 是 checkpoint 与 pretrain 的参数状态差，不是 optimizer 的逐步更新量。

| method | ER stage 0 → 10 → 25 → 49 | fresh gap stage 0 → 10 → 25 → 49 | relative update stage 49 |
| --- | --- | --- | ---: |
| finetune | 8.3196 → 8.2921 → 8.2466 → 8.4136 | 0 → −0.025 → −0.075 → −0.175 | 0.02754 |
| EWC | 8.3196 → 8.2754 → 8.1889 → 8.0939 | 0.025 → 0 → 0 → 0 | 0.05494 |
| online-EWC | 8.3196 → 8.2784 → 8.1871 → 8.0495 | 0 → 0 → 0 → 0 | 0.04829 |
| SI | 8.3196 → 8.3334 → 8.3284 → 8.3258 | 0 → 0 → 0 → −0.05 | 0.00552 |
| MAS | 8.3196 → 8.3239 → 8.3186 → 8.3193 | 0 → 0 → 0 → −0.025 | 0.00318 |

在这个受限 protocol 中，没有方法在后期出现稳定的正 `fresh_gap`；因此不能说已经观察到固定预算 LoP。ER、drift、parameter update 的变化说明方法确实产生了不同的状态轨迹，但它们是 predictor/diagnostic，不是 outcome 的替代品。EWC/online-EWC 的 ER 下降和 SI/MAS 的 ER 稳定也不能单独解释为塑性损失。

## 数据 shift 的初步观察

`isruc-noise-s0p5-local` 使用同一 subject、同一 checkpoint、同一 probe budget，只把输入替换成 label-preserving 的相对通道噪声 0.5。clean 的 ER 为 `8.3196/8.2921/8.2466/8.4136`，noise 的 ER 为 `8.6002/8.5407/8.4297/8.5913`（stage 0/10/25/49）；noise 下 `acc_gain` 也更高，但这可能只是标签/分割/难度改变后的 supervised-oracle 适应增益。它证明“数据变化会改变表示谱和适应曲线”，不证明“数据变化必然造成 LoP”。

自动生成的 paired report 给出 noise-minus-clean ER delta `+0.28063/+0.24858/+0.18304/+0.17776`，以及 fresh-gap delta `0/−0.025/0/+0.075`（stage 0/10/25/49）。这些是同一 checkpoint 的输入条件差异；参数 relative update 的条件 delta 为 0，因为两条件复用了同一 checkpoint，进一步说明输入 shift 与参数更新是两个独立因素。

要把 shift 与 LoP 区分开，正式矩阵必须同时包含 clean、shift、fresh initialization、equal-budget probe、相同 subject order 和至少 3 个真实 seed；还要检查 old-task/generalization、label integrity、class balance 和 calibration。若只看到 clean-to-shift accuracy 下降，应称为 domain-shift degradation，不能直接称为 LoP。

## 当前结论与下一步

当前系统状态是“本地证据链已跑通，科学证据仍不足”：ISRUC 的多方法 stage 轨迹、FACED 的输入契约、受控 noise manifest、参数更新范数、trajectory audit 和版本归档均可复现；尚没有 FACED 正式多 seed 连续 CL，也没有 ISRUC/FACED 的 ≥3 seed LoP 结论。下一步应优先补 seed 4322/4323 的真实连续 checkpoint，并以同一 matrix manifest 重跑 clean/shift；在此之前不扩大攻击、防御或编译后端研究范围。

多 seed preflight 已把这一缺口具体化：ISRUC finetune subject 2 的 stage 0/10/25/49 在 seed 4321 为 ready，而 seed 4322/4323 的 8 个 cell 因三份模型参数文件不存在而 blocked。系统没有复制 seed 4321，也没有把 blocked cell 写入科学分析。

无标签诊断 smoke 还显示 ISRUC subject 2 pretrain 的 normalized predictive entropy `0.50197`、mean confidence `0.71406` 和 noise-0.05 prediction agreement `1.0`。这些量说明模型在该小批量上的预测稳定性，但没有经过参数更新，也没有提供 LoP outcome；正式在线适应实验必须把它们与离线标签评估分开保存。

FACED subject 1 pretrain 的同协议 smoke 为 normalized entropy `0.30097`、mean confidence `0.75749`、prediction agreement `0.85`。ISRUC 与 FACED 的数值不应直接横向比较为“哪个更容易 LoP”，因为类别数、采样率、输入通道和数据语义不同；它们首先证明两个数据集都能进入同一无标签诊断契约。

clean/noise 0.5 的无标签矩阵中，noise-minus-clean normalized entropy delta 为 `−0.01289/+0.00029/+0.00001/+0.00637`，prediction-consistency delta 为 0（stage 0/10/25/49）。这说明在该小预算下 confidence/consistency 对这个 noise shift 不一定敏感，不能用单一无标签指标替代 LoP outcome。
