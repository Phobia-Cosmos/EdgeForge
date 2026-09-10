# RA-EEG 论文主张评估（2026-08-20）

本报告把 `/home/undefined/Desktop/IPhone/论文/10_持续学习与适应安全` 中与 LoP、遗忘、replay、谱退化和投毒/防御相关的主张，与 BrainUICL 的 ISRUC、FACED 和 EdgeForge LoP 结果对齐。证据文件由 `scripts/analyze-paper-claims-gaeeg.py` 只读生成；BrainUICL 源码、数据和 checkpoint 均未复制或修改。

## 结论先行

EEG 上已经有三类可复现信号：持续学习确实存在旧任务性能下降；replay/正则化会改变 stability-plasticity 权衡；任务顺序、代理扰动和监控防御会显著改变结果。LoP probe 也观察到未来适应增益与 Transformer effective rank 的负相关，但目前只有 seed 4321，且 probe 是 supervised-oracle fixed-budget heldout，因此只能作为机制线索，不能作为统计或因果结论。

论文中关于 Atari、视觉或 RL 的绝对结论不能直接迁移到 EEG。当前最可靠的表述是“在给定 RA-EEG 协议下观察到相容现象”，而不是“已经证明 EEG 具有同样机制”。

## 主张与现有证据

| 论文主张 | EEG 证据 | 当前判断 |
| --- | --- | --- |
| 持续学习会同时出现遗忘和可塑性下降 | ISRUC/FACED clean、aligned/full49 与正则化结果包含 `old_fr`、`bwt_acc`、`final_old_acc`；例如 ISRUC `persist_eeg_probe_v2` Plain ER 的 `old_fr=0.1181`、`bwt_acc=-0.0194` | 支持“存在遗忘/稳定性损失”；尚未证明是单一 LoP 机制 |
| Replay 能缓解遗忘但可能牺牲当前适应 | `persist_eeg_probe_v2` Plain ER：`mean_current_acc_gain=+0.0115`、`old_fr=0.1181`；order/persistence 矩阵显示不同 schedule 差异明显 | 支持协议内 trade-off；不能跨数据集排名 |
| EWC/SI/MAS 等重要性正则化保持旧知识 | FACED clean summary 中 SI 的 `old_fr=0.0173`、`bwt_acc≈+0.0001`；ISRUC T2T clean 中 SI `old_fr=0.0109`、`bwt_acc=-0.0014` | 支持“某些协议下遗忘较小”；需要统一 aligned protocol 和多 seed |
| 谱塌缩/低有效秩与 LoP 相关 | EdgeForge expanded LoP：stage 0/10/25 rank `26.666/25.962/13.107`，对应 ACC gain `-0.01/+0.09/+0.23`，Pearson `-0.9281` | 机制线索；单 seed、3 stages，且 smoke 与 expanded 相关性方向不同，不能作因果结论 |
| probe budget/样本量会影响 LoP 观察 | cap=4 smoke 的 Pearson `0.5705`、mean gain `0.0`；cap=16/full-eval 的 Pearson `-0.9281`、mean gain `0.1033` | 明确支持“评测敏感性”；后续必须固定预算并做多 seed |
| 投毒可使 continual learner 遗忘 | `attack_smoke_brainwash1` 与 `attack_smoke_pacol1` 保存了 BrainWash/PACOL 风格 smoke；`proxy_dual_harm_*` 在 proxy stream 下 old accuracy/BWT 下降 | 仅支持“代理攻击管线可运行且会改变指标”；不是原论文攻击的严格复现 |
| 防御/监控可以拒绝有害更新 | `icml2026_t2t_clean49...` 记录 T2T `detected_pairs`、`rejected_updates`；robust-feature 记录 protected fraction/lambda/defense loss | 证明监控信号被记录并影响更新；尚无攻击-防御曲线或统计显著性 |
| 任务顺序会决定结果 | `persist_eeg_order_matrix_v1` 覆盖 uniform/stratified/late random、ISRUC/FACED、EWC/Plain ER，含 3 seeds；ISRUC uniform EWC final-old-ACC mean change `-11.06 pp`，late random EWC `-15.51 pp` | 支持顺序效应；这些是 proxy/persistence 协议，不应和 clean baseline 混合 |
| Self-Purified Replay 对 EEG 一定安全 | SPR/PuriDivER 结果存在，但当前实验为 EEG 迁移/反馈协议，且已有笔记指出 embedding hubness、少数类和 buffer 污染风险 | 不支持“安全保证”；需要标签噪声、输入投毒、buffer contamination 的独立实验 |

## 重要协议边界

- `aligned-full49`、`method-transfer`、`historical-unclassified`、`raeeg-lop-posthoc-v1` 必须分开统计。
- clean baseline、proxy attack、natural LoP 和 supervised-oracle probe 不是同一种实验。
- 当前 LoP 可用 checkpoint 只有 seed 4321；没有 seed 4322/4323 的对应 milestone checkpoint，因此多 seed LoP gate 为 `blocked-by-checkpoint`。
- `old_fr`、BWT、ACC gain 的数值只能在相同数据集、任务顺序、训练预算、BN 设置和 seed 集合内比较。

## 对项目的调整

1. 将 LoP 作为 EdgeForge 的可观测性工作流：固定 stages、probe steps、loader budget、seed 和 manifest，并同时归档 rank、stable rank、sigma max、weight norm、gradient/activation 稀疏度。
2. 先补齐真实 seed 4322/4323 checkpoint，再做多 seed LoP；在此之前不发布 Pearson 或方法排名。
3. 把 poisoning 分为 label flip、input/proxy shift、PACOL/BrainWash 风格三条轨道，每条都提供 clean、attack、defense 三组配对 runs。
4. 对 replay/正则化使用统一 `aligned-full49` 协议，报告 old/new ACC、macro-F1、BWT、forgetting 和 current gain 的均值、标准差与置信区间。
5. 将 T2T/robust-feature 先定位为监控和防御原型，不宣称已经复现 2026 ICML 理论的最坏情形保证。

## 运行分析脚本

```bash
cd /home/undefined/Desktop/EdgeForge
python3 scripts/analyze-paper-claims-gaeeg.py
```

默认输出 `logs/paper-claims-gaeeg-20260820.json`。该文件只保存指标、协议和证据路径，不保存 EEG 样本或模型参数。
