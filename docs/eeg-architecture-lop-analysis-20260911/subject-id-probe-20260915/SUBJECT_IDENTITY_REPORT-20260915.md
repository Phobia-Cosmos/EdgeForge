# EEG subject-id 身份可分性实验（2026-09-15）

## 研究问题与结论

本轮实验只把 `subject_id` 作为目标，不读取或训练睡眠阶段、情绪等数据集标签。我们询问的是：对一个已经在 enrollment 中出现过的被试，能否根据另一段完整 EEG sequence/trial 判断其来自哪个被试；同时比较原始波形、BrainUICL 前端分支、fusion、Transformer 各层、分类器输入和 logits 的身份信息保留量。

结论是：两套数据都存在稳定的跨 sequence/trial 个体指纹，而且当前 BrainUICL 前端比直接下采样波形更容易被线性 probe 识别。ISRUC 的 EEG branch 身份准确率为 0.7223（机会水平 0.0102），FACED 的前端/fusion 为 0.8313（机会水平 0.0081）。Transformer 深度增加后身份可分性逐步下降，最终 logits 接近机会水平但仍高于机会水平。这个结果证明“表征中存在 subject 信息”，不等于未知个体识别、医学身份确认或 LoP 已经发生。

## 严格实验协议

每个 ISRUC `.npy` 文件（20 个 30 s epoch）或 FACED `.npy` 文件（20 个 trial epoch）作为一个不可拆分的 sequence/trial 观察。对每个 subject 的完整 group 做确定性随机一半划分：一半 enrollment/train，另一半 test；因此同一 sequence 内没有 epoch-level 泄漏。probe 是 `StandardScaler → PCA(128) → multinomial LogisticRegression`，PCA 只用于让 98/123 类的线性 probe 在全量数据上可重复运行；原始表示维度在表中仍按真实维度记录。报告 top-1、balanced top-1、top-5 和机会水平 `1/N_subjects`。

原始波形表示：每个 epoch 在时间轴平均池化到 128 点，再对 20 个 epoch 取 median；同时提供逐 epoch RMS 增益归一化版本。学习到的表示：使用真实 seed=4321 pretrain checkpoint，提取每个 sequence 的 20 个 token，并对 token 取 median。ISRUC 的 `eeg_branch` 是 6 个 EEG 通道分支，`eog_branch` 是前 2 个 EOG 通道分支；FACED 的当前模型是 32 通道共享 `features` 单分支，因此 `eeg_branch`、`eog_branch` 和 `fusion` 在本实验中是同一个共享前端 tap，不能解释为独立 EOG 生理分支。

## 全量结果

### ISRUC

数据包含 98 个 subject、4,276 个完整 sequence（共 85,520 个 epoch）；每个 subject 的机会水平为 1/98 = 0.0102。

| representation | top-1 | balanced top-1 | top-5 | feature dim |
| --- | ---: | ---: | ---: | ---: |
| raw_waveform | 0.0347 | 0.0331 | 0.1275 | 1,024 |
| raw_waveform_gain_normalized | 0.0189 | 0.0190 | 0.0772 | 1,024 |
| eeg_branch | **0.7223** | 0.7209 | 0.9210 | 512 |
| eog_branch | 0.4806 | 0.4802 | 0.7699 | 512 |
| fusion | 0.5698 | 0.5694 | 0.8512 | 512 |
| transformer_layer1 | 0.5129 | 0.5123 | 0.8230 | 512 |
| transformer_layer2 | 0.4570 | 0.4562 | 0.7884 | 512 |
| transformer_layer3 | 0.4261 | 0.4253 | 0.7703 | 512 |
| classifier_input | 0.2657 | 0.2624 | 0.5841 | 128 |
| logits | 0.0471 | 0.0446 | 0.1673 | 20 |

### FACED

当前 processed 副本包含 123 个 subject、984 个完整 trial（每个 subject 最多 8 个 trial）；每个 subject 的机会水平为 1/123 = 0.0081。

| representation | top-1 | balanced top-1 | top-5 | feature dim |
| --- | ---: | ---: | ---: | ---: |
| raw_waveform | 0.0407 | 0.0407 | 0.1037 | 4,096 |
| raw_waveform_gain_normalized | 0.0427 | 0.0427 | 0.1057 | 4,096 |
| eeg_branch (= shared frontend) | **0.8313** | 0.8313 | 0.9533 | 512 |
| eog_branch (= shared frontend) | **0.8313** | 0.8313 | 0.9533 | 512 |
| fusion (= shared frontend) | **0.8313** | 0.8313 | 0.9533 | 512 |
| transformer_layer1 | 0.5772 | 0.5772 | 0.8313 | 512 |
| transformer_layer2 | 0.4776 | 0.4776 | 0.7480 | 512 |
| transformer_layer3 | 0.4268 | 0.4268 | 0.6931 | 512 |
| classifier_input | 0.2480 | 0.2480 | 0.5488 | 128 |
| logits | 0.0264 | 0.0264 | 0.0833 | 20 |

## 如何解读

1. **输入确实含有身份线索。** 原始波形的简单线性 probe 仅为 ISRUC 3.47% 和 FACED 4.07%，但都显著高于各自机会水平；FACED 的旧版手工波形/频谱/连接特征 probe（独立审计）曾达到 98.61%，说明幅值、频谱、通道关系或采集条件中包含很强的个体/批次信息。增益归一化使 ISRUC 原始 probe 降至 1.89%，提示部分信号来自幅值尺度；FACED 则没有下降，需进一步做参考和 session 校正。
2. **BrainUICL 前端保留或放大身份可分性。** ISRUC EEG branch 0.7223、FACED shared frontend 0.8313，远高于原始下采样波形。该现象可能来自稳定的通道拓扑、生理差异，也可能混合电极增益、参考、采集批次和任务状态，不能直接做生理归因。
3. **后层逐步压低身份信息。** Transformer layer1→3、classifier input 和最终 logits 的 top-1 下降，说明任务头的低维瓶颈和时序混合丢弃了一部分身份信息。logits 仍高于机会水平，因此“任务输出不含身份”尚未得到严格证明。
4. **这不是 LoP 证据。** 身份可分性是数据/表征机制变量；LoP 仍需 fresh-vs-warm、固定更新预算、至少 3 个 seed、旧任务 retention、AULC/fresh gap 等持续学习结果。

## 局限与下一步

- 当前划分是 group-disjoint，不是 session-disjoint。ISRUC 主要是每人一晚记录，FACED processed 副本也没有可靠 session 元数据；因此要发布结论前必须补充按 recording/session 的划分，或从原始元数据构造 session。
- 睡眠/情绪标签虽然没有输入 probe，但不同 subject 的 class 分布可能成为隐式身份代理。下一轮应按 task class 做 class-balanced enrollment/test，并加入标签置换、冻结随机网络和仅统计特征基线。
- 闭集分类不能回答“完全未见过的新 subject 是谁”。对于真实身份系统，应增加 verification（与 enrollment embedding 的距离、ROC/AUROC、EER）和 open-set identification（未知拒识率）。
- 至少用 seed=4321、1234、2026 三个 probe split/模型 seed 报均值和 95% CI；当前表是 seed=4321 的一次可复现实验。

## 产物与复现

- 运行脚本：[subject_identity_probe.py](../../../scripts/subject_identity_probe.py)
- ISRUC 数值、混淆矩阵和 PCA：[ISRUC/](ISRUC/)
- FACED 数值、混淆矩阵和 PCA：[FACED/](FACED/)
- 两个目录中的 `subject-id-summary.json` 保存了数据根目录、checkpoint、split manifest、阶段维度和所有指标；`REPORT.md` 是自动生成的简表。
- 原始 EEG、BDF 和完整 checkpoint 没有复制进 EdgeForge；结果目录约 12 MB，可安全纳入版本控制。

复现示例（使用共享 `brainuicl` 环境）：

```bash
PYTHONPATH=/home/undefined/Desktop/bci/code/tta_security/BrainUICL \
/home/undefined/Disk/python-envs/brainuicl/bin/python scripts/subject_identity_probe.py \
  --dataset ISRUC \
  --data-root /home/undefined/Disk/datasets/brainuicl/processed/isruc_group1_npy_float32 \
  --brainuicl-root /home/undefined/Desktop/bci/code/tta_security/BrainUICL \
  --checkpoint-root /home/undefined/Disk/ai-storage/BrainUICL/model_parameter/ISRUC/Pretrain \
  --checkpoint-seed 4321 --batch-size 16 --gpu 0 --seed 4321 --probe-dim 128 \
  --plot-max-sequences 5000 \
  --output docs/eeg-architecture-lop-analysis-20260911/subject-id-probe-20260915/ISRUC
```

FACED 只需将 `--dataset`、`--data-root` 和 `--checkpoint-root` 替换为本报告中 `subject-id-summary.json` 记录的值。
