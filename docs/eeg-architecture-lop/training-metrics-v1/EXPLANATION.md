# EEG 训练指标可视化

本目录中的图来自 `rms-equalized-tcn-source-epochs10-diagnostics-v1` 的结构化诊断结果，共 240 行：8 个 target stage、3 个 seed、fresh/warm 两个 arm 和 5 个 adaptation budget。误差线是跨 stage-seed cell 的标准差。它不是 raw `continuous-medium-v0.18.0-r2` 的内部诊断图；因此这里的 fresh-gap 数值不能直接与 raw baseline 的最终 fresh-gap 混用。

## 先回答 warm/fresh 看起来“不变”的问题

它们并非完全不变。当前图画的是 24 个 stage-seed cell 的均值，而不是某一个 subject 的完整轨迹；跨个体差异和标准差很大，所以均值曲线看起来平滑。采样点也只有 adaptation budget `0/5/10/25/50`，没有记录每一个训练 step，因此不能从这张图读取连续的 stream-age 变化。

在这份诊断中，warm embedding effective rank 为 `3.091 → 2.909 → 3.269 → 3.084 → 2.989`，warm gradient nonzero fraction 为 `0.596 → 0.575 → 0.585 → 0.588 → 0.596`；它们没有单调坍缩。fresh embedding rank 为 `4.391 → 5.562 → 6.001 → 11.369 → 3.459`，实际上变化更大，只是跨 stage 的方差更宽。`classifier_input_near_zero_fraction` 在所有 cell 都是 `0`，这一个指标才是真正没有变化的。

因此，读图时应优先看 `stage-heatmaps.png` 或单个 stage/seed 轨迹，而不能只看总体均值；若要验证长期 LoP，还需要在每个 stream checkpoint 保存诊断，而不是只在五个 budget 终点采样。

## fresh 的角色不是防护算法

LoP 的主要研究对象是 warm 模型：它携带旧任务训练后的状态，随后在 target stream 上继续适应。fresh 模型是每个 target stage 重新初始化的反事实参照，用来计算 `fresh-gap = accuracy_fresh − accuracy_warm`。正 gap 表示在相同固定预算下 fresh 适应得更好，提示 warm 可能存在可塑性损失；它不是一种防护或正则化方法。真正的防护实验应另行比较 replay、正则化、ReDo、reset 或 adapter 等干预。

本目录中的 `fresh-gap-by-budget.png` 是 LoP 的结果对照图，不是研究目标本身。主分析应改为：`warm` 的目标适应、旧任务 retention、内部 rank/谱/激活/梯度状态，以及这些状态是否先于后续 warm 性能恶化；fresh 只保留为必要的反事实基线。

## 改变 EEG 数据后的图目前有什么

已经有三类改变数据后的结果：

- `condition-*/visualizations-*`：gain drift、噪声/SNR、baseline drift、channel crosstalk、polarity 等输入层的 RMS、PSD、PCA、subject distance 和 waveform 图；
- `condition-dose-curve-*/eeg-lop-dose-curves.png`：各扰动条件的 fresh-gap/性能层对照；
- `training-metrics-v1/`：仅针对 RMS-equalized TCN pilot 的 warm/fresh rank、梯度和参数更新诊断。

目前尚未为 gain drift、SNR、baseline drift、crosstalk 等每个条件重新采集完整的 warm 内部诊断，因此还没有可信的“改变数据后 rank/谱/梯度/激活变化图”。这不是数据没有变化，而是实验运行时没有打开 `--checkpoint-diagnostics`。后续比较必须对 raw 和每个扰动条件使用相同 subject、sequence、seed、budget 和 calibration manifest，再绘制 warm-only 内部指标，并把 fresh 作为参考线。

## 图 1：`training-metrics-by-budget.png`

六个面板分别显示 embedding effective rank、normalized effective rank、首个卷积块 effective rank、梯度 L2 范数、梯度非零比例和参数相对更新量。warm 是跨 stage 传递的 checkpoint，fresh 是每个 target stage 重新初始化的模型。梯度范数使用对数纵轴，因为不同 cell 的数值跨度较大。图中只能看预算相关的即时变化，不能把横轴 budget 当成长期 stream age；要判断论文意义上的“训练越久越失去可塑性”，还需要沿 stream age 保存连续 checkpoint。

当前最重要的读法是：warm embedding ER 没有单调下降，先降后升再回落；gradient nonzero fraction 约保持在 0.58–0.60；classifier-input near-zero fraction 在原始表格中所有 cell 都为 0（因此没有额外画成一条完全重合的零线）。这不支持“整体谱坍缩或全网络失活”的结论。

## 图 2：`fresh-gap-by-budget.png`

纵轴是 `fresh-gap = accuracy_fresh − accuracy_warm`，0 线以上表示 fresh 在固定预算后更好，0 线以下表示 warm 更好。误差线显示 24 个 stage-seed cell 的离散程度，文字标注是 fresh-gap 为正的 cell 比例。它是 LoP 的主要结果图，但均值为正仍不够；严格 gate 还要求跨 seed、transition 的方向一致并且不确定性下界为正。

## 图 3：`metric-vs-fresh-gap.png`

每个点是一个 warm stage-seed cell，颜色表示 adaptation budget。它用来检查 rank、梯度可达性或梯度范数是否与 fresh-gap 同步变化；散点混合或相关系数接近 0 时，不能把该指标称为 LoP 机制。当前 warm 初始 embedding ER 与最终 fresh-gap 的探索性相关约为 -0.112；本图的预算级 cell 相关如下：

- `embedding_effective_rank` vs fresh-gap：Pearson r = 0.036。
- `block1_effective_rank` vs fresh-gap：Pearson r = 0.028。
- `gradient_nonzero_fraction` vs fresh-gap：Pearson r = 0.135。
- `gradient_norm_l2` vs fresh-gap：Pearson r = 0.029。

## 图 4：`stage-heatmaps.png`

每个热图的纵轴是连续 target stage（同时标出 subject），横轴是 adaptation budget；每个格子是 3 个 seed 的均值。它用于发现被总体均值掩盖的局部异常，例如某个 subject 的 rank 突然降低或某一 stage 的 fresh-gap 方向相反。热图仍然是描述性结果，不能把单个异常格子直接解释为 LoP。

## 当前结果的结论

图中可以直观看到 rank、谱代理和梯度指标如何随 adaptation budget 变化，但它们不是独立的 LoP 证据。当前连续 medium 的 budget=50 最终 fresh-gap 均值约为 -0.008（TCN），且严格 gate 未通过；因此这些图支持“transfer sensitivity/预算依赖变化”的描述，不支持“ISRUC 已出现自然 LoP”。详细数值表见 [training-metrics-summary-20260912.md](../../eeg-architecture-lop-analysis-20260911/training-metrics-summary-20260912.md)。

注意：这里的 rank 是 calibration 表征矩阵的 effective rank，`sigma_max` 也只是激活表征的最大奇异值代理，不是完整参数奇异值谱。后续若要绘制真正的参数谱曲线，应在每个 checkpoint 保存完整 singular-value summary，并固定矩阵展平方式、层名和 calibration manifest。
