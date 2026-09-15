# BrainUICL 训练后分层稳定性图说明

本目录是 5 个监督训练 epoch 的 medium ISRUC pilot。训练只使用 source subjects；所有身份 probe 都按完整 sequence 留出。它不是完整 98 被试的正式 checkpoint 结果。

`brainuicl-stage-pca.png` 每个点是一条 sequence 的 20 个 epoch embedding 中位数，颜色是 subject。每个面板从左到右、从上到下对应 EEG branch、EOG branch、fusion、Transformer layer 1/2/3、classifier input 和 logits。相同颜色的点在不同 sequence 间越集中，表示该层个体画像越稳定；同色点散开表示个体内部状态/采集漂移。

`brainuicl-stage-stability.png` 左图比较 epoch token 和 sequence median 的身份 probe，右图是标准化表征空间的 between-subject centroid distance / within-subject radius。右图低于 1 表示被试内变化大于被试间中心差异，不能称为强个体不变性。

这次训练后，sequence-level probe 约为：EEG branch 0.25、EOG branch 0.23、fusion 0.23、Transformer layers 0.18/0.18/0.22、classifier input 0.12、logits 0.12；前后半 profile correlation 约 0.39–0.59，between/within 比值约 0.22–0.71。结论是睡眠任务训练会重塑并削弱部分个体统计稳定性；这些数值不能解释为认证性能或 LoP 证据。

详细数值、embedding 和训练历史位于分析归档 `docs/eeg-architecture-lop-analysis-20260911/brainuicl-subject-stability-trained-pilot-20260914/`。
