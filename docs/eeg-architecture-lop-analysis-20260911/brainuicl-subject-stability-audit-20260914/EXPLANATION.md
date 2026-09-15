# 图表说明

`brainuicl-stage-pca.png` 每个点是一个 sequence 的 20-epoch 中位 embedding，不是单个 epoch；颜色是 subject。横向 spread 表示不同个体/sequence 的表征差异，纵向同色散开表示被试内状态或采集漂移。PCA 只用于可视化，不是训练目标。

`brainuicl-stage-stability.png` 左图比较 epoch token 与 sequence median 的身份 probe。sequence median 更接近个体画像，epoch probe 更容易受睡眠阶段影响。右图是标准化空间中的 between-subject centroid distance / within-subject radius；大于 1 才表示被试间中心差异超过被试内波动。

本次是 random-init architecture audit。若某层准确率较高，只说明随机卷积/Transformer 仍保留了输入幅度、频谱和通道结构，不能说模型已经学会个体身份。训练 checkpoint 到位后必须重跑并报告差值。
