# Clean ISRUC medium: complete sequence waveform views

每个 sequence 文件包含 20 个按时间排序的 30 秒 epoch；E00--E09 是 adaptation，E10--E19 是 held-out evaluation。这里读取原有 float32 数组，不在绘图脚本中重新滤波或重采样。睡眠标签仅写作 class 0--4，避免在没有核对上游映射时擅自赋予阶段名称。

每个入选 sequence 有三张图：`waveform-stack` 使用同一个 sequence-wide 尺度叠放 channel 0 的 20 条完整波形，因此可以比较 epoch 间真实相对振幅；`all-channels` 用 8 个热图查看每个通道的 20×30 秒变化；`characteristic-epochs` 单独放大具有代表性的完整 epoch。颜色深浅不能跨不同图片直接比较，因为每张图片使用自己的 robust scale。

每个 sequence 还会生成 `characteristic-epochs` 图，自动放大低 RMS、中位 RMS、高 RMS、首次标签变化和最大相邻 RMS 跳变所在的完整 30 秒 epoch；如果多个规则选到同一个 epoch，图中会合并显示这些原因。通道在图中只做垂直错位，所有通道共用该 sequence 的 robust amplitude scale，因此可同时观察波形形状和相对振幅。

选择规则优先保留睡眠标签转换最多的 sequence 和 epoch RMS 变化最大的 sequence。图中的 low/median/high RMS 只表示该 sequence 内的相对振幅，不自动等于坏数据或某种睡眠阶段。

## Subject 2, sequence 45

选择原因：most label transitions。标签转换 11 次，epoch RMS 变异系数为 0.230；低/中位/高 RMS 代表 epoch 分别是 E17、E02、E07。

- `subject-2-sequence-45-waveform-stack.png`：从上到下按 E00--E19 阅读。每条曲线都是 channel 0 的完整 30 秒原始输入；曲线颜色是 class code。右侧 RMS 轨迹用于定位振幅突变，空心标记表示第一次标签改变。先比较相邻 epoch 的形状，再检查高 RMS 是否只出现在单个 epoch，最后比较 adaptation 与 evaluation 两半。
- `subject-2-sequence-45-all-channels.png`：每个子图对应一个通道，每一行对应一个 epoch。横向纹理表示 30 秒内部的波形变化，纵向连续纹理表示多个 epoch 的共同结构，孤立的深色行更可能是高振幅瞬态或伪迹。不同通道同时变化更像整体状态/增益变化，仅少数通道变化更像局部导联差异。
- `subject-2-sequence-45-characteristic-epochs.png`：每个面板是一个完整 30 秒 epoch，不是把多个 epoch 拼接起来；标题直接给出 E##、class、选择原因和 RMS 相对序列中位数。先看同一面板中 8 个通道是否同步出现突发/振荡，再比较不同面板的振幅与形状。它用于定位‘值得进一步分析的 epoch’，不能单独证明某个睡眠阶段或 LoP。

## Subject 2, sequence 14

选择原因：largest within-sequence RMS variation。标签转换 8 次，epoch RMS 变异系数为 0.593；低/中位/高 RMS 代表 epoch 分别是 E00、E07、E10。

- `subject-2-sequence-14-waveform-stack.png`：从上到下按 E00--E19 阅读。每条曲线都是 channel 0 的完整 30 秒原始输入；曲线颜色是 class code。右侧 RMS 轨迹用于定位振幅突变，空心标记表示第一次标签改变。先比较相邻 epoch 的形状，再检查高 RMS 是否只出现在单个 epoch，最后比较 adaptation 与 evaluation 两半。
- `subject-2-sequence-14-all-channels.png`：每个子图对应一个通道，每一行对应一个 epoch。横向纹理表示 30 秒内部的波形变化，纵向连续纹理表示多个 epoch 的共同结构，孤立的深色行更可能是高振幅瞬态或伪迹。不同通道同时变化更像整体状态/增益变化，仅少数通道变化更像局部导联差异。
- `subject-2-sequence-14-characteristic-epochs.png`：每个面板是一个完整 30 秒 epoch，不是把多个 epoch 拼接起来；标题直接给出 E##、class、选择原因和 RMS 相对序列中位数。先看同一面板中 8 个通道是否同步出现突发/振荡，再比较不同面板的振幅与形状。它用于定位‘值得进一步分析的 epoch’，不能单独证明某个睡眠阶段或 LoP。

## Subject 12, sequence 21

选择原因：most label transitions。标签转换 11 次，epoch RMS 变异系数为 0.540；低/中位/高 RMS 代表 epoch 分别是 E04、E11、E03。

- `subject-12-sequence-21-waveform-stack.png`：从上到下按 E00--E19 阅读。每条曲线都是 channel 0 的完整 30 秒原始输入；曲线颜色是 class code。右侧 RMS 轨迹用于定位振幅突变，空心标记表示第一次标签改变。先比较相邻 epoch 的形状，再检查高 RMS 是否只出现在单个 epoch，最后比较 adaptation 与 evaluation 两半。
- `subject-12-sequence-21-all-channels.png`：每个子图对应一个通道，每一行对应一个 epoch。横向纹理表示 30 秒内部的波形变化，纵向连续纹理表示多个 epoch 的共同结构，孤立的深色行更可能是高振幅瞬态或伪迹。不同通道同时变化更像整体状态/增益变化，仅少数通道变化更像局部导联差异。
- `subject-12-sequence-21-characteristic-epochs.png`：每个面板是一个完整 30 秒 epoch，不是把多个 epoch 拼接起来；标题直接给出 E##、class、选择原因和 RMS 相对序列中位数。先看同一面板中 8 个通道是否同步出现突发/振荡，再比较不同面板的振幅与形状。它用于定位‘值得进一步分析的 epoch’，不能单独证明某个睡眠阶段或 LoP。

## Subject 12, sequence 39

选择原因：largest within-sequence RMS variation。标签转换 7 次，epoch RMS 变异系数为 1.184；低/中位/高 RMS 代表 epoch 分别是 E01、E15、E18。

- `subject-12-sequence-39-waveform-stack.png`：从上到下按 E00--E19 阅读。每条曲线都是 channel 0 的完整 30 秒原始输入；曲线颜色是 class code。右侧 RMS 轨迹用于定位振幅突变，空心标记表示第一次标签改变。先比较相邻 epoch 的形状，再检查高 RMS 是否只出现在单个 epoch，最后比较 adaptation 与 evaluation 两半。
- `subject-12-sequence-39-all-channels.png`：每个子图对应一个通道，每一行对应一个 epoch。横向纹理表示 30 秒内部的波形变化，纵向连续纹理表示多个 epoch 的共同结构，孤立的深色行更可能是高振幅瞬态或伪迹。不同通道同时变化更像整体状态/增益变化，仅少数通道变化更像局部导联差异。
- `subject-12-sequence-39-characteristic-epochs.png`：每个面板是一个完整 30 秒 epoch，不是把多个 epoch 拼接起来；标题直接给出 E##、class、选择原因和 RMS 相对序列中位数。先看同一面板中 8 个通道是否同步出现突发/振荡，再比较不同面板的振幅与形状。它用于定位‘值得进一步分析的 epoch’，不能单独证明某个睡眠阶段或 LoP。

## Subject 14, sequence 21

选择原因：most label transitions。标签转换 12 次，epoch RMS 变异系数为 0.103；低/中位/高 RMS 代表 epoch 分别是 E09、E01、E17。

- `subject-14-sequence-21-waveform-stack.png`：从上到下按 E00--E19 阅读。每条曲线都是 channel 0 的完整 30 秒原始输入；曲线颜色是 class code。右侧 RMS 轨迹用于定位振幅突变，空心标记表示第一次标签改变。先比较相邻 epoch 的形状，再检查高 RMS 是否只出现在单个 epoch，最后比较 adaptation 与 evaluation 两半。
- `subject-14-sequence-21-all-channels.png`：每个子图对应一个通道，每一行对应一个 epoch。横向纹理表示 30 秒内部的波形变化，纵向连续纹理表示多个 epoch 的共同结构，孤立的深色行更可能是高振幅瞬态或伪迹。不同通道同时变化更像整体状态/增益变化，仅少数通道变化更像局部导联差异。
- `subject-14-sequence-21-characteristic-epochs.png`：每个面板是一个完整 30 秒 epoch，不是把多个 epoch 拼接起来；标题直接给出 E##、class、选择原因和 RMS 相对序列中位数。先看同一面板中 8 个通道是否同步出现突发/振荡，再比较不同面板的振幅与形状。它用于定位‘值得进一步分析的 epoch’，不能单独证明某个睡眠阶段或 LoP。

## Subject 14, sequence 30

选择原因：largest within-sequence RMS variation。标签转换 7 次，epoch RMS 变异系数为 0.448；低/中位/高 RMS 代表 epoch 分别是 E14、E05、E19。

- `subject-14-sequence-30-waveform-stack.png`：从上到下按 E00--E19 阅读。每条曲线都是 channel 0 的完整 30 秒原始输入；曲线颜色是 class code。右侧 RMS 轨迹用于定位振幅突变，空心标记表示第一次标签改变。先比较相邻 epoch 的形状，再检查高 RMS 是否只出现在单个 epoch，最后比较 adaptation 与 evaluation 两半。
- `subject-14-sequence-30-all-channels.png`：每个子图对应一个通道，每一行对应一个 epoch。横向纹理表示 30 秒内部的波形变化，纵向连续纹理表示多个 epoch 的共同结构，孤立的深色行更可能是高振幅瞬态或伪迹。不同通道同时变化更像整体状态/增益变化，仅少数通道变化更像局部导联差异。
- `subject-14-sequence-30-characteristic-epochs.png`：每个面板是一个完整 30 秒 epoch，不是把多个 epoch 拼接起来；标题直接给出 E##、class、选择原因和 RMS 相对序列中位数。先看同一面板中 8 个通道是否同步出现突发/振荡，再比较不同面板的振幅与形状。它用于定位‘值得进一步分析的 epoch’，不能单独证明某个睡眠阶段或 LoP。

## Subject 15, sequence 37

选择原因：most label transitions。标签转换 9 次，epoch RMS 变异系数为 0.584；低/中位/高 RMS 代表 epoch 分别是 E05、E06、E10。

- `subject-15-sequence-37-waveform-stack.png`：从上到下按 E00--E19 阅读。每条曲线都是 channel 0 的完整 30 秒原始输入；曲线颜色是 class code。右侧 RMS 轨迹用于定位振幅突变，空心标记表示第一次标签改变。先比较相邻 epoch 的形状，再检查高 RMS 是否只出现在单个 epoch，最后比较 adaptation 与 evaluation 两半。
- `subject-15-sequence-37-all-channels.png`：每个子图对应一个通道，每一行对应一个 epoch。横向纹理表示 30 秒内部的波形变化，纵向连续纹理表示多个 epoch 的共同结构，孤立的深色行更可能是高振幅瞬态或伪迹。不同通道同时变化更像整体状态/增益变化，仅少数通道变化更像局部导联差异。
- `subject-15-sequence-37-characteristic-epochs.png`：每个面板是一个完整 30 秒 epoch，不是把多个 epoch 拼接起来；标题直接给出 E##、class、选择原因和 RMS 相对序列中位数。先看同一面板中 8 个通道是否同步出现突发/振荡，再比较不同面板的振幅与形状。它用于定位‘值得进一步分析的 epoch’，不能单独证明某个睡眠阶段或 LoP。

## Subject 15, sequence 35

选择原因：largest within-sequence RMS variation。标签转换 6 次，epoch RMS 变异系数为 1.101；低/中位/高 RMS 代表 epoch 分别是 E15、E01、E09。

- `subject-15-sequence-35-waveform-stack.png`：从上到下按 E00--E19 阅读。每条曲线都是 channel 0 的完整 30 秒原始输入；曲线颜色是 class code。右侧 RMS 轨迹用于定位振幅突变，空心标记表示第一次标签改变。先比较相邻 epoch 的形状，再检查高 RMS 是否只出现在单个 epoch，最后比较 adaptation 与 evaluation 两半。
- `subject-15-sequence-35-all-channels.png`：每个子图对应一个通道，每一行对应一个 epoch。横向纹理表示 30 秒内部的波形变化，纵向连续纹理表示多个 epoch 的共同结构，孤立的深色行更可能是高振幅瞬态或伪迹。不同通道同时变化更像整体状态/增益变化，仅少数通道变化更像局部导联差异。
- `subject-15-sequence-35-characteristic-epochs.png`：每个面板是一个完整 30 秒 epoch，不是把多个 epoch 拼接起来；标题直接给出 E##、class、选择原因和 RMS 相对序列中位数。先看同一面板中 8 个通道是否同步出现突发/振荡，再比较不同面板的振幅与形状。它用于定位‘值得进一步分析的 epoch’，不能单独证明某个睡眠阶段或 LoP。
