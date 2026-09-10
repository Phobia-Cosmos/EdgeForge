# Data-induced synthetic EEG LoP result

本实验只改变派生 synthetic 数据的 target 标签条件关系，不冻结模型、不缩放梯度、不修改 ISRUC 原始数据。它用于寻找可控的数据诱导机制，不能作为真实 EEG 的科学结论。

## 成功方法：target label reversal

Source 数据的标签由 feature-0 的正极性决定；所有 target subject 的标签改为 feature-0 的负极性。波形本身保持不变，target 仍是 EEG-shaped `(8, 3000)` 数据。实验使用 3 个 seed（4321、4322、4323）、8 个连续 target transition 和严格 fresh-gap gate。

| adaptation budget | 平均 fresh-gap | transition 方向 | gate |
|---:|---:|---:|---|
| 5 | +0.517 | 8/8 positive | candidate |
| 10 | +0.535 | 8/8 positive | candidate |
| 25 | +0.484 | 8/8 positive | candidate |
| 50 | +0.190（后段转负） | mixed/negative | blocked-inconsistent-direction |

## 解释

在有限预算 5–25 时，warm 模型携带 source 的标签极性先验，target 数据要求相反决策边界，因此 fresh 模型更快适应，形成稳定正 fresh-gap。预算增加到 50 后，warm 模型也逐步学会反转关系，LoP 优势消失。这是“持续先验冲突 + 有限适应预算”诱导 LoP 的证据。

## 与其他数据 shift 的比较

- 单个强 anti-label shortcut：前几个 transition 可产生正 gap，但后续会恢复，未通过全程 gate。
- 旋转 anti-label shortcut：transition 方向混合，未通过 gate。
- target label reversal：budget 5/10/25 均通过 gate，是目前最稳定的数据诱导方案。

该结果说明：要让数据本身诱导 LoP，需要让 target 的标签语义与 source 学到的判别先验系统性冲突，并把观察限制在模型尚未完成重学习的有限适应窗口内。不能把这种人为标签变换直接解释为 ISRUC 中的自然 LoP。
