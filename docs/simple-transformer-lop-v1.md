# 简单 Transformer EEG LoP 诊断协议 v1

该协议用于验证 EdgeForge 是否能把与 EEG 网络结构一致的 LoP 指标接入编译器/运行时实验链路。模型输入仍是 ISRUC `(20,8,3000)`，前端保留时域统计和五个 EEG 频带，Transformer 只承担序列建模和逐 epoch 分类，因此诊断结果可与 BrainUICL 的输入契约对齐。

`fresh_gap = accuracy_fresh − accuracy_checkpoint`。正值只代表在相同 adaptation steps、学习率、数据划分和 seed 下 fresh 起点略高，属于 LoP candidate；负值表示 checkpoint 起点更有利。`retention_accuracy_drop = accuracy_initial − accuracy_final`，负值表示 retention accuracy 上升。effective rank 使用中心化 hidden matrix 的奇异值能量熵，stable rank 为 `||H||_F² / ||H||_2²`，attention entropy 是注意力分布熵；后两类是 representation/diagnostic 指标，不是 LoP outcome。

本协议目前是 supervised-oracle：target labels 参与 adaptation，因而不能回答无标签在线 TTA 下的 LoP。为了形成论文级证据，需要把同一 protocol 扩展到更长连续阶段、FACED、更多真实 seed，并采用按 seed 聚类的 bootstrap；同时固定 retention set、class balance、probe budget 和 stage order。任何输入 shift（噪声、通道丢失或跨被试）都必须与 fresh control、label integrity 和 calibration 一起报告，不能把 domain-shift degradation 直接命名为 LoP。

脚本入口为 `scripts/simple-transformer-lop.py`。它写出 `metadata.json`、每个 seed 的 `run.json`、`summary.json` 以及轻量级 checkpoint；运行输出存放在共享 AI storage，版本归档只保存 JSON 结果、命令、环境和 SHA-256 清单。
