# EdgeForge V8：Model Registry 与 Capability Gate

## 目标

V8 将 V7 的实验事实转换成可审计的模型发布判断：模型先以 candidate 注册，只有同一 workload、protocol 与 comparison group 下的固定策略可以评估它；Gate 只产生 accepted 或 rejected，不会自动进入 production。生产切换与回滚是显式 Operator 动作，Agent、Worker 和实验完成事件均无权直接发布。

```text
ExperimentRun + immutable metrics
              │
              ▼
      Model(candidate)
              │
              ├── scope mismatch ──► reject evaluation request
              │
              ▼
   immutable GatePolicy
              │
              ▼
 immutable GateEvaluation
       │               │
     PASS             FAIL
       │               │
   accepted         rejected
       │
       └── explicit operator promote ──► production
                                             │
                       explicit switch/rollback
                                             ▼
                                        rolled_back
```

## Registry 与证据身份

`models` 保存模型名称、workload、protocol、comparison group、来源 experiment、可选 checkpoint SHA-256、描述符、状态和注册版本。V8 不假设 checkpoint 已上传：V7 Catalog 中的 checkpoint component digest 可以放入 descriptor，只有真正存在单一受控 checkpoint blob 时才填写 `checkpoint_digest`。

注册时控制面读取来源 `experiment_runs`，并要求三元 scope 完全一致。缺失 comparison group 的旧实验不能直接注册；应先建立有明确协议身份的新实验，而不是由调用方猜测分组。当前状态机为：

- `candidate → accepted`：所有 Gate rule 通过；
- `candidate → rejected`：任一 rule 失败，或 Operator 在评估前显式取消；
- `accepted → production`：Operator 提供原因并显式 promote；
- `production → rolled_back`：另一个已通过 Gate 的同 scope 模型被 promote，或执行显式 rollback；
- `rolled_back → production`：作为已通过 Gate 的历史版本被显式恢复。

同一个 scope 由 SQLite partial unique index 保证最多只有一个 production 模型。

## 不可变 Policy 与 Evaluation

`gate_policies` 只允许创建，不提供更新接口；重复 id 会被拒绝。每条规则包含 metric、比较运算符、有限数值 threshold、可选 namespace 和 step。首版支持 `>=`、`>`、`<=`、`<`：

```json
{
  "metric": "summary.final_old_acc",
  "operator": ">=",
  "threshold": 0.7,
  "step": null
}
```

`summary.<path>` 从 ExperimentRun 的摘要解析；其他 metric 名称从该 run 对应的 `experiment_metrics` 查询。时序指标必须用 namespace/step 消除歧义，并且最终只能解析为一个值。Gate 不做“挑选最好 step”或跨 context 聚合，因为这种隐式自由度会改变策略语义。

`gate_evaluations` 每次插入新记录，保存 policy snapshot、metric snapshot、逐规则 actual/threshold/PASS 结果、来源 experiment id 和实际 experiment task id。后续策略或实验记录不会改写已有判断；系统也不提供 evaluation 更新、删除或重新置为 candidate 的接口。

## RA-EEG aligned 策略

`config/raeeg-aligned-gate-v1.json` 只适用于 `raeeg / eeg-cl-v1-aligned / aligned-full49`：

- `summary.final_old_acc >= 0.70`；
- `summary.final_seen_acc >= 0.64`；
- `summary.bwt_acc >= -0.02`。

这些阈值是 V8 基础设施验收策略，不是新的科研结论，也不应被复用到 SPR/PuriDivER 的 method-transfer 协议或 LoP post-hoc smoke。阈值若需修改，必须创建新 policy id/version，不能覆盖 v1。

## API、CLI 与权限边界

Model、Policy 和 Evaluation 分别通过 `/api/v1/models`、`/api/v1/gate-policies`、`/api/v1/gate-evaluations` 查询或创建。模型状态动作使用 `/api/v1/models/{id}/promote|reject|rollback`；每次人工动作必须提供非空 reason，并写入不可覆盖的版本事件。

CLI 提供 `models`、`model-register`、`gate-policies`、`gate-policy-put`、`gate-evaluate`、`gate-evaluations`、`model-promote`、`model-reject` 和 `model-rollback`。当前共享 Bearer Token 只能证明控制面调用权限，还不是完整的个人身份与审批系统；生产环境下一步应将 Operator 身份接入独立认证和双人审批。即使如此，V8 已保证 Worker 完成实验、Gate PASS 或未来 Optimization Agent 都不会自动执行 promote。

## 当前边界与 V9 入口

V8 Gate 判断的是既有实验提供的研究能力证据，不编译模型，也不测量模型级 latency、显存或 graph break。V9 将把已 accepted 的 checkpoint 送入 4070S `PyTorch eager → torch.compile` Pipeline，并在 Compiler Artifact 上再次执行数值正确性、能力回归和性能门禁。任何跨 protocol 排名、LoP smoke 科研解释、或仅因速度更快而绕过能力 Gate 的行为仍被禁止。
