# Constraint-Control Experiment Tracker

更新时间：2026-09-10（Asia/Shanghai）。用户报告 P001 单 source GPU smoke 为
`completed=3, planned=3, rejected=0`。这只通过 replay 基础设施 gate，不构成幻觉机制
结果。主线 transition mechanism audit 已实现，真实 contrast run 待执行。

| Run ID | Priority | Claim | Configuration | Status | Gate / expected artifact |
|---|---|---|---|---|---|
| P001 | MUST | infrastructure | Llama; controlled source; seeds 0,1,2 | one-source GPU smoke: 3/3 completed, 0 rejected; full run pending | replay fidelity only; no mechanism claim |
| P002 | MUST | candidate-route validity | frozen anchors; explicit onset/span contrasts | transition/source-unit audit implemented; real run pending | signed delta/current effects, hop ledger, pointwise closure |
| D001 | MUST | onset association | natural QA; source-disjoint discovery/confirmation | planned | matched onset/pseudo-onset offset curve |
| D002 | MUST | annotation validity | D001; verifier + stratified human audit | planned | span/onset accuracy, agreement, uncertain bucket |
| C001 | MUST | causal necessity/sufficiency | matched constraint pairs; selected paths + shams | planned | bidirectional exact restore/remove table |
| C002 | MUST | local mediation | onset carrier cut; paired free-run | planned | mediated effect and factual endpoint |
| R001 | MUST | scope | Qwen-family; frozen protocol | planned | independent-model replication matrix |
| R002 | MUST | scope | Gemma-family; frozen protocol | planned | second-family replication matrix |
| T001 | MUST | scope | summarization transfer | planned | task-transfer effect matrix |
| T002 | MUST | scope | data-to-text transfer | planned | structured-source transfer matrix |
| H001 | MUST for detector claim | detection | deployed features vs strong baselines | planned | held-out AUPRC, calibration, lead time |
| H002 | MUST for detector claim | transfer | zero-retrain cross-task/model | planned | transfer matrix and failure boundaries |
| A001 | NICE | mechanism | seed/intervention/negative-control ablations | planned | ablation and null-control tables |
| F001 | MUST | reporting | passed runs only | planned | claim/evidence ledger and reproducibility bundle |

## Decision log

- 2026-09-10：P002 改为同一事件内的 onset-to-rollout audit。主估计量是与 discovery
  对齐的 adjacent remote attention delta；当前 native remote write 单独报告。
- 2026-09-10：四个 coarse source group 仅作为 provenance partition；每个
  `source_unit_id` 单独传播，不再按 coarse group 命名幻觉机制。
- 2026-09-10：用户报告 P001 三个 seed 全部通过。这里只更新基础设施状态，不据此判断
  正常/幻觉机制。
- 2026-09-09：主图改为 answer-conditioned causal backbone，不为所有生成 token 构建
  完整 DAG。
- 2026-09-09：机制 oracle 特征与部署检测特征强制分离；不预设普遍机制。

## Result-to-claim gate

- Candidate-route claim：P002 的真实 held-out contrast run、matched onset controls 与
  source-level interval 通过。
- Causal mechanism-in-scope claim：C001、C002、至少一个跨模型和一个跨任务 run 通过。
- Detector claim：H001 超过强基线且 H002 至少一个迁移设置成立。
- 任一阶段失败都保留原始结果并收窄 claim，不通过追加筛选或改标签定义挽救结论。

## Verification

- 本地代码与测试状态以当前提交前 CI 结果为准；真实 8B P002 尚未运行。
