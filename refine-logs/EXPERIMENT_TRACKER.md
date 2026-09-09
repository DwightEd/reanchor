# Constraint-Control Experiment Tracker

更新时间：2026-09-09 16:15:19（Asia/Shanghai）  
状态：计划阶段；尚未实现或运行实验。

| Run ID | Priority | Claim | Configuration | Status | Gate / Expected artifact |
|---|---|---|---|---|---|
| P001 | MUST | infrastructure | Llama; 20 QA/controlled sources; seeds 0,1,2 | planned | generation/replay token-logit fidelity report |
| P002 | MUST | Claim 1 validity | P001 targets; VJP/JVP vs finite difference | planned | signed-effect and cut-closure tests pass |
| D001 | MUST | Claim 1 discovery | Llama; natural QA; 100 sources × 3 seeds | planned | frozen lookback rate and R/B/U/O/L definitions |
| D002 | MUST | annotation validity | D001; verifier + stratified human audit | planned | span accuracy/agreement and uncertain bucket |
| C001 | MUST | Claim 1 causal | Llama; matched constraint pairs; top paths | planned | restore/remove + sham intervention table |
| C002 | MUST | Claim 1 confirmation | Llama; source-held-out QA confirmation | planned | preregistered mechanism effect with source CI |
| R001 | MUST | Claim 1 scope | Qwen3-8B non-thinking; frozen protocol | planned | independent-model replication matrix |
| R002 | MUST | Claim 1 scope | Gemma-2-9b-it; frozen protocol | planned | second-architecture-family replication matrix |
| T001 | MUST | Claim 1 scope | summarization transfer | planned | task-transfer effect matrix |
| T002 | MUST | Claim 1 scope | data-to-text transfer | planned | structured-source transfer effect matrix |
| H001 | MUST for Claim 2 | Claim 2 | deploy features vs 3 baseline families | planned | held-out AUPRC/calibration/lead-time table |
| H002 | MUST for Claim 2 | Claim 2 | zero-retrain cross-task/model | planned | transfer matrix and failure boundaries |
| A001 | NICE | both | feature deletion + negative controls | planned | ablation and null-control tables |
| F001 | MUST | reporting | all passed runs only | planned | claim/evidence ledger and reproducibility bundle |

## Decision log

- 2026-09-09：主图改为 answer-conditioned causal backbone；不再为所有生成 token 构建完整 DAG。
- 2026-09-09：lookback、causal reanchor、constraint-restoring reanchor 使用三级定义。
- 2026-09-09：机制 oracle 特征与部署检测特征强制分离。
- 2026-09-09：主张上限为“跨任务/架构反复出现的机制家族”，不预设普遍机制。

## Result-to-claim gate

- Claim 1 只有在 C001、C002、至少一个跨模型 run 和至少一个跨任务 run 通过后才可标记 supported。
- Claim 2 只有在 H001 的最强基线增量成立且 H002 至少一项迁移成立后才可标记 supported。
- 任一阶段失败都保留原始结果并收窄 claim，不通过追加筛选或改标签定义挽救结论。
