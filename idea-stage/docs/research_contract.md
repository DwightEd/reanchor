# Research Contract: 条件绑定支持模型 G0

## Selected Idea

实现 docs/graph_method_proposal.md 最终作者修订：冻结 LLM，source-only evidence states 与 source-blind address states 分离；候选条件化两层 source read，直接 softmax(f) 训练。先实现 G0，不引入未经必要性检验的 G1。

## Core Claims（待验证，不是结果）

1. 显式条件/指代两种正确历史角色的配对任务能教会条件绑定。
2. 这种支持模型能迁移到自然 response-first-error 检测。
3. 图必要性和原生生成器机制均未成立，本次不声称已实现。

## Method Summary

至少六条程序事实，目标与未在历史断言的 donor 交换绑定，保持历史真且不变。目标直接来自程序；不用真假伪标签。主损失为 binding CE + centered neutral energy variance。候选为 native top-32、source token IDs、观测 token 的并集。部署读出 max(f)-f(y)，用无标签/source-balanced 分层参考校准，不保证正常 FPR。

## Experiment Design

M0: 无下载的 CPU 单元/集成测试与 seeded tensor witness。
M1: 单卡冻结表示提取、程序绑定训练/dev 选择、隔离 test 配对准确率；默认小规模工程 pilot。
M2: RAGTruth 无标签输入适配、全 token 冻结评分、独立 labels join 的首错指标。
基线：native negative margin；source-null 为架构不变量测试。图比较、全自然泛化和原生路径干预不包含在这次最小实现里。
width=128, blocks=2, seed=42；超参数由 CLI 指定。GPU 小时待实测，本轮不远端执行。

## Current Results

尚无新方法真实 GPU 结果。干净 CPU 环境 30 项测试通过，独立代理按 RUNBOOK 原样执行也通过。
微型随机 Llama 的结果仅作软件测试，不是方法有效性证据。

## Key Decisions

新 CLI 与 scripts/run_g0.sh 为主入口，run_pilot.sh 转发新入口；旧 factorial 审计保留为 audit 子命令与 run_audit.sh。所有输出拒绝覆盖，不使用未来 token，不读取 H 标签选模。schema、seed、source splits、backbone/checkpoint 指纹与运行配置写入结果。

## Status

- [x] Idea selected
- [x] Main method implemented
- [x] Local sanity verified (CPU only)
- [ ] Representative GPU results
- [ ] Natural first-error results
- [ ] Graph necessity / native causal mechanism
