# G0 重构与工程验证

日期：2026-09-11。实现依据：graph_method_proposal.md 最后作者修订。
本次由 experiment-bridge 实现、run-experiment/compute-env-contract 验证；AUTO_DEPLOY=false。

## 已交付

- 主 CLI 从旧四格审计拆为 G0 generate/extract/train/score/calibrate/evaluate/pipeline。
- G0：冻结 Llama；source-only evidence value；source-blind address；两层候选条件化读源。
- 训练只用程序答案与 neutral 方差；自然标签只在冻结评分后评估。
- 首错、逐官方 span 起点、全幻觉指标分开；source-balanced bootstrap 与 negative-margin 对照。
- 实际权重/全部缓存张量/代码/提取配置的内容身份校验；标签 join 核对 source 和 instruction。
- run_g0.sh / run_pilot.sh 为新主线；run_audit.sh 保留旧实验。

## 工程证据

| 验证 | 实际结果 |
|---|---|
| 独立 CPU 环境 | Python 3.13.9, torch 2.8.0+cpu, transformers 4.57.1 |
| seeded matmul | WITNESS, CPU, shape [8,8] |
| 作者最终回归测试 | 30 passed in 14.87s |
| fresh agent 按 RUNBOOK 最终逐字复验 | 30 passed in 15.14s；六条命令均 exit 0 |
| Ruff | All checks passed |
| Bash 语法 | run_g0 / run_pilot / run_audit 均通过 |
| CLI | 新 pipeline 与旧 audit 的 --help 均 exit 0 |

测试包括微型随机 Llama 的完整流水线、未来后缀隔离、空源能量、配对程序真值、
源划分、校准 ties、缓存篡改、权重变化、数据变更、相邻/重叠 spans 和脚本失败/拒绝覆盖。
共享 conda 的早期运行曾打印 DLL access-violation 诊断后仍退出 0；独立环境无该诊断。
未修改共享 conda，未上传环境、缓存、数据、权重、密钥或原始实验输出。

## 审查与边界

独立初审发现缓存身份、source identity、相邻 span onset 三项阻断，均已修复并补测试。
复审追加的数据准备配置/划分身份问题也已修复并补测试。
第二轮最终结论 PASS，无残余 BLOCKING；同模型家族审查仅为 provisional 工程审查。

上述是工程结果，不是检测效果：本轮没有运行远端 Llama 8B，没有新的自然数据 AUROC/AP。
没有实现 G1 图模型，没有证明图必要性、创新性、通用幻觉机制或原生生成器的因果路径。
默认 64 程序 sources + 32 自然 sources 是探索性 pilot；未排除 discovery sources 时
不能称独立确认性实验。混合无标签参考分位数也不提供正常 FPR 保证。

## 远端入口

在仓库内激活已有 GPU Python 环境后：

```bash
git pull --ff-only && bash scripts/run_g0.sh
```

自然指标在 outputs/g0_<时间>_<进程号>/evaluation/summary.json，完整日志为同名 .log。
只验证程序链路可用 PROGRAM_ONLY=1 PROGRAM_SOURCES=8 EPOCHS=1 bash scripts/run_g0.sh。
