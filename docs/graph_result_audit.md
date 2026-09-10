# Graph 结果审计

## 冻结结果

输入日志包含 153,513 个已评估 token、688 个 source；先冻结无标签分数，再 join labels；指标采用 equal-source weighting 和 200 次 source-cluster bootstrap。

| Cohort | Positive prevalence（source-balanced） | Score | AUROC | AUPRC | AUPRC / prevalence |
|---|---:|---|---:|---:|---:|
| all | 0.11738 | attention displacement | 0.71216 | 0.28221 | 2.40× |
| all | 0.11738 | constraint displacement | 0.54586 | 0.12601 | 1.07× |
| all | 0.11738 | negative margin | 0.58609 | 0.14595 | 1.24× |
| continuation | 0.11408 | attention displacement | 0.71511 | 0.28085 | 2.46× |
| continuation | 0.11408 | constraint displacement | 0.54554 | 0.12247 | 1.07× |
| continuation | 0.11408 | negative margin | 0.58077 | 0.13849 | 1.21× |
| onset | 0.00607 | attention displacement | 0.63309 | 0.00917 | 1.51× |
| onset | 0.00607 | constraint displacement | 0.55344 | 0.00685 | 1.13× |
| onset | 0.00607 | negative margin | **0.70778** | **0.01627** | **2.68×** |
| onset | 0.00607 | relative position | 0.50972 | 0.00581 | 0.96× |

## 结果允许说什么

1. `attention_displacement` 是目前最强的全 token / continuation 筛查信号，但首错表现明显下降；它更像“错误已经写入上下文后的锁定”指标，而不是首错机制。
2. 首错上最强的现有单变量是 `negative_margin`。它说明首错附近存在决策不稳定性，但绝对 AUPRC 只有 0.01627，不能单独承担实用检测。
3. 当前 `constraint_displacement` 在 onset 上几乎只比 prevalence 高一点，不能支持“首错就是回看节点违反远程约束”的结论。
4. onset prevalence 约 0.6%，因此 AUROC 0.71 也可能对应极低 precision。后续主指标必须是 onset AUPRC、固定 false alarms/source 下的 recall 与 detection delay，不能只报告 AUROC。

## 结果不允许说什么

- 不能从 CHARM 的监督可分性推出无监督一定可识别；无监督需要额外的 majority-normal、invariance 或 causal-control 假设。
- 不能把 attention mass 当成信息传递或因果使用。当前 `support` 只是按 message norm 分摊整 head 的 margin，不是 source-specific signed causal effect。
- 不能把旧实验中“onset 更关注远处历史 token”解释为“重新读取 prompt evidence”。旧定义中的 far history 是较早的 response token，而且 onset 时 prompt share 多数反而下降。
- 不能用后续错误 token 来提高在线首错分数；这会把 retrospective span confirmation 泄漏进 onset detection。

## 对方法设计的直接约束

- 将首错和延续建模为两个状态：`NORMAL -> FRACTURE -> LOCKED`。
- 将 reanchor 从硬门槛改为连续调节量；旧 hard-anchor 候选覆盖仅 23/688（3.34%），不足以作为总体检测入口。
- 核心信号必须区分“有 source lineage 的合法历史 relay”和“对当前预测影响强但已与 source 脱钩的 orphan relay”。
- 旧 attention、margin 与位置特征保留为对照或不确定性门控，不再被命名为机制本身。

## 继续核查发现的评价边界

`control_graph/audit_evaluation.py` 的 onset cohort 是 `label==0 OR (label==1 AND previous==0)`，所以它排除了全部 continuation token；现有数字是“首错 vs 正常”，不是“完整生成流中的首错定位”。后续必须额外报告 onset vs all non-onset，并使用一对一告警事件匹配。

代码把 response 第一个 token 的 previous 设为 -1，因此从回答开头开始的 hallucination span 不进入当前 onset cohort。新评估应以显式 span 起点/已知序列起点处理；对未知标注边界仍标记 unknown，不能自动当正常。

`relative_position=response_index/(response_tokens-1)` 使用最终长度，因此只是事后位置诊断；在线 detector 的条件校准只能使用绝对位置等当时已知信息。

这些是本轮读代码确认的定义限制，没有重新计算或改写用户已跑出的指标。

## 当前 constraint proxy 的具体限制

在 graph 的 `control_graph/audit.py::_support_edges` 中，每个 group 的值由 `share * head_margin` 再除以所有有效 head 的绝对 margin 总和得到。share 来自 message norm，因此同一 head 的不同来源被分配了同一符号。

一个线性玩具例子：evidence 的真实投影为 -1、history 为 +2，整 head 净值为 +1。按 norm 比例 1/3、2/3 分摊 +1，会把 evidence 也记为 +1/3，而看不出它原本反对该候选。这不证明实际样本一定存在该抵消，却证明当前 proxy 无法识别本轮所关心的这种情况。

因此“constraint displacement 弱”否定的是这项具体代理测量的实证支持，不能据此推出“模型内部不存在约束反对信号”。要测试后一个问题，需要新的有符号计算或有限干预，旧结果不可转换为新测量。
