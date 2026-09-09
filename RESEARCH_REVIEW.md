# Reanchor 当前证据与主张边界

- review date: 2026-09-10
- reviewed baseline: `8ef256d10c04d2538c8a48a128f6caa8408dd680`

## 结论

当前主线已经收敛为“无标签 reanchor 候选发现 + 固定轨迹上的解析传播审计”，
尚未收敛为一个经验证的无监督幻觉检测器，也没有证据支持“普遍幻觉机制”。
`completed=3, planned=3, rejected=0` 只证明 P001 的三条 capture/replay 基础设施样本
通过门禁，不是方法收敛、检测性能或机制结果。

## 主线实际回答的问题

1. 在固定生成轨迹上，重建相邻 response query 的完整 causal attention。
2. 用当前 query 的同一 source partition 测量从 local 到 remote 的注意力增量。
3. 将 layer/head 搜索压成 sparse 与 broad 两个 token 统计量。
4. 用独立 `source_id` 的最大值参考分布校准并校正预声明 family。
5. 合并连续显著 token 为 episode，只在最强 anchor 做昂贵传播。
6. 事件冻结后才连接 N/H 标签，并以独立 source 为统计单位。

这套流程发现“相对于校准来源极端的远程重取 transition”，而不是直接输出
hallucination probability。自然 source-max 是经验参考分布，并非机制意义上的真实
null。

## `graph` 中旧统计能支持什么

`graph/experiments/reanchor_flow/message_dag/LOOKBACK_EVENTS.md` 记录了用户提供的
真实 V3 统计表：总体 matched-H 更 local、更平稳；标注 onset 往往更 remote、TV 更大。
该文件同时明确：

- 结果来自保存的统计表，本环境没有重跑原 8B forward；
- 7,097 与 259 是目标/事件数，独立 source 数分别只有 303 与 189；
- 1,024 个物理 head 彼此相关，不能当作独立重复；
- onset 增加的远处来源主要是旧 response history，不是 prompt constraint；
- “全部 H”与“onset H”是不同 aggregate contrast，不能拼成同一条
  “远看 → local 自强化 → 出错”的因果轨迹。

因此它是真实的探索性结构关联，也是一个值得确认的假设；“总是如此”以及因果/普遍
机制都不成立。

## 本次机制审计修正

1. 主估计量改为与 discovery 对齐的相邻行 remote attention delta；当前整段 native
   remote write 仅作为另一参考估计量。
2. 四个 coarse source group 降为完整 provenance partition，不再命名四种失败机制。
3. 每个 `source_unit_id` 单独传播并排序，防止不同材料在 content 内抵消。
4. normal、`N -> H` onset、continuing-H 对称报告；只在同一 event 内检验 onset 与
   后续 `2+` hop rollout，且后续 target 必须位于下一次 reanchor 之前。
5. margin/layer/root 按元素分别做 additive closure；resume identity 哈希所有上游
   artifact、source annotation 与 contrast。
6. 当前结果仍显式标为 linearized candidate route，`claim_supported=false`。

## 获得机制信心所需的确认性实验

- 同 sample/source 内，为 `N -> H` onset 建立两侧全正常匹配 pseudo-onset；匹配位置、
  token 类型、logprob、entropy，用 source-cluster permutation/max-T 检验完整时序曲线。
- 对同一事件同时要求 onset 的 error-favouring transition effect，以及下一次 reanchor
  前 continuing-H target 的负向 `2+` hop effect，报告所有分母和无事件覆盖率。
- 做无重归一化的选中 value-path 删除、正确 source restore 和双向 intervention，并用
  same-distance/random-head/random-layer/magnitude-matched shams。
- onset 后切断 local carrier，检验后续幻觉 span 概率/长度以及 remote effect 的中介衰减。
- 从干预点 paired free-run 解码，验证答案事实性真的改变，而不只 teacher-forced margin。
- 至少跨两种 model family、两种 task family、source-disjoint confirmation 复现，并做
  leave-one-domain-out 异质性分析。

全部通过后，最稳妥的主张仍是“在限定模型/任务中反复出现的机制家族”，而不是所有
LLM 幻觉的普遍机制。
