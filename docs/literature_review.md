# 文献综述：从图可分性到首错的条件约束控制

## 核验范围

- 对 19 篇核心 arXiv 工作运行了确定性存在性核验；`verified_papers.json` verdict 为 `PASS`，hallucination rate 与 pending rate 均为 0。
- 重点阅读方法、监督信号、推断时可用信息和 failure mode，而不是仅比较摘要中的总体指标。
- 本文中的“无监督”严格指：不使用人工幻觉标签、LLM 伪标签或由这些标签训练的分类器；在无标签数据上估计 robust normal law/阈值仍属于允许范围。

## 四条文献主线

| 主线 | 代表工作 | 已证明的事实 | 对本项目仍缺的东西 |
|---|---|---|---|
| 监督图表征 | [CHARM](https://arxiv.org/abs/2509.24770) | attention/activation 构成的 attributed graph 经 GNN message passing 后具有强监督可分性；graph structure 的确有增益 | 训练目标直接使用 token/answer 标签；没有首错专门评估，也没有 source-world 干预，因此不证明无监督可识别或因果机制 |
| attention / internal-state 检测 | [Lookback Lens](https://aclanthology.org/2024.emnlp-main.84/)、[ReDeEP](https://arxiv.org/html/2410.11414)、[CORTEX](https://arxiv.org/abs/2606.31033)、[TPA](https://arxiv.org/html/2512.07515) | prompt/response attention、parametric-vs-context contribution、with/without-reference hidden-state delta 都含有效信号 | Lookback、ReDeEP、CORTEX、TPA 最终均学习有监督判别器；attention mass 不能单独证明因果使用；总体 token/span 分数容易利用 continuation persistence |
| 首错与错误延续 | [First Hallucination Tokens Are Different from Conditional Ones](https://arxiv.org/abs/2507.20836)、[How Language Model Hallucinations Can Snowball](https://arxiv.org/abs/2305.13534) | 首错的 entropy/perplexity 信号显著强于 conditional hallucination；早期错误会诱发后续合理化和一致性锁定 | 仍是现象学/retrospective 关联，未判定首错是在哪条内部信息路径上失去证据约束 |
| 无监督边界与异常检测 | [BOCPD](https://arxiv.org/abs/0710.3742)、[Anomaly Transformer](https://arxiv.org/abs/2110.02642)、[Rigorous Evaluation of Time-Series Anomaly Detection](https://arxiv.org/abs/2109.05257)、[Range PR](https://arxiv.org/abs/1803.03639) | run-length posterior、robust normal profile 和 first-passage 适合检测稀有状态转移；事件级评估必须防止 point adjustment 和未来泄漏 | 通用 graph/time-series anomaly score 没有 RAG 的 source provenance 语义，也不能自行区分合法信息 relay 与错误自回归 relay |

## 最接近的工作及真正差异

### CHARM：证明“图中有可监督学习的结构”，没有证明“无监督一定能恢复”

CHARM 将 token 作为节点、attention 作为边、activation/attention 作为属性，再通过 GNN 进行 message passing。论文明确把图与标签组成 `(G, y)` 并训练预测器。它最重要的启示不是“照搬 GNN 然后去掉标签”，而是：**单 token 局部特征不足，关系结构中确实存在额外可分信息。**

但是从 supervised separability 推出 unsupervised identifiability 在逻辑上不成立。一个有监督网络可以利用任意与数据集标签相关的 nuisance direction；无监督方法必须另加可检验假设，例如：大多数位置属于正常状态、正常状态的 source-control law 在 relation/task/position 条件下稳定、首错是该 law 的稀有 first-passage departure。

### CORTEX：它发现“前文可以合法携带证据”，恰好暴露了本项目的关键缺口

CORTEX 比较 with-reference 与 without-reference 的内部表征，并加入 contextual residual，避免把“当前 token 不直接读取文档、但通过前文继承文档信息”误报为幻觉；其 smoothing 又利用 hallucination label 的 span persistence。这个设计说明：**Past-token dominance 本身不能作为异常，合法的证据压缩也会产生这种外观。**

CORTEX 的 detector 使用 token labels 训练，并用 span smoothing 改善总体检测；因此它没有给出本项目要求的无标签在线首错方案。我们的候选差异不是估计完整 `lineage`，而是在同一 query 上测量 history-read 对 source-read 的有符号条件性作用，再用独立 source-world 绑定交换检验语义解释。前者是检测读出，后者是构念验证，不能混称为一个已被证明的 provenance estimator。

### TPA / ReDeEP：贡献分解很接近，但 source bucket 仍然太粗

TPA 将 next-token probability 分解为 Query、RAG、Past、Self、FFN、FinalLN、Initial 七类来源，然后用 126 维 POS-conditioned features 和 5 个 XGBoost 组成监督 ensemble。ReDeEP 以 Copying Heads 与 Knowledge FFNs 的代理量做回归。二者都说明“来源构成”有价值，却把 `Past` 或 `RAG` 当作整体 bucket。

关键反例是：一个 Past token 可能携带正确证据，也可能成为错误延续的条件；甚至前文完全正确，当前也可能把它用于错误的条件。相似的 source bucket mass 不足以区分这些情况。因而本项目不再只问“来自 Past 多少”，而问“历史读取是否削弱了证据对当前候选的反对”。所有来源在同一 head 中按无符号份额分摊净贡献，可能掩盖来源之间的相反方向；当前 graph proxy 的这一限制见结果审计。该反例是测量论证，不是自然数据已证实的机制。

### First Hallucination Tokens：支持把 onset 与 continuation 分开，但不提供机制

该工作在 RAGTruth 上表明首个 hallucination token 的 entropy/perplexity 比后续 conditional tokens 更有判别性；这与当前 graph 结果中 onset 的 `negative_margin` 优于 attention displacement 相符。它支持三状态视角：正常、首错边界、错误锁定；但 entropy burst 只描述模型在边界处“不确定”，不能说明为什么错误候选战胜正确候选。

## 与本项目历史结果的联合解释

把文献和现有实验放在一起，可以提出以下受证据约束的两阶段假设，但不能据此还原每个 span 的因果轨迹：

1. 首错与较低决策 margin 有关联；它是强基线，不是已证明的触发 gate。
2. 首错时远处 response-history attention 上升，但 prompt evidence share 并未同步上升；这更像旧回答 carrier 被重新使用，而不是证据被重新获取。
3. 首错之后 local-history attention 上升，与局部延续/lock-in 解释相容，但 attention 分配尚未证明后续具体使用了哪个错误 token。
4. attention-only 分数因此在 continuation 上最好；当前粗粒度 constraint score 在 onset 上接近随机。

这个组合排除了两种过度解释：

- “远程 attention 增加 = 正确回看 evidence”；
- “首错之后持续 local attention = 首错产生机制”。

## 候选研究空白

截至本轮检索，已有工作分别研究了 attention graph、source contribution、with/without-reference internal delta、首错的不确定性和错误 snowballing；尚未找到直接把以下对象作为首错 estimand 的工作：

> 在同一生成决策内，通过有符号 source-read × history-read 干预，测量历史读取是否遮蔽了直接证据对当前候选的反对，并以无标签参考分布定位首错。

早期曾考虑以历史 carrier 对 source 的低敏感性定义“谱系断裂”；正确的证据压缩构成反例，因此最终设计收紧到上述条件性作用，语义解释仍需受控事实验证。

这是一个**候选空白**而不是已经完成的 novelty proof；正式投稿前仍需做独立 novelty check 和更广的引用追踪。

## 由文献导出的设计原则

1. 检测的主要测量对象是有方向的生成决策效应，不是 raw attention mass；本方案不新增图网络，也不声称恢复完整因果图。
2. source type 还不够；独立 source-world 条件绑定交换负责验证语义解释，query-local 四格本身不提供完整 provenance。
3. 在线 onset score 只使用 `<= t` 的状态；后续 local lock-in 只做 retrospective confirmation。
4. 无监督不等于自由拟合一个 HMM/GNN。应固定机制拓扑，只从多数正常样本估计 robust conditional reference law。
5. reanchor 应是连续 moderator：检验回看强度是否调节 signed motif 与 onset 的关联，而不是先假定所有 onset 都在 hard reanchor event。
6. 评价必须报告 onset AUPRC、固定 false alarms/source 下 recall、detection delay，并禁止 point adjustment。

## 补充检索与方法修正

继续检索补充了三个不可省略的先例：

- [Analyzing the Source and Target Contributions to Predictions in Neural Machine Translation](https://lena-voita.github.io/posts/source_target_contributions_to_nmt.html)（ACL 2021，作者说明页）：使用 LRP 研究 source 与 target-prefix 的贡献；过度依赖生成前缀与幻觉的关系早已存在。因此“模型更信自己”不是足够的新颖性。
- [OPERA](https://arxiv.org/abs/2311.17911)（CVPR 2024）：利用 attention 中的 knowledge aggregation/over-trust 与回溯修复多模态幻觉；所以“聚合点之后错误自强化”的宽泛描述也不是新发现。
- [CAD](https://aclanthology.org/2024.naacl-short.69/)（NAACL 2024）：with/without-context 的分布对比即可改善 faithfulness。新方法必须证明第二个 history 因子提供单一 source contrast 没有的信号。

[CausalGaze](https://arxiv.org/abs/2604.11087) 进一步提高了“因果图检测”的新颖性门槛，其图干预与监督 detector 的梯度/训练相关；本方案需要与 generator decision margin 的有限路径干预区分。[DICD](https://doi.org/10.1016/j.knosys.2026.116819) 的出版页摘要显示它以 trigger/entity 的 context manipulation 检验 event-extraction faithfulness；本轮没有取得全文，因此仅记为必须继续核查的近邻工作。

**对早期 insight 的修正**：低 source 敏感性也可能来自正确的证据压缩，不能单独称为“谱系断裂”。最终方法应检验带方向的条件性作用：历史支持当前候选、直接 evidence-read 在历史抑制时反对该候选、历史读取削弱该反对。这里的 finite path intervention 仍非语义真值或完整 provenance 估计，需要 A/B 事实编辑与正常 relay 控制来验证。

文献中关于 future smoothing、无监督与监督的区别，会直接约束最终方法的评价与训练接口。不会把新近工作的全 token AP 与本项目 onset-only AP 放在同一排行榜中。

## arXiv API 补充命中

项目 helper 以 `all:hallucination AND (all:prefix OR all:graph OR all:onset)` 检索，返回以下与主线紧密相关的原文，已另行浏览：

- [Quickest Detection of Hallucination Onset: Delay Bounds and Learned CUSUM Statistics](https://arxiv.org/html/2606.12476)（2026 预印本，v3）：明确研究 onset 的 false-alarm–delay tradeoff，并报告低误报操作点下 recall 仍低；其速度收益很大部分来自更好的逐 token score。这意味着“改写为变化点问题”本身不新，检测延迟必须与漏检一起报告。文中的渐近理论/分布假设未在本项目证明，不能把其数值下界移植为我们的保证。
- [Evidence Graph Consistency in Retrieval-Augmented Generation: A Model-Dependent Analysis of Hallucination Detection](https://arxiv.org/html/2606.06748)（2026 预印本）：在其 evidence-graph embedding 定义下报告跨模型的检测方向反转。它不直接否定内部 attention graph，却进一步说明“图异常 = 幻觉”的方向不能未经验证地跨模型固定。

以上两篇与 OPERA、CAD 共四个补充 arXiv ID 通过存在性核验，记录于 `verified_additional_papers.json`。累计 23 个 ID 的存在性核验不等于已完成 23 篇全文审查；这里仅对实际阅读的段落作方法归纳。
