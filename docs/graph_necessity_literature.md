# 本轮文献补充：图的必要性、绑定与模型目标

检索时间：2026-09-10。本轮检查本地ReDeEP前三页，并使用web原始论文与arXiv helper；Zotero/Obsidian未配置，未创建research wiki。两条只读文献分工的结果按canonical arXiv ID去重，以下14个ID经verify_papers.py存在性核验PASS。核验不是全文结论认证。除注明的只读范围外，不声称穷尽查新。

| Paper | 作者 | 年份/出处 | 核心方法 | 对本项目的限制/边界 |
|---|---|---|---|---|
| [How do Language Models Bind Entities in Context?](https://arxiv.org/abs/2310.17191) | Feng & Steinhardt | 2024 / ICLR | activation swap与binding-ID向量干预 | 受控绑定任务，不是无监督幻觉检测；绑定与内容可分开研究 |
| [Representational Analysis of Binding in Language Models](https://arxiv.org/abs/2409.05448) | Dai et al. | 2024 / EMNLP | PCA/ICA寻找ordering subspace，activation patch改变属性取回 | 简单非图绑定基线；真实谓词和自然文本泛化受限 |
| [Mixing Mechanisms: How Language Models Retrieve Bound Entities In-Context](https://arxiv.org/abs/2510.06182) | Gur-Arieh, Geva & Geiger | 2026 / ICLR | 位置/词汇/reflexive causal mixture拟合干预分布 | 机制模型学习不是检测正确性；0.95是JS similarity而非accuracy |
| [Finding Alignments Between Interpretable Causal Variables and Distributed Neural Representations](https://arxiv.org/abs/2303.02536) | Geiger et al. | 2024 / CLeaR | DAS用symbolic counterfactual训练分布式对齐 | 有高层模型和目标；不能称严格无监督发现所有语义 |
| [Do I Know This Entity? Knowledge Awareness and Hallucinations in Language Models](https://arxiv.org/abs/2411.14257) | Ferrando et al. | 2025 / ICLR | SAE实体认识方向与事实提取干预 | feature选择与评价用correctness；不证明图必要 |
| [Interpreting Key Mechanisms of Factual Recall in Transformer-Based Language Models](https://arxiv.org/abs/2403.19521) | Lv et al. | 2024 / preprint | 路径patch、OV/MLP输出分解 | 内容变换比attention mass丰富；窄实体属性实验 |
| [Explaining the Reasoning of Large Language Models Using Attribution Graphs](https://arxiv.org/abs/2512.15663) | Walker & Ewetz | 2025 / preprint | CAGE跨生成token的归因DAG与线性路径传播 | source→history→output追踪已有；非负化忽略抑制/非线性 |
| [CausalGaze](https://arxiv.org/abs/2604.11087) | Kong et al. | 2026 / Findings ACL | detector梯度驱动边gating与GAT分类 | 人工幻觉标签训练；非原生成器真实干预响应拟合 |
| [Why Retrieval-Augmented Generation Fails: A Graph Perspective](https://arxiv.org/abs/2605.14192) | Guo et al. | 2026 / preprint | transcoder attribution图与图Transformer correctness detector | LLM judge labels；图的结构统计、routing控制已有 |
| [Learning to Attribute with Attention](https://arxiv.org/abs/2504.13752) | Cohen-Wang, Chuang & Madry | 2025 / preprint in inspected source | AT2从随机source ablation的真实响应学习head权重 | 已有amortized intervention-response学习；加性非图，必须对比 |
| [Twin Worlds: Equivariance-Based Abstention for Evidence-Grounded Reasoning](https://arxiv.org/abs/2608.28018) | Nguyen et al. | 2026 / preprint | 实体bijective substitutions后重生成并inverse-map比较 | 答案等变性弃答已有；受实体结构和长文本适用性限制 |
| [Cross Paraphrastic Invariance Learning for Hallucination Detection](https://arxiv.org/abs/2606.08157) | Lin et al. | 2026 / preprint | 文档claim paraphrase对比学习 | 幻觉标签与继承标签，不是label-free |
| [Dual-Pathway Circuits of Object Hallucination in Vision-Language Models](https://arxiv.org/abs/2605.13156) | Liu et al. | 2026 / preprint | 联合与单组件patch分解、正负作用翻转 | circuit筛选使用正确/错误标签；VLM object任务 |
| [Likelihood Ratios for Out-of-Distribution Detection](https://arxiv.org/abs/1906.02845) | Ren et al. | 2019 / NeurIPS | background model修正生成似然的背景统计 | likelihood ratio非创新，也非事实正确性保证 |

补充原始来源：[Li et al., ACL 2026](https://aclanthology.org/2026.acl-long.914/) 的正文§3定义SSR/SAS两个指标，明确attention模式随任务变化，而FFN grounding关联更稳定。不能把摘要中的机制语言当成已验证普遍因果结论。[DICD](https://doi.org/10.1016/j.knosys.2026.116819) 目前只有出版摘要可读，其contextual binding与sensitivity/invariance构成待进一步全文核查的近邻，未计入上表方法级结论。

## 归纳

第一，图确实可以表示来源到历史再到输出的多跳关系，但这种表示并不新。CAGE和新的RAG attribution graph已经覆盖大量路径追踪概念。CHARM有监督成功说明某些特征可读，不证明图拓扑是无标签事实性判别的充分统计量。

第二，绑定与信息来源是不同问题。Feng与Steinhardt、Dai等和Mixing Mechanisms把实体内容、位置地址和词汇机制区分开来，给出比“远程attention变大”更细的对照。但其中的构造数据与预定causal变量不能被隐去，不应冒充无需语义假设的自然幻觉检测。

第三，原模型的功能辨识与事实性判断之间有缺口。AT2可以忠实预测错误答案的干预响应；DAS也可以忠实描述一个错误计算。因此更高的功能预测精度并不保证更好的幻觉识别。将其残差直接称hallucination score仍需额外可证伪假设，不能靠GNN自动解决。

第四，创新不能落在已有零件的串联：图、干预、等变性、归因、密度比、source copy都已有先例。本轮保留的候选问题是：能否用一个不允许历史native states直接注入证据value的source-rooted预测器，保留合法relay同时减少错误绑定？图只是候选归纳偏置，其必要性要由同目标的非图模型证明。source-rooted也不保证选择了正确事实，语义泛化仍是风险。

## 路线淘汰记录

- 四标量control graph：保留机制审计，不宣称graph representation learning。
- 下一子图重建：项目已有相关负结果，不能直接恢复为主线。
- 响应预测残差：可忠实拟合错误，缺乏事实性目标；本轮初稿交由独立审查。
- source-rooted shadow predictor：仅候选；需与pointer/cross-attention/sequence grounded decoder精确对比，新颖性尚未确证。

## 阅读范围

绑定方向来自只读分工的原论文方法/限制提取；主线程复核Mixing Mechanisms HTML。归因方向主线程复核CAGE、AT2、Twin Worlds、RAG Graph Perspective和Dual-Pathway原文；Li et al. PDF已读到指标定义。DICD全文未读。各模型论文的数值未与本项目onset表混排。

## 检索失败

一次web调用出现connection failed，后续调用恢复。没有用该失败当作“未发现先例”的证据。

## 新增近邻：不能把“self-supervised graph”当作空白

[SiGHT，Chen、Chiang、Peng，AISTATS 2026](https://proceedings.mlr.press/v300/chen26d.html) 已提出自称self-supervised的图幻觉检测。出版摘要明确其用prompt合成幻觉内容，再以word-level关系图和GAT判别；“不需要人工标签”不等于本项目的“不用真假伪标签”。本轮PMLR出版页已核验，但PDF抓取先遇到octet-stream解析错误、再发生网络连接重置，未完成全文方法审读，故不作摘要以外的实现结论。

[Pointer-generator，See、Liu、Manning，ACL 2017](https://aclanthology.org/P17-1099/) 已利用source copy改善事实细节，同时保留生成通道。新候选若仅改名为“source-rooted”，不足以形成创新；必须证明历史只作query/relay、source ports不先池化、以及无幻觉标签学习目标的不可删作用。注意这些是待验证区别，不是该旧方法无法表达它们的定理。

## 同名消歧与理论边界补查

- [CAGE: Cognitive Attribution Graphs for Faithful Inline Citation Generation](https://arxiv.org/abs/2607.24236)，Yan 等，2026。摘要描述 answer-centered support subgraphs 与后续引用生成，和 Walker/Ewetz 的 attribution CAGE（2512.15663）不是同一篇。当前仅核验摘要，不引用未读训练细节。
- [(Im)possibility of Automated Hallucination Detection](https://arxiv.org/abs/2504.17004)，Karbasi 等，2025。仅核验摘要。其理论处于特定 language-identification 框架，不能简化为“所有现实无监督检测都不可能”；它也不能证明本方案可行。

## 本轮收敛后的修订

source-cloze shadow predictor 已被独立审查否定为当前主线：只会 source 补词不等于条件绑定。替代候选是 paired-binding evidence energy：source 绑定交换与两种正确历史角色的组合任务。source-rooted 仅是输入约束，不是贡献或正确性证明。当前首先考虑非图 cross-attention 版本 G0；来源保留图 G1 必须在同任务、同预算下产生自然首错增益才能保留。相对于上述近邻，创新性仍未确立。
