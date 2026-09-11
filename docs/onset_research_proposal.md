# 从信息流到约束控制：无监督定位幻觉首错的研究方案

> 历史方案：本页的四格干预不是当前主线。当前要求见 [回看节点与约束归属](graph_method_proposal.md)，先控制采样与前缀差异，再分析具体读取过程。此前设计审阅的 READY 不代表机制已经通过实验。

状态：待实现、待验证的方法方案。本文交付的是方法定义与验证协议；query-local 四格尚未实现，也没有新的 GPU 结果。当前代码的文本 A/B 实验属于另一测量接口。

## Problem Anchor

- **Bottom-line problem**：在完全不使用幻觉标签训练检测器的前提下，定位 RAG 回答中第一个事实性错误 token，并检验该首错是否由一次“回看/重新锚定”过程中的证据约束失效所触发。
- **Must-solve bottleneck**：现有 attention/graph 分数主要擅长识别幻觉 span 已形成后的局部自回归延续，不能区分两种外观相似的历史依赖：合法的“证据经先前回答 token 继续传递”与错误 token 脱离原始证据后成为伪证据。
- **Non-goals**：不把任意远程注意力增加定义为幻觉；不以一个硬阈值的 reanchor 事件筛掉大部分样本；不靠 RAGTruth 标签、伪标签或监督神经分类器学习决策边界；不把 span continuation 的高分冒充首错检测。
- **Constraints**：复用现有生成模型与 RAGTruth traces；标签仅在所有分数冻结后用于评估；检测时不得看未来 token；机制验证允许在事后使用成对 source/prefix 干预，但不得用 gold hallucination label 选择内部节点或调参。
- **Success condition**：提出一个可识别、可干预、可证伪的首错机制；在 source-disjoint、source-balanced 的 onset-only 评估上，显著超过负 margin、位置、attention displacement 和静态图异常基线，并以有限干预证明该信号不是位置、token identity 或 span persistence 的替代变量。

## 技术缺口与主判断

CHARM 证明了监督图可分性；它不保证无监督几何会自动按事实性聚类。我们自己的 GCN PCA-kNN 曾得到 AUROC 0.6982，但有 role/position 混杂，且不是 onset-only 结论。现有结果更直接地约束主线：onset 的 negative margin 为 0.70778 / 0.01627（AUROC / AUPRC），attention displacement 为 0.63309 / 0.00917，constraint displacement 为 0.55344 / 0.00685。当前 proxy 没有证明首错机制。

历史实验中，首错的 far-response attention 常上升，prompt attention 常下降；后续错误以 local-response attention 为主。这支持一个两阶段假设，却不能从不同样本的群体平均拼出每个 span 的因果轨迹。

**主 insight：信息被再次使用，不等于约束仍然有效。首错可能发生在历史内容保留了影响力，却遮蔽了当前证据对这一用法的反对。**

例如 source 写“工作日 9 点，周末 10 点”。前文“工作日 9 点”完全正确。在生成“周末则是 9 点”时，旧的“9”可能被再次调用，但“工作日”这一适用条件没有约束当前输出。首错之前不需要已有错误 token。随后新生成的错误“9”又成为局部延续的依据。

这个例子只是说明假设，不是实验发现。它也解释了为何早期“无 source 敏感性 = orphan relay”的定义不够：正确证据压缩到前文后也可能低敏感，因此必须测量**带方向的证据反对，以及历史对该反对的遮蔽**。

## 方法主张与贡献范围

一句话方法：在每个生成决策上测量 source-read × history-read 的有限干预交互，寻找“历史支持当前候选、证据原可反对、历史削弱这种反对”这一有符号四格交互的稀有出现。

主贡献是一个明确的、当前位置按 key-region 门控的有符号 logit-margin interaction，及其与首错的关联检验。受控事实实验通过后，才进一步解释为条件性证据控制失效。支持贡献是同一四格测量的无标签检测器；不将其包装为新 GNN 架构或因果推断定理。

两条路线比较：

| 路线 | 优点 | 最主要风险 | 决定 |
|---|---|---|---|
| 更大 self-supervised GNN 学 attention graph，再做异常检测 | 可批量复用 traces | 已有 endpoint/layout loss 改善但事实性几乎随机；重建目标未必保留首错信号 | 保留 first-order GCN 基线，暂不加网络 |
| 原生计算上的有符号 source × history 干预 | 直接问证据有没有改变当前决定；每 token 可定义，不需已知正确答案 | finite ablation 可能离开自然分布，且仍不是事实真值 | 作为主路线，先过可证伪 pilot |

## 核心方法

### 输入、时间与候选

目标是刚采样但仍在缓冲区的 token \(y_t\)。用于预测它的 query 位置是 \(q=t-1\)，第一 response token 的 query 为 prompt 最后位置。所有干预只使用 prompt 与 \(y_{<t}\)。先采样 y_t、运行分支评分、再释放/标记该 token，期间不生成 y_{t+1}。token-delay=0 仍有分支计算的 wall-clock latency，必须单独报告。若部署选择先释放再评分，则另外报告已经暴露的 token 数，不能沿用缓冲模式的零暴露口径。

固定候选 \(v_t=\arg\max_{v\ne y_t}z_{11,t}(v)\)，即原生分布中除 observed token 外最强的 token；所有干预共用这个候选。四格都测同一 margin：

\[
m_{ab,t}=z_{ab,t}(y_t)-z_{ab,t}(v_t).
\]

这里没有 correct/error 标签；runner-up 也不保证正确。D<0 只说明指定路径使 observed-minus-rival margin 降低，甚至不保证 observed 已输给 rival。因此自然 RAG 的首要结论只能是 motif-associated onset signal。语法、同义词或子词 rival 都可能产生 false positive；语义机制要靠后述冻结的 A/B 事实实验建立。

### 明确干预对象

将 query \(q\) 的每层每头 attention 写入按 key 位置分为：

- \(E\)：retrieved evidence 的内容 token；
- \(H\)：严格早于 \(q\) 的 response-history token；
- \(B\)：query/instruction、special tokens、当前 self token 等其余项。

三者互斥且完备；数据适配器从 prompt 模板给出 evidence 区间。缺失可靠区间时标记 measurement unavailable，不能把整段 prompt 冒充 evidence。local/far 只是 H 的事后分层，不是另一个检测器。

在 layer \(\ell\)、head \(h\)：

\[
w_q^{\ell h}(a,b)=a\sum_{j\in E} A_{qj}^{\ell h}(a,b)V_j^{\ell h}(a,b)
 +b\sum_{j\in H} A_{qj}^{\ell h}(a,b)V_j^{\ell h}(a,b)
 +\sum_{j\in B} A_{qj}^{\ell h}(a,b)V_j^{\ell h}(a,b).
\]

在 output projection 前缩放各分组的 AV message，不重归一化 attention。各分支从相同的原生 prefix KV cache（截至 q 之前）和 q 的 embedding 开始，逐层重新计算 query、attention、residual、FFN 和 logits。每层的 attention 按本分支状态重算。分支产生的 KV 不写回原生 cache。

主 pilot 取 \(a,b\in\{0,1\}\)。对 \(j<q\)，\(V_j(a,b)\) 是所有分支共享的原生 cache；对 self 的 \(j=q\)，它随分支重算。\(A(a,b)\) 也由各分支重算；这里没有固定 attention 图的近似。四格含义：

| | history-read 原生 \(b=1\) | history-read 阻断 \(b=0\) |
|---|---|---|
| evidence-read 原生 \(a=1\) | \(m_{11}\)：原生预测 | \(m_{10}\)：抑制当前 query 的历史读取 |
| evidence-read 阻断 \(a=0\) | \(m_{01}\)：抑制当前 query 的直接 evidence 读取 | \(m_{00}\)：两种当前读取均抑制 |

**干预只切当前 query 的读取路径。** 上游 prompt/response cache 与 self token 仍可能携带 evidence 信息；因此 \(a=0\) 绝不能命名为“模型完全没有证据”，\(b=0\) 也不是“完全没有历史”。这是固定已实现前缀后的路径干预，不是 natural indirect effect，也不能单凭它恢复完整语义谱系。

### 可检验的四个量

\[
E_t=m_{11}-m_{01},\qquad D_t=m_{10}-m_{00},
\]
\[
H_t=m_{11}-m_{10},\qquad R_t=E_t-D_t.
\]

- \(E_t\)：原生历史读取存在时，当前直接 evidence-read 对候选 margin 的作用。
- \(D_t\)：当前 history-read 被抑制时，同一路 evidence-read 的作用。
- \(H_t\)：保留 evidence-read 时，当前 history-read 对候选的作用。
- \(R_t\)：history-read 对 evidence-read 作用的调节；保留符号，不能取绝对值。

具体假设模体为：

\[
H_t>0,\quad D_t<0,\quad R_t>0.
\]

历史读取支持 observed token；历史抑制后可测到 evidence-read 反对 observed token；原生历史读取削弱了这种反对。它比“history 很强”或“evidence 很弱”多了一个可操作的有符号交互。

玩具数值（非实测）：\(m_{11}=1,m_{01}=1.5,m_{10}=-2,m_{00}=0\)。则 \(E=-0.5,D=-2,H=3,R=1.5\)。证据在原生计算中仅使错误候选 margin 降低 0.5；压低历史读取后，证据反对强度变为 2，且 observed 候选失去优势。

正常历史 relay 可以拥有很大的 H，只要它没有同时掩盖 evidence 对当前用法的反对，就不会因 H 大而自动触发该模体。仍可能存在正常 false positive，必须由 matched controls 和真值评估检验，而不是通过定义排除。

### 可识别性、自检与明确的漏检类型

对任意四个角点，可精确重参数化为 \(m(a,b)=c+\alpha a+\beta b+\gamma ab\)。于是 \(D=\alpha,E=\alpha+\gamma,H=\beta+\gamma,R=\gamma\)。这不是网络在中间门控值上双线性的假设。K 只检测一个符号区域，不覆盖所有 history dominance。

纯加性覆盖是必要反例：若 evidence 反对强度为 2、历史加性支持为 3，而交互为 0，则 observed 仍可能是错误 winner，但 K=0。证据完全未被读取、证据理解本身错误、H 为空的第一个 response token，也都可能 K=0。必须报告各类型覆盖和全流漏检，不能据此筛掉样本。

四格不能区分“条件语义丢失”与 LayerNorm、query rerouting 等一般非线性。受控实验要在数字/实体集合不变时只交换条件绑定，并加入正确 relay 和等能量 placebo；未通过则保持模型内受控交互的描述，不宣称完整 evidence provenance。

### 无监督检测：固定机制，学习参考分布

先冻结有向分数：

\[
K_t=\min\{[H_t]_+,[-D_t]_+,[R_t]_+\}.
\]

三个条件必须同时出现。它们共享 logit-margin 单位，min 没有可监督调整的权重。参考数据仅用于估计 K 的正常范围；不把 low density 本身当事实定义。

source-disjoint 无标签 reference split 上，按 model、task、绝对生成位置的预定 log2 bins 估计 source-balanced 经验尾部：

\[
s_t=-\log\widehat{\Pr}_{ref}(K\ge K_t\mid model,task,position\ bin).
\]

每个 stratum 内每 source 总权重为 1，按其在该格的 token 数均分；若少于 30 个 source，依次去掉 position、再去掉 task，始终不跨 model。最粗 model 格仍不足 30 个 source，则只输出原始 K，不给校准告警。要求 1% 尾部操作点时还需至少 199 个 reference source；不足则继续向粗格回退，仍不足时标记 budget_unresolved，不以小格外推 1% 告警。经验尾部保守计入 ties，加一个 source 单位的平滑质量：(1 + weighted exceedance)/(1 + source count)，避免零尾概率。该尾部是混合无标签分布的经验量，不承诺 conformal coverage，也不等于真实 false-positive rate。

主要假设：reference 的多数位置正常，且同一 model/task/position 条件下 reference 与目标数据的机制分布足够稳定。若 reference 本身经常有同类错误，该方法会失效。测试时冻结 reference，不在线吸收异常 span。

在线边界取 \(s_t\) 超阈值的首次上穿；不累加到序列开头，不使用 HMM/GRU 或双向 smoothing。预定 reference 告警预算为 1% scored tokens，仅是操作点，不是真实 FPR 保证。报告完整 threshold curve。

negative margin 是必须比较的强基线。本轮删除融合器：主分数只有 K 的无标签尾部读出。必须在 native margin 分层后检验其增量；K 与 margin 的融合不能用来挽救失败的机制假设。

### 回看节点的位置

reanchor 是 moderator，不是 inclusion gate。定义固定 W=16 的 far-response 集，并在比较 q 与 q-1 时使用同一个共同 key 集，避免“local 变 far”的移动边界造假。连续量为相邻 query 对共同 far-history 集的 attention share 增量；BOS/Self 单独计入控制。

核心检验是：首错的 K 异常是否富集在高 reanchor 处；正常 reanchor 是否仍然保有 evidence control；非 reanchor 首错占多少。如果首错多数不在回看节点，只能放弃“回看是通用入口”叙事，仍可保留每 token 的机制检测。

### 复杂度与集成

冻结 LLM，不新增可训练网络。每个 token 有一次原生计算和三次 query-only 重算；共享 prefix KV，不重算整条前缀。teacher-forced 离线实现也必须为每个 q 使用原生历史 cache，不能把在所有 q 上同时切 H 的一条全序列 forward 当作等价实现。

输出至少包含 sample/source/model/task、t/q、observed/rival token id、四格 logits/margins、E/D/H/R/K、source mask、intervention strength、native replay error、continuous reanchor、measurement status。labels 独立文件，只在评分和 reference 冻结后 join。原始 logits 先转 FP32 再相减；native parity 在同 kernel、batch=1 的无门控重算上与参考比较，容差由无标签重复运行误差冻结。branch 必须从相同只读 prefix cache fork；若 cache 会原地更新，显式 clone 或静态视图回滚，且验证原生 cache 不变。role masks 必须逐 token 互斥完备，E/H 为空的输出也进入覆盖计数。

当前 reanchor 的文本 A/B source × forced-prefix 四格与本方法的计算路径四格**不是同一个 estimand**。现有 `source_onset/source_followup/prefix_followup/coupling` 继续作为文本机制审计；文本 interaction 新增 signed 字段，保留旧绝对 coupling 的语义。新增独立 query-intervention schema，禁止把已有结果直接重命名成 K。

### 首个实现的完整合同

定义 \(F_t(a,b;y_t,v_t,M,\mathcal E,\mathcal H,\mathcal B)\) 为冻结模型 M 在原生 prefix cache 上重算 q、按上述 AV 分组门控后得到的 observed-minus-rival logit margin。首个实现限定 Llama decoder、eager attention、eval/dropout=0、batch=1、无 padding；暂不声称任意 decoder 都兼容。

- 原生与分支使用同一 tokenizer、chat template、position_ids 和 RoPE 配置。prefix cached K 已施加 RoPE，禁止二次旋转；q 的 K/Q 按原生规则重新生成和旋转。
- GQA/MQA 按原模型的 query-head→KV-head 映射读取缓存；E/H/B 由 token index 定义，对所有 query heads 使用相同 token mask。不创造独立 KV heads。
- 分组 AV 相加之后执行原生 output projection；若有 projection bias，只加一次。residual、各层 RMSNorm/LayerNorm、FFN、final norm 与 lm_head 都照原模型执行，不缓存原生 q 的 norm 输出。
- self K/V 是该分支当前层重算结果，只参与当前 q 的 self edge；分支不可向父 cache append。初版仅处理普通 causal attention；其他 attention bias、padding、滑窗或模型变体显式报 unsupported，不能静默降级。
- eager 分支与 eager native 逐 token 检查 logits/margin parity；若与此前 flash/SDPA trace 比较，单独测量 backend 差异，不把它并入机制效应。固定容差、cache 不变与逐层无门控复现是运行前数值门。
- 本合同是待实施规格，不代表这些 checks 已执行。

### 失败模式与边界

1. **Ablation 分布偏移**：主机制结论必须同时经预定小幅门控（1→0.75）与等能量 placebo 支持；在四角点 1/0.75 上，E/D/H 除以步长 0.25，R 除以 0.25²，独立校准，不能与 1/0 的 K 混在同一 reference。二者的符号一致率与语义方向单独报告。只在全切除出现的效果不具备原生机制解释，也不得看 test 标签后把另一强度改成主结果。
2. **词法历史仍在**：query-local cuts 未删除 prefix token identity；不能把 K 称为全部历史来源效应或证据谱系的完整估计。
3. **替代候选语义无效**：固定原生 runner-up 是预算内可用信号，不保证事实竞争；报告多 token 候选敏感性，但不按 test labels 选择 rival。
4. **Evidence 缺失/冲突/事实被误解**：D 未必指向正确答案；机制范围是可用证据被历史遮蔽这一子类，不能覆盖所有 hallucination。
5. **从准确前缀到首错**：H 可以来自真实但适用范围不同的信息；实验必须包含这种样本，避免把已错误前缀的 snowballing 当成首错起因。
6. **旧标签参与过研究构思**：本设计受已有标注统计启发。可声称 detector fitting/scoring 不用标签，不可声称整项研究从未看过标签；confirmatory test 必须使用未用于提出方向的新 source。

## 最小的三组核心实验

### 实验 1：原生 RAGTruth 全流首错

执行顺序是实验 2 的工程 pilot 与独立构念检验 → 本实验 → 实验 3 的迁移。先冻结旧 688 sources 为 discovery，新增/保留未见 source 分成无标签 reference 与 confirmatory test。按 model/task/source 分组后再处理 token。在受控 pilot 通过后，做无标签抽样的 100 responses 全 token 速度测试；这些标签结果不能冒充新 confirmatory 结论。

当前会话没有完整远端 manifest，不能声称已经留出多少未见 source。运行前必须输出 model/checkpoint revision、tokenizer/chat-template digest、decoding 配置、prompt/response token IDs、source split manifest 及哈希。自然机制主结果仅纳入可用权重且可核对原生成器与 prompt 的回答；closed-weight 回答不进入白盒生成原因分析。

特别区分：拿 Llama-3.1 去重放由 Llama-2/GPT 生成的 RAGTruth 文本，可以评价这个新模型的检测特征，但不是原生成器产生该错误时的机制。若原生条件不能复现，只能标为 proxy-model / replayed-prefix 分析。重新生成新回答则需要独立标注；不能直接继承原回答标签。

主结果必须含两种 denominator：

- onset vs normal：对齐现有结果，排除 continuation，只作诊断比较。
- onset vs all non-onset：包含 continuation 作为 non-onset，衡量真实全流首错定位；以这个和告警事件匹配为主要结果。

以每个 hallucination span 的首 token 为正，同时单独报告每条 response 的第一个 span onset（首个事实错误）。回答从第一个 token 就进入标注 span 时计入 onset；未知标注边界保持 unknown。标注 span 起点与真正语义分歧可能差一个子词，exact 为 benchmark 目标，±1/±3 仅预定辅助。

主指标为全流 exact onset AUPRC：在每个明确的评价 cohort 中，source s 有 n_s 个有效标注 token，每个 token 权重 w_i=1/(S*n_s)，S 为该 cohort 的 source 数；unavailable 但有标签的 token 保留并赋最低分。它与 reference 的权重分别计算，不共享标签信息。操作指标同时报告 recall、precision、每 1,000 正常 token 的误报数、每正常 source 的告警数。额外报告 clean-response 暴露 token 数/误报 episode 数，命名为 empirical ARL proxy；零误报时报告暴露量和区间，不输出无限可靠性的结论。有限且会恢复的 hallucination span 不直接满足永久变化点的渐近理论，不套用文献的“1.3 token 下界”。

预定 L=32：每个 span 在 [onset, min(onset+L, span_end)] 内匹配最早尚未匹配的告警；未检出记 delay=L。报告这种明确命名的 capped miss-penalized delay、仅命中条件下 delay 和 miss fraction。提前告警是误报，不可回填为负延迟命中。±1/±3 使用独立一对一匹配。所有阈值/episode reset（首次上穿，降至阈下后重新允许告警）由无标签 reference 固定。

不因 K=0、没有 H、没有显著 reanchor 而删除 token。measurement unavailable 同时报告 coverage 和全体评估（无可用分数的 token 设为最低告警优先级）；相同 score ties 用标准 AP 处理，不随机打破。source-cluster bootstrap 至少 1,000 次，比较配对差值 CI。

必要基线：negative margin、attention displacement、constraint displacement、position、同四格的 E/D/H 单变量与无交互 K ablation、GCN PCA-kNN（在相同 token/splits 上重新评估）。CHARM 是监督参照，不与本项目旧数字直接拼表。

支持性分析检验 K 与 continuous reanchor 的联合富集，控制 source/task、绝对位置、native margin 和 token 类型；这是事后诊断模型，不进入 detector training。报告非回看 onset 的比例。

决定门：K 必须对 onset 提供超过 margin 和单变量的增量，并在全流事件指标上保持；若仅 continuation 有效，主假设失败。

### 实验 2：带条件的成对事实干预

第一步在现有本地模型路径对应的 Llama-3.1-8B-Instruct（实际 revision 运行前核对）上预先构造 30 个受控事件，每类 10 个，不做统计普遍性声明。扩展 reanchor 现有六 margin 文本因子实验，覆盖 temporal、entity-binding、numeric/negation 三类无歧义事实，至少包含“前文完全正确，但当前适用条件改变”的样本。source world A/B、选项和字面 placebo 在模型评分/标签查看前冻结。交换 A/B，等长替换实体/数字，保持候选 tokenization 可比。

所有事件在承诺前 q 与承诺后分别测量；不以已知幻觉 onset 来决定 q。人类只验证 source edits 是否表达预定事实，不给 detector 训练 hallucination labels。使用自然生成的首个选择作为 outcome，forced-prefix 四格测干预响应。

关键 construct-validity 子实验保持事实词汇集合不变，仅交换绑定：world A=工作日9/周末10，world B=工作日10/周末9；控制组继续相同条件，实验组切换条件。自然生成或预定准确前缀在各自 world 中都必须事实正确。这样可区分旧词/数字重复、合法 relay 与条件绑定使用失败。由此只验证所构造关系的语义方向，不把该正确性赋给原生 runner-up。

在所有预定事件上记录 K；修复统一使用与主测量相同的全 H 分组，不加入 source-unit selector 或 top-k 路径选择器。

固定 source world \(w\) 的事实支持选项 \(o_w\) 与冲突选项 \(o_{\bar w}\)，独立语义结果量为：
\[
g_w(b)=z_{a=1,b}(o_w)-z_{a=1,b}(o_{\bar w}),\qquad
\Delta_{\mathrm{repair},w}=g_w(0.75)-g_w(1).
\]
q、选项及绑定世界由输入模板预定，不按 K 或生成正确性筛选。主受控测试限于实际 tokenizer 中的单 token、等 token 长度选项；前缀拼接不得改变 tokenization。多 token 选项另外报告序列 log-probability 分析，不混进此 q-local estimand。native 选择不属于两个选项时记 off-option，仍计算预定 g 并计入全部事件的 margin 分析；行为修复率中单独列 off-option，禁止自动视为正确或只删除它们。

未定向的固定 A-minus-B margin 在 world A/正确A 下应随修复上升，在 world B/正确B 下应下降；检验的是预定语义方向的变化，不要求每一个事件都跨越零或拥有相同效应大小。正常 relay 控制继续同一条件，风险事件切换条件；两类由模板预定，而非按实际是否发生错误命名。

placebo 固定为原生 history attenuation 实际施加的逐层 message 改变量的等范数随机方向替代：用预定 seeds 17/23/47 的正交符号/坐标变换保留每层改变量范数，在 q 的相同位置施加，但不沿 H 的原方向。报告 real-versus-placebo 的配对效果，以及正常事件正确率损害。随机方向的均值不能预设为零；用差值 CI 检验特异性，不以“placebo 不显著”当作其无效的证明。

30-event pilot 仅作数值/接口与方向探索；数值门通过不等于机制成立。正式构念检验使用另外冻结的 120 个 base facts（上述三类各 40，paired worlds 与 relay/switch 条件同属同一 fact cluster）。主 K、0.75 剂量、选项与测试集均在看该集结果前冻结。放行方向性机制解释的门为：风险组平均 \(\Delta_{\mathrm{repair}}\) 的 source-cluster bootstrap 95% CI 下界 >0；real-minus-placebo 及 risk-minus-normal-relay 的配对 CI 下界 >0；两个 source worlds 的定向效应均为正。任何未通过项都报告，不扩大事件集追到显著，也不替换主 K。通过后仍只支持这些关系上的构念效度，普遍性需自然结果及跨模型验证。

正式构念 bootstrap 的独立单位固定为 base_fact：先在一个 fact 内汇总 paired worlds、relay/switch 和三个 placebo seeds，再重采样 120 个 fact cluster；不能把这些分支或 seeds 当作独立实验样本。

这里需要防止循环验证：K 已要求 H=m11-m10>0，因此相同 m10 干预降低 observed margin 只是定义重述，不计为独立证据。主 K 固定使用 1/0 四格；独立机制 outcome 在预先固定、未用于主 K 的 1→0.75 剂量下评分，并使用 source 定义的完整语义选项（不是 native runner-up）、条件绑定 swap 和新的自由延续正确性。全部预定事件都进入该验证，不能只保留高 K 或事后能修复的样本；同时报告正常事件被破坏的比例。matched placebo 使用相同剂量/预算。后续 token 仅用于验证 outcome，不回填在线分数。

这才将模型内交互与可独立检查的事实使用关联起来。任一验证失败，只保留四格数值事实，不能把“信息传递但条件未传递”写成已发现的生成机制。

### 实验 3：删除检验与迁移

同样本比较 full K、只用 E（类似当前直接 context sensitivity）、只用 H、绝对值替代符号的 graph anomaly、无交互版本。对预定小幅门控、等 message-norm 的非证据/无关历史 placebo、真实证据 unit 与干扰 unit 分开验证，并在第二个模型家族重复。

最关键否证：若合法 relay 与幻觉首错在 matched-margin、token 类型和位置条件下同样频繁出现 K，或随机/无关路径 cuts 产生同样修复，或小幅门控方向不稳定，则不能称为 evidence-control mechanism。若效果只存在一个任务/一个模型，缩小适用范围，不用“普遍机制”。

## 文献定位

- [CHARM](https://arxiv.org/abs/2509.24770)：关系表征有监督可读性；本方案以固定有符号交互模体替代标签训练的图读出。
- [First Hallucination Tokens](https://arxiv.org/abs/2507.20836)：首错与 conditional tokens 必须分开；本方案增加决策前的受控路径测量。
- [CORTEX](https://arxiv.org/abs/2606.31033)：合法历史 relay 的最接近竞争者；本方案不把 source 缺敏感性当错误，而测“被 history 遮蔽的反对”。
- [ReDeEP](https://arxiv.org/html/2410.11414)、[TPA](https://arxiv.org/html/2512.07515)：source contribution 的机制启发；本方案保留 signed interaction，并以 source 单元干预校验语义。
- [CAD](https://aclanthology.org/2024.naacl-short.69/)：with/without context 的输出对比；本方案的差异是第二个 history-read 因子，检测条件性 source effect。
- [OPERA](https://arxiv.org/abs/2311.17911)：attention aggregation/over-trust 与回溯并非新发现；本方案针对文本 RAG 的 exact onset 和 finite signed interaction。
- [Voita et al. 的 source/target contribution](https://lena-voita.github.io/posts/source_target_contributions_to_nmt.html)：prefix over-reliance 早已有研究；新颖性不能放在“模型更信自己”这句话上。
- [CausalGaze](https://arxiv.org/abs/2604.11087)：需对照其监督 detector-gradient 图干预与本方案的 generator-margin 干预；“用了因果图”本身不新。
- [Quickest Detection of Hallucination Onset](https://arxiv.org/html/2606.12476)：已经提出 onset 的 false-alarm–delay 范式；本方案不宣称变化点任务新颖，借鉴其漏检与延迟联合评价，保留 exact first-token 目标。
- [Evidence Graph Consistency](https://arxiv.org/html/2606.06748)：其图表示的跨模型方向反转是反例，要求本方案逐模型报告；它不是 attention-graph 的直接否证。

以上只是相对于已读工作的具体差异，不构成完成的 novelty proof。DICD（2026，event extraction 的 context manipulation）也已检索到，其全文尚未取得，不对其未读实现作排他性断言。

## 与近邻方法的精确差异

| 方法 | 判别训练信号 | 实际操作/结果量 | 首错与时序 |
|---|---|---|---|
| CHARM | hallucination labels，GNN | attention/activation attributed graph → token/answer score | 主评估不是 exact onset |
| CORTEX | token labels，MLP | with/without reference hidden delta + historical contextual residual | post-hoc；双向 persistence smoothing |
| TPA | labels，XGBoost | 按来源分解 probability，POS 聚合 | 主要为回答级判别 |
| CausalGaze | labels，GNN/detector gradient | detector-guided attention-edge intervention | 不是本方案的无标签 generator-margin 四格 |
| ReDeEP | regression coefficients | context/parametric proxies 与模块干预 | 不以无标签首错为主要问题 |
| CAD | 无新增 detector 训练 | with/without context output contrast | 改变解码，不是首错定位器 |
| First Hallucination Tokens | 标签用于分析 | entropy/perplexity 按 span 位置分组 | onset 与 conditional 差异 |
| Quickest Detection | 监督 score/label dynamics | sequential score、ARL 与 delay | 首错变化点已经是先例 |
| 本方案 | 无标签 conditional reference | 同一生成器 q 处 E/H key-region 四格，固定候选 margin 的有符号混合差分 | y_t 缓冲后评分，不看 y_{t+1}；全流 onset |

最接近的是 CORTEX 的 historical reference influence 与 CAD 的 source contrast；新增差异集中在第二个受控 history-read 因子及其有符号交互。这个差异尚需实证证明有用，也可能在尚未取得全文的近邻工作中已有相近形式。

## 实施交接

1. reanchor 保留现有文本因子实验；新增 `query_read_intervention.py` 与独立四格输出 schema，native replay parity 先行。
2. graph 增加新 schema 的 reader，计算有符号量及无标签参考分布；沿用 frozen-score/label-join 隔离，扩展 full-stream onset 评估。
3. 先上 100 responses 的全 token 速度/数值 pilot，再决定确切样本量。不可从旧 attention traces 伪造 intervention margins；缺少模型状态时必须重新运行模型。

成本假设：一个 7–8B 模型、单卡 24–80GB，BF16，microbatch=1 起步。每 token 三个额外 query forward，共享原生 KV；共有 4 次分支计算；真实 wall-clock 取决于 eager attention、cache、Python/kernel launch，不能据此宣称 4 倍延迟。没有本机 benchmark 前不报告实测 GPU-hours。按 pilot 的 measured seconds/token 乘目标 token 数、模型数和干预分支数估算预算。

最高风险不是拟合器选择，而是该 signed motif 是否真的在自然首错中出现。先证明这一点，再考虑低秩近似或模型压缩。
