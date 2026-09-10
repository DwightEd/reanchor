# 研究方案：学习“事实在何种条件下成立”，而不是学习图是否罕见

状态：未验证的研究候选。2026-09-11 已添加最小 G0 代码与一键 pipeline（见 README）；真实 GPU 训练、自然泛化、图必要性和创新性尚未验证。G1 未实现。本页保留完整研究目标，实际首版范围以 docs/RUNBOOK.md 为准。

## Problem Anchor

- **Bottom-line problem**：在完全不使用幻觉标签训练检测器的前提下，定位 RAG 回答中第一个事实性错误 token，并检验该首错是否由一次“回看/重新锚定”过程中的证据约束失效所触发。
- **Must-solve bottleneck**：现有 attention/graph 分数主要擅长识别幻觉 span 已形成后的局部自回归延续，不能区分两种外观相似的历史依赖：合法的“证据经先前回答 token 继续传递”与错误 token 脱离原始证据后成为伪证据。
- **Non-goals**：不把任意远程注意力增加定义为幻觉；不以一个硬阈值的 reanchor 事件筛掉大部分样本；不靠 RAGTruth 标签、伪标签或监督神经分类器学习决策边界；不把 span continuation 的高分冒充首错检测。
- **Constraints**：复用现有生成模型与 RAGTruth traces；标签仅在所有分数冻结后用于评估；检测时不得看未来 token；机制验证允许在事后使用成对 source/prefix 干预，但不得用 gold hallucination label 选择内部节点或调参。
- **Success condition**：提出一个可识别、可干预、可证伪的首错机制；在 source-disjoint、source-balanced 的 onset-only 评估上，显著超过负 margin、位置、attention displacement 和静态图异常基线，并以有限干预证明该信号不是位置、token identity 或 span persistence 的替代变量。

## 问题边界

原问题仍是无幻觉标签的首错检测，以及回看时信息使用为何出错。撤回两个不能闭合的目标：拟合原模型的干预响应只能描述其行为；source-cloze 可能只学复制，不能自动转化成自然回答中的条件正确性。改成 source-derived relational self-supervision：学习当前实体/条件对应哪份证据。训练有自动构造的绑定目标，但没有真实或伪造的“幻觉/正常”分类标签。若用户要求连构造任务目标也禁止，则本路线不满足那种更强的 training-free 限制。

## 简洁性约束

一个主贡献候选：用两种不同的历史角色约束一个候选证据能量模型。图 G1 从主线删除，保留为必要性比较分支；先实现非图 G0。一个小可训练模型，冻结原 LLM；不再训练第二个完整语言模型，不加异常分数融合、change-point 网络、SAE、LLM judge 或新的检索器。图版本和非图版本互相竞争，不把更复杂版本默认命名为主线。

## Technical Gap 与待检验 insight

首错可以发生在此前每个 token 都正确时：旧事实在旧条件下正确，但条件已经切换，模型继续使用旧事实。候选机制是“事实被传递了，成立条件却没有随之更新”。随后错误 token 成为新的语言上下文，延续阶段才出现局部依赖增强。这解释了为何“远处历史→局部历史”的时间现象值得研究，但现有 attention 统计没有测到绑定，更没有证明这个解释。

受控例：source 说“北馆周一开放；南馆周四开放”。正确历史“北馆周一开放。至于南馆，它在周……”下一 token 应支持“四”，不是“一”。首错前没有错误历史。“一”和“四”都能回溯到 source，路径存在/源注意力高/数值复制都不能判别。真正要学的是事实和当前条件的兼容关系。

## Contribution Focus

- 主贡献候选：用 source 世界的绑定交换和正确历史的角色交换，学习 token 级条件证据支持能量，而不学习幻觉标签。
- 支持贡献候选：在冻结检测后，检验高风险首错是否对应原生成器中可干预的条件错绑定。
- 非贡献：GNN、graph rarity、copy、counterfactual equivariance 本身均非新概念；不承诺所有幻觉共享此机制，也不从 CHARM 的监督有效推出无监督必然有效。

## Proposed Method

### 系统接口

输入 source S、用户指令 U、已生成前缀 H=y_<t、实际刚生成的 token y_t；不读取 y_>t，也不把 y_t 输入用于预测它的上下文编码。冻结生成器 p_phi 提供候选和语言表示。模型输出 f_theta(v;S,U,H)：候选 v 在当前条件下的证据支持能量。f 是可训练非线性关系模型，不是预先统计的 distance/entropy/margin 四维向量。

最小版本采用 frozen contextual features + 两层候选条件化 cross-attention 支持头，hidden width 128；输入保留 source 和 prefix 的 token 身份、顺序与分区，不提前池化成几个特征。source 表示仅由 S 编码；地址/指代表示由 U+H 的 source-blind replay 编码，避免直接把 source-conditioned final hidden state 当答案旁路。source-blind 不保证没有语言先验或旧答案，训练角色对照仍然必要。

事实 value 从 source 表示读取；prefix 可控制读取谁和怎么组合，但不直接以 value residual 接到证据 readout。空 source 时 f 恒为 0。该限制只保证“无源不注入事实 value”，不保证语义真，也不能阻止 query gate 选择错误的 source。不能把它宣称成安全性定理。

采用共享候选 embedding 的兼容打分：b_t(v)=Read_theta(S,U,H,e_v)，f_theta(v)=dot(W e_v,b_t(v))。所有候选使用相同参数，没有每词独立的正确性标签参数。Read 是支持头，候选 embedding 是查询/匹配端，不得作为绕过 source 的 value。原 LLM 只作冻结表示和候选，不作训练事实裁判；检测器不修改原 LLM 的生成，也不承担完整语言生成。

### 图是否必要：两种实现，同一个目标

G0：直接 source cross-attention 支持头。prefix 的 source-blind 表示可以解析代词；允许多层组合。它是必须先跑的最小路线。

G1：同预算的稀疏、来源保留的 message-passing 支持头。source token 是根；prefix token 是 relay；当前预测位置是终点；边有 source-read、history-read、residual 三类。历史节点仅转发从 source 注入的证据向量，source ID 在多条路径汇合时保留，不能把同一个来源的多次转述当独立票数。对每个源单元 s：
B_v[s] = sum_{u in Pa(v)} alpha_theta(Q_v,Q_u,e_uv,e_candidate) W_type B_u[s]。
B 在 source 根初始化；其他节点零初始化；每个 source 的聚合归一化；最后才进行候选条件化 source 聚合。source token/句子边界是单位索引，不是人工标出的“正确事实节点”。

初版 G1 使用冻结 LLM 的因果 attention support 构造 prefix relay 候选边，每个 query/head top-16 后取并集，记录舍弃质量；source→当前预测节点保留全 source-read 候选，不允许原生成器看错来源就使支持模型永远无法看对。计算图不输入 y_t 及未来。第一版只取最后四个均匀层的 support，shadow head 共两层、width 128；调整宽度而非给 G1 额外深度以匹配 G0 参数/FLOPs。工程 pilot 必须比较未裁剪小图，不能把截断例静默剔除。

这是用于证据支持的 shadow graph，不是原生成器的完整因果图；冻结 routing 只能做结构候选。原始图边改变影响 f，证明的是支持头使用拓扑，不证明原生成器如何产生幻觉。

模型学习的是条件下的源选择、关系组合与 relay 规则，而不是重建邻接矩阵。若 G0 同目标、同数据、同预算达到同等效果，删除 G1。Transformer 可以表示同样的图，所以不存在“图在信息论上不可替代”的论证。图的合理性只能是相同预算下对未见关系组合/relay 深度更有效的归纳偏置。

### 自监督任务：不能只交换名字，必须交换绑定

从可精确求值的 records/programs 构造两世界。第一阶段用人为定义关系程序随机生成事实表与自然语言模板，词表可来自无标签 source，但不把任意自然段落的 LLM 抽取当金标准。若 RAGTruth Data2txt 原始结构化字段可用，另开记录驱动分层；当前尚未确认本地原始 records 可用，不能假定已经存在。自然 QA/summarization 不自动拥有金标准关系程序。

一个表含实体 e、条件 c、关系 r、值 v；至少六个对象，日期/值允许重复、独立随机。源世界交换当前目标与一个历史从未提及 donor 的 value，保持全 source 值多集不变、此前所有历史断言完全不变且为真。例如：
S_A: north→Mon, south→Thu, west→Fri；
S_B: north→Mon, south→Fri, west→Thu。
两个世界的历史逐字相同：“北馆周一开放。至于南馆，它在周……”；支持的 next-token 分别是“四”和“五”。其余对象和事实作为独立干扰，不能用“取不同于历史的唯一另一个值”求解。

二实体 north/south 互换只能展示关系与词值不同，不能直接作此 source-only 干预：否则历史里的 north 事实也被改掉，source 与 prefix 同时变化形成混杂。source-only 对照必须固定 H；另开 prefix-only 配对研究正确历史角色。程序直接给出当前条件支持的值/token，而非“该句为幻觉”的二元标签。

每个 source 世界交叉两类全部正确的历史：
1. 显式条件：当前问题已经指定 south，改写先前无关但正确的 north 事实/顺序，支持值应不变。迫使模型不把最近/最显著旧事实当新证据。
2. 指代必需：当前问题说“后者/该馆”，历史决定其所指是 north 还是 south，支持值应相应改变。迫使模型不能一概忽略历史。
历史必须由世界内 records 程序生成，并在该世界中保持正确；不靠人为植入一个错 token 教模型识别伪造负例。

每批同时含直接条件、条件切换、合法指代、1–3 跳组合；开发集按 source/模板/实体隔离；长 relay/新组合只用于 heldout 验证。训练答值词不直接出现在用于预测它的当前句位置；source 包含它是任务需要。数值、实体、顺序同时随机化以阻断位置/词频捷径。

### 一个目标，不堆叠特征损失

原生成器冻结。对于每个世界和历史，支持目标 v* 由程序计算。对候选 C：
q_theta(v|S,U,H,C) = softmax_C(f_theta(v;S,U,H))。
L_binding = mean_over_paired_worlds_and_histories[-log q_theta(v*)]。

配对平衡和参数共享施加绑定/角色约束，不再额外加 equivariance 正则。为避免只在值 token 训练却给所有词位打分，程序同时生成完整回答及其语义槽位。对确定与事实值无关的中性位置，训练同一个头给全部候选相同能量：先中心化 f_bar(v)=f(v)-mean_C f，再使用 L_neutral=mean_v f_bar(v)^2。训练批次等量采样 binding 与 neutral 位置，总目标 L=L_binding+L_neutral；两项分别按样本均值，权重固定为 1，不用 H 标签调参。没有新增 token 分类器，也不在自然测试中使用程序槽位门控。

“中性”由已知程序语义确定，不等于所有功能词：否定、比较、量词、单位和关系词都可能改变事实。这些语义关键位置必须纳入可求值 binding 目标；程序未覆盖的语义现象属于 OOD 风险，不能借 neutral 标签强制清零。这里只在构造任务中知道槽位，不声称在任意自然文档上已有这个 oracle。这个目标仍是程序监督，不能把它说成只观察自然图就能学习真伪。训练两世界所有答案均出现且均被正确支持；不把某个表面词固定成正/负例。f 直接学习当前条件下的候选支持分布，不再加 log p_phi 后训练、却把 f 单独读出。softmax(log p_phi+f) 仅保留为消融，不能默认等同于纯证据支持。训练监督应精确称为“无幻觉标签、程序可验证的 source-derived supervision（任务自监督）”，不是“无训练目标”，也不是从未标注自然生成中神奇学出真伪。

不存在足以保证上述任务覆盖自然 RAG 的分布假设。第一阶段能合法解释的范围仅是受控实体/条件/数值绑定；Data2txt 原始 records 可核验后可扩展为记录到文本。自然 QA/summarization 是未通过的外推终点，而不是既得能力。原始 Problem Anchor 不变：如果不能迁移到自然首错，则整个项目目标仍未完成，而不是悄悄把受控任务改名为目标完成。

初版训练 source 程序族规模上限 20k records、4种历史交叉；这是待运行配置，不是已运行结果。early stopping 只看 heldout binding CE、两类历史角色准确率，不能看 RAGTruth 的任何幻觉指标。泛化到自然文本是首要未解决风险，不能靠模板得分掩盖。

### Token 级推断、证据能量差与参考校准

候选 C_t = 原 LLM top-32 next-token ∪ 当前可见 source 中全部非 special token IDs（去重）∪ {实际 y_t}。不使用 gold 正确 token、实体 span 或幻觉标签，不作学得的候选检索器；source 集合可在整条回答复用。没有额外 top-N source 截断；计算按候选分块，不得静默裁剪候选或筛样本。所有比较模型使用相同候选集合，训练也用同一构造规则；训练目标是已知程序答案，不以测试 gold 补候选。

这个补充允许把原模型 top-32 外、但 source 内的正确 token 纳入竞争，避免完全继承原生成器的候选盲区。它仍不覆盖所有正确改写/推导、不同 tokenization 或 source 外知识，不能称保证检测。候选数变大可能增加正常词位极值误报，需在同候选集下比较 neutral objective 和各基线；没有自然 token 的正确候选 oracle 时不能从 H span 标签虚构 coverage。程序任务直接算覆盖；自然评估需要单独冻结、仅作评估的 source-supported 修正候选核验，不能回流训练或调参。多 token 值训练按 teacher-forced next token 求和，检测仅用已见历史；值首个分歧 token 才能出现条件区分，分词边界单列。

原始证据支持差：
D_t = max_{v in C_t} f_theta(v;S,U,H) - f_theta(y_t;S,U,H)。
这是学习到的证据能量差，不是原生成器不确定性。q 与 f 的训练/读出语义一致；q 的 top1 或 -log q 只作预注册删除对照，不在 test 上选优融合。

D 大表示候选集中存在一个相对更被当前证据支持的候选；不等于逻辑证明 y_t 为假。最大值候选可能只是同义词/语法替代，这是必须在正常词位上量化的误报风险。无 source 时 D=0 表示没有此类绑定证据，不表示回答正确；报告这一盲区。

全 source 候选扩大了极值误报的机会，因此不直接跨候选规模比较裸 max。预注册按候选数 |C|（<128、128–511、512–2047、≥2048）、source 长度（<1024、1024–4095、≥4096）与 task 分层，在 source-disjoint、无 H 标签的 calibration 集上计算 D 的加权经验 CDF。每个 source 总权重相同，得到部署主分数 R_t=F_ref(D_t | nuisance stratum)。某层不足 50 个独立 source 时，按固定顺序先合并 source 长度、再合并候选数、最后跨 task；回退层和覆盖率都输出。原始 D 与校准 R 同时保存和报告；没有基于 H 标签选择二者或校准边界。

这是混合未标注参考分布的分位数校准，不是已知正常 null，更不是保证真实 FPR 的 conformal p-value。比较模型接受同一候选集与同样的 nuisance 校准协议；评估同时报告未校准结果，避免校准本身制造结构收益。长度/C/task 只用于消除测量尺度差，不是替代完整关系模型的三个主特征。若校准后的自然首错增益消失，就不能报告原 max 的收益为事实性规律。

即使直接训练 f，也必须用去 source、随机 source、只用 p margin 对照检查其是否真正利用条件证据；不可把 score 高默认解释成原生成机制。阈值只可由未标注 calibration 预定分位数冻结，不声称保证给定 FPR。

### 为什么这不是“几个特征”

最终风险是一个标量不构成问题；关键是它由候选与整个 source-prefix 关系计算得出，参数可学习未见条件的选择、组合与传播规则。可用保留节点/词值/边权统计而交换对应关系的测试检验它是否真的区分结构。它仍是候选模型，不保证学到的结构具有人类语义；非图模型也可能学到这些规则，不能把“非图”误等同“手工特征”。

## 与最相近文献的边界

- Binding ID 与 Mixing Mechanisms 已经研究实体绑定和不同检索机制；这里不能宣称首先提出“绑定错误”。
- AT2 已经从真实 ablation 学响应；这不是本方法的 truth criterion。
- CAGE (Walker/Ewetz, 2512.15663) 已有 source-history-output 路径归因；来源传播图本身不新。
- Twin Worlds (2608.28018) 已有实体置换后的回答等变与拒答；这里拟区分的是“换名字”与“同词集合换关系”，以及正确历史在显式条件/指代条件下应遵守不同的变换规则。是否足以构成贡献仍待对照，不能声称已确立新颖性。
- SiGHT 已有合成监督的关系图幻觉检测；这里不合成/预测 H 标签，但程序任务也是额外监督假设，必须公开承认。
- CAGE (Yan et al., 2607.24236) 是另一篇同缩写的 citation-support 图工作，不能与归因 CAGE 混淆；本轮只核验其摘要。
- 没有保证对任意问题、任意 source 无监督识别真伪的定理。source-relative support 与世界真实分开。

## 三块最小验证

1. **先证明目标与结构**：同一训练目标下比较 G0、G1、固定四统计/MLP、纯序列小 Transformer、去 source。受控测试保持词值集合和度数/质量统计，改变绑定；同时要求显式条件时忽略无关历史、指代时利用必要历史。成对世界合取准确率是主指标；未见组合/relay 深度是迁移指标。单纯 real-vs-rewire accuracy 不算图必要。G1 只有同时通过成对绑定合取准确率、未见 relay/关系组合和自然首错三项增益才升为主线；差异以 source cluster bootstrap 置信区间报告，任一缺失则图不是已确立贡献。只观察 shadow 图边变化导致分数变化不足以保留图。
2. **自然首错泛化**：旧 688 source 仅 discovery。新 source-disjoint fit/calibration/test；标签在分数/候选/配置冻结后 join。主终点明确为每条回答的第一次事实错误，而不是把同回答后续所有 span onset 作为独立主样本。单列两个不可混用的任务：(a) 在每个 H span onset 与正常 token 上的旧 cohort 对齐分数，用于复核旧表；(b) 全流第一次错误的监测，首错前所有正常 token 纳入误报，首响应错误不得排除，延迟限制 L=32，未检测记 miss，首错后的报警不算 exact-onset hit。全流 score 表保留所有 token，包括 continuation；辅助报告 each-span onset AP 与全流 exact-first-onset AP，并声明后者将 later errors/continuations 当非目标而非正常事实标签。另报 false alarms/1000 pre-first-error token、每 source 平均精确召回和报警时间。以 source 为 bootstrap 单位，多个回答不能当独立 source。原负 margin .7078 AUROC/.0163 AP 来自排除 continuation 的旧 onset cohort，只能按相同分母比较，不能与全流 AP 混比。控制 task、绝对位置、copy overlap/词频；错误集中在数值实体时收窄声明。
3. **原模型机制而非 shadow 自证**：对独立、程序可验证的 source 条件交换，保留此前历史正确，观察原生成器 A/B logit margin 与再生成结果。按冻结候选路径规则选择 head/edge，做局部路径干预、等幅随机/位置匹配 placebo、正常 relay 对照；干预只改变 detector score 不算通过。对 natural onset 的节点规则在查看标签前冻结。回看指标用 W=16 的远历史注意力质量作为连续 moderator，不设硬门。若原生成器的选择变化不符合 binding-recovery 预测，只能保留检测器结果，不能宣称回看首错机制。第二模型/任务复现是范围要求，未执行不能写普遍。

## Failure Modes、预算与交付边界

最大风险依次是程序任务到自然文本迁移、正常词位误报和大候选集极值偏差、source 外推导/改写候选缺失、影子模型与原生成机制不一致、图增益为零。每项都可能终止路线，不用更多融合特征补救。矛盾 source、抽象推理、source 之外的真实知识不在当前机制的可保证范围。

先用 100 个冻结前缀做工程吞吐/内存与候选覆盖 pilot；source-blind replay 有额外大模型计算，不能称零成本或实时。没有测到硬件吞吐前不捏造 GPU 小时。现有 traces 若缺少 source/prefix hidden states 则需要重放；本轮没有授权/启动 GPU 任务。

当前方法是可审阅、可证伪的候选，尚未通过三个 gate；不把同族模型审阅评分当作创新或有效性的实验依据。

## 最小闭合条件

这是一种 source-grounded auxiliary verifier 候选，不是已经学到原生幻觉电路的模型。三件事分别需要证据：自然首错检测有效；显式图优于同预算非图实现；原生成器确实发生可干预的条件错绑定。任何一项不能借另外一项的成功替代。原始项目目标仍未完成。

“source 监督”公开包含程序可验证任务目标；没有真实或合成的 hallucination labels，不等于没有监督假设。如果严格限定为仅学习未标注注意力图分布、排除所有 source 任务目标，本方法不满足该更强限制；目前也没有证据保证这种严格设定能识别所需事实性。

## 主要参考链接

- [How do Language Models Bind Entities in Context?](https://arxiv.org/abs/2310.17191)
- [Mixing Mechanisms](https://arxiv.org/abs/2510.06182)
- [Learning to Attribute with Attention / AT2](https://arxiv.org/abs/2504.13752)
- [Attribution CAGE](https://arxiv.org/abs/2512.15663)
- [Twin Worlds](https://arxiv.org/abs/2608.28018)
- [SiGHT](https://proceedings.mlr.press/v300/chen26d.html)

更完整的文献方法、监督来源和阅读限制见 graph_necessity_literature.md；引用这些论文不代表它们支持本候选已经有效。
