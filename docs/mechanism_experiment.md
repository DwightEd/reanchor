## A2-v1完整负结果与v2受控迭代（2026-09-13T15:46:56.948789+08:00）

v1训练/自然预测/评价/48来源实际擦除均完成。固定graph差分source-balanced AUROC全词0.505927、首错后0.512073，base NLL分别0.535477/0.516545；36答/6源/4733词是反复使用的开发集，6答来源进source-only预训练，30source-unseen同样无改善。结果/实现/完整分母见 graph/docs/GROUNDED_GRAPH_V1_RESULTS_20260913.md。

固定query擦来源X后坐标指针概率降0.399276，生成gain drop为−0.000657；原始gain本就−0.001492，因此不能把full擦除的正drop叫正收益被移除。source-copy任务中H_full已经含完整source，图生成分支可能被绕过。当前v2保持240来源、划分、9220锚点、参数、10epoch和loss，只换为真实all-source-erased H_empty查询及解码基底，原X保持；新特征/训练/预测/评价代码已工程闭合，fresh-doc见证正在启动真实capture，来源依赖诊断待审查。尚无v2数值。

当前方案与可执行指令：graph/docs/GROUNDED_GRAPH_RESTORATION_V2_20260913.md、GROUNDED_GRAPH_RESTORATION_V2_RUN_20260913.md。source48 anchor正增益+固定query真实X擦除降益为机制门控；全词和strictpostfirst两个差分分别报告，不翻方向。准确回看、自然适用约束、原模型路由/聚合、连续范围仍未解决。外部Codex MCP不可用，科学审计不称PASS；保留所有旧文件/未跟踪文件和分支。

以下均为保留的历史状态；当前以最新段落及实际manifest为准。

## 当前联合图模型与实际状态（2026-09-13T15:07:49.713783+08:00）

主线已写成单一 GroundedGraphAdapter：完整source图的4096维特征，两步128维消息传递，真实预测前h[t−1]查询，门控残差经冻结LMhead。固定自然检测score=logp_base−logp_adapter。当前方法/执行入口为 graph/docs/CURRENT_METHOD_20260913.md、GROUNDED_GRAPH_MODEL_20260913.md 和 GROUNDED_GRAPH_TRAIN_RUN_20260913.md。

Qwen来源模板实验因确证错误owner停止：6/240源，19/48机械可用但有段落归属错误，已落盘7298forward，SIGINT130；中断批额外forward未知。所有原始文件保留，不用于训练。替代的来源原文坐标重建已完成240源、9220锚点，192/48全文SHA隔离；训练特征135519token/240forward，自然36答5840token/4733词/36forward均exit0。自然开发6源中13717在source-only训练内，单独报告重合，不称独立test。

训练/自然预测/评价/实际payload擦除代码均已实现并完成工程检查。训练执行见证正在启动，尚无新自然检测数值；不称方法收敛。旧SourceRel95.4%仍只属弱字段检索；全量17790的post-first弱/负结果保留。准确回看、自然适用约束、原LLM路由vs聚合、连续影响范围仍未闭合。

以下为保留的历史检查点；当前状态以上述入口和真实manifest为准。

## 最新核验与方法修正（2026-09-13T14:00:27.286414+08:00）

完整来源库存已实际完成并独立审计：2965来源/17790引用，1241059组件，unknown3049、长文本字段4083、mapping failure0；CPU339.217秒，执行与独立审计均exit0。报告 graph/refine-logs/constraint_inventory_doc_witness_20260913.md。SourceRel与post-first的完整负结果保持，不能将字段重建95.4%当自然归属准确率。

用户指出反复局部实验的死胡同后，进一步明确：冻结Qwen推理图只作为外部语义参照，不足以回答内部信息是否能判定归属。新 docs/REASONED_GRAPH_METHOD_20260913.md 和 next_iteration/reasoned_graph{,_runner}.py 已写，工程审查尚未完成，未prepare/未GPU执行；其当前提示图仅field/record/context粗粒度，完整组件图仍保存在库存，不能混称。当前还在审查联合来源指针与grounded token重建的轻量内部图模型目标，未实现/训练，不宣称已选定有效结构。

当前没有GPU实验在运行。准确回看、适用归属、路由/聚合、连续范围四项仍未闭合。保留所有旧文件与分支，以下运行状态均为历史。

---

## 最新进展：SourceRel自然迁移失败已定位；修完整候选图（2026-09-13T13:26:02+08:00）

SourceRel-Mini完整运行结束：883来源、38541辅助4096维向量、20epochs。来源验证top1=95.40%，但same-field/other-record仅77.27%（TFIDF100%）；60错中55错在此组。自然旧36回答/1619槽位全部保留，622Data2txt给候选、997域外；两来源均已在source_train。实际出现地点→name高置信、地址数字候选只有评分、overall rating→review rating、每天营业→单个weekday；长评论和unknown-valued owner还被旧候选池排除。**完整源字段重建训练完成不等于自然约束归属成功，当前头停止推广/追加训练**。具体表格、反例与checkpoint见 graph/docs/SOURCEREL_RESULTS_20260913.md。

全量post-first CPU分析也结束，无新模型调用：17790响应，严格首错后810750 tokens/148814 errors，旧all/first的504组指标对完全复核。post-first source-balanced组AUROC按token加权均值：entropy0.5480、history-smallJS0.5127、source-smallJS0.5533、history-minus-source-support0.5222；不是pooled指标，也不支持连续错误检测已解决。报告 graph/refine-logs/population_postfirst_results_review_20260913.md。

当前实现 next_iteration/constraint_inventory.py：全部未知值字段/长文本、精确decoded-to-raw组件、field/context/record包含关系和七天key库存；QA/Summary也保留原文节点。工程复核中，完整2965来源CPU编译待执行。只有源拓扑完整性，不冒称语义图真值；已删除假定正确的q_relation/q_owner/q_condition桶、all-slot遮蔽和同句同事件硬绑定。自动query scorer仍在细化，不运行尚未设计好的新GPU模型。

主线graph、机制reanchor；回看/归属/路由与聚合/连续范围四项仍未闭合。全部旧代码和结果、两分支、未跟踪文件保留。以下“当前”与待跑项为历史。

---

## 当前进展：全量机制完成，约束归属模型开始完整训练阶段（2026-09-13T12:54:03+08:00）

RAGTruth population 已 COMPLETE：17790/17790，failed0，原进程退出、GPU释放。完整评价覆盖2903263 tokens、156478 error tokens及36个task×generator×split组。按token数加权的组内AUROC均值：all entropy0.577、source JS0.446、history JS0.577；首错及之前 entropy0.799、source JS0.623、history JS0.449。这不是合并AUROC；没有独立post-first指标。saved-token observer不一致率24.17%，不能当原生成器机制或新检测器效果。核验见 graph/refine-logs/ragtruth_full_evaluation_review_20260913.md。

typed-hours全17790行CPU编译已完成，6198条Data2txt仅2个合法B/A（train1/test1）。唯一train窗口完成90次原生forward，自动选择query707–710/layers16–31联合组；observer原本已偏正确A，origin半剂量反向、有效控制不足，强证书0。不能称准确回看、幻觉修复或稳定采纳。结果见 graph/docs/TYPED_HOURS_RESULTS_20260913.md、TYPED_NATIVE_RESULTS_20260913.md。旧record-root元数据实际指向name anchor，已单独追加370来源更正 outputs/typed_hours_owner_metadata_correction_20260913.json；原源值、token对齐和干预数值未改，未重跑GPU。

当前主线结构见 graph/docs/CURRENT_METHOD_20260913.md。SourceRel-Mini数据/小头/特征/训练CLI已实现：official-train Data2txt共883来源，source train/validation=720/163、7064 queries；完整长度预检38541视图/3954619 tokens/max1200，无截断。仅来源字段指针重建训练，无幻觉标签。词面validation top1=80.60%、top5=99.00%，是必须比较的强基线。固定两来源特征21forward、2epoch真实头sanity已完成；接下来完整特征编码和20epoch训练，自然36回答候选迁移已写、工程复核中。**尚无完整训练结果，也未解决自然归属、路由/聚合或连续错误范围**。

后续以source留出与自然迁移结果决定模型结构，不继续相同有限标签批次，不靠增加营业时间模板、降低门限或更深GNN宣称成功。两个仓库分支及所有未跟踪文件保留。下面各“当前”、PID和待运行项属于历史。

---

## 当前修订：语义入口负结果已确证，转向可核验关系与原生定位（2026-09-13T11:27:21+08:00）

surface owner 完整36回答的合格B/A仍为0，native0。独立实现复核未发现必修bug；新增10个固定请求的CPU完整词表读出诊断完成30次forward，标签总概率0.998963–0.9999998，自然输出全部为字母+EOS，CPU/GPU类别10/10一致。这排除了这些请求的格式/归一化解释，**没有修复语义判断**。停止原样重跑Qwen有限标签入口。结果与边界见 graph/docs/SURFACE_OWNER_V1_RESULTS_20260913.md。

已精读PropSegmEnt、Claimify、GraphFC、LLM2Vec和Qwen3Embedding/Reranker的方法部分，笔记 graph/refine-logs/fact_scope_primary_methods_20260913.md。新增fact_scope与fact_scope_mask：重叠/非连续事实成员、共享变量的三值约束求解、坐标级目标值遮蔽，独立23项CPU测试通过；它们只在给定关系假设下求解，不是自动语义真值模块。

当前先实现Data2txt明确字段关系的 typed_hours 分支，用七天量词约束与字段坐标核验营业时间，同时保持同句WiFi约束。真实9273可自动构造6 PM→4 PM单值对照，原始token前缀和全部7天来源端点已对齐，无标签。36条开发清单的工程检查中只有1个合格对照，不能当作广泛覆盖或回看成功。新完整17790行CPU编译器已写，独立工程审查进行中；尚未启动全量编译或typed native GPU。QA/Summary通用语义入口仍未解决。

下一步冻结完整清单的编译与拒绝分母，再在预定train候选上检验自动query×layer集合、正确证据V消息介导和有条件路由，不做“关系影响决策”简单实验。主线尚未实证收敛；100%回看、聚合失效区分及连续错误范围均未证明。

原population 16281/17790，failed0，running，PID162289；未暂停或修改其冻结文件。两个仓库分支和全部未跟踪文件保留。以下状态是历史。

---

## 当前：surface owner 完整负结果，先解决事实范围与语义读出（2026-09-13T10:42:48.137386+08:00）

surface_owner_v1 已完整完成 B36/C36，child/wrapper 均 exit0。2377 个辅助特征文档、595 次特征前向；1619 个目标、44 个进入门限、145 个候选对进行有限验证。**合格正误对照0，native干预0，没有标签评价**。2199 次 reader 前向、2344 次返回（145 缓存命中），无 reader_error。输出 graph/outputs/surface_owner_v1_20260913，独立执行报告 graph/refine-logs/surface_owner_v1_doc_witness_20260913.md。

不能把有候选当成归属成功。程序按顺序给出的第一拒绝为：完整edited base124、语义保留16、value-only3、归属2。重新检查全部五个门，非互斥失败为：完整base124、语义保留125、value-only145、归属51。故不能将124都解释成多事实父句问题；词面候选存在动词/日期等不合适替换，有限首token读出还出现明显语义不一致。独立只读故障复核正在进行。下一步用可重叠、非连续原文token集合表示事实及共享约束，先定义/验证可适用关系，不再仅增加候选或调低阈值后重跑。

当前仍未证明准确自动回看定位、内部信息足够辨别当前约束归属、错误路由与聚合/采纳失效区分、连续错误范围。主线graph尚未实证收敛；reanchor全量是observer机制测量，不能当新方法效果。原population恢复PID162289，独立见证14883→14961、失败0，旧14883份manifest、4元数据和55份surface代码及快照不变。最新读取 15020/17790，状态running。保留全部旧结果、未跟踪文件与分支。以下“当前”和PID都是历史快照。

---

## 当前修订：停止设计无效的 soft graph v1；修复槽位与归属（2026-09-13T09:44:17.892527+08:00）

soft_graph_v1 已主动早停，实际落盘 A36/B36/C27/D0；没有 merge、没有新标签评价。核对 child PID155856 后用 pidfd 发送 SIGINT，audit exit -2、wrapper exit1，旧结果完整保留。wrapper 已恢复 population PID159091；恢复时仍 loading_model、13669/17790，实际进度增长由独立见证继续核验，不能把启动等同成功。

停止原因是结构目标未成立：完整A/B的426个claim、1704条来源边中，source-pointer prepare仅10个可构造候选，且出现残缺单词、语法破坏与未保留时间范围。1394条要求整体事件改写、222条subject关闭、54条response owner未解析、23条source owner未解析、1条quoted关闭。10个候选尚未有限语义验证，不能视为正确对照。原soft版还把地址/不完整anchor当边界，把bag级风险赋给整句词；target-only D不解决这些缺陷。B完成72次特征前向，native干预0。

下一版删除自由SRL坐标硬前置：完整sentence base → 原文surface slot → 带父句/字段记录的source occurrence → 遮住目标值的owner匹配 → 固定候选的有限归属/局部关系/保留条件验证 → 合格单槽位B/A接回原生定位、origin、paired routing与MLP。masked重编码是辅助表示，不冒充原生成轨迹；定位仍使用原始token化和完整正误事件差。详细设计见 graph/refine-logs/surface_slot_owner_matcher_revision_20260913.md。

已实现CPU：source_pointer_contrast 22项检查且独立C0/R0；graph_boundaries 15项检查且独立C0/R0；surface_graph 13项检查，尚待独立复核。新surface_owner编码/匹配实现中，未跑GPU。当前尚未证明准确自动定位、适用约束归属或错误采纳区分；没有100%保证，主线尚未收敛，继续按这些具体失败迭代。保留所有未跟踪文件与冻结版本。

## 最新运行：soft graph v1 完整36回答（2026-09-13T08:41:31.225750+08:00）

新的主线检测器已冻结并由独立文档见证启动，child PID 155856，wrapper session67861，输出 graph/outputs/soft_graph_v1_20260913。settings digest 9833b851b479f6ffe53d4a601a947d53fc13eb4d161946daf5f2e9e33fa32cd8，文件SHA fb13d3dabe6333c15f00ae722993148018cecf8c949010f8b01172788753bec4，47份代码冻结。72项相关CPU检查通过，独立runner/pipeline复核9通过，Critical0/Required0。当前没有新效果评价。

执行A结构图、B高维特征/候选、C有限关系判断、D每span原始输入介导/定位、merge全词四版本评分。完整结构见 graph/docs/SOFT_GRAPH_METHOD_20260913.md，运行指令见 graph/docs/SOFT_GRAPH_V1_RUN_20260913.md。原全量暂停于13669/17790、失败0，由同一独占wrapper完成后恢复。未修改任何旧冻结代码/结果，未跟踪文件均保留。

本版仅以观测回答span的完整logP测依赖，不把它冒充正误决策差。严格A/B层仍单独保留，本批不声称已区分错误路由和正确路由后聚合失效。标签仅在全部预测完成后评价。v3全部弃权/native0的完整负结果已归档；以下状态均按时间视为历史。

---

## 最新状态：v3 完整负结果与主线架构调整（2026-09-13T08:05:20.112574+08:00）

v3 的 A/B/C/D/merge 各36/36完成；4733词、260标注错误词，语义与机制覆盖0、错误召回0、实际native forwards 0。368问题中271 uncertain、97 invalid。独立完整性审计通过；这不支持自动回看、约束归属、路由/采纳或连续错误机制主张。结果见 graph/docs/NATIVE_AUDIT_V3_RESULTS_20260913.md。三个旧批次完整保留，停止仅修格式后重跑。

主线设计改为：全span固定能量图检测器 + 严格因果证据覆盖层。高维残差特征/角色与事件拓扑提出竞争来源；冻结Qwen有限选项估计完整来源及局部关系；正向且对照调整的native依赖调制图风险；仅明确前向reuse传播，纠正/引用/新话题阻断。Unknown向0.5收缩；局部未提及不推出全来源缺失。不训练幻觉二分类，不按RAGTruth标注调权重。四个同reader对照必须同批输出。审查规格见 graph/refine-logs/soft_graph_fixed_energy_review_20260913.md；当前正在编码，尚无新检测结果。

原全量已恢复并验证PID 153564，本次读取12802/17790、失败0。原wrapper/native均exit0；12574份旧manifest、4份冻结文件及8份代码哈希保持。实时状态以population progress.json为准；以下“当前”均为历史。

---

## 运行中快照：v3 与下一版结构工作（2026-09-13T07:09:44.864444+08:00）

v3 完整36条开发回答仍在执行，当前阶段 A，该阶段已完成 18/36，native PID 147669。最近已发布的18份 A 产物包含148个问题：110不确定、38无效，2327词中语义评分覆盖0，尚无有效风险对照供给。这里是无标注的运行中诊断，不能作为36条最终评价；尚未加入本版评价标签。v1/v2的完整负结果继续保留。

原 RAGTruth 全量机制任务暂停在12574/17790、失败0，由独占调度器在v3结束后恢复；不同时加载另一GPU作业。v3冻结代码、参数、阈值未改。运行指令与评价入口见 graph/docs/NATIVE_AUDIT_V3_RUN_20260913.md。

下一版结构工作已完成原文指针/字段图的CPU模块 route_graph/source_event_graph.py，26项检查通过，独立工程审查必修项已关闭。这只证明原文坐标、角色指针与字段结构完整性，不证明事实归属。outputs/source_event_inventory_20260913 已生成36回答/6来源清单，其中2个Data2txt来源字段图为42/51节点；47个清单文件哈希复核一致，无模型调用、无标签。

针对当前候选供给失败，已详细阅读 SRLScore、FENICE、SIRG、CORTEX、CausalGaze、HalluSpan、SiGHT、FGWEA；审查在 refine-logs/unsupervised_alignment_structural_review_20260913.md。保持不以幻觉标签训练的锚点，正在收窄高维角色候选与事件联合约束的匹配接口；不把相似度/null质量当作真假，不把监督或合成幻觉二分类训练换成主线。自动回看定位、适用约束、路由/采纳与连续错误范围仍未实证收敛。

以下旧“当前”和PID均为对应时刻历史。

---

## 当前更新：v2 完成负结果，v3 接口修订中（2026-09-13T06:30:50.935275+08:00）

36 条同源开发回答的 v2 已完整执行并评价：119 个问题中 93 个无效、26 个不确定，语义覆盖 0，错误召回 0，native forward 0。故自动回看、路由与采纳、连续错误边仍未获得本批次的实证验证。详情见 graph/docs/NATIVE_AUDIT_V2_RESULTS_20260913.md（在 graph 仓库中为 docs/ 下同名文件）。

v3 将纯角色 JSON 问题改为保留原句的单槽位 mask，未解析条件也保留为显式条件；重复文本按关系角色与原始坐标绑定。来源端不接收未遮蔽陈述或候选答案，解析仅接受完整 JSON 根并保留原始边界。所有请求的严格解析、边界恢复与失败进入冻结结果。47 项相关 CPU 检查通过；独立复审未结束，GPU 尚未启动。v1/v2 输出与负结果均保留，未改变因果阈值或使用标注挑样本。

原全量运行 PID 145947，12488/17790，失败 0；以下旧 PID/“当前运行”按历史记录解读，实时状态以 population progress.json 为准。研究任务仍在继续，软件可运行不等于方法有效。

---

## 最新状态：v1自然批次失败，v2原子槽位集成中（2026-09-13T05:36:30.868925+08:00）

36/36条v1已完成A/B/C/D/merge并独立评价：4733词、260标注错误词，语义覆盖287词（6.06%），错误召回0，native实际forward0。172问题中103无效、56不确定、13被reader判支持，无有效风险窗口。不能称回看/约束归属/路由检测已验证或方法收敛。详见 graph/docs/NATIVE_AUDIT_V1_RESULTS_20260913.md；原结果和executed_code快照保留。

主线graph正在集成v2：原子事件角色、精确跨度/指代绑定、确定性隐藏槽位提问、逐角色证据表核对；保持全上下文与弃权分母，重新跑同36条开发清单，无标签调阈值。真实自然效果待跑。旧全量reanchor任务恢复PID141861并继续产出；原wrapper因120秒验真超时exit1的事实保留，后续恢复已有独立证据。新的900秒等待修复已复审。

以下状态与PID均按时间保留为历史。

---

## 最新状态：完整自然批次启动预检（2026-09-13T04:47:53+08:00）

graph 的 A/B/C/D/merge 主线实现已完成部署前复核，native validation、pipeline、scheduler、evaluation 均无未关闭 Required。完整CPU回归最新独立结果87 passed，随后N position-only不得计入internal覆盖的专项回归通过；这些不代表方法有效性。

冻结执行输出：graph/outputs/native_audit_v1_20260913；36条输入；settings digest 9e5c7ae743f7157c08ffd5a1c2e93672e009c4e265408e99c99d5f217b80f85f。独立文档执行代理已被派发完整批次调用，当前是启动预检阶段，尚无新机制结果。运行文档：graph/docs/NATIVE_AUDIT_RUN_20260913.md；中文模型结构说明：graph/docs/NATIVE_METHOD_MODEL_20260913.md。

关键落实：native搜索不看reader证据候选；输入来源与key位置分开验证；连续错误绑定到先前具体答案槽位；同角色原始输入对照按答案长度冻结；正确恢复单独控制；N必须两个模板均通过位置与来源验证；所有无法判定与未覆盖词保留分母，标注只在预测全量冻结后加入。仍未验证：准确自动回看定位、当前约束归属可靠性、路由/MLP机制覆盖、连续错误范围与语义层之外的增益。

---

# 回看窗口与约束归属：本轮可执行实验

## 当前：原生消息图完整自然验证，部署前（20260913_042134）

以下旧“当前”、O6和旧PID均为历史。O6未运行，不再排队；不再证明已知的“关系影响决策”。

方法规格完成五轮真实GPT-5.5审查：6.22→6.90→7.12→7.79→8.30；规格READY_TO_IMPLEMENT，研究仍REVISE/实证NOT_READY。A/Qwen独立来源问答与逐条件引用、B/Llama候选盲的消息组搜索、C/不看native效应的固定候选核对、D/完整事件E/V/输入介导/跨角色/MLP交互、CPU全词覆盖与历史边已编码，入口route_graph.audit_runner。部署前工程复审尚在闭合，未启动新GPU任务。

新自然清单已冻结：outputs/native_audit_design_20260913，官方train每任务2来源×6生成器，共36回答，未读labels/quality，排除14375。36条原token对齐通过，最大输入1757。语义读者的预测不是评价真值；图教师是否提供增量仍需自然数据验证。

支持的纠正控制单独处理：当前E/此前C且问题全部条件等价，才以此前错误答案构造明确错误alternative；历史依赖正值不自动继承幻觉。N仅对预选content位置及其origin做第二withholding模板，N paired/MLP暂不测、不发完整证书。原文分组保留跨span长边。

执行上限320真实forward/Claim、1280/response，含B/A两分支、donor、逐组sham；最多4个不同Claim，有合格纠正控制时3风险+1控制。该保守cap不要求消耗完，失败不回填候选。它是机制教师，不是已验证的高速全量检测器。

原冻结population仍运行：10635/17790，失败0，PID20383。未修改其冻结代码、结果或未跟踪文件。新独占调度与恢复脚本正在复审。近期父进程23项相关CPU检查通过，独立审查员全graph tests 78 passed；正在修复的编排另须专项复验，不能将测试数写成方法效果。

未验证：来源问答准确率、自动位置/介导覆盖、连续错误vs纠正、图相对同读者非图增益、实际成本。RAGTruth没有唯一回看点或因果影响终点真值，不承诺100%定位。下一步是完整36条自然A–D验证，并按最大的实际失败环节继续迭代。


## 2026-09-13 方法优先修订（当前；以下旧“下一项O6”已失效）

核对时间 2026-09-13T02:36:21.408455+08:00。用户明确停止简单关系实验，要求先解决方法结构。O6单来源缺失约束行为实验已取消作为当前下一项，没有启动。现有全量任务继续：7219/17790，失败0，PID 20383；它仍是冻结observer机制测量，不能当新检测器结果。

本轮已按research-lit/research-refine精读QA事实核查、图事实核查、内部归因、无监督检测及分段方法，论文与固定代码笔记位于graph/refine-logs/*architecture_lit_20260913.md、grounding_qa_lit_20260913.md。两个完整草案接受真实GPT-5.5协作方法审查，6.22/6.90分，均REVISE；Codex MCP不可用，未冒充该后端或科学有效性审计。

正在修订的完整架构为 graph/refine-logs/architecture_round2_20260913.md：冻结来源语义锚点与真实计算图分开、完整陈述对比、同目标控制、自动联合query组搜索、区分事实区间与条件影响区间。**外部reader锚定并不证明内部图独立识别归属**；跨角色路由、聚合覆盖与自然证书覆盖仍是必须补齐的问题，未宣称收敛。

仅新增graph/route_graph/causal_contrast.py计算内核，7项CPU测试通过；它不是端到端新方法。正在进行结构复审及独立代码审查，没有新GPU实验、没有删除未跟踪文件或变更全量冻结代码。


## 2026-09-13 当前进展：O4/O5 已完成，全量继续

核查时间 2026-09-13T01:46:22+08:00：RAGTruth **5514/17790**，失败 **0**，状态 `running`，当前 PID **20383**。实时值以 `reanchor/outputs/ragtruth_population_20260912/progress.json` 为准。

本轮按 monitor-experiment、analyze-results、experiment-plan、research-refine、run-experiment 和 code-review-and-quality 工作流继续迭代；科学审稿后端不可用，未生成通过裁决。

O4 已完成 4 个关系世界、160 干预；数字/位置固定，仅交换 Before/After，模型 4/4 随关系选择。最终 query 的 X（来源内容）0/4 翻转，E（来源内连接）4/4；冻结 N/A/G 读出仅 3/4、2/4、2/4，未形成有效自动归属检测。
O5 据此追加并完成 167 条件：132 个单 query E 仅 t131 能翻转，32 个单层均不能；去掉 t131、此前全部 query 联合 E 仍翻转。因此没有唯一必需回看点的证据；来源读取正增量 top8 对该单点充分集合召回 0/1。

O4/O5 都已结束，当前 GPU 正在跑原冻结全量，不是继续重复局部案例。全量当前日志 `reanchor/runs/ragtruth_population_resume_relation_routing_20260913.log`。两次短调度均已安全恢复；O5 独立见证 5057→5101 新增 44 条，旧结果/settings/冻结代码未变。

4077 条已发布回答的独立 CPU 评价已完成，其中 4 个 QA 生成器组各 989 条完整。它是部分结果；没有全量 COMPLETE。Llama2-7B 官方 QA test 首错 entropy AUROC 0.8989、全部 token 0.5996；history JS 分别 0.3055/0.6058，评价总体不同，不能称为检测器提升。GPT4 仅 1 个首错。QA test 已用于诊断，后续改法须另设未查看的确认性评估。

尚未验证项按优先级：缺失适用约束/开始补具体时长的决策 → 独立来源归属读出 → 自动单点/联合回看候选 → 延续/纠正/换题的有符号影响 → 保留跨 span 边的自适应分组 → 图相对同输入非图增益。O6 目前仅设计，未执行。

主线仍在 graph，reanchor 负责机制执行。完整属性图和干预算子可运行；**新图→自动风险主线尚未收敛**。不把 O4 构造阶段规则的成功当成原自然错误修复；原来源没有给当前洋葱阶段唯一时长。

当前权威材料：graph/docs/RELATION_MECHANISM_RESULTS_20260913.md（数字/解释）、MECHANISM_ITERATION_20260913.md（待验证清单/部分全量）、MISSING_CONSTRAINT_PLAN_20260913.md（下一项）；graph/refine-logs/FINAL_PROPOSAL.md 和 EXPERIMENT_PLAN.md 已按负结果重写。


## 历史记录（截至 2026-09-12，旧 PID 和未运行状态不代表当前）

## 当前执行任务：RAGTruth全部17790回答

通用8条件机制扫描已通过6条GPU小批次及两次恢复验证；17790条全量已后台启动，PID 16928。执行状态、输出位置和完成数以`graph/docs/RAGTRUTH_POPULATION_RUN_STATUS_20260912.md`为准，协议见`RAGTRUTH_POPULATION_MECHANISM_PLAN_20260912.md`。这批是本地Llama3.1重放机制测量，尚不提供自动合法归属结论。

## 最新：来源/历史、跨层采纳与单层穷举已实际执行

O1 68条件、O2 98条件、O3 32输入世界+64单层干预均完成。错误洋葱数值依赖取出前阶段的来源约束，正确grill依赖烤制来源；全层最终query X、动态端点交换可翻转，任一单层X均不能。逐节点干预支持这两个数值端点处的跨层读取，不证明唯一最早回看点或自动100%定位。完整当前结果见graph/docs/OWNERSHIP_MECHANISM_RESULTS_20260912.md；协议OWNERSHIP_FACTORIAL_PLAN、OWNERSHIP_MULTILAYER_PLAN、OWNERSHIP_SOURCE_LAYER_PLAN（均20260912）。原始输出保存在outputs，分析表/曲线在results/ownership_analysis_20260912与ownership_source_layer_analysis_v2_20260912。下文的原13案例和早期负结果仍保留。

主线检测器属于 graph；本仓库实现机制验证。回看是待检测的计算事件，
不能直接标为幻觉。熵是完整原生词表分布的熵，不施加 temperature/top-p。

## 固定的执行顺序

1. `match-entropy`：读取已完成的原始特征，验证回答摘要和 token 连续性。
   在测试来源上比较首错与正常位置：相同 token、同回答相同 token、
   相同 token 且都位于独立规则识别的句/分句起点。匹配只使用 token、
   位置和边界，不使用熵。严格起点对照要求整个正常分句无幻觉标注；
   不在规则边界的首错必须计入未覆盖数，不能移标到实体或句首。
2. `mechanism`：程序构造两个对象、两个约束值，交换归属但保持词汇集合、
   生成前缀、目标候选、token 长度不变。两个对象分别询问，防止永远选
   第一个对象或第一个数值也通过实验。每次保存完整候选分布的熵、两个
   指定候选的 logit 差、逐层 attention/MLP 更新的词表读出。
3. 同一输入重复前向，直接比较完整 logits；采样 seed 不参与前向。
   正确值由构造事实决定；强制观察错误候选不等于模型实际生成幻觉。
4. 分别切断当前回答到正确事实、竞争事实的直接 attention 边，并运行
   全前缀的真实局部窗口 attention。所有条件重新前向，不能裁剪缓存
   attention 后称作干预。直接边切断保留间接中继，不能解释为删除事实。
5. 将另一个归属世界的中间层 MLP 更新替换进当前世界，与原世界的
   同位置更新比较。记录候选偏好是否朝另一个世界移动。MLP 的普通
   logit-lens 差值只是读出；跨世界替换才是对应的干预。
6. 对已有自然样本的正确时长、错误洋葱时长、头饰和正常转换窗口，
   固定保存的 token ID，记录局部到远处转移及之后的 span。分别切断
   source、远处 response 与最近 response 的直接边，检查原生候选
   分布如何变化。案例只用于验证，不能用其手工位置调回看阈值。

## 回看量化与图的接口

对相同的可见历史 key，按当前步固定远近划分。各 head 分别计算远处
attention 增量和局部 attention 减量，取两者正部的较小值，再平均。
这测量局部向远处的净转移，不把所有分布变化都叫回看。source 与远处
response 都属于远处；持续特殊 token 不参与。保留连续分数，不用
当前案例选择阈值；分别检查窗口 8、16、32 的定位稳定性。

图保留 token 身份和层方向。检测目标是：回看之后，约束上下文是否
经直接或历史中继维持到后续回答。归属交换时，读出应跟随关系改变；
只跟随数值词、attention 总量或稳定续写不能算成功。

## 判定规则

- 同 token 对照差异消失：原始熵优势可能包含词形/边界混杂。
- 正常的多候选证据检索也可熵高；不能把熵高定义成错误路由。
- 归属交换使候选偏好正确翻转：模型状态对该关系敏感。
- 遮挡正确事实与遮挡竞争事实效果不同：直接读取有内容选择性。
- MLP 替换改变归属偏好：该层段 MLP 更新参与本对照的关系选择。
- 自然窗口中只有错误之后才异常：属于错误延续信号，不能声称提前检测。
- 合成事实没有实际错误生成时，不能报告其幻觉检测准确率。

输出只保留设置、逐项数值和检查结果。每个完整案例立即保存，可按相同
设置跳过已完成案例。实验结果不能直接替代 graph 的来源独立检测评估。

## 2026-09-12 已执行结果

数值保存在 [同 token 对照](../results/entropy_controls_20260912/numbers.json) 和
[机制对照](../results/mechanism_controls_20260912_v3/numbers.json)。
13 个案例全部完成；32 个同世界 MLP 替换检查和 5 个自然窗口的完整 logits
重复检查最大误差均为 0。GPU 为 RTX 4090，torch 2.8.0、transformers 4.57.1，
Llama-3.1-8B-Instruct，bfloat16，eager attention。归属世界的生成前缀和
prompt token 数量、词汇多重集合相同；仅交换两个值的归属。

### 同 token 的观察性对照

使用已完整提取的 110 条测试回答，61 个回答首错位置。探测模型为 Llama-3.1，
回答原生成模型为 Llama-2，不能当作原生成模型的不确定性实验。

| 对照 | 可匹配首错 | 首错减正常的平均熵，nats | 首错更高的比例 |
| --- | ---: | ---: | ---: |
| 同 token | 58 | 0.4905 | 68.97% |
| 同回答、同 token | 29 | 0.5700 | 62.07% |
| 同 token、同为规则分句起点 | 36 | 0.2265 | 55.56% |

严格边界匹配降低了可覆盖样本数，也减弱了平均差异。不能从原始 AUROC
直接声称控制词形、边界后仍普遍熵高。后两行控制不同因素，不能视为
在同一批位置上连续增加控制。规则分句边界不等同人工语义 span 边界；
跨样本对照尚未控制全部难度、位置和历史内容。

### 归属交换与因果干预

| 条件 | 正确值为全词表 top-1 | 平均熵，nats | 正确值相对另一个值的平均 logit 差 |
| --- | ---: | ---: | ---: |
| 完整读取 | 32/32 | 0.1093 | 8.9453 |
| 切断所需事实的直接边 | 2/32 | 1.9342 | -2.4395 |
| 切断竞争事实的直接边 | 32/32 | 0.1463 | 10.3887 |
| 全前缀局部窗口 16 + 初始 4 token | 0/32 | 3.4659 | 0.0020 |
| 全前缀局部窗口 64 + 初始 4 token | 32/32 | 0.0893 | 8.7617 |
| 替换另一归属世界的中间层 MLP 更新 | 32/32 | 0.1430 | 6.2930 |

这里有 8 个构造模板，每个模板 2 个对象 × 2 个归属世界，并非 32 个独立
自然样本。完整读取没有产生错误值；这些数值验证归属敏感性和路由依赖，
不能报告为自然幻觉检测准确率。MLP 替换层为零基编号 10–20；它削弱
正确偏好但未翻转最终选择，不能归因为 MLP 单独决定了路由。

### 自然样本：错误也依赖 source

正确烤制时长：切断 source 的直接连接后，t96 的 top-1 从 10 变为 5，
t99 从 14 变为 15。错误洋葱时长也出现 t128 从 10 变为 5、t131 从 12
变为 15。故“错误时已经不使用 source”不符合这些窗口的干预结果。

错误 t128：切断远处 response 连接后，top-1 仍为 10，而熵从 0.4993
降到 0.1827 nats。远处历史影响了此处的不确定性，但删除它没有纠正
错误值。这个结果不能独立证明被删除的历史就是错误语义路由。

所有干预固定实际 token 序列，使用同一种完整前向执行方式。完整前向
与原 KV-cache 采样不是逐位一致的数值协议，干预只与本轮完整前向基线
比较。自然窗口的 candidate_margin 使用该位置预先固定的两个候选；
候选对在不同位置可以不同，不能直接横向比较该 margin 的大小。
自然样本的连接干预作用于整个已生成前缀的相应边，数值读取聚焦指定
窗口；因此其效应尚不能归因到某一个回看节点。MLP 更新替换则只作用
于显式读取的 query 位置和指定层段。

回看分数确实在正常条目转换和部分组织回答的位置增大。错误 for→空格
t127 的分数为 0.01815，正确对应 t95 为 0.00943；但全窗口还有更高的
列表/段落边界，he→address 也不是窗口最大值。当前结果不支持声称已
完成可靠的回看定位，更不支持把高分直接作为幻觉分数。

### 本轮收敛

模型能表示和使用对象—约束归属；检索缺失、候选竞争与错误归属需要
分别识别。自然错误中仍存在 source 依赖，且错误生成可以变得很确定。
因此 graph 的主线读出必须保留约束上下文与当前对象之间的关系及其
后续维持；单独的回看、熵、稳定路径或 source 依赖均不足以完成判定。

## 2026-09-12 新读出的独立验证

原13案例的协议、输出与结论不变。另以 graph 的 source-context 算子完成128个固定前缀条件（8模板、2对象、2世界、4布局）；修正版均全覆盖、模型128/128正确。主层冲突前缀读出选对26/32，direct 27/32；无实际错误预测，不能声称错误检测或图必要性。新的结果、逐条件输入及执行代码在 [results/binding_validation_v3_20260912](../results/binding_validation_v3_20260912/numbers.json)，设计、自然基线和失败分析见 graph/docs/METHOD_ITERATION_20260912.md。

复现：`OUTPUT_DIR=outputs/new_binding_run bash scripts/run_binding_validation.sh`；该入口默认128条件，单模板加 `LIMIT_TEMPLATES=1` 为16条件。旧目录的代码设置与本版不同，须使用新目录。

补充覆盖核验：128条件中仅62条同时将两种构造数值纳入原生top4（原始5、逆序6、干扰28、冲突23）。其余条件的“两个source候选”可能包含空格/标点，不能当作两种数值间的归属竞争；见新结果包 constructed_value_coverage.json。该核验仅用于冻结预测后的评价，不修改候选或分数。

最终执行版为v3：按条件标识派生独立可复现的置换seed，保存实际端点/候选移动比例；v2及更早输出仍保留。主层冲突前缀G/D/置换为26/27/26（各32条）；原生logits和G/D与v2逐项完全相同。工程第三轮修复审查通过，研究主张仍未获支持。

## 2026-09-12 有符号消息与后续影响探针

后续新问题、预声明查询和v1后数值迭代记录在
graph/docs/EVIDENCE_ADOPTION_PLAN_20260912.md。主算子graph/route_graph/adoption.py，
驱动src/decoding/adoption_probe.py；没有改动原13案例代码或结果。

既有00012自然样本，t96/99/128/131/145/166/180共7查询：真实末层V/O闭合相对误差
0.170%–0.241%；加入末层MLP的fp32局部导数对1%消息削弱的主候选差估计误差
0.8%–3.4%，整组删除最大误差200.8%。记录大幅非线性而不当作精确分解。

访问干预固定历史[119,133)，等长对照[85,99)，均从预测t134首次生效；固定后文。
两者生效前logit误差0，sham全程0。生效后的全词表JS均值分别0.014000/0.004564，
均改变3次原生top1。错误片段干预的最大影响在t157的Note（JS0.320459），因此
错误跨度、因果依赖后代与陈述转换须分别评价。没有给出自动边界或检测准确率。

[原始结果](../results/adoption_pilot_v2_20260912/summary.json)、
[逐项数字](../results/adoption_analysis_v2_20260912/numbers.json)、
[影响曲线](../results/adoption_analysis_v2_20260912/influence.svg)。所有新代码/结果未提交。
