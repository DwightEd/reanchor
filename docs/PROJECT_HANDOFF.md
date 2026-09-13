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

# 项目交接：回看、约束归属与无监督图检测

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

## 最新任务：RAGTruth全量机制验证

用户已授权开始全量。已实现全部17790回答的8条件机制入口；6条真实GPU smoke和两次恢复见证均通过。全量已后台启动，PID 16928，输出reanchor/outputs/ragtruth_population_20260912，日志reanchor/runs/ragtruth_population_20260912.log。当前状态统一看`graph/docs/RAGTRUTH_POPULATION_RUN_STATUS_20260912.md`，冻结协议见同目录`RAGTRUTH_POPULATION_MECHANISM_PLAN_20260912.md`。此前O1–O3仅局部来源/窗口，不是全量；新图→风险主线仍未验证，O4关系归属对照仍未运行。

最新用户收敛：先用RAGTruth，优先补齐可运行graph主线。当前新方法仅完成属性图表示与机制干预，自动图→归属风险读出及批量评价未实现。不得用旧run_route_evaluation冒称新方法已收敛，也不要继续用机制案例替代主线交付。数据支持/预检/旧基线命令见graph/docs/LOCAL_DATASETS_AND_RUN_STATUS_20260912.md。

## 最新续研：O1–O3已完成

不要停留在“完整图已捕获”或重跑旧13案例。新实验已完成：ownership_factorial（68干预含sham）、ownership_multilayer（98条件）、ownership_source_layer（32输入世界+64单层X）。对应脚本scripts/run_ownership_{factorial,multilayer,source_layer}.sh，输出目录均为outputs/同名_20260912。

一个真实错误洋葱窗口跟随来源中取出前阶段的时长；四个同源seed的正确grill跟随烤制时长。全32层最终数值前query的X或端点交换足以翻转两个窗口，32个单层逐个均不充分。合法动作归属尚未自动识别；12改成14不叫修复，因为来源没有给取出后洋葱合法时长。候选差、完整分布及正常控制均在graph/docs/OWNERSHIP_MECHANISM_RESULTS_20260912.md。

已按结果两次追加实验、完成工程复审和原始文件哈希核验。所有旧未跟踪文件与结果保留，GPU作业已结束。下一项待做：数值不变、只交换动作/阶段归属，检验高维节点+拓扑是否能读出关系合法性；不是马上训练GNN或报告检测准确率。科学审稿仍REVIEW_UNAVAILABLE。

更新于 2026-09-12。本文记录已核对的项目进展，供新的 Codex 会话继续工作。
先读本文和 `mechanism_experiment.md`，再根据任务读算法；不要把历史设想当成当前实现。

## 研究目标与边界

目标是在不预先标注实体关系、不使用幻觉标签训练检测器的条件下，从冻结模型的
内部计算识别：模型何时重新读取远处证据，以及这次读取是否维持了当前对象/动作
所需的约束，能否在首错发生前或当步定位风险及其后续 span。

回看是需要重点检查的决策事件，正确和错误生成中都可能出现。不能把 attention
转向、attention 熵高、logits 熵高、source 依赖或稳定续写直接定义成幻觉。
attention 熵只描述读取分布分散；不能据此断言模型正在竞争语义路由。
logits 熵还受表述多样性、边界、难度等影响，错误选择后也可能很确定。

主线检测算法必须放在 **graph**；**reanchor** 用于机制实验、实例分析与算子验证。
用户要求代码直接、函数适度短、尽量复用真实中间状态；不堆叠未经验证的特征，
不添加复杂报告框架。保存原始逐项数字、必要设置和完成标记即可。

## 仓库与环境

远端研究根目录：`/share/home/tm902089733300000/a903202310/lys/research/`。

| 项目 | 分支 | 本次核对的版本与状态 |
| --- | --- | --- |
| reanchor | `agent/direct-sampling` | 本服务器 HEAD `196de4a`；本轮新增机制验证代码、结果和文档尚未提交 |
| graph | `agent/graph-structure-audit` | 本服务器已安全快进至 `4a904c4`；本轮归档、报警与上下文诊断尚未提交 |

两个项目分别对应 `https://github.com/DwightEd/reanchor.git` 和
`https://github.com/DwightEd/graph.git`。本地位于 `D:/projects/python_projects/research/`。

服务器 Python：`/share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python`。
模型：同一 `lys` 目录下 `models/Meta-Llama-3.1-8B-Instruct`；RAGTruth：
`data/RAGTruth/dataset`。已执行环境为 RTX 4090 24GB、torch 2.8.0+cu126、
transformers 4.57.1、bfloat16、eager attention。系统 Python 不含完整研究依赖。

服务器两个仓库都有未跟踪的旧实验/文件；本地 graph 有未跟踪的
`docs/ANNOTATION_AND_REANCHOR.md`。全部保留，不能用 reset/clean 清理。
远端 `reanchor/src/reanchor/` 是未跟踪旧目录，当前代码包是 `src/decoding/`。

graph 的 `docs/RESEARCH_STATUS.md` 已更新为本轮实际结果；原四边说明归档于
`docs/RESEARCH_STATUS_before_20260912.md`。不要恢复 control、四边评分或自编码训练。

## 2026-09-12 续研结果

最新澄清：用户要求“高维节点属性+关系拓扑”的整样本图，先解决真实正误窗口的
约束归属可辨问题；窗口分组不截断其他信息。已用graph/route_graph/sample_graph.py
和本库src/decoding/sample_graph_capture.py重捕00012/00013全层图，补旧采样缺失的
prompt内部边。原始图在outputs/attributed_samples_20260912，730/681节点、32层、
32头、4096维，约6.1GiB；独立原样执行退出0，2项新tiny模型测试通过。图表示
实现完成，归属可辨性/图增益/回看定位100%均未证明。详见graph/docs/
ATTRIBUTED_SAMPLE_GRAPH_20260912.md及REAL_WINDOW_CONSTRAINT_INVENTORY_20260912.md。

第二轮最新状态：用户要求研究连续错误影响和信息采纳，已在graph新增
`route_graph/adoption.py`，reanchor新增`src/decoding/adoption_probe.py`。完成1条
既有自然回答7查询的真实V/O、末层MLP后缀与历史访问干预；1%消息削弱的局部贡献
误差0.8%–3.4%，整组删除不能用线性近似代替。错误处可以低竞争，历史影响跨句
到Note；尚未实现或证明连续错误检测、自动影响终点、多层图/GNN。完整报告在
graph/docs/EVIDENCE_ADOPTION_RESULTS_20260912.md，方法与协议为同目录
EVIDENCE_ADOPTION_METHOD_20260912.md和EVIDENCE_ADOPTION_PLAN_20260912.md。
原始包`results/adoption_pilot_v2_20260912/`，曲线与数字
`results/adoption_analysis_v2_20260912/`。复现需新目录：
`bash scripts/run_adoption_probe.sh --output outputs/new_adoption_run`。v1和旧文件均保留。

本轮详细报告在 graph 的 `docs/METHOD_ITERATION_20260912.md`，后续先读该报告和本文件；`mechanism_experiment.md` 的原13案例与数字继续保留。

- 新算子实现于 graph/route_graph/context.py；reanchor/src/decoding/binding_validation.py 仅构造和执行机制条件，正确关系不进入读出。
- 修正候选上下文覆盖、对照有效集和随机置换后，完整复跑128条件：模型128/128正确，读出全覆盖；主层冲突前缀context 26/32、direct 27/32。不能报告幻觉检测准确率，未支持路径必要性。
- 正式复跑小型结果包：`results/binding_validation_v3_20260912/`，包含逐条件输入/数值、模板配对、settings、执行源码和哈希清单。运行 `OUTPUT_DIR=outputs/new_binding_run bash scripts/run_binding_validation.sh`；单模板可加 `LIMIT_TEMPLATES=1`。
- graph旧捕获恢复217完整回答、47,880 token；108条适用测试回答上residual首错AUROC 0.4361，entropy 0.8442；冻结训练阈值报警命中16/59和24/59。旧捕获缺原manifest，属于探索性基线。
- 当前两个研究主张均未获支持。停止将本版上下文读出接入默认检测，下一轮先验证同对象不同动作的归属及有符号消息贡献。自动回看定位、实际错误条件、等实际预算主比较和完整新捕获仍待验证。
- 研究审稿后端不可用，保留 pending Codex review 状态；工程测试或代码审查不等于科研结论通过。未跟踪旧文件保留，本轮没有push。

## 已完成的实现

reanchor 入口为 `main.py`，主要路径如下：

| 模块 | 已实现的工作 |
| --- | --- |
| `src/decoding/sampling.py` | 同次生成保存实际 token ID、attention、候选 logits、原生完整词表熵 |
| `fixed_prefix.py`、`decisions.py` | 固定已生成 token 前缀补取表征；检查候选、历史读取和同前缀一致性 |
| `reading_graph.py`、`route_readout.py` | 回看原型、严格层方向的多跳 source 路径、表征关系残差；尚未证明检测有效 |
| `entropy_controls.py` | `EntropyControls.run()`：首错与正常位置的同 token、同回答、分句起点匹配 |
| `interventions.py` | `CausalReadout.run()`：真实 attention 边切断、局部窗口、MLP 更新替换；`local_to_remote()`：局部向远处净转移 |
| `mechanism.py` | `MechanismExperiment.run()`：8 个归属交换模板及 5 个自然窗口，逐例保存，可复用完成案例 |

graph 已实现 `prepare -> extract -> detect -> evaluate`：
`RagtruthPreparer.run()` -> `FrozenGraphCapture.run()` -> `PathEncoder.encode()` ->
`RouteDetector.run()` -> `RouteEvaluator.run()`。算法位于 `route_graph/`。
固定路径算子比较真实端点与保持角色/段落/距离质量的条件置换期望，对候选状态
读取 observed/null/residual/signal，再用来源隔离的无标签参考评分。
这是实际图路径计算，但尚未证明比非图基线更有效；新回看和归属实验尚未接入主线。

## 已执行实验与可以支持的结论

结果已随 reanchor 代码推送：

- [同 token 对照数字](../results/entropy_controls_20260912/numbers.json)
- [机制对照数字](../results/mechanism_controls_20260912_v3/numbers.json)
- [自然窗口逐项数字](../results/mechanism_controls_20260912_v3/focus.csv)
- [完整实验协议及限制](mechanism_experiment.md)

### 1. 相同 token 不等于相同前缀

同模型、相同完整输入、相同执行协议，预测分布由前向计算决定。seed 影响采样选择，
温度/top-p 改变采样分布；它们不会让同一前向的原生 logits 因未来选对/选错而不同。
同一个输出 token 出现在不同上下文时，熵当然可以不同。

使用 110 条完整测试回答、61 个回答首错位置进行观察性匹配：

| 对照 | 匹配数 | 首错减正常平均熵（nats） | 首错熵更高 |
| --- | ---: | ---: | ---: |
| 相同 token | 58 | 0.4905 | 68.97% |
| 同回答、相同 token | 29 | 0.5700 | 62.07% |
| 相同 token、同为规则分句起点 | 36 | 0.2265 | 55.56% |

严格边界控制下的证据减弱，不能说首错普遍熵高。这三行不是同一批位置的嵌套控制。
此批是 **Llama-3.1 读取 Llama-2 生成的 RAGTruth 回答**，不能称为原生成模型机制。
标签不移到实体；冠词、空格、标点若属于标注起点就如实保留。

### 2. 归属对照证实了关系敏感性，未完成幻觉检测

8 个程序模板，每个 2 对象 × 2 归属世界，共 32 条件。保持生成前缀、prompt
长度和 token 多重集合，交换约束值的归属；正确值由构造事实独立确定。

| 条件 | 正确值为全词表 top-1 | 平均熵（nats） |
| --- | ---: | ---: |
| 完整读取 | 32/32 | 0.1093 |
| 切断所需事实直接边 | 2/32 | 1.9342 |
| 切断竞争事实直接边 | 32/32 | 0.1463 |
| 局部窗口 16 + 初始 4 token | 0/32 | 3.4659 |
| 局部窗口 64 + 初始 4 token | 32/32 | 0.0893 |
| 另一世界中间层 MLP 更新替换 | 32/32 | 0.1430 |

MLP 替换零基层 10–20 后，正确候选偏好平均由 8.9453 降到 6.2930，但选择未翻转。
这支持模型能使用归属关系、直接读取有选择性；不证明 MLP 单独决定路由。
完整条件全部正确，因此不能把这些数值报告成自然幻觉检测准确率。
32 个条件属于 8 个模板，统计时不能当成 32 个独立自然样本。

### 3. 自然错误仍依赖 source

烹饪样本 source 14375、seed 0、trace `00012`：正确时长为 10–14，错误洋葱
时长为 10–12。切断 source 直接边时，正确位置 t96 的 top-1 从 10 变 5、
t99 从 14 变 15；错误位置 t128 从 10 变 5、t131 从 12 变 15。
**有读取证据不等于约束用对对象**；“错时不再使用 source”不符合这些结果。

错误 t128 切断远处 response 后，熵从 0.4993 降到 0.1827，top-1 仍为 10。
历史影响不确定性，但删除这些连接没有修复错误，不能称为找到了错误语义路由。

局部向远处转移分数：错误 for→空格 t127 为 0.01815，正确 t95 为 0.00943
（窗口 16）。正常段落/条目转换也有更高峰；he→address 不是窗口最大值。
尚不能声称已可靠定位所有目标回看节点。手工窗口不是自动检测标签。

### 4. 数值验证与解释限制

13 个案例已全部完成，37 次重复/同世界替换检查完整 logits 最大误差为 0。
上一轮代码的 52 项测试全部通过（45 项主测试及 7 项 Git Bash 发布测试），
修改范围 Ruff 通过；真实 8B GPU 实验已执行。本文是文档变更，不新增测试结果。

- `margin` 是指定 logits 差，不是传播扰动；top-2 margin 与固定候选差也不同。
  不同自然位置候选对可能不同，不能直接横向比较其大小。
- 自然 attention 干预作用于整个生成前缀相应边，读数聚焦窗口；尚未将因果效应
  定位到单个回看节点。直接边切断保留间接中继，不等于删除事实。
- 本轮完整前向与原 KV-cache 采样存在执行差异，只在同协议下比较干预。
- 旧 routes/sampling 的熵用 bits，本轮机制实验用 nats；比较前统一单位。
- attention 权重不是完整信息贡献；value、输出投影、残差与 MLP 都可能改变读出。
- 表征关系残差不是归属真值；同义改写、软对应收缩、合法跨句组合也可能产生残差。

## 已有数据与长时间 graph 提取

远端 reanchor 的原始采样为 `outputs/samples_20260911_145421_235`，
补取状态为 `outputs/states_samples_20260911_145421_235`。大 NPZ 留在服务器，
随 Git 同步的是 `results/` 中的小型数值和生成文本，不要忽略或上传全部 outputs。

旧 graph 输出为 `outputs/routes_20260911_110359_256`。2026-09-12 上次审计：
计划 256 条、217 条完整、下一条部分写入，共 47,989 行，约 8.5 GB；缺完成
manifest 和检测评价结果，当时已无运行进程。不是已完成的主线评估。
旧版本逐 token 路径计算很慢；写入中间特征，但没有断点续跑，也不自动复用旧
688 条 attention 摘要。不要直接重跑并覆盖；先验证完整样本并设计逐样本复用。
110 条完整测试回答已用于上面的熵匹配，部分样本没有被伪装成完整回答。

## 接下来按此顺序收敛

1. **确定回看事件的定位协议。** 在共同可见、非特殊的历史 key 上，各 head
   计算 `min(远处质量增量的正部, 局部质量减量的正部)`，再汇总。它比一般 TV
   更接近局部转远处，但仍是候选算子。参考分布只用过去/无标签来源隔离数据；
   比较窗口 8/16/32，预先固定阈值规则和事件预算。独立标注小批正确/错误窗口，
   同时检查普通段落边界；报告定位率、误报和离首错距离，不用错误标签挑阈值。
   旧 W1+过去基线原型保留作对照，不能未经比较宣布新算子已胜出。
2. **检验归属能否被内部状态自动读出。** 构造数值不变、只换归属的难对照，
   加入改写、干扰对象、顺序变化及确实诱发错误的条件。自然样本只用于外部检查。
   对照中已知关系只用于机制判定，检测输入不带实体清单或正确关系。
   在定位窗口执行局部边/消息和匹配 sham 干预，分别检查 attention、历史中继、MLP。
   比较是否保留对象上下文的读出与数值词/总 attention 读出；目前没有通过验证的
   自动归属分数，不能将假设直接命名为“正确证据概率”。
3. **在 graph 集成一个最小结构读出。** 保留 token 身份、层方向以及
   source→历史→当前 query 的直接/多跳路径，读取回看后约束上下文是否维持。
   以已有条件路径残差为起点，先定义一项可证伪的关系读出；不要把所有诊断量拼接。
   必须设置边/端点置换、无路径聚合、只用 entropy/margin/位置等同输入对照。
   若保留拓扑没有稳定增益，就不宣称图必要；固定算子替代 AE 本身不构成创新。
4. **做来源隔离的首错评估。** 先实现按样本保存/复用和真实进度，利用已有数据，
   再补缺失状态。阈值和无监督参考仅来自 fit 来源；标签仅在分数冻结后接入。
   主指标聚焦固定报警预算下首错召回、提前量、事件误报，以及图相对非图增益；
   全 span token 分数为辅助，避免连续错误占比造成虚高。按 source 计算不确定性，
   通过后再跨模型验证；原生成模型机制与跨模型探测分别报告。

候选论文主线是“决策窗口中的约束关系如何经历史中继保持或失配，以及无标签图
读出能否识别这种失配”。目前已具备实验工具和部分机制证据，尚未完成机制闭环、
自动归属检测、图必要性或论文创新性证明。

## 新会话首先做什么

先确认 cwd、两个 Git 分支及未提交文件，阅读本文、`mechanism_experiment.md`、
`mechanism.py`、`interventions.py` 和 graph 的 `route_graph/operator.py`。
按用户本轮任务继续；不要默认重跑已完成的 13 个案例，不要重新开始泛泛讨论。
若需要复核现有实验，在远端 reanchor 根目录运行 `bash scripts/run_mechanism.sh`；
同设置会跳过完成案例，改变实验设置时显式指定新的 `OUTPUT_DIR`。

补充覆盖核验：128条件中仅62条同时将两种构造数值纳入原生top4（原始5、逆序6、干扰28、冲突23）。其余条件的“两个source候选”可能包含空格/标点，不能当作两种数值间的归属竞争；见新结果包 constructed_value_coverage.json。该核验仅用于冻结预测后的评价，不修改候选或分数。

最终执行版为v3：按条件标识派生独立可复现的置换seed，保存实际端点/候选移动比例；v2及更早输出仍保留。主层冲突前缀G/D/置换为26/27/26（各32条）；原生logits和G/D与v2逐项完全相同。工程第三轮修复审查通过，研究主张仍未获支持。
