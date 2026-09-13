# 项目交接：回看、约束归属与无监督图检测

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
