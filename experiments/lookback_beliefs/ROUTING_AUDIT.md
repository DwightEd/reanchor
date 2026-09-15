# 约束可读，但 QK 选错了吗？

当前入口：`python -m experiments.lookback_beliefs.routing_audit --resume`。
这是**机制审计，不是已验证的无监督幻觉检测器**。原 LLM 冻结；小探针和候选子空间
只使用构造记录的语义标签，不读自然幻觉标签。探针正确不等于指针被模型实际使用。

## 本次替换了什么

删除旧自然实验的整段限定/载荷 V 切断、单一 carrier 恢复及符号判读入口；它们证明过
来源和历史依赖，但不能识别错误绑定。旧文件和完整结果可在 `6184c25` 中追溯，
服务器 outputs/samples/states 不删除。原论文的 `run.py --mode scan/dcm`、数据、
patching、子空间和测试保留，避免把已取得的指针/载荷基准实验丢掉。

新代码仅四个核心文件：
- `routing_data.py`：构造正交条件与四个既有自然窗口。
- `routing_engine.py`：一个物理 head、一个 query 的 Q/K/V 分离干预与原模型后续计算。
- `routing_math.py`：有限信道信息、参考集子空间、独立语义读出、精确 QK 对比分解。
- `routing_audit.py`：采集、冻结、留出比较、有限干预及输出。

## 要区分的失败，不预先宣布哪种存在

| 观察组合 | 可继续追查的解释 |
|---|---|
| 独立参考读出不能恢复所需绑定 | 约束可能未形成/传到这里，也可能读出器失败；不能证明信息不存在 |
| 绑定可读，Q 投影后的同标签读出弱 | Q 投影可用性下降的线索，不是容量损失定理 |
| Q 随条件变化，但对竞争 K 的区分方向分量小 | 条件变化没有控制这两个地址的相对选择 |
| 路由按正确关系变化，内容写入变化，但最终候选仍错 | 继续检查 OV 后的整合和读出 |
| 只在晚层改变答案，缺少预定语义组合行为 | 不称为绑定/指针修复 |

“正确指针已经编码”不是输入假设。`hypothesis_counts.json` 统计未参与拟合的样本中：
正确候选、绑定选错、词表 top1 不在候选中的回答格式/门控失败，以及独立读出正确但
实际候选选择错误。若没有错误，就报告0；不能用故意改坏 hidden 制造原生幻觉。

## A. 受控标尺：不是再次证明替换状态会改输出

每组两条记录，稳定身份 A/B，查询需要同时匹配 item 与 phase。

```
Record A: item-X; phase: early; value: amber.
Record B: item-X; phase: late; value: violet.
Question: item-X 在哪个 phase 的值？
```

独立改变：C=查询 early/late，B=phase 与记录身份的对应，O=两条记录的先后顺序。
八个世界全部保留：正确记录为 `C XOR B`。默认16组，前8组仅作 reference、后8组
作 heldout，payload 词汇库也分开。8条其他 item 的干扰记录默认保留。模型自然答错和
答对均报告，不筛选 both-correct。另各一份内容更换和保持绑定的问法改写，共10世界/组。
固定回答前缀 `The requested value is` 让候选与 source 的 leading-space token 一致；
仍单列 top1 不在候选集合的情形，不把它偷换成绑定失败。

预先规定的局部操作：

| 操作 | 改什么 | 留住什么 | 机制假设所预测的结果（并非保证） |
|---|---|---|---|
| Q | 取 C-flip donor 的查询 | 接收世界 K/V | 接收材料另一记录的值 |
| K | 取 B-flip donor 的两个地址 | 接收世界 Q/V | 接收材料另一记录的值 |
| QK | 同时取上述两种 donor | 接收世界 V | 两次翻转相抵，原记录的值 |
| V | 取新内容 donor 的值 | QK 和选择关系 | donor 的新内容，不是选择翻转 |
| 同绑定 Q | 问法改写但目标记录不变 | 接收 K/V | 仍取接收材料的原记录 |

检查的是这些**组合规则**能否在留出上下文成立，而不是 logits 任意变化。
只改一个 head 时完全可能不改变 top1，因此同时保存候选选择、QK log-odds、真实
post-WO 差向量、完整词表 logits；零效应可来自冗余，不直接证明该 head 无用。

## B. 子空间不是由错误标签或节点重构训练出来的

对每层 normalized residual x，参考组中收集使目标从 A 转到 B 的配对差分 D。
对改变记录顺序、以及同时翻转 C/B 但保持目标身份的差分 N，取其前2个方向作为
显式 nuisance 候选。对 `D - D P_N` 做 SVD，固定前4个有效方向 U。
源位置的 address 候选子空间则用 B 配对差分，去除 O 方向。

这些是**候选语义差分子空间**，不是先验已经确认的 pointer/address。
有用语义也可能与 nuisance 重叠，所以空空间/失败要如实保留。原始 Q/K 完整替换仍
作为对照，不把 SVD 的能量与维度命名为知识bit数。

Q子空间干预为：

```
delta_x = P_U (x_donor - x_base)
delta_q = R_query W_Q delta_x
```

除了该变化，还做相同范数随机方向、同绑定 donor 对照；随机对照默认单次，
不能据此宣称得到显著性p值。对固定接收端 K，取地址判别子空间
`D_K = span{k_j-k_0}`，分开只注入 delta_q 的 D_K 分量和正交分量。
正交分量不改变两个指定地址的 log-odds 是代数性质，不是研究发现；它仍可能改变
对其他 token 的注意力和输出。真正的结果是 U 中的变化有多少落入有效判别方向，
这种变化是否正确支配留出任务选择，而不是只有范数变大。

默认只干预 reference 组条件路由信息量最大的1个物理 layer/head。
所有 head 的原始测量都保存，heldout/natural 不参与选择。`--channels`可预先扩大范围。
不按测试错误挑头、不在结果出来后选最好层、不合并 heads 后声称找到了真实通路。

另外替换该读取层前一层同 query 的 attention 写入和 MLP 写入，观察后续 Q、地址
对比和输出。它只定位最近一个上游写入环节，不冒称已完成全模型电路发现。

## C. 信息论在这里具体做什么

1. **已知的实验变量** C 均衡二选一，在固定 group/B/O 下有1bit。
2. 路由 J 是“读到指定值位置A、位置B、其他位置”的完整三类信道：
   `I(C;J | group,B,O) = JS(p(J|C=0),p(J|C=1))`。
   不把两个候选的质量单独归一化。一个几乎完全不看这两处的 head 不应虚显高信息。
3. 输出同样记录两候选及词表其他 token 的分布。高MI不意味着选择正确：完全反向
   的二元信道仍有1bit。因此同时报告正确地址质量、实际候选正确率和门控失败。
4. reference-only 的 PCA+逻辑读出测 C、目标记录和 source phase 的可读性；heldout 上
   `1 - cross_entropy / ln(2)` 是该读出给出的**经验变分下界估计**，允许为负。
   它不是高维 hidden 的精确MI，不是已经优化到全体读出函数的 V-information。
5. residual、Q、K读出采用相同构造语义目标。下界与路由MI之差只作并列诊断，
   **不能叫作丢失的事实bit数或率失真保证**。有限样本、读出能力、条件空间均有差别。

`cluster_interval` 对完整构造组重采样，不把同组八个世界或同一个token跨层当独立样本。
自然案例只有两个 source，无资格估计稳定的自然真假分布或总体AUROC。

核心匹配对比是：

```
Lambda = q dot (k_B-k_A) / sqrt(d)
Delta_Lambda = Delta_q dot Delta_k_base / sqrt(d)
             + q_base dot Delta_(k_B-k_A) / sqrt(d)
             + Delta_q dot Delta_(k_B-k_A) / sqrt(d)
```

Q/K donor raw向量在**接收者位置**施加RoPE。分解使用实际干预文件中的接收坐标，
不是混用不同长度prompt的旋转后向量；双向差异和交互项保留，不强行线性化。

## D. 自然原例：不再只在“12”已经准备补完时观察

复用原 samples 精确 token IDs，读取 source+seed，不生成新回答，不重套原模板。
原状态缓存只验证身份，不把最后归一化状态误贴回 block。旧汇总没有新的 Q/K 数据，
因此需要新前向；每个世界一次 incremental-KV 回放可同时记录三个位置。

| 原例 | 三个预测位置（目标词尚未输入） | 边界 |
|---|---|---|
| cooking_onion /14375 seed0 | for、10、12 | 原洋葱步骤无唯一正确时长，不指定1–2或10–14为答案 |
| cooking_grill_control /同答 | for、10、14 | 这条局部烤制关系正确，不认证整篇菜谱 |
| headdress /14315 seed2 | and、headdress、gold | 比较泛指和专属主体的适用范围 |
| clothes_lengths_scope_control /seed3 | with、knee、ankle | 不自动认定范围省略为假 |

每例三个世界：原始、明确改变原文绑定/权限、保持语义的改写对照。
烹饪的反事实把10–12从较早步骤移到洋葱步骤；头饰把Only the Inca改为All the Incas。
这些是**改变局部陈述适用性**的受控 donor，不是发现了原问题正确答案。
原回答后缀 token 全部保持；自然反事实单独标记，不能算原RAGTruth数据。

`natural_pair_geometry.csv` 对所有head、所有三位置记录这些改变如何影响 x、Q、
地址判别方向及原词支持；语义保持对照同样记录。两个指定anchor只取辨别性词的
最后子词（12/14、headdress/pins、knee/ankle），不是把整条事实都压在一个词上。
完整注意力行保留，其他所有 token 是明确的 outside。该设计可能遗漏真正地址。

自然干预不指定 truth target，不将简单ordinal-record探针当成已经验证的自然绑定读出。
“正确指针已编码但没用”必须先在独立留出标尺成立，再评估向自然语义的迁移；本代码
不会从两个案例自动生成这句话。自然对照只能提供具体匹配/使用环节的证据。

## 执行和成本

```
python -m experiments.lookback_beliefs.routing_audit \
  --samples outputs/samples_20260911_145421_235 \
  --output outputs/lookback_routing_v1 --device cuda:0 --resume
```

默认16组×10世界+4自然窗口×3世界=172次基础回放；保存每个世界的原始分量。
干预限于reference选出的1个head、前2个heldout组（不挑错误）及4个自然窗口的第0位置。
这些是预先设定的计算预算，不是按效果筛选。可用 `--site-indices 0 1 2`检查后续位置，
但换参数必须换输出；不后台，不预估真实8B耗时。原生eager仍有单层密集attention成本。

阶段入口：`--phase capture`、`--phase analyze`（CPU）、`--phase intervene`。
先完成capture才能analyze；intervene使用已经冻结的subspaces/points，不能重新选。
`--suite controlled`仅跑标尺；默认both同时处理自然原例。
老 `natural.py` 入口已经删除，不会悄悄运行另一套同名实验。

原自然基线继续对 top-logits/log-normalizer/存储精度attention核验，不提高容忍度。
所有干预独立KV、没有旧世界缓存串用；Q/K/V只在指定head单次读取替换，GQA兄弟head
不被无意改变。每层其他位置和后续非线性计算继续原生运行。相同世界和早期query
不变属于**软件校验**，不列为机制发现。

## 先读哪些输出

```
hypothesis_counts.json          实际有多少绑定错误、多少仅门控失败、多少“可读但选错”
heldout_binding_readout.csv      逐层、逐例；不是仅输出均值
selected_channel_binding.csv    residual可读 vs Q可读 vs K可读 vs 实际读取
channel_measures.csv            每层每head的信息与正确方向（不要只选最大的MI）
natural_pair_geometry.csv       四例三位置的条件改变/语义保持对照
intervention_effects.csv        组合预测是否命中、路由变化、post-WO与输出变化
qk_decomposition.csv            接收坐标下Q项、K项、交互与代数闭合
capture/*.npz                   原生h/x/Q/K/V/attention/写入/完整logits
interventions/*.npz             每个干预的逐头分量及实际输出
subspaces.npz + subspaces.json  reference冻结的小读出与候选子空间
```

“只换Q能改输出”不够；需同时看是否按预定来源重新取值、QK双翻转/同绑定对照是否
符合规则、是否超过随机子空间。未出现错误时报告未检验该错误机制。主张只到测得的
层次，不从单头无效推断全模型无机制，不从成功修复推断原运行已经有正确指针。

本地没有用户8B权重和原NPZ。测试验证数值、索引、隔离和端到端软件；没有自然效果保证。
文献依据及理论不可外推的边界见 `RESEARCH_BASIS.md`。
