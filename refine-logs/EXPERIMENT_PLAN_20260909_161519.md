# Answer-Conditioned Constraint Control：机制发现实验计划

更新时间：2026-09-09 16:15:19（Asia/Shanghai）  
计划状态：设计冻结前草案；本阶段只定义研究与工程方案，不实现实验代码。

## 1. 研究对象与核心问题

本子项目研究的不是“模型有没有看远处”，而是：模型在生成事实性答案时，是否把材料中的**实体—属性—限定条件关系**恢复为对当前候选 token 的有效控制；当答案出错时，这种控制在哪个阶段断裂。

工作假设是：

> 一类幻觉并非源于模型完全没有读取相关材料，而是内容信息已经被接收甚至形成中继，但限定条件未被正确绑定、未进入答案决策，或其正确作用在晚层被局部续写路径覆盖。错误 token 之后的局部自依赖主要放大既成错误，而不一定是首因。

实验必须允许这一假设失败。它需要区分五个阶段，而不是把所有现象归结为 attention locality：

| 阶段 | 问题 | 工作量化 |
|---|---|---|
| Receipt（R） | 实体、属性、限定条件是否到达候选回看节点/中继？ | 来源分组到中继状态的因果效应与可解码性；探针只作辅助 |
| Binding（B） | 中继是否保留正确的实体—属性—条件组合，而非只提取相关词？ | 匹配约束交换对上的交互效应与关系选择 margin |
| Utilization（U） | 正确组合是否实际改变当前答案候选？ | 中继/路径干预对候选特异 logit margin 的效果 |
| Override（O） | 正确作用是否在后续层被反向路径抵消？ | 早期正向贡献与晚期负向贡献的符号、位置和抵消比例 |
| Local rollout（L） | 首个错误决策后，局部历史是否继续放大错误？ | 错误起点后的历史 token 路径占比及错误片段延续概率 |

## 2. 可主张内容：先冻结边界

### Primary Claim 1：机制主张

在一部分源材料约束的事实生成错误中，存在可重复的“内容已接收、约束控制缺失/被覆盖”因果表型；它能通过候选特异的输出 margin、匹配反事实和双向干预与一般的远距回看、低置信度或局部依赖区分。

最低证据：

1. 回看事件在看标签前筛选，机制量在来源隔离的确认集上复现；
2. 条件路径对正确/错误候选 margin 的有符号效果，不只是 attention 权重或线性可解码性；
3. 对正确轨迹移除该路径会伤害正确候选，对错误轨迹恢复该路径会提高正确候选 margin；
4. 匹配 content-only、随机位置、距离、熵、词类和层深等对照后仍成立；
5. 至少两个模型架构族、两个任务类型出现同方向的阶段级效应。层号/head 身份无需相同。

允许的最终措辞是“发现了一类反复出现的约束控制失效机制/机制家族”。三个模型和若干任务不足以声称“所有大模型幻觉的普遍机制”。

### Primary Claim 2：检测主张

不使用标准答案或错误标签构造的 R/B/U/O/L 在线近似特征，能在来源隔离测试上比输出不确定性、attention locality 和普通 hidden-state probe 更早、更稳定地检测事实错误起点，并具有跨任务或跨模型迁移能力。

最低证据：

1. 部署特征在推理时不读取 gold answer、错误 span 或“正确/错误候选”标签；
2. 训练、阈值选择和测试按 source/question 分组隔离；
3. held-out AUPRC 相对最强基线的配对 bootstrap 置信区间下界大于 0；
4. 报告校准、低误报区间和相对错误起点的提前量，不能只报告 AUROC；
5. 至少一次跨任务和一次跨模型零重训评估。

### Supporting Claims

- “回看节点”只是 label-free 候选；只有位于答案目标因果骨架上的事件才叫 **causal reanchor**。
- 只有干预证明其恢复限定条件对候选的控制时，才叫 **constraint-restoring reanchor**。
- 输出条件化骨架可以比“每个 token 的完整前向 DAG”更小，同时保留与目标答案相关的路径。

### Anti-claims / 必须主动排除

- 仅仅 attention 看得更远或更分散就等于正确推理；
- 线性探针读出条件信息就证明模型使用了它；
- 晚期 local attention 必然是幻觉根因；
- 只在错误样本中选择事件后得到的差异可以推广到总体；
- teacher-forced replay 与真实采样轨迹天然完全一致；
- 在单个模型、单个任务上观察到效应即可称为普遍机制。

## 3. 相对已有工作的创新切口

以下单点本身都不构成主要创新：attention-gradient 图、local/global 差异、远距回看、隐藏态探针、late-layer override。

本项目要验证的组合创新是：

1. **从“信息是否流动”转向“限定条件是否获得输出控制权”**：分别追踪 content path 与 constraint path，并以当前答案候选 margin 为终点。
2. **从事件出发的全未来 DAG 转向答案目标条件化的稀疏因果骨架**：先定义首个事实分歧 token/事实 span，再反向寻找真正到达输出的路径。
3. **保留丢失账本**：主图只显示到达目标的路径；旁路账本记录条件信息到达中继后在 binding、utilization 或 override 边界损失的位置，避免“只画成功路径”导致无法解释失败。
4. **双向因果证据**：错误轨迹的恢复干预与正确轨迹的移除干预必须方向相反；只做单向 patch 不足以声称机制。
5. **机制与检测双防火墙**：研究期 oracle 指标可使用 gold/hallucinated candidates；部署期特征必须完全去除这些信息，再独立检验检测价值。

如果只有自然样本上的统计差异，结果只能叫“相关表型”；如果加入候选特异干预，可叫“机制证据”；如果再跨架构/任务复现，可叫“反复出现的机制家族”。

## 4. 完整数据流

默认执行路径固定为：

```text
读取问题与证据材料
-> 模型自由自回归采样
-> 保存 token、logit、采样元数据
-> 对同一 prompt + 已采样回答做 teacher-forced replay
-> 校验 replay 与 generation 的逐 token logits
-> 捕获 Q/K/V、attention/MLP write、residual 状态
-> 生成完成后再做事实与 token span 标注
-> 冻结首个分歧 token / 事实 span 目标
-> label-free 筛选 lookback candidates
-> 从答案 margin 反向提取 output-conditioned causal backbone
-> 对 top paths 做精确 patch / ablation
-> 计算机制统计、检测特征与报告
```

自由采样与 teacher-forced capture 是两次前向过程。第二次不是重新生成答案，而是高效重放已经采样出的 token 序列。只有逐 token logit/probability 在预设容差内一致，样本才进入机制分析；否则标记为 replay mismatch 并排除。

### 每条 trajectory 的必要记录

- prompt、source/evidence units、question、sample seed、模板和 tokenizer hash；
- sampled token ids/text、逐 token logits/top-k、entropy、停止原因；
- 每层 Q/K（含 RoPE 后语义）、V、head output、attention/MLP write、residual in/out；
- attention 历史或可由 Q/K 确定性重建所需的信息；
- 模型 revision、dtype、设备、软件版本和捕获 schema version；
- 生成后加入的事实 span、首错 token、错误类型、证据单元和标注置信度。

特殊 token 不参与 lookback candidate 统计，但保留在序列索引和因果守恒检查中。

## 5. 研究对象如何确定

### 5.1 先生成，后标注

不能先挑“看起来会错”的 token 再生成。模型按固定采样配置独立生成 3 个 seed；生成和状态捕获结束后，才执行：

1. 结构化/确定性 verifier（可用时）；
2. source-grounded entailment 与答案规范化；
3. 只对歧义项使用 LLM judge；
4. 对正确、错误和不确定层各做分层人工抽检。

标注单位是事实 span，不是单个 subword。目标 token 默认取**首个使事实候选发生分歧的生成 token**；同时以整个事实 span 的长度归一化 margin 做稳健性分析。

### 5.2 Lookback、causal reanchor 与错误目标分开

- `lookback candidate`：排除特殊 token 后，由 adjacent-step local→non-local attention 变化和 max-null 校准筛出；不看正确性标签。
- `causal reanchor`：lookback candidate 位于当前答案目标的有符号因果骨架上。
- `constraint-restoring reanchor`：进一步通过条件交换及恢复/移除干预证明其控制限定条件。

这样不会把几乎所有 token 当成研究对象，也不会把“有远距 attention”直接解释成机制。

### 5.3 答案条件化因果目标

机制研究的主目标为：

```text
M_t = log p(gold_candidate | prompt, generated_prefix)
      - log p(emitted_error_candidate | prompt, generated_prefix)
```

正确样本用匹配的易混淆候选构成 margin；没有可靠候选对的样本可用于检测评估，但不进入候选特异机制主分析。

## 6. 因果图与阶段指标

### 6.1 图的样式

不为每个生成 token 保存一张完整 DAG。每个事实目标只生成一个压缩骨架：

- 终点：首个分歧 token 或事实 span margin；
- 节点：`(layer, token, stream)`，stream 至少区分 residual、attention write、MLP write；
- 边：对目标 margin 的 signed causal contribution；
- 来源类型：constraint、entity/content、其他材料、generated history、special；
- 裁剪：先用 VJP/JVP 一阶贡献筛 top paths，再用精确有限干预确认；
- 守恒：被裁剪、被抵消和未到达目标的质量进入 loss ledger，不伪装成完整解释。

图主要用于定位，主结论来自路径级数值与干预，不来自图像观感。

### 6.2 工作指标

- `receipt_content` / `receipt_constraint`：来源单元到 reanchor 中继的路径效果；
- `binding_interaction`：实体交换 × 条件交换对关系选择 margin 的差分中的差分；
- `utilization_effect`：中继/路径恢复后对目标 margin 的精确变化；
- `override_ratio`：reanchor 后与正确条件相反的贡献绝对值 / 先前正向贡献绝对值；
- `local_rollout_ratio`：首错后来自近期生成历史的目标贡献占比；
- `constraint_control_ratio`：有符号条件路径效果相对内容路径效果的归一化比例。

这些是预注册前的工作定义。M0/M1 只允许修正数值稳定性和索引问题；一旦在 discovery split 上冻结，就不能根据 confirmation 标签改公式。

## 7. 数据、切分与模型

### 7.1 任务梯度

1. **主发现任务：source-grounded QA**。问题、证据材料、答案和条件单元边界清晰，适合定义首个分歧 token。
2. **受控识别任务：匹配约束交换对**。从自然 QA 记录构造只改变时间、主体、范围、否定或关系限定的一处编辑，并验证答案随之改变；仅用于确认 B/U/O，不替代自然分布结果。
3. **迁移任务：source-grounded summarization 与 data-to-text**。分别检验文本证据和结构化字段条件下的阶段级复现。

现有 `attention_audit_v3` 仅用于迁移/索引 sanity check；主结果必须从原始问题材料重新自由采样，不能把旧 teacher-forced answers 当作新轨迹。

### 7.2 切分

- `discovery`：仅用于探索连续 R/B/U/O/L 表型并冻结指标；
- `calibration`：只定 lookback/null 阈值、图裁剪 K 和检测阈值；
- `confirmation`：一次性检验机制主张；
- `transfer`：官方 test 的其他任务/模型，仅做迁移。

所有切分按 `source_id/question_id` 分组，不能按生成序列随机切；同一问题的 3 个采样 seed 必须在同一 split。首轮建议 discovery/calibration/confirmation 为 50%/20%/30%，官方 test 保持独立。

### 7.3 模型顺序

| 角色 | 模型 | 原因 |
|---|---|---|
| 主发现 | `meta-llama/Llama-3.1-8B-Instruct` | 与当前 Llama 风格捕获/解析能力最接近，先降低实现风险 |
| 独立复现 | `Qwen/Qwen3-8B`，`enable_thinking=False` | 独立模型系列；关闭显式 thinking 以保持答案协议一致。官方卡给出了 non-thinking 的采样配置 |
| 架构压力测试 | `google/gemma-2-9b-it` | 第三模型系列，检验阶段级结论是否依赖 Llama/Qwen 的具体实现；需提前确认权重许可 |

模型卡：[Llama 3.1 8B Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)、[Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B)、[Gemma 2 9B IT](https://huggingface.co/google/gemma-2-9b-it)。Qwen3 主实验采用其官方 non-thinking 建议作为起点（temperature 0.7、top-p 0.8、top-k 20），但每个模型的采样配置分别冻结并在报告中显式记录，不能混作同一温度效应。

第一轮不研究显式 chain-of-thought；它会额外引入“思考文本是否忠实”的变量。若主机制成立，再把 thinking mode 作为独立扩展。

## 8. 子项目工程结构

计划在 `experiments/constraint_control/` 下实现，默认入口和数据流保持线性：

```text
experiments/constraint_control/
|-- main.py                 # 只解析参数，构造 ConstraintControlExperiment(config).run()
|-- experiment.py           # generate -> capture -> annotate -> trace -> evaluate 编排
|-- config.py               # 少量成组参数的 typed dataclass 与边界校验
|-- dataset.py              # 问题、证据单元、约束单元和 source-level split
|-- generation.py           # 自由采样、token/logit 记录、replay fidelity 校验
|-- capture.py              # 统一 capture schema；调用模型适配器
|-- annotation.py           # 生成后事实/span 标注，不参与事件筛选
|-- targets.py              # 首错 token、事实 span 与候选 margin 定义
|-- causal_backbone.py      # target-seeded VJP、路径裁剪和 loss ledger
|-- interventions.py        # 条件交换、恢复、移除与 sham 干预
|-- evaluation.py           # 机制统计、检测基线、bootstrap 与迁移评估
|-- models/                 # 三个实际模型的薄适配器；共享最小 protocol
|-- tests/                  # 索引、replay、反事实、因果守恒和防泄漏测试
`-- README.md               # 一键运行、产物 schema 与 claim 边界
```

复用 `src/reanchor` 中已经成立的 artifact、label-free discovery 和解析逻辑；不复制旧 `message_dag` 执行链。只有三个实际模型适配器都需要相同操作时才建立共享 protocol，不预建 registry/factory。

### 输出结构

```text
runs/<run_id>/
|-- manifest.json
|-- trajectories/
|-- captures/
|-- annotations/
|-- targets/
|-- backbones/
|-- interventions/
`-- reports/
```

每个阶段以 manifest 中的输入 hash、模型 revision、schema version 和完成状态为恢复边界；标签与捕获物理分离，避免标签泄漏进事件筛选。

## 9. 核心实验块

### B1. 自然轨迹机制发现（MUST）

- **支撑主张**：Claim 1 的表型发现。
- **任务/数据**：source-grounded QA；discovery split；每个 source 3 个采样 seed。
- **系统**：Llama 3.1 8B Instruct。
- **指标**：R/B/U/O/L 连续量、reanchor 发生率、首错前后路径构成；source-level bootstrap CI。
- **设置**：事件筛选完全 label-free；正确/错误组在位置、距离、熵、词类、答案长度上匹配或回归调整。
- **成功标准**：得到少量可解释表型，并在 calibration 上冻结定义；不以 discovery p-value 作为论文证据。
- **失败解释**：若效应由熵/位置解释，放弃“约束控制”主线；若只有少数 task subtype，收窄 claim。
- **产物**：表型二维图、阶段 Sankey/transition table、预注册签名。

### B2. 受控反事实与双向干预（MUST）

- **支撑主张**：Claim 1 的因果识别，排除 generic content retrieval。
- **任务/数据**：从自然 QA 分层抽取的实体/时间/范围/否定/关系约束交换对；confirmation 前冻结生成规则。
- **系统**：Llama 主分析；Qwen 复现。
- **指标**：binding interaction、path-specific utilization effect、override ratio、恢复/移除干预的 margin 变化和 answer flip rate。
- **设置**：一阶 VJP/JVP 只筛 top paths；最终证据来自 exact activation/path patching。对照为 content-only、随机同层位置、距离匹配位置、sham donor。
- **成功标准**：错误轨迹恢复条件路径提高 gold-vs-error margin，正确轨迹移除同类路径降低该 margin；两方向 CI 均排除 0，且显著强于 content-only/sham。
- **失败解释**：只有 probe/梯度差异而无精确干预效果时，只保留相关性结论；单向恢复无移除效果时不能声称必要性。
- **产物**：主机制表、配对干预图、正常/错误压缩骨架案例。

### B3. 冻结机制的跨模型/任务确认（MUST）

- **支撑主张**：Claim 1 的机制家族范围。
- **任务/数据**：held-out QA、summarization、data-to-text；source-disjoint transfer split。
- **系统**：Llama、Qwen、Gemma。
- **指标**：冻结签名的方向一致率、标准化效应量、随机效应汇总与异质性。
- **设置**：层位置按归一化深度比较；不得追求相同 head 编号。模型适配器先过 replay 与有限差分测试。
- **成功标准**：至少两个架构族、两个任务类型出现同方向的 B/U/O 核心效应；报告不符合的模型/错误类型。
- **失败解释**：若只在一个模型成立，claim 改为 architecture-specific；若只在 QA 成立，claim 改为 grounded QA mechanism。
- **产物**：模型×任务效应矩阵、异质性图、适用范围表。

### B4. 幻觉检测可用性（MUST for Claim 2）

- **支撑主张**：Claim 2。
- **任务/数据**：所有任务的 token-level onset；按 source 隔离 train/calibration/test。
- **系统**：每模型独立训练一次轻量分类器；另做跨模型/任务零重训。
- **特征**：部署版 R/B/U/O/L 近似，不含 gold/error candidate；oracle 版本只作为机制上界。
- **三类基线**：输出 uncertainty；attention locality/lookback；普通 hidden-state linear probe。
- **指标**：AUPRC（主）、AUROC、ECE/Brier、FPR@固定 recall、相对首错的 lead time；配对 source bootstrap。
- **成功标准**：部署特征的 held-out AUPRC 超过最强基线且 CI 下界大于 0；信号在错误发生前或当下出现，而不是只在错误后出现。
- **失败解释**：oracle 有效但部署版无效，说明机制分析可成立但不能主张检测；只在同任务有效则只称 task-specific detector。
- **产物**：检测主表、校准图、提前量曲线和 oracle/deploy gap。

### B5. 简化、负对照与失效案例（NICE-TO-HAVE，投稿前必须）

- **支撑主张**：证明贡献不是复杂特征堆叠。
- **任务/数据**：B2/B4 的确认集，不重新挑样本。
- **系统**：主模型 + 一个复现模型。
- **指标**：删除 R/B/U/O/L 单项后的机制/检测变化；random label、random source、future-token、special-token 与层置换负对照。
- **成功标准**：约束绑定/利用项提供不可由 locality/uncertainty 替代的增量；负对照接近 null。
- **失败解释**：若 uncertainty 单独解释全部提升，则检测创新不成立；若 random source 同样有效，因果定位有漏洞。
- **产物**：消融表、负对照表、诚实 failure taxonomy。

## 10. 统计陷阱与解决方式

| 陷阱 | 设计约束 |
|---|---|
| 先看错误再选回看节点 | lookback 在 join labels 前计算并冻结；主分母包含全部 candidates |
| 一个 source 的多个 token/seed 当独立样本 | source/question 为聚类单位；hierarchical/bootstrap 或 mixed effects |
| token 位置、距离、熵、词类混杂 | 匹配 + 协变量调整；报告未调整与调整结果 |
| 大量 layer/head/token 多重检验 | discovery/confirmation 分离；max-T 或层级 FDR；主指标预注册 |
| attention 被误当因果 | attention 只做候选生成；主证据必须含目标 margin 和 exact interventions |
| probe decodability 被误当使用 | probe 只证明 R；U 必须由干预验证 |
| teacher-force 暴露未来或重放偏差 | causal mask + 已生成 prefix；逐 token generation/replay logits 校验 |
| patching 离开数据流形 | 匹配 donor、局部小幅 patch、sham 与双向干预、报告 KL/hidden norm shift |
| 用 gold 构造检测器导致泄漏 | oracle/deploy 两套特征与代码路径；测试断言 deploy schema 不含 gold 字段 |
| 只分析能成图的样本形成幸存者偏差 | 报告所有 candidates 的去向和 loss ledger；无法闭合者单独计数 |
| post-hoc 改机制定义 | discovery 后版本化冻结指标；confirmation 一次性运行 |

## 11. 执行顺序、成本与决策门

硬件未知，因此先用 M0 实测“每 1k 输入 token、每 100 输出 token、每条 exact patch”的成本，再更新预算。以下为 A100 80GB 级单卡的保守暂估，不作为运行承诺。

| 阶段 | 内容 | 暂估 GPUh | 决策门 |
|---|---|---:|---|
| M0 Sanity | 20 个受控问题；生成/replay/index/有限差分/守恒测试 | 1–3 | replay logits 达容差；已知干预方向正确，否则不扩大 |
| M1 Pilot | 100–200 个 QA source × 3 seeds；自然错误率和事件覆盖 | 8–20 | 有足够首错 span；事件不过密；R/B/U/O 数值非退化 |
| M2 Main causal | Llama discovery/confirmation + top-path exact patch | 20–60 | 双向干预成立，否则机制 claim 降级/停止 |
| M3 Replication | Qwen、Gemma、两类迁移任务 | 40–100 | 两架构族×两任务同方向，否则收窄范围 |
| M4 Detection | 部署特征、三类基线、迁移与校准 | 5–15 | held-out 增量和提前量成立，否则删除 Claim 2 |
| M5 Polish | 消融、负对照、图表和复现包 | 5–15 | 只在主结论通过后运行 |

MUST 总预算暂估 74–198 GPUh；M0 后按真实吞吐重新估算。exact patch 是主要成本，因此只对一阶筛出的 top paths 与分层样本执行；不能用更便宜的一阶近似替代最终因果证据。

## 12. 首批三个可执行 run

1. **P001 — generation/replay fidelity**：20 个 QA/受控约束问题，Llama，3 seeds；验证 token 对齐、逐 token logits、特殊 token 排除和 capture schema。
2. **P002 — answer-conditioned backbone sanity**：对 P001 的已知正确/错误候选构建 margin，验证 VJP/JVP 与有限差分方向、cut closure 和 loss ledger。
3. **D001 — natural QA pilot**：100 个 source × 3 seeds，自由采样后标注；在不看标签的情况下冻结 lookback candidate rate，再评估首错附近的 R/B/U/O/L 分布。

P001 或 P002 不通过时，不启动大规模 capture。D001 若把大部分 token 都选为 lookback，先修 null calibration 和 episode collapsing，而不是扩大样本。

## 13. 实现与验收顺序

后续代码采用垂直切片：

1. `generate -> replay -> one trajectory artifact`；
2. `annotate -> one frozen decision target`；
3. `target -> one causal backbone + loss ledger`；
4. `one matched counterfactual -> restore/remove intervention`；
5. `one held-out report -> mechanism + detection metrics`；
6. 再增加 Qwen/Gemma adapters 和任务迁移。

每个切片先写行为测试。完成标准包括：单元/集成测试通过、`ruff`/现有测试通过、toy finite-difference 对齐、deploy feature 防泄漏断言、README 中能从入口追踪到结果。

## 14. 当前结论

现在还不能声称已经发现普遍机制。可以声称已经形成一个可证伪的机制发现方向：**答案条件化的约束控制追踪**。它是否成为论文级机制贡献取决于 B2 的双向因果干预和 B3 的跨架构/任务确认；是否成为幻觉检测信号则由信息防火墙下的 B4 独立决定。
