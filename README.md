# Reanchor：出错前在回看什么

新会话先读 [项目进展与后续方向](docs/PROJECT_HANDOFF.md)：已完成工作、真实结果、
尚未验证的假设、两个仓库的版本和下一轮实验顺序。

回看与约束归属的控制实验见 [执行方案](docs/mechanism_experiment.md)：

```bash
bash scripts/run_mechanism.sh
```

入口 `main.py mechanism` 调用 `MechanismExperiment.run()`：归属交换、真实
attention 连接干预、MLP 更新替换，再检查五个已有自然样本窗口。输出逐项
`tokens.csv`、`layers.csv` 与完成检查；相同设置重跑会跳过已完成案例。
使用已有 GPU Python 环境，`OUTPUT_DIR`、`SAMPLES_DIR`、`STATES_DIR` 可以显式指定。

同 token 熵对照使用 `main.py match-entropy --features ... --labels ... --output ...`。
这里的标签只用于比较首错与正常位置，不能作为检测算法的输入。

本仓库用于机制验证、具体样本分析和候选算子的实验。主线检测方法在
[graph 项目](https://github.com/DwightEd/graph) 中实现、集成和评价。
这里的回看定位、多跳来源和关系残差是实验原型；“实验流程已接通”不表示主线方法已经完成。

入口只有 `main.py`：

- `sample` → `SamplingExperiment.run()`：加载模型，逐 token 采样，在同一次前向保存 attention 和候选 logits。
- `inspect` → `AttentionAnalysis.run()`：读取保存的数据，输出具体 token 选择和历史读取变化。
- `routes` → `RouteAnalysis.run()`：对具体样本计算 source 上的 head 内散布、head 间分歧与逐步转向。
- `decisions` → `DecisionInspection.run()`：固定窗口的原始候选、指定历史边、相同前缀数值检查。
- `states` → `FixedPrefixStates.run()`：固定原生成前缀，补取各层表征和完整 logits 熵。
- `revisits` → `RevisitAnalysis.run()`：内容回看、多跳路径与关系保持读出。
- `compare` → `WindowComparison.run()`：独立案例窗口的数字对照。

算法在 [sampling.py](src/decoding/sampling.py)、[attention.py](src/decoding/attention.py)
和 [routes.py](src/decoding/routes.py)。
新增的具体检查集中在 [decisions.py](src/decoding/decisions.py)。
`io.py` 只负责 JSON 读写及拒绝覆盖已有输出。没有训练、概率校准、分组评估、bootstrap 或 report。
当前研究要求见 [回看节点与约束归属](docs/graph_method_proposal.md)。
该文区分已观察结果、本轮检查和待验证算法；其他历史笔记不作为当前实现说明。

## 回看与证据路径验证实验

```bash
git pull --ff-only origin agent/direct-sampling && bash scripts/revisits_and_push.sh
```

脚本选择并打印最新的 outputs/samples_*，也可传入原有样本目录：
`bash scripts/revisits_and_push.sh outputs/你的采样目录`。

执行顺序是固定前缀状态补取 → 全流图分析 → 案例窗口对照 → 推送数字结果。
第一步需要原模型和 GPU，严格使用已有 token ID，不重新采样，不应用温度或 top-p。
补取 embedding、各层表征和完整 logits 熵，逐步核对原缓存 top logits、log-normalizer 和 attention。
默认最大差异 1e-4，超出立即停止。可用 DEVICE、STATES_DIR、LOGIT_ATOL 指定实际环境。
完成的状态按样本复用，未完成样本重做；状态放在 outputs/states_*，不进入 Git。

只计算、不推送：

```bash
export PYTHONPATH="$PWD/src"
python main.py states --samples outputs/你的采样目录 --output outputs/fixed_states
python main.py revisits --samples outputs/你的采样目录 --states outputs/fixed_states \
  --output outputs/revisit_analysis --window 16 --quantile 0.95 --context 4 --hops 3
python main.py compare --analysis outputs/revisit_analysis \
  --cases examples/decision_cases.csv --output outputs/revisit_analysis/comparison.csv
```

每条回答完成后保存：

- tokens.csv：query_step/query 与被预测 token，读取变化、历史阈值、active/event、原生 margin/熵。
- layers.csv：每层的内容 W1、回看分数、head 散布/分歧与有效秩。
- paths.csv：所有 token、每层、每跳的来源分歧和关系读数；展示三个来源端点。
- edges.csv：事件附近逐 head 的强内容连接与增强的历史连接，保留 token 身份和原始权重。
- graph.npz：完整 head 平均图边、各跳全部来源分布与读数，留在服务器，不推入 Git。
- comparison.csv：原有六个具体窗口的逐步数字对照；窗口不参与算法，未标注首错就不报首错准确率。
- state_checks.csv：每条轨迹的最大 logits/attention 差异，用于核对两次前向的一致性。

算法分为三个直接调用的模块：

1. [RevisitSignal.run()](src/decoding/reading_graph.py) 在共同可见的正文和历史键上计算
   相邻步 W1。排除指令、special、self 及刚变成历史的旧 query，避免把正常增加上下文当回看。
   减去各 head 过去位移的中位数，乘两时刻较小的原始内容质量，再对 head/层平均。
   这是过去基线校正，不是移动平均。active 标识超过过去分位数的区间，event 标识起点。
2. 同文件的 ReadingGraph.run() 按原生边计算直接来源，以及经历史节点的 2..hops 路径。
   层严格递减，历史 token u 的输入行是 u+1；prompt/source 是终点。默认最多三跳，
   可用 --hops 或脚本 HOPS 调整，有效最大值为模型层数。计算保留全部端点，无 top-k 截断。
   原始逐 head 图仍在采样 NPZ；分析图将中间 heads 平均，不能声称保留了每条 head 路径。
3. [RouteReadout.run()](src/decoding/route_readout.py) 先比较直接来源和多跳来源的 JS 分歧。
   再固定 token 到 source 的对应 T：实际路径权重乘输入 embedding 相似性。
   用同层、源文均值中心化的表征 x，比较生成端关联与对应后关联：

```text
mapped_i = sum_s T(i,s) * x_source[s]
residual(i,j) = max(0, dot(x_i,x_j) - dot(mapped_i,mapped_j))
relation_residual(i) = 历史读取边加权的 residual
```

另存 source 身份循环移位的 relation_shuffled 对照，以及有效关系覆盖率。
它们都是无需关系标签的读数，不是关系真值或错误概率；同义改写、软对应范数下降、
合法跨句组合也可能产生残差。无来源或退化表征留空。中间状态只能使用已经输入的 token：
第 t 行 query=P+t-1，预测 response[t]，不能把当前 query 的关系残差称为未生成 token 的知识。

[实现方案和验收边界](docs/revisit_route_plan.md) 说明当前算子、时间对齐和局限。
已有测试覆盖数值算子、无未来泄漏、真实微型模型端到端、状态身份检查和发布失败行为；
真实 8B 模型上的区分效果需要服务器结果。程序关系监督与低秩投影仍是备选，未参与本实现。
算法没有读取实体清单或幻觉标签。正文边界来自原有 RAGTruth QA passage 标记，
因此 states 的正文对齐目前限定此 prompt 格式。

目录中的两个层级现在用不同名称：

```text
reanchor/                 Git 项目根目录，在这里 pull 和运行 scripts
  main.py                 命令行入口
  src/decoding/           Python 代码包
    sampling.py           加载模型、生成与保存中间状态
    attention.py          查看具体历史 token 的读取变化
    routes.py             分析 source 上的关注分布
    io.py                 文件读写
  outputs/                服务器上的原始采样，Git 忽略
  results/                随 Git 同步的分析结果和生成原文
```

原来的 `reanchor/src/reanchor` 是项目名与 Python 包名相同，不表示又 clone 了一份仓库。
`git pull` 不会创建新的嵌套项目。当前算法文件是 `reanchor/src/decoding/routes.py`。

## 运行

在 reanchor 仓库、已激活的 GPU Python 环境中：

```bash
git fetch origin && git switch agent/direct-sampling && git pull --ff-only origin agent/direct-sampling && bash scripts/run_samples.sh
```

脚本用现有服务器的 Llama-3.1-8B-Instruct 和 RAGTruth 路径，
对 14304、14315、14325、14375 各运行 seeds 0、1、2、3，
最多生成 512 token，temperature=0.7，top-p=0.9，cuda:0，bfloat16。
`MODEL_PATH`、`RAGTRUTH_DIR`、`OUTPUT_DIR`、`DEVICE`、`DTYPE`、`PYTHON_BIN` 可覆盖对应参数。
相对路径以仓库根目录为基准。需要调整采样数量时直接修改脚本中的参数。
采样和分析均有进度条；任一步失败即停止。

```text
outputs/samples_<时间>_<进程号>/
  settings.json        实际运行参数，保存一次
  prompts.jsonl        选中的 source_id 和原始 prompt
  samples.jsonl        每个回答的原文、seed、停止原因、NPZ 文件名
  00000.npz ...        实际 token ID、显示文本、attention、前五候选 logits
  reading/
    tokens.csv         每一步实际选了什么，备选是什么，概率和 margin
    attention.csv      每一步每个 head 回看哪里，权重如何变化
```

模型使用自身 chat template 和 generation_config 中的终止符；没有二次前向或扰动。
NPZ 保存完整 attention、前五候选 logits，以及完整词表的 `logit_entropy`（bits）；不保存 hidden states。
已有 NPZ 缺少 `logit_entropy` 时，routes 对应列留空，不用 top-5 近似，也不要求重新生成。
完整 attention 以 float16 保存，logits 以 float32 保存，分析小于量化精度的变化时应谨慎。
inspect 只读取 NPZ；routes 另读原始 prompt，并加载本地 tokenizer 检查字符与 token 的对应。
两种分析都不加载模型权重，不需要 GPU。
routes 从已保存的 prompt token ID 还原前缀并验证原文，不重新套用可能带日期的 chat template。
本版 inspect 需要新增的 token 显示文本和所选 logit 字段，请用新脚本生成对应数据。

## 直接检查首错之前

先结合 `samples.jsonl` 的回答、`prompts.jsonl` 的材料和 `reading/tokens.csv`，
人工确定**这个新回答**第一个错误 token 的 `step`，从 0 开始。
例如确认 `00000.npz` 的第 37 号 token 首次出错：

```bash
export PYTHONPATH="$PWD/src"
python main.py inspect \
  --samples outputs/你的采样目录 \
  --trace 00000.npz --first-error 37 --before 16 --after 0 \
  --min-distance 16 --output outputs/first_error_37
```

这个命令是用法示例，37 不是自动推断的标签。
它导出 step=21～37；首错前的其他步骤仍保留，便于看转向发生在哪里。
不指定 `--first-error` 时导出全部步骤，可以检查正确回答以及冒号、转折、结论等位置。
不要把原 RAGTruth 另一个回答的字符跨度套到新回答上。

`token_id` 和 `token_piece` 保留模型实际分词，`token` 是单 token 的解码显示。
它们不是词序号或字符序号；一个实体可能跨多个 token，部分字节 token 单独显示可能不完整。
原文以 `samples.jsonl` 的完整解码为准。
如果标注段从空白、标点或引导词开始，应保留这个事实，不能自动把标签移到后面的实体。

## 两张表怎么读

`tokens.csv` 每行是一次预测：实际生成的 token、原模型概率、top-1/top-2 候选及其概率，以及：

- `top2_margin = z(top1) - z(top2)`：最大的两个原始 logits 之差。
- `chosen_margin = z(实际 token) - max z(其他 token)`：实际选择相对最强替代项的优势，采样时可以为负。

这里的概率来自未施加 temperature/top-p 的完整词表 softmax。
margin 等价于对应候选的 log 概率比，不是扰动后的变化量。
相邻步骤的候选对应不同的下一个 token 任务，不能直接当作同一个实体候选在持续竞争。

`attention.csv` 保留 layer/head，不先平均。设 prompt 长度为 P：

- 第 t 行读取的是 query=P+t−1 对历史 key 的 attention，用来预测生成 token t。
  `error_offset=0` 仍是首错 token 生成**前**；负数表示提前几次预测。
- `previous_peak` 是上一步在共同历史 key 中注意力最大的具体 token。
- `read_token` 是当前步在远处历史中权重增加最多的具体 token；
  `read_position` 是完整序列位置，`distance=query_position-read_position`。
  `read_context` 给出其附近最多九个 token 的显示文本，并截断到当前已知前缀。
- `previous_attention`、`attention`、`gain` 分别是这条边的前值、现值和差。
  若所有远处边都下降，gain 也可以为负；这不是回看增强。
  第一步没有可比较的前值，差分留空。

突变大小 `shift` 只在两步**共同可见**、非特殊的历史 key 上计算。
新 query key 从两边一起排除；分别归一化后计算半个 L1 距离：

```text
J = {j < query_position，且 j 不是特殊 token}
p(j) = A_current(j) / sum_J A_current
p_previous(j) = A_previous(j) / sum_J A_previous
shift = 0.5 * sum_J |p(j) - p_previous(j)|
```

shift 越大，过去读取的分布变化越大；需要结合具体边的原始权重和 gain 判断，
否则很小的历史读取质量经过归一化也可能呈现大变化。
远处定义为 query-key 的 token 距离至少 `--min-distance`，默认 16。
选取增加最大的远处边只便于查看，全部边仍在 NPZ 中；没有用任意阈值把某一步判成幻觉。

这能直接检查“在首错前，哪个 head 开始增加对哪段历史的读取，随后选择了什么 token”。
正确回答也需要同样查看。attention 变化是读取行为的代理；
仅凭它和 margin 的同时变化，还不能证明筛选了错误的语义路由，或该路由造成幻觉。

## 分析这四个具体例子

已有 16 条采样结果时，在仓库中运行：

```bash
git fetch origin && git switch agent/direct-sampling && git pull --ff-only origin agent/direct-sampling && bash scripts/analyze_cases.sh
```

脚本选择 `outputs/samples_*/samples.jsonl` 修改时间最近的采样目录并打印路径。
也可以明确指定：

```bash
bash scripts/analyze_cases.sh outputs/你的采样目录
```

脚本只分析现有数据；没有重新采样步骤。首次缺少 matplotlib 时会安装该绘图库。
需要 `samples.jsonl`、`prompts.jsonl`、`settings.json` 和四个对应的 NPZ。
`TOKENIZER_PATH` 可指定移动后的本地 tokenizer 目录；默认使用 settings 中的模型路径。
`PYTHON_BIN`、`SAMPLES_DIR`、`OUTPUT_DIR` 可覆盖解释器、输入目录、输出目录。
每次默认建立新的 `routes_<时间>_<进程号>` 子目录，原始文件不会改写。

## 把真实结果一起推送

在项目根目录执行以下命令，会分析已有采样，把结果提交并推送到当前分支：

```bash
git fetch origin && git switch agent/direct-sampling && git pull --ff-only origin agent/direct-sampling && bash scripts/analyze_and_push.sh
```

也可指定已有采样目录：

```bash
bash scripts/analyze_and_push.sh outputs/你的采样目录
```

结果保存到 `results/routes_<UTC时间>_<进程号>/`。除了每例的 CSV 和图，
还复制本次输入的 `samples.jsonl`、`prompts.jsonl`、`settings.json` 和观察窗口 `cases.csv`。
这样拉取仓库后，就可以对照生成原文、材料和真实数值分析。大体积原始 NPZ 留在服务器。
仅提交此次结果目录；未跟踪的其他文件不会被加入提交。
有未提交的代码或暂存修改时会停止，以保证结果对应已提交代码；分析失败也不会提交、推送半成品。
Git 推送失败时结果与本地提交仍保留，脚本不会强制推送或重写远端历史。

你在会话中提供的 16 条真实生成文本已另存于
[results/pasted_samples_20260911/samples.jsonl](results/pasted_samples_20260911/samples.jsonl)。
该文件只有生成文本，没有 attention 数值；它不是上述分析脚本的运行结果。

[examples/route_cases.csv](examples/route_cases.csv) 只指定样本和待看的原句：

| source_id / seed | 观察窗口 | 原文对照的用途 |
| --- | --- | --- |
| 14375 / 0 | 第 5 步：洋葱与 10–12 分钟 | 查看时间与动作对象的错误组合 |
| 14315 / 2 | `and wore a headdress...` | 查看帝王限定条件的丢失 |
| 14315 / 3 | 及膝、及踝的描述 | 这两个长度有原文依据 |
| 14304 / 0 | `3. Star:` | 查看正常条目转换 |

按 source_id + seed 查找真实文件名，不假定它们一定叫 00012 等。
原句必须在对应回答中恰好出现一次，否则停止，不猜测另一批生成结果的位置。
原句不是首错标注，也不参与结构量计算；字符定位沿实际生成的 token ID 解码，
不会重新分词后套用另一组 token。跨多个 byte token 的字符全部纳入窗口。

每个样本输出一个以 trace 文件名命名的目录，例如 `00012/`：

- `tokens.csv`：完整回答的 step、实际 token、观察原句标记、top-2 margin、完整词表熵（若已保存）。
- `source_tokens.csv`：原始 passage 正文的 token 位置及上下文，排除题目、指令、passage 标签和特殊 token。
- `trajectory.csv`：整个回答的逐步、逐层结构数值。
- `reading.csv`：观察原句前 16 步、原句内部、后 8 步，每个 layer/head 的具体读取位置与原始权重。
- `trajectory.png`：全部层的四张热图。红框只标出待看的原句；空白表示该值没有定义。

调用路径为 `main.py routes -> RouteAnalysis.run()`，核心计算集中在 routes.py。
默认观察窗口和历史基线可从命令行调整：

```bash
PYTHONPATH=src python main.py routes \
  --samples outputs/你的采样目录 --cases examples/route_cases.csv \
  --before 16 --after 8 --baseline 16 --output outputs/route_analysis
```

## 结构数值的定义

当前 TV/shift 是读取分布变化的描述，逐词复制也会产生大值。
不将它的峰直接当作回看或幻觉。图距离的改进与 head 选择要求见当前研究文档。

在每层固定的 passage token 集合 S 上，各 head 单独归一化：

```text
mass[h,t] = sum_j∈S attention[h,t,j]
p[h,t,j] = attention[h,t,j] / mass[h,t]
dispersion[t] = mean_h H(p[h,t])
disagreement[t] = H(mean_h p[h,t]) - dispersion[t]
shift[t] = mean_h 0.5 * sum_j∈S |p[h,t,j] - p[h,t-1,j]|
```

H 的单位是 bits。前两项分别表示 head 内分散、head 间分歧；
不会先平均 head 再只算一个熵。shift 捕捉连接分布的变化，包括熵不变的位置转移。
`shift_baseline` 是同一层之前最多 16 步的有效 shift 中位数，不含当前步或未来步。
没有依据这四例拟合阈值、选择层、训练分类器或自动赋予幻觉标签。

`source_mass` 是所有 head 的原始 source 权重均值。
零 source 权重的 head 不参与 D/J；TV 要求同一个 head 在前后两步均有正 source 权重。
没有有效 head 时留空；`valid_heads` 记录当前参与 D/J 的数量，第一步 TV 和基线留空。
很小的 source 权重也可能归一化出明显的变化，必须同时查看原始质量，不能单看熵或 TV。

`source_mass` 就是分给 passage 正文的总 attention 权重，不是另外训练或拼接的检测特征。
例如 `[0.001, 0.001]` 与 `[0.4, 0.4]` 归一化后同为 `[0.5, 0.5]`，熵相同，
但前者仅有 0.002 的注意力分给 source，后者有 0.8。保留它是为了分清这两种情况。

reading 中 `peak_*` 是当前最强 source 边，`gain_*` 是前后至少一次有权重的边中增量最大的一条。
若这些边全部下降，则保留负 gain；第一步没有前值和 gain。
这里只导出便于逐行检查的两条边，完整连接仍在原始 NPZ 中。
层、head、step 都从 0 开始。第 t 步 attention 的 query 是 P+t−1，用于预测 response token t。

先在 trajectory 看转向相对近期基线何时增大，再到 reading 看相同 step/layer 的具体 source，
最后通过 tokens 对齐原句；正确对照也按相同步骤检查。固定 source 集合控制了历史长度增长，
但目前只研究对原始 passages 的直接注意力，不覆盖通过已生成答案间接回看的路径。
这些数值是关注结构的描述，不能单独证明某条路由导致了幻觉。

## 固定回看窗口：候选、历史对象与限制词

`analyze_cases.sh` 和 `analyze_and_push.sh` 现在同时执行此检查，结果位于 `decisions/`。
只需新检查时直接运行，无须重复 routes 或加载模型：

```bash
PYTHONPATH=src python main.py decisions \
  --samples outputs/你的采样目录 --cases examples/decision_cases.csv \
  --output outputs/decision_inspection
```

[decision_cases.csv](examples/decision_cases.csv) 固定六个观察窗口：三种烹饪时间、
头饰限制、正常衣长描述和 Star 条目转换。`start` 包含、`stop` 不包含，都是 response step。
`text` 必须与该窗口保存的单 token 显示文本拼接完全一致，否则停止；不能套到新一批生成上。
`history_steps` 是指定的 response key，`prompt_positions` 是完整输入里的 prompt key。
这些编号只用于实例核验，不是自动回看检测或预先指定实体的模型输入。

输出四张直接读数的表：

- `tokens.csv`：实际选择的原始概率、chosen margin、top-2 margin 和已保存的完整 logits 熵。
  `chosen_rank` 只表示其在保存的 top 候选中的排名；不在其中时留空。
- `candidates.csv`：每个保存候选的 ID、文本、原始 logit、完整词表概率、是否实际选中。
  概率不施加 temperature/top-p，也不对 top-5 重新归一化，因此概率和通常小于 1。
- `reads.csv`：指定 key 在**全部层和全部 heads** 上的原始权重、前值、增量。
  包括低权重、非最大边和零权重。新 key 的前值/增量留空；不可见 key 不导出。
  `distance=0` 是 query 自身，不是回看。某个 response token u 的自身 query 在预测行 u+1。
- `same_prefix.csv`：整个采样目录内，各回答相同 token 前缀的候选数值比较。
  包括首次抽样分叉那一步，不比较分叉之后不同前缀的状态，也不比较不同选择的 chosen logit。
  `max_top_logit_error` 比较按排名保存的 logits，需结合 `candidate_ids_equal` 读取；
  另给出 log-normalizer 的误差。没有重复前缀时只有表头，不表示通过了完整激活一致性检查。

输入的 settings、samples、prompts 和 cases 也会复制保存，原始 NPZ 留在服务器。
旧缓存缺少 logits 熵时留空，不反推。此命令不依赖模型权重或 tokenizer；
只读取固定缓存，seed/温度不会参与分析计算。

相同前缀检查只能验证保存数值的一致性。不同前缀下正确/错误段落的差异仍受上下文、
位置和表达影响；相同 seed 或改用 greedy 不能自动消除这些因素。
