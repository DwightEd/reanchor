# Reanchor：采样与内部状态分析

在远端 reanchor 仓库、已激活的 GPU Python 环境中运行：

```bash
git fetch origin && git switch agent/direct-sampling && git pull --ff-only origin agent/direct-sampling && bash scripts/run_samples.sh
```

`run_samples.sh` 使用服务器现有的 Llama-3.1-8B-Instruct 和 RAGTruth 路径，
对 14304、14315、14325、14375 各采样 4 个 seed，每次最多生成 512 token，随后输出 attention 数值 CSV。
结果保存在新建的 `outputs/samples_<时间>_<进程号>/`，包括 `attention.csv`。
路径和设备可通过 `MODEL_PATH`、`RAGTRUTH_DIR`、`OUTPUT_DIR`、`DEVICE`、`DTYPE`、`PYTHON_BIN` 覆盖；相对路径以仓库根目录为基准。

自然样本实验从 `python main.py sample` 开始。执行链只有：

```text
main.py → SamplingExperiment.run() → 加载模型 → 逐 token 采样并保存当步状态
main.py → AttentionAnalysis.run()  → 读取 attention → 输出数值 CSV
main.py → SpanEvaluation.run()     → 读取已有方法的分数和标签 → 分组指标 CSV
```

- [sampling.py](src/reanchor/sampling.py)：加载本地模型，使用 KV cache 生成，在同一次前向取 attention、hidden states、候选 logits。没有第二遍 replay。
- [attention.py](src/reanchor/attention.py)：WAAD / FAI 的基线复现，逐 token、layer、head 输出。它们来自 [Li 等的论文](https://arxiv.org/html/2510.13554v2)，不是本项目提出的新方法。
- [span_evaluation.py](src/reanchor/span_evaluation.py)：区分首错、连续错误段开头与段内延续，检验全 token 指标是否掩盖起点表现。

在已安装依赖的环境运行（从源码运行时先 `export PYTHONPATH="$PWD/src"`）：

```bash
python main.py sample --dataset /path/to/RAGTruth/dataset --model /path/to/Llama \
  --source-ids 14304 14315 14325 14375 --seeds 0 1 2 3 \
  --max-new-tokens 512 --output outputs/samples
python main.py analyze-attention --samples outputs/samples --output outputs/attention.csv
```

参数默认 temperature=0.7、top-p=0.9、cuda:0、bfloat16。模型使用自身 chat template，输入一条 user 消息，即 source_info 中的原 prompt；不额外要求思维链。终止符取模型 generation_config。到长度上限的回答记为 max_new_tokens。

输出只有实验输入、原始数据和数值：

```text
settings.json    本次实际参数，保存一次
prompts.jsonl    本次选中的原问题与材料，保存一次
samples.jsonl   source_id、seed、回答、token 数、停止原因、张量文件名
00000.npz …     token_ids、prompt_length、special_mask、attention、hidden_states、
                top_ids/top_logits、entropy、原模型 logprob、采样分布 sampling_logprob
attention.csv  每个 token/layer/head 的 WAAD、FAI、可用未来 query 数和最高 attention 来源位置
```

不写 schema、identity、replay_mode 等轨迹说明字段，不生成 report、HTML 或叙述性分析。样本行在每次生成完成后落盘。非空输出目录拒绝覆盖。

张量 `attention[L,H,t,j]` 是在预测响应 token t 时，query `P+t-1` 对 j 的注意力。
`hidden_states[L+1,t,D]` 同样是预测之前的状态，含 embedding 层。存储为 float16；
top_logits 是未施加 temperature/top-p 的原模型 float32 logits。
完整 token ID 可配合 settings 中的 tokenizer/model 路径恢复到具体 token；不要将 response 的词或字符序号当成模型 token 序号。

WAAD 默认窗口 10；FAI 默认使用当前生成 token 后距离 10～100 的 query，区间端点包含。
FAI 是事后量，不能放入该 token 生成前的检测器。没有可用未来 query 时 CSV 留空，并给出 fai_queries=0。
两项采用原始 attention，包含特殊 token；peak_source 必须结合 special_mask 排除 sink 的解释。
它们只描述距离与后续读取，不能自动判断幻觉。

## 检验连续 span 假设

每个方法导出同一测试集的完整 token 分数 CSV，列为：

```text
source_id,response_id,token_index,token_count,label,score
```

label 为 0/1，score 越高表示越可能幻觉；每个回答 token_index 必须完整覆盖 0～token_count-1。
新采样回答需要独立标注，不能套用原数据另一个回答的标签。该入口不从回答文字猜测标签。

```bash
python main.py evaluate-spans --input charm_tokens.csv --output charm_metrics.csv --bootstrap 200
```

输出 all、onsets、first_error、continuations、first_error_clean_prefix 的计数、阳性比例、AUROC/AP 与 source bootstrap 区间，并直接打印同一数值表。连续段按二值标签游程定义；回答 token 0 若为阳性也算开头。最后一组仅保留首次阳性及之前的正常 token，以及完全正常回答。前四组使用相同正常 token 集，source 在每组内等权；组间重加权后不能直接用原始计数拼回总 AUROC。AP 随阳性比例变化，应同时查看 prevalence。当前未实现语义类型／位置匹配、逐 span 等权和固定误报率的报警评估。

要区分“attention 结构有效”和“连续标签容易平滑”，需在相同 source split、输入表示与调参预算下比较 CHARM、无消息传递、仅相邻 token 链的消息传递，并分别评估起点与延续；同时核查方向、池化和归一化是否允许未来回答影响当前输出。不能根据一次无图消融或 span 比例就断定 CHARM 的效果全部来自连续性。

[First Hallucination Tokens Are Different from Conditional Ones](https://arxiv.org/html/2507.20836v1) 已研究段内位置差异；对其测试的 logit 信号，首 token 更易区分。因此我们应检验不同方法的收益落在哪类位置，而不是预设延续对所有方法都更容易。CHARM 原论文的 token 实验使用 NQ/CNN；RAGTruth 上的发现需要单独复现，不能直接替代其原始实验。

## G0 条件绑定支持模型

另有 **G0 训练与检测流程**。
G0 使用冻结 Llama 表示、候选条件化 source cross-attention，直接学习 `softmax(f)`。
旧四格实验保留为独立的 `audit` 子命令。

这是最小研究实现，不代表新方法已经有效。未实现 G1 图版本或原生成器内部路径干预。

## 远端一键运行

先进入远端 reanchor 仓库并激活已有 GPU Python 环境：

```bash
git pull --ff-only && bash scripts/run_g0.sh
```

原入口 `bash scripts/run_pilot.sh` 现在也启动 **G0**。
旧审计请改用 `bash scripts/run_audit.sh`；不要混用两类结果。

默认读取已存在的本地文件，不下载模型：

- 模型：`/share/home/tm902089733300000/a903202310/lys/models/Meta-Llama-3.1-8B-Instruct`
- RAGTruth：`/share/home/tm902089733300000/a903202310/lys/data/RAGTruth/dataset`
- 设备：`cuda:0`；dtype：`bfloat16`
- 程序任务：64 个 base sources，每个 source 8 个配对上下文；按 source 分 train/dev/calibration/test。
- 自然数据：按无标签 hash 选择 32 个 sources，约 1/4 校准、3/4 测试；这是小规模探索性运行。
- G0：width 128，2 个 source-read blocks，10 epochs，seed 42。

覆盖路径或设备：

```bash
MODEL_PATH=/path/to/llama RAGTRUTH_DIR=/path/to/RAGTruth/dataset DEVICE=cuda:1 bash scripts/run_g0.sh
```

先只做程序任务工程验证：

```bash
PROGRAM_ONLY=1 PROGRAM_SOURCES=8 EPOCHS=1 bash scripts/run_g0.sh
```

参数可用环境变量覆盖：`PYTHON_BIN`、`OUTPUT_DIR`、`PROGRAM_SOURCES`、`RAGTRUTH_SOURCES`
（0 = 全部）、`EPOCHS`、`BATCH_SIZE`、`MAX_TOKENS`、`CANDIDATE_CHUNK`、
`GENERATOR_FILTER`、`EXCLUDE_SOURCES`。后者为已使用 source IDs 的 JSON 数组文件。
更多参数：`bash scripts/run_g0.sh --help`。

脚本不抢占或终止已有进程。默认拒绝已有显存占用 ≥500 MiB 的 GPU；
确认可以共享时才显式设置 `ALLOW_BUSY_GPU=1`。也可先设置 `CUDA_VISIBLE_DEVICES`，
此时 `DEVICE=cuda:0` 指可见设备中的第 0 张。

## 结果保存在哪里？

每次创建新目录，已有目录/日志一律拒绝覆盖：

```text
outputs/g0_<时间>_<进程号>/
  run.json                         参数、软件环境、Git revision
  program/                         配对任务、source split
  natural/                         剥离标签后的自然输入与 source split
  cache/                           冻结 source/address 表示、候选及 token offset
  head/head.pt                     dev objective 最优的 G0
  head/history.jsonl               每轮训练与 dev 指标
  scores/program/test/program_metrics.json
                                   程序目标、角色准确率、A/B 合取准确率
  scores/natural/{calibration,test}/scores.jsonl
                                   无标签 raw scores 与 negative margin
  calibrated/natural/scores.jsonl   校准分数；不是正常 null p-value
  evaluation/summary.json           自然首错、span onset、全幻觉指标与 bootstrap
  evaluation/alarms.jsonl           首报警、早报、漏检、截断延迟
  summary.json                     整条 pipeline 完成后才写入
outputs/g0_<时间>_<进程号>.log       包括异常 traceback 的完整日志
```

`PROGRAM_ONLY=1` 不创建 natural/evaluation 部分。
退出成功只说明流程完成，请看测试集指标；低指标不会被改写为“方法通过”。

## 模型与数据约束

- source 与 source-blind instruction/history 分别编码；head 的 evidence value 仅来自 source。
- 每个候选读完整的 source/address 表示，不是四个汇总特征。空 source 的原始能量恒零。
- 标签是程序可验证的答案目标，不是 hallucination labels；自然数据不参与训练或选模。
- 两世界只交换当前目标和未被历史断言的 donor。历史保持真实且不变。
- 当前程序族仅覆盖 opening-day binding 和显式/指代角色；不冒充任意关系组合或自然理解。
- 损失为 binding CE + 中性位置能量方差；不用原模型概率加到 f 后训练。
- 候选 = native top-32 ∪ 所有 source token IDs ∪ 当前观测 token。无 gold candidate 补入。
- decoder 因果重放可共享整段计算，但 head 对位置 t 只读取 <t 的 slice。
- cache 绑定实际权重、tokenizer/template、提取配置/代码和全部张量的内容摘要；
  分阶段恢复拒绝混用。首次会顺序读取权重文件计算 SHA256，不是重新下载。
- 自然分数携带 source/instruction 摘要及原始数据文件身份，标签回填前必须一致。
- 校准按 task/source 长度/候选数分层、每 source 等权；小层按固定顺序回退。
  用严格左侧经验分位数处理 ties，避免恒零 detector 把所有 token 判成最高分。
- **仅 evaluate 读取原始 labels 做评估。** prepare 阶段解析原始 JSON，但不访问 labels 选样本。
- 自然任务是统一 Llama observer replay，不是各 RAGTruth 原生成模型的因果电路。
- 默认是探索性 source split。旧 discovery sources 需用 EXCLUDE_SOURCES 显式排除，
  不能把默认小样本运行称为独立确认性实验。

## 分阶段运行与恢复

所有阶段可单独执行，不需要重新采集已经完成的 cache。失败产物保留，
未完成的 cache 不生成 manifest；恢复请为失败阶段指定新的空目录。

```bash
export PYTHONPATH="$PWD/src"
python main.py --help
python main.py generate --output outputs/program --sources 64
python main.py extract --input outputs/program/train.jsonl --output outputs/cache-train --model /path/to/llama
python main.py train --train-cache outputs/cache-train --dev-cache outputs/cache-dev --output outputs/head
python main.py score --cache outputs/cache-test --checkpoint outputs/head/head.pt --output outputs/raw-test
python main.py calibrate --reference outputs/raw-calibration --scores outputs/raw-test --output outputs/calibrated
python main.py evaluate --scores outputs/calibrated --dataset /path/to/RAGTruth/dataset --output outputs/evaluation
```

上面的分阶段示例需先分别提取 dev/calibration/test cache、生成 calibration scores；
完整依赖顺序由一键脚本自动处理。`evaluate` 只接受完整自然 all-token 分数，
拒绝拿程序选定词位冒充自然首错。

## 安装与本地验证

Python ≥3.10，PyTorch ≥2.5，Transformers ≥4.46 且 <4.58。先激活正确的 CUDA 环境；
脚本不会自动重装/升级 CUDA PyTorch。需要安装项目依赖时：

```bash
python -m pip install -e ".[test]"
python -m pytest -q
python -m ruff check src tests main.py
```

测试使用临时创建的小型随机 Llama/safetensors，不下载真实权重；其结果仅证明软件流程。
远端实际 Llama 8B 的显存、吞吐和检测效果仍需运行验证。环境说明见 [运行契约](docs/RUNBOOK.md)。

## 研究资料

- [方案及未解决问题](docs/graph_method_proposal.md)
- [文献与图必要性](docs/graph_necessity_literature.md)
- [旧四格机制审计方案](docs/onset_research_proposal.md)
- [已有 graph 结果审计](docs/graph_result_audit.md)
- [本次实现契约](idea-stage/docs/research_contract.md)
- [工程验证记录与未验证边界](docs/IMPLEMENTATION_VERIFICATION.md)

数据字段依据 [RAGTruth 官方说明](https://github.com/ParticleMedia/RAGTruth)；
默认标签策略包含官方全部 spans（包括 implicit_true），衡量 source-relative 支持，
不是判断外部世界真假。
