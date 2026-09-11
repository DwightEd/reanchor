# Reanchor：出错前在回看什么

入口只有 `main.py`：

- `sample` → `SamplingExperiment.run()`：加载模型，逐 token 采样，在同一次前向保存 attention 和候选 logits。
- `inspect` → `AttentionAnalysis.run()`：读取保存的数据，输出具体 token 选择和历史读取变化。

算法在 [sampling.py](src/reanchor/sampling.py) 和 [attention.py](src/reanchor/attention.py)。
`io.py` 只负责 JSON 读写及拒绝覆盖已有输出。没有训练、校准、分组评估、bootstrap 或 report。
旧实验可从 Git 历史恢复；`docs/` 中的研究笔记不是当前实现说明。

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
NPZ 不保存当前分析没有使用的 hidden states、entropy 和采样分布派生指标。
完整 attention 以 float16 保存，logits 以 float32 保存，分析小于量化精度的变化时应谨慎。
分析只读取 NPZ，不再加载模型或 tokenizer，也不需要 GPU。
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
