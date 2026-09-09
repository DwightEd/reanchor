# Constraint-control mechanism discovery

这个子项目实现答案条件化的约束控制实验。当前完成的是 P001 捕获阶段：让模型自由采样答案，再对完全相同的 token 序列做 teacher-forced replay，验证两条执行路径的 logits 一致后保存中间状态。

后续的错误标注、首个分歧 token、答案条件化因果骨架和双向干预会沿新的公开 seam 逐项加入；当前入口不会把尚未实现的阶段报告为已完成。

## 入口与主接口

唯一 Python 入口是 `main.py`。服务器上直接运行默认的 RAGTruth QA
smoke capture：

```bash
conda run --no-capture-output -n research \
  bash scripts/run_constraint_control_p001.sh
```

`meta-llama/Llama-3.1-8B-Instruct` 是 gated model。首次使用时，先在其
[Hugging Face 页面](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
接受许可，再在同一个 `research` 环境登录并确认身份：

```bash
conda run --no-capture-output -n research hf auth login
conda run --no-capture-output -n research hf auth whoami
```

如果服务器已有可访问的本地权重，不需要登录 Hub：

```bash
MODEL=/absolute/path/to/Llama-3.1-8B-Instruct \
OUTPUT_ROOT=runs/p001_ragtruth_qa_local \
  conda run --no-capture-output -n research \
  bash scripts/run_constraint_control_p001.sh
```

脚本默认读取
`/share/home/tm902089733300000/a903202310/lys/data/RAGTruth/dataset`，输出到
`runs/p001_ragtruth_qa_llama31_8b`，并运行 Llama-3.1-8B-Instruct、RAGTruth
train split 的 20 个 QA source、每个 source 三个采样 seed。需要改变任务、划分
或数量时使用环境变量，例如：

```bash
TASK=Summary SPLIT=test MAX_SAMPLES=5 \
  conda run --no-capture-output -n research \
  bash scripts/run_constraint_control_p001.sh
```

完整参数调用为：

```bash
python -m experiments.constraint_control.main \
  --input /share/home/tm902089733300000/a903202310/lys/data/RAGTruth/dataset \
  --input-format ragtruth \
  --task QA \
  --split train \
  --output runs/p001_llama \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --device cuda:0 \
  --dtype bfloat16 \
  --seeds 0 1 2 \
  --max-new-tokens 64 \
  --temperature 0.7 \
  --top-p 0.9 \
  --top-k 20 \
  --max-samples 20
```

执行路径保持为：

```text
parse arguments
-> ExperimentConfig
-> ConstraintControlExperiment(config).run()
-> RagTruthDataset（或通用 SourceDataset）
-> HuggingFaceBackend.sample_and_replay()
-> GenerationRecorder.run()
-> run/index.json
```

## RAGTruth 输入边界

RAGTruth 原始目录必须同时包含：

- `source_info.jsonl`：读取 `prompt`、`task_type`、问题和外部材料；
- `response.jsonl`：只读取 `source_id` 和 `split`，用于保留官方 train/test
  source 划分。

P001 不把 RAGTruth 已有的 `response`、`labels`、`quality` 或 generator model
输入模型，也不靠这些字段选择样本。原因是本实验会让 Llama-3.1-8B-Instruct
重新自由采样，新答案与 RAGTruth 原答案不是同一段文本，原有字符级标签不能直接
移植。新答案是否幻觉必须在捕获后单独判定。

适配器把 RAGTruth prompt 转成两条消息，并给出可复核的字符区间：QA question
和任务指令为 `constraint`，每个 passage 为 `content`；Summary 的任务指令为
`constraint`、文章为 `content`；Data2txt 的任务指令为 `constraint`、structured
data 为 `content`。

## 通用 JSONL 输入格式

输入是 UTF-8 JSONL，每行一个 source-grouped record：

```json
{"sample_id":"q1","source_id":"doc-1","split":"discovery","task":"QA","messages":[{"role":"system","content":"Answer only from the supplied evidence."},{"role":"user","content":"Use 2020. Ada won. Who won?"}],"evidence_units":[{"unit_id":"constraint-1","kind":"constraint","text":"2020","message_index":1,"char_start":4,"char_end":8},{"unit_id":"content-1","kind":"content","text":"Ada won","message_index":1,"char_start":10,"char_end":17}]}
```

约束如下：

- 同一个 `source_id` 不能跨 discovery/calibration/confirmation split；
- 同一 source 的多个采样 seed 始终由同一次运行产生并留在同一 split；
- capture 输入不包含 correctness、error span 或 gold candidate；
- `evidence_units.kind` 只能是 `constraint`、`content` 或 `other`。
- 每个 evidence unit 必须用 `message_index` 和左闭右开的字符区间定位；区间文本必须与 `text` 完全一致。

## 输出格式

```text
runs/p001_llama/
|-- index.json
`-- trajectories/<split>/<task>/<sample_id>/seed_<seed>/
    |-- trajectory.json
    `-- capture.npz
```

`trajectory.json` 保存输入 identity、模型 revision、采样参数、回答、停止原因和 replay 误差。`capture.npz` 当前包含：

- 完整 `token_ids`、token pieces、`special_mask`、`response_start` 和 prediction row positions；
- generation/replay 的 emitted-token logits；
- raw-model log probability、sampling-distribution log probability 和 entropy；
- 每步 raw logits 的 top-k token ids/logits；
- replay 的逐层 residual states；
- replay 的 eager attention weights。

生成完成前不读取任何结果标签，manifest 明确记录 `labels_used_for_capture=false`。如果 generation 与 replay 的全词表最大 logit 误差超过 `--replay-atol`，该 source/seed 会标成 `rejected`，不会进入后续机制分析。

运行时 `tqdm` 以 trajectory 为单位显示总进度，并在 postfix 中显示当前 `sample_key` 与 seed。脚本会在输入文件不存在或目标目录已经包含完整 `index.json` 时直接退出，避免误跑和覆盖已完成实验。

## 当前资源边界

P001 使用 eager attention 验证语义和索引，因此适合 20 条左右的短上下文 sanity run。attention 存储量约为 `layers × heads × response_tokens × sequence_tokens`；不要直接把它用于完整数据集。P002 将以目标 token 为终点，改为分层 Q/K/V 捕获和按需重建，避免长期保存所有 token 的完整 attention DAG。

## 文件职责

| 文件 | 职责 |
|---|---|
| `main.py` | 解析外部参数并调用主接口 |
| `config.py` | 显式配置与参数验证 |
| `records.py` | label-free message/evidence domain records |
| `dataset.py` | JSONL 读取与 source split 防泄漏 |
| `ragtruth.py` | 原始 RAGTruth prompt、官方 split 与证据/约束单元适配 |
| `model.py` | Hugging Face 自由采样与精确 replay |
| `generation.py` | fidelity gate、artifact schema 与原子落盘 |
| `experiment.py` | records × seeds 的线性编排及 manifest |
| `tests/` | 从公开接口验证行为 |

## 验证

```bash
python -m pytest -q experiments/constraint_control/tests
python -m pytest -q
python -m ruff check src tests experiments/constraint_control
```
