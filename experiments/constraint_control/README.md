# Constraint-control mechanism discovery

这个子项目实现答案条件化的约束控制实验。当前完成的是 P001 捕获阶段：让模型自由采样答案，再对完全相同的 token 序列做 teacher-forced replay，验证两条执行路径的 logits 一致后保存中间状态。

后续的错误标注、首个分歧 token、答案条件化因果骨架和双向干预会沿新的公开 seam 逐项加入；当前入口不会把尚未实现的阶段报告为已完成。

## 入口与主接口

唯一入口是 `main.py`：

```bash
python -m experiments.constraint_control.main \
  --input data/questions.jsonl \
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
-> SourceDataset
-> HuggingFaceBackend.sample_and_replay()
-> GenerationRecorder.run()
-> run/index.json
```

## 输入格式

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

## 当前资源边界

P001 使用 eager attention 验证语义和索引，因此适合 20 条左右的短上下文 sanity run。attention 存储量约为 `layers × heads × response_tokens × sequence_tokens`；不要直接把它用于完整数据集。P002 将以目标 token 为终点，改为分层 Q/K/V 捕获和按需重建，避免长期保存所有 token 的完整 attention DAG。

## 文件职责

| 文件 | 职责 |
|---|---|
| `main.py` | 解析外部参数并调用主接口 |
| `config.py` | 显式配置与参数验证 |
| `records.py` | label-free message/evidence domain records |
| `dataset.py` | JSONL 读取与 source split 防泄漏 |
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
