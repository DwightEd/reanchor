# Reanchor：条件绑定支持模型 G0

主入口现已改为 **G0 训练与检测流程**，不再把四个 factorial 坐标当作图学习模型。
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
