# 从已保存 logits 检查语义选择：不重新跑模型

之前的 `export_semantic_readout.py` 只作为会话附件提供，仓库没有该文件，因此服务器报
`can't open file ... [Errno 2]`。现在根目录保留同名兼容入口，实际实现统一在
`experiments/lookback_beliefs/routing_readout.py`，避免两份脚本分别维护。

## 直接运行

在 reanchor 根目录，已激活 research 环境：

```bash
git pull --ff-only origin main
python -m experiments.lookback_beliefs.routing_readout \
  --audit outputs/lookback_routing_v1 --decode
```

原先的命令现在也能运行：

```bash
python export_semantic_readout.py --audit outputs/lookback_routing_v1 --decode
```

不使用 `set -e`。`--decode` 只加载 config.json 指向的本地 tokenizer，不加载模型权重。
换了服务器可加 `--model /本地tokenizer目录`。缺少 tokenizer 时保留数值 ID 并报告
`tokenizer_status=unavailable`，不联网下载、不伪造 token 文本。不解码时只依赖 NumPy。

## 这次实际修正的判读问题

1. **不再只读取做了干预的两个受控样本。** 导出所有已保存 baseline 的全部观测位置。
   留出核心统计排除 content/style 对照；每个 query 只计一次，不因层/head重复计数。
2. **候选内偏好与全词表选择分开。** 同时输出两候选概率、总质量、名次、log-odds、
   指定目标的全词表概率、以及 top-k 具体文字。top1不在候选内不能自动算作幻觉，
   候选内排序正确也不能充当完整答案准确率。只观察下一词不能判断整段生成是否答错。
3. **保留后续位置的作用。** 原先汇总只读干预点；新表导出该运行保存的所有位置，
   标记 before/at/after。在commitment做干预、观察后面10/12，是原干预的后续作用，
   不是在10/12上执行了新干预。
4. **饱和概率下仍可观察支持变化。** 同时给出 log p(y) 和
   `logodds_rest = z_y - logsumexp(z_{v != y})`；不通过 `log(p/(1-p))` 计算，避免
   浮点概率已经舍入为1时溢出。竞争集合是完整其他词，不冒称正确答案。
5. **V的新载荷目标单独评分。** 即使目标不在接收者的两个候选中，也输出其真实概率和
   排名，不再拿原候选判定载荷交换成败。目标只在原声明干预点有含义，不随意传给后续词。
6. **同案例、同干预点、同观测点、同head进行对照。** 输出q/k子空间与相应随机方向、
   同绑定或改写对照的差值。它们是描述性差值，不是显著性检验，不自动证明语义特异性。
7. **不再用source anchor当自然真值。** 自然案例的12/14等是来源代表词，不是当前词
   的真假二选一。自然行的correctness字段留空，保留人工案例状态但不构造token标签。
8. **明确指针辨识边界。** v1的Q donor只改变条件、没有更换载荷。若干预目标等于donor
   的答案，该输出不能单独区分“指针”和“内容”。表中报告这个重合，不把target_hit
   当作纯指针证据。此次没有追加新donor、新子空间训练或GPU实验。

## 输出

```
outputs/lookback_routing_v1/semantic_readout_review/
  baseline_readout.csv        所有原生/对照世界的基线逐位置读出
  candidate_details.csv       每项干预在全部已保存位置的候选/目标/原词变化
  top_tokens.jsonl            原生top-k、干预前后top-k及可选token文字
  control_comparisons.csv    相同位置/通道上的真实操作减对照操作
  natural_world_effects.csv  原始材料与改变绑定/保义改写的完整世界比较
  export_status.json         分母、候选外首选频数、缺失/无效文件及版本
  cases.json                 原样本与反事实定义，便于核对目标和条件
  README.md                  结果阅读提示
```

自动生成 `outputs/lookback_routing_v1/semantic_readout_review.zip`，包含这些表与现有关键
汇总，不包含大NPZ、模型权重。原始capture/interventions/config/旧结果完全不修改。
重复运行只更新导出目录的指定文件和结果包；不会把目录里无关文件打进zip。

先看 baseline_readout.csv / export_status.json：受控测试中有无真正选错候选的运行？
再看 candidate_details.csv：Q确实改变路由时，输出候选margin有没有相应改变？
然后看 control_comparisons.csv 和后续位置：效果是否超过同类扰动，能否影响后续选择？
自然完整世界的差异必须与单head干预分开；改写材料使原陈述有依据不叫原事实下纠错。

## 缓存与完整性

此读出独立于采集代码的版本hash，所以不会因新增读出要求重跑已完成的172次回放。
不改 routing_data/math/engine/audit 四个采集文件，不更改已有resume身份。
捕获文件中存在case_signature时核对原始case；始终核对原坐标、候选ID范围、query和词表
维度。先于干预点的logits与same_world必须在原数值界内保持不变。

缺文件不会填0：继续导出其余可用部分，并记录missing；坏坐标/数值记录invalid，
不把该操作的部分位置混入结果。缺总CSV时可以读取interventions/*.json的已完成操作。
正常完整退出码0；缺失、未完成或无效退出码2，结果包仍保留。`--allow-partial`只允许
缺失/未完成，不能允许无效数据。只有顶层CSV/JSON没有NPZ的包，不能恢复完整词表。

测试：

```bash
python -m pytest experiments/lookback_beliefs/tests/test_routing_readout.py -q
```

本次软件验证使用构造logits、临时磁盘目录及用户提供的真实metadata结构。没有本地
8B权重或服务器capture/interventions NPZ，因此没有新增自然机制结论或检测成绩。
