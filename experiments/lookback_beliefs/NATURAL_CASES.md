# 烹饪／穿衣案例：读取、限定条件与中继作用

这是 `experiments/lookback_beliefs` 的**真实案例扩展**，不改原 scan/DCM 入口。
没有训练分类器，没有重构损失，没有逐事件 JVP；使用原 token 做有限干预。
选例来自既有记录，不是新的随机独立测试集。输出不是无监督检测 AUROC。

## 找回的实际数据

| 案例 | source / seed / 历史 trace | 解释边界 |
|---|---|---|
| cooking_onion | 14375 / 0 / 00012.npz | 将较早烹饪步骤的10–12分钟附给后续洋葱/啤酒步骤；后者只注明小火继续烹饪 |
| cooking_grill_control | 同一回答 | 烤肠10–14分钟在passage 2明确存在；只作为这一局部关系的正对照 |
| headdress | 14315 / 2 / 00006.npz | 原文 Only the Inca 限定被泛指 They ... and wore 省去 |
| clothes_lengths_scope_control | 14315 / 3 / 00007.npz | 膝长、踝长本身均有原文支持；性别限定省略不是自动的错误真值 |

这几个是用 RAGTruth **source passage 再次采样**的 Llama 回答，**不能套用数据集其他
response 的人工 hallucination span**。`natural_cases.json` 明确记录人工解释及局部窗口，
不用它训练或选择统计显著的 head。烹饪 passage 3 的1–2分钟是黄油中炒洋葱，不是
啤酒中继续煮洋葱的正确替代数字，因此本实验不提供一个伪造的“正确时长”。

可核对的仓库来源：
- `results/pasted_samples_20260911/samples.jsonl`
- `results/routes_20260911_091806_237/prompts.jsonl`
- `examples/decision_cases.csv`
- `src/decoding/fixed_prefix.py` 定义原状态数组
- `scripts/run_mechanism.sh` 定义下面的历史缓存路径

默认缓存：
```
outputs/samples_20260911_145421_235/
outputs/states_samples_20260911_145421_235/
```
通过 source_id + seed 查找真实 trace，而不靠文件编号猜身份。trace、source mask、原 token
序列、模型和 dtype 都核对；原文到 tokenizer offset 的往返必须精确等于全部原 token IDs。
不重新生成、重套 chat 模板、借用其他回答标签或截断原 prompt。
默认目录不在时只在唯一匹配的既有 samples 索引间回退，遇到多份匹配明确要求 --samples。

## 一键运行

从 reanchor 根目录，已激活 research 环境：

```bash
python -m experiments.lookback_beliefs.natural --resume
```

默认四个窗口，两个独立来源，原模型路径/dtype读取 samples/settings.json，cuda:0。
首先读取旧 attention/hidden，再加载同一模型进行有限干预。不会扫描全部RAGTruth。

只看已经保存的数据，不加载大模型（需要本地原 tokenizer）：

```bash
python -m experiments.lookback_beliefs.natural --phase inspect --resume
```

显式指定目录：

```bash
python -m experiments.lookback_beliefs.natural \
  --samples outputs/samples_20260911_145421_235 \
  --states outputs/states_samples_20260911_145421_235 \
  --output outputs/lookback_beliefs_natural_v1 --resume
```

可用 `--select cooking_onion headdress` 只跑两例；更改 cases/layers 等参数用新 output。
`--layers 15 19 22 26` 为本次探索层，来自此前受控实验的层区间，但**不预设自然案例也有
相同 pointer/payload 分工**。禁止最终 block/直接预测位置注入，减少直接塞入答案的解释。

## 检验逻辑

### A. 不重新推理的缓存分析

按原预测坐标 q=P+t-1 导出逐 head 的限定条件、载荷、控制、历史种子 attention；
证据内有效来源数为 2^H，H 是注意力来源分布熵，不是事实知识量。
同时保留原完整词表 logits 熵及各层 carrier 与查询状态余弦。
所有逐 token、逐 head 原数值在 NPZ；CSV 只作分区域逐 head 汇总，不是先平均头再计算。
余弦相似不证明可解码，更不证明功能性绑定。

### B. 原模型中的有符号有限作用

对指定来源集合 S，实际移除：
```
m_S(q,l) = W_O[l] concat_h sum_{j in S} a[q,j,l,h] V[j,l,h]
attention_output[q,l] -= m_S(q,l)
```
不重新归一化 attention，不删除原词，也不改模型权重。每层使用当前世界的原生 Q/K/V，
只重算少量查询行并核对 aV 与原生 o_proj 输入（允许相对误差至 .02）。
RoPE 和 GQA 均参与，原 W_O bias 保留。后续层允许正常非线性响应；这不是一阶近似。

限定条件和载荷分别移除，并加入：相同 token 数量的来源控制、两者共同移除、真实历史种子
与较早位置控制。控制不保证完全同距离或同 attention 质量，不称它是交换性检验或p值。
这只消除**选定位置的直接 V 消息**，不能声称完全擦掉事实（其他状态仍可承载它）。

读出固定为原回答 token y_t 的完整词表 log p(y_t)。竞争词取原基线最强非 y_t，冻结后不再改。
竞争词**不是正确答案**。所有结果保存干预减基线：
- 切断限定条件后原错误词更可能（delta>0）：与“限定信息曾在反对该输出”相容。
- 切断载荷后原错误词更不可能（delta<0）：与“载荷被错误陈述利用”相容。
两者都需与各自控制比较，不能单独证明机制。
另给完整词表 JS 和 C/V 双切的有限非加性 `base-cutC-cutV+cutCV`；它不是语义互信息。

### C. 中继是否承载了这份作用

预先指定一个**目标之前的已生成位置** b（如 onions、They），不是每个位置都穷举。
在 source-constraint message-cut 世界：
1. 在某层将 b 恢复成完整原运行的状态，观察后续被测词偏好是否回到原基线。
2. 反向将 cut 世界的 b 移进完整原运行，观察偏好是否朝相反方向移动。
3. 在首个目标 query 之前的另一个历史位置做恢复。这个位置位于同一切断影响区内，
   不是一个从未被切断影响的恒等对照；同时报告它的实际状态变化范数。
   它不保证语义无关或精确距离匹配，只排查恢复效果是否对所选承载位置具有选择性。

恢复的是**原运行**，它本来可能输出错误，所以 recovery 不是事实正确率。
只改 b 的整状态也不能宣称提取出了“纯指针”；本实验把状态的中介作用与真假判断分开。
切断后的其他直接路线仍被切断，所以低恢复可能来自旁路、冗余或非线性，而非证明 b 无信息。

原 `hidden[0]` 是 embedding；`hidden[l+1]` 可作为非最终 block l 输出；`hidden[L]` 已经过
最终归一化，不能拿去替换最后 block 输出。对每个缓存 carrier 做真实同世界替换检查；
若误差大于 --cache-atol (.02)，明确记录未复用该量化状态，改用本次同世界重新采集的少量
状态，不悄悄混用。原轨迹 top logits / log-normalizer 与新 baseline 差异超过 --replay-atol
(.1) 时停止并保存 checks.json，先核查模型/数值环境，不要直接放宽阈值追求“跑通”。

### D. 延续检验严格不看未来

history_seed 只取问题片段中的第一个预先指定 token；从它已经出现之后才切断对它的读取。
检查之前所有 logits 完全不变。后续仍 teacher-force 原来的文字，因此结果是条件生成中的
依赖作用，**不是自由生成纠错率，也不自动确定整个错误 span**。
`after_target` 只是目标之后的窗口，不被自动标为正常。

## 输出如何读

```
outputs/lookback_beliefs_natural_v1/index.html
  <case>/case.json                       原回答、来源短语、实际 token 坐标
  <case>/tokens.csv                      熵轨迹、目标标识
  <case>/head_reads.csv                  逐层逐头、各角色读入质量
  <case>/cached_observations.npz         完整逐head观测与几何
  <case>/effects.csv                     各切断／恢复对原输出偏好的变化
  <case>/mediation.csv                   原carrier和控制位置的恢复对比
  <case>/checks.json                     原缓存复现、数值误差、是否复用缓存状态
  <case>/mechanism_summary.json          有符号效应，不自动输出机制成立标签
  <case>/<arm>.npz                       每个token的logp、margin、entropy、JS
  <case>/<cut>.messages.npz              carrier/首个事实查询的逐头pre-WO删除代码
```
后者乘回对应 head 的 W_O 即为删除消息；它是选定来源集合的逐head和，不是全量逐边消息库。
正常情形默认每窗口约29次有限前向（1 baseline + 1零切 + 7切断 + 4层×5状态对照），
不是对每一个事件再做一次全路径JVP。只保存小状态与窗口数值，逐例释放；**未测真实8B时间**。

数据量只有2个来源。不得将这些图、恢复分数或软件测试写成总体检测性能；没有自动发现
正确事实的算法，也没有要求人工提供新的 binding packets。新的检验结论要看实跑结果。

## 软件验收

```bash
python -m pytest experiments/lookback_beliefs/tests/test_natural.py -q
```
测试包含原身份、禁止末层规范化状态误注入、逐head消息相加、QK/RoPE/GQA重建、零干预、
双向状态置换、历史切断的前缀不变、hooks清理、原参数不变与完整小模型输出。支持安装了
transformers时额外跑随机tiny Llama；本地缺此依赖时明确skip，不冒称跑了自然样本。
