# Lookbacks to Track Beliefs：指针、载荷与绑定干预

**这是论文核心机制实验的独立 PyTorch 实现，不是 RAGTruth 幻觉检测器，也不是官方代码的完整复现。**
基于本会话提供的 `lookback_patching_demo.py` 整理；不修改 reanchor 现有入口或模型权重。
不需要 NNsight、NDIF、API key、新模型下载或预先跑完整 attention/JVP。模型仅从本地加载。

## 1. 实验究竟检验什么

两次正常前向得到 base / donor 激活；将 donor 的指定残差状态移植到 base，
让**原模型剩余层继续执行**。这是有限激活干预，不是最终 LM head 的静态投影。

| base | donor（换顺序及取值） |
|---|---|
| Bob → bottle → beer | Carla → cup → tea |
| Carla → cup → coffee | Bob → bottle → water |
| 查询 Carla/cup：coffee | 同一查询：tea |

若移植的是“第一个状态”的 **pointer**，base 应转而输出 **beer**；若移植的是
**payload**，应输出 **tea**。前者既不是 base 答案，也不是 donor 答案，能区分两种假设。
程序同时记录 pointer/payload 两种目标的逐层 IIA，不挑最高的一层冒充已经证明机制。

- `--experiment answer`：在最终输入 token 移植整状态，沿层扫描指针到载荷的变化。
- `--experiment binding`：保持两个状态词相同，只反转记录顺序，在两边**相同状态词**
  位置同时替换；比较另一条绑定的答案。这个模板是 state address/payload 操作的简化适配。
- `--mode dcm`：在明确选定的一层，学习小子空间的稀疏 mask，检验语义干预能否实现。
- 自定义 JSONL 支持多位置替换和指定层的 base 状态恢复，可组织 visibility 等对照。
  **未内置**全部 visibility 数据、attention knockout、逐层累计 source 替换、BigToM 或论文图表。

本生成器与作者完整 CausalToM 不同；8B 的层号、行为与 70B/405B 不能直接等同。
程序不将自然回答的幻觉标签用于生成、训练或评价这些机制实验。

## 2. 从 reanchor 根目录运行

```bash
# 不加载大模型：展示高层机制应当给出的不同答案。
python -m experiments.lookback_beliefs.run --mode symbolic

# 默认从已有 population/settings.json 读取 observer 模型路径。
# 每对样本：两次 baseline + 一次 self-patch 检查 + 每个待测层一次真实干预。
python -m experiments.lookback_beliefs.run \
  --mode scan --experiment answer --samples 16 \
  --output outputs/lookback_beliefs_answer_v1 --device cuda:0 --resume

# 单卡本地模型可显式指定；不套 chat template 是默认行为。
python -m experiments.lookback_beliefs.run \
  --mode scan --experiment binding --samples 16 \
  --model /path/to/local/Meta-Llama-3.1-8B-Instruct \
  --output outputs/lookback_beliefs_binding_v1 --device cuda:0
```

默认扫描模型全部层（零基编号），用 `--layers 10 15 20` 可先检查少数层；没有套用论文的
70B 固定层段。`--chat-template` 明确启用模型模板。默认 dtype=bfloat16、batch=1，
只保留目标位置的 CPU 激活，LM head 仅计算最终位置 logits；不保留完整 attention 矩阵。
仍需本地模型能正常装入显存，DCM 的下游反向图有额外开销；没有实测加速/显存保证。

`--resume` 只续跑相同设置下已完整保存的**逐层扫描**。DCM 使用新输出目录，不假装能从
不完整训练步骤续跑。不要两个进程同时写同一输出目录。

## 3. DCM：学习的不是重构

首先按 `source_id` 将样本拆成 train/validation。默认只保留 base 与 donor 都回答正确的
**训练对**；验证对全部保留，报告无条件及 both-correct 条件下的结果与分母。没有反复补采
直到凑够成功样本，也不把原始错误的样本悄悄丢掉。`--include-errors` 是显式的训练对照。

从训练对的激活做未中心化 SVD 得到正交行基 B。**这是本实现本地估计的候选基**，
不是作者未随仓库提供的奇异向量。实际秩受训练样本的数值秩限制，输出记录有效秩。

```
P(m) = (diag(m) B)^T (diag(m) B)
h_patch = h_base + (h_donor - h_base) P(m)
loss = -logit(语义干预目标) + lambda * ||m||_1
```

和官方脚本一样，连续 mask 的系数是 m²，而不是 m；训练后 clamp 到 [0,1] 并 round。
不生成 D×D 投影矩阵，实际计算为 `((delta @ B.T) * m.square()) @ B`。
只优化 mask，语言模型 `eval().requires_grad_(False)`；梯度经过原模型后续计算传回 mask。
目标是**反事实语义监督**，不是完全无监督算法；没有训练节点/边重构器。

```bash
# 20 只是明确指定的层示例，不是声称 8B 的指针位于该层。
# 发现层段与验证尽量使用不同来源；不要根据验证结果重复选层和稀疏系数。
python -m experiments.lookback_beliefs.run \
  --mode dcm --experiment answer --samples 64 \
  --layer 20 --hypothesis pointer --rank 32 --epochs 1 \
  --output outputs/lookback_beliefs_dcm_l20_v1 --device cuda:0
```

验证同时输出 `full_state`、`dcm`、`zero`（空子空间）和 `rank_matched_random`。
零 mask 应恢复 base 输出；随机对照在相同 SVD 候选基中随机取相同数量的方向，
不是随机方向的多次显著性检验。当前未做多 seed 统计或论文量级的 GPU 实验。

## 4. 输入、输出与指标

内置模板的 token 位置由 tokenizer offsets 定位。完整输入止于 `Answer:`，不含答案。
候选必须是与前缀对齐的**单个 continuation token**；多子词会报错，不偷偷取第一个子词。
候选概率使用完整词表 softmax，不在小候选集合内重新归一化。greedy 命中则比较实际预测词
经空白/大小写归一化后的字符串。保存精确 token IDs 以便核查词首空格差异。

输出目录：

```
config.json / pairs.jsonl       完整设置和具体输入故事
samples/<pair>.npz             层、候选概率/logits/margin、实际token/干预位置
samples/<pair>.json            greedy 输出、base/donor 正确性、每种假设命中
summary.json / curves.csv      逐层/逐干预臂 IIA（没有全数据挑最佳层）
subspace.npz / split.json      DCM 专有：候选基、soft/binary mask、损失、来源划分
complete.json                  此次完整运行结束
```

`IIA = 干预后输出符合指定高层干预预测的比例`，不是原输入的准确率、AUROC 或幻觉率。
每行包括 `n_all/iia_all` 与 `n_both_correct/iia_both_correct`。后者分母为零时为 null。
原始模型不擅长此任务时先看 baseline 和分母，不能宣称无效干预就是没有这个机制。

自定义输入用 `--pairs custom.jsonl`，每行字段见 `examples/pair.jsonl`：
- `base_prompt/donor_prompt`：已格式化完整输入；除非指定 `--chat-template`，不额外包装。
- `base_answer/donor_answer/hypotheses`：各自答案与高层干预预测，不默认 donor 就是干预目标。
- `sites`：`{"base":"last","donor":"last"}`，或选择器 `{"text":"coffee","occurrence":0}`，
  或 `{"span":[start,end]}`；两边选中 token 数必须一致。
- `restore` 可选：如 `{"layer":25,"base":"last"}` 将该层该位置恢复成 base 状态。
  同层恢复与干预重叠直接拒绝。跨层恢复是控制，不要在无解释时用于掩盖干预。

## 5. 阅读与调试顺序

`data.py → patching.capture_states → patching.patch_logits → run.summarize`。
DCM 再读 `subspace.svd_basis / fit_mask`；没有改动原项目 main.py。
在 `patching.py` 的 `updated[:, positions] = replacement` 附近打断点，观察 base/donor
选择位置与张量。命令通过 `python -m` 从仓库根目录运行。

```bash
python -m pytest experiments/lookback_beliefs/tests -q
```

本地软件测试覆盖：第三答案、相同词项绑定交换、offset定位、空/整子空间、m²与稠密投影
等价、hook清理、真正下游梯度只到mask、base恢复、来源留出、逐样本保存和扫描续跑。
软件通过不代表真实模型已经表现出指针/载荷阶段。

## 来源与范围

- Prakash et al., *Language Models use Lookbacks to Track Beliefs*: https://arxiv.org/abs/2505.14685
- 作者说明：https://belief.baulab.info/
- 官方代码：https://github.com/Nix07/belief_tracking/tree/0579347e3cf963d13d55edf041c7b595b0dcd88b
- 对应函数：`get_reversed_sent_diff_state_counterfacts()`、`get_reversed_sentence_counterfacts()`、
  `validate()`、`get_low_rank_projection()`（single-layer patching）。

只复用所述实验思想与公式，未复制作者完整数据和脚本；简化模板、自制 SVD 基、单样本
PyTorch hooks、结果保存是本实现的适配。作者的硬编码token位置、NDIF和隐藏依赖没有照搬。
