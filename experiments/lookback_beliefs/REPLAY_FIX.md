# 0.25 缓存复现误差：先对齐计算路径，再解释干预

原采样的 `sampling._generate()` 是 prompt prefill 后逐 token、use_cache=True，
模型原 LM forward 给出 logits。旧 natural 审计却一次送入完整前缀，
use_cache=False，再将若干预测位置分块送入 LM head。两者数学上因果等价，
但低精度矩阵计算的形状和执行次序不同，不保证逐位一致。

用户报告的0.25是所存top候选logits或log-normalizer的最大绝对差，
不是概率差25%，也不是幻觉机制效应。不能仅凭0.25认定一定是BF16误差。
本修复改变执行路径，不提高原 .1 的阈值，不覆盖或改写旧采样数据。

## 修改

- `natural_replay.py`：同原采样，prefill一次，然后依次输入原保存token。
  不重新采样，不apply chat template，不复用另一干预世界的KV。
- 全部 baseline、限定/载荷V消息切断、carrier恢复均采用同一执行路径。
  使用原模型返回的eager attention及真实V；不再以另一个QK矩阵形状
  重构这些权重。W_O bias不删除，逐head删除代码仍保留。
- carrier替换只作用于其实际被输入那一步；后层KV正常记录替换后的结果。
  同层先前计算出的KV不能被错误地当成后层更新。每个世界从空KV开始。
- `replay_tokens.csv` 拆开每个原预测位置的top-logit误差、log-normalizer
  差、对应log-prob误差、top1与量化到原float16存储方式的attention差。
  `checks.json` 保存执行模式、torch/transformers/CUDA、dtype及各自误差。
- 任一原缓存检查失败，在切断前停止。共同logit偏移即使不改变概率也只报告，
  不以此自动绕过原缓存校验。若KV回放仍不符，需核对本地权重与依赖版本。
- 新默认输出为 `outputs/lookback_beliefs_natural_v2`。原案例定义、统计输出
  格式和审计目标不变（仍是native-message-cut schema v1）；config.execution
  纳入续跑身份，旧目录不可混入新结果。`--execution full`仅用于旧参考对照。

完整输入会逐token重复执行多次干预，比整段并行回放慢；不是逐事件JVP，
只针对四个预选窗口。不声称有加速或实测显存保证。

## 运行（不使用 set -e）

在reanchor根目录，research环境中：

```bash
python -m experiments.lookback_beliefs.natural \
  --phase all --execution incremental_kv \
  --output outputs/lookback_beliefs_natural_v2 --device cuda:0 --resume
```

成功完成后：

```bash
python -m experiments.lookback_beliefs.mechanism_review \
  --audit outputs/lookback_beliefs_natural_v2
```

不删除旧samples/states，不需要重新生成回答；第一次新审计重做未完成的案例。
新目录内以后可按完整案例续跑。不能续接未完成的单个切断臂。

## 这项审计不是“替换后答案变了，因此证明了指针”

隐藏状态是载体；pointer/address/payload是通过语义干预定义的功能。
论文关键的区分是：把“第一个记录”的选择从donor移入base时，
base输出自己的第一条记录内容，而不是donor的答案词。
该第三答案不是任意层扰动的必然后果；完整状态替换仍不能证明纯子空间。

当前自然实验只检验指定来源集合V/WO写入对原词的有符号有限作用及中继恢复。
限定切断使原错误更容易、载荷切断使其更难，且同一词上超过对照，
才与限定抑制/载荷支持的分离相容。它不区分所有QK错误来源、不发现真值，
也不能证明当前carrier是纯指针或直接修好了错误。

“移除来自限定token的V消息”不是“移除所有限定语义”：后面token的状态
可能已整合限定，QK匹配也可能携带它。弱效应只能记为该干预未检出。

本地只完成PyTorch软件测试；没有服务器8B权重与原缓存，没有新自然机制结果。
可选HuggingFace随机tiny Llama测试仅在安装transformers时运行。

依据：PyTorch Numerical accuracy（Batched computations or slice computations）；
Language Models Use Lookbacks to Track Beliefs（arXiv:2505.14685，§3.3与§5.1）。
