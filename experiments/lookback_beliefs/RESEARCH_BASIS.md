# 文献依据与本实现的边界（2026-09-15）

以下将原论文内容与本实验选择分开。新审计不是这些论文的完整复现，更不是把它们的
公式放到自然RAG上就得到的检测定理。

## 1. Hallucination is a Consequence of Space-Optimality

原文：https://arxiv.org/html/2602.00906v7

随机稀疏事实成员判断；满足给定失真条件所需的最优每事实存储由事实/非事实分数分布
之间的最小KL刻画。对数损失下存在真事实与部分非事实高置信分数碰撞。它研究的主要
瓶颈是事实记忆，外部非参数记忆会改变信息条件。没有定位Transformer指针/地址/载荷，
没有证明每次RAG绑定错误来自参数容量。

本实现采用的启发：不将下一词高置信当作事实依据，独立测任务要求的选择变量。
没有把注意力范数转换为知识bit数，没有从B/n下界推断某个head的幻觉。

## 2. A Theory of Usable Information Under Computational Constraints

原文：https://arxiv.org/abs/2002.10689

Xu等的predictive V-information将观察者函数族与计算约束纳入可预测信息的定义。
计算可以增加相对于受限读出器的可用信息，不应把“可解码”与“实际下游算子可用”混同。

本实现只用reference拟合的小语义读出，在heldout上报告1-CE/log(2)，并与原QK信道
及组合干预并列。它是一个读出实例给出的经验下界/可预测性测量，不能宣称算出了整个
函数族上的最优V-information，更不能把差值当作确定丢失的bit。

## 3. Language Models Use Lookbacks to Track Beliefs

原文：https://arxiv.org/html/2505.14685v3
项目：https://belief.baulab.info/

CausalToM控制条件中，互换选择引用与互换载荷给出不同的预期答案，尤其是donor选择
在接收故事中取回第三个内容。指针和地址只需经Q/K后相容，不必在原表示空间相同。
低秩干预是功能验证，不只看探针准确率。作者的成功任务证据不直接解释自然幻觉。

本实现保留原scan/DCM复现为基准；新审计将Q/K/V分别固定或改变，并用C/B/O的
组合预测、同绑定跨问法、内容更换和随机子空间作对照。完整Q/K不叫纯指针/地址。
候选子空间来自配对差分而不是论文的DCM优化，因此不是声称复制了官方低秩结果。

## 4. How do Transformers Learn Implicit Reasoning?

原文：https://arxiv.org/html/2505.23653

受控隐式组合推理任务中，中间实体早已可读，跨上下文一致且能被下游组合使用的状态
几何更接近推理能力的形成。语义patch应产生接收问题对提供者中间实体的组合结果，
而非照抄提供者答案。

本实现学习对记录身份变化敏感、对顺序及保持目标的联合变换相对稳定的候选空间，
在未见payload/组上检查行为。仅SVD几何好看不计为成功。自然任务是否复用该空间
未预设成立。

## 5. Attention Deficits in Language Models: Causal Explanations for Procedural Hallucinations

原文：https://arxiv.org/html/2602.19239

研究受控长上下文任务中的门控/回答模式与候选绑定错误，比较可用与被利用的信息，
结合错误样本上的读出和干预。这个区分促使本实现把top1不在候选集合与真正候选选错
分开，且不能仅在原模型都答对的配对上解释错误。

需要格外小心信息论方向：高错误率不推出低互信息。均衡二元J=1-C有1bit且全部
答反；Fano给出的必要界不能反向当作信息少的证据。本代码加入这个反例测试，
同时报告MI与正确方向，避免由高MI宣布理解了真实约束。

## 6. A Mathematical Framework for Transformer Circuits / In-context Learning and Induction Heads

原文：https://transformer-circuits.pub/2021/framework/index.html
原文：https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html

QK描述读取选择，OV描述内容写回。Q/K/V-composition强调前序层写入如何形成后序
query/key/value。该分析启发本实现把前一层attention/MLP写入与下一层QK变化连接，
不再将“最终attention热图”冒充指针形成过程。RMSNorm、RoPE、GQA使用原生实现，
不将简化线性电路当作整个Llama的精确计算图。

## 7. Information Flow Reveals When to Trust Language Models

官方代码：https://github.com/rxu0112/RAG-information-flow

作者用内部贡献/信息流评价来研究可靠性，并比较基于路径与相关性的信号。它不支持
“凡是删除来源后logits有变化就发现了幻觉机制”。本次不重新复现通用源删除，而明确
区分条件变量、匹配地址与被运用的内容。旧来源切断结果保留为历史，不作为新方法创新。

## 8. How Does Reasoning Flow? Tracing Attention-Induced Information Flow for Targeted RL in LLMs

原文：https://arxiv.org/html/2606.10646

FlowTracer构造attention诱导DAG，以目标可达势作Doob h重加权，从输入前向分配
流量，并据此调整有可验证奖励的token级强化学习更新。它不是节点边重构模型，也
不是自然幻觉的无监督分类器，聚合attention路径仍非完整残差/MLP因果电路。

本次仅借鉴目标相关对比与信号应有明确用途；不把它的RL结果或路径概率叫作本实验
效果。graph中的原FlowTracer子模块不属于这次reanchor局部重构范围。

## 本次自己的推导，而非文献原结论

固定接收坐标和两个候选key时，log(a_B/a_A)=q·(k_B-k_A)/sqrt(d)，Q、K双变化
可精确拆成Q项、K项、双线性交互。地址差张成的子空间决定这两个位置的相对logits。
这些是代数恒等式，测试通过不算发现。研究结果必须是语义条件相关的状态改变经过
哪些分量、是否按正确关系改变检索与最终选择，且在独立组和正常/错误比较中成立。
