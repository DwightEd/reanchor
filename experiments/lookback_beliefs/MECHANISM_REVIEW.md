# 读到了，但用错适用条件：怎样检查已有干预结果

当前自然案例主实现是 `350247a9` 的 `natural.py/natural_engine.py/natural_data.py`。
它已经在 main 中，采用原输入上的 V/WO 消息移除和双向 carrier 恢复。
**不要再应用旧 `reanchor_natural_lookback.patch`：那个补丁使用 evidence echo，
与当前原生消息干预不是同一种实验，并会覆盖同名入口。**

本次新增 `mechanism_review.py`，只读取现有自然实验的冻结输出，不修改任何原结果、
不重新跑模型、不训练分类器、不读 RAGTruth 幻觉标注。输出描述性符号核验，不声称
已经发现正确答案、证明纯指针、修复原回答或获得总体幻觉检测成绩。

## 先运行，再直接读取判读

在 reanchor 根目录、research 环境下：

```bash
python -m experiments.lookback_beliefs.natural --phase all --resume
python -m experiments.lookback_beliefs.mechanism_review \
  --audit outputs/lookback_beliefs_natural_v1
```

已有自然实验完成时，只运行第二行；不加载 tokenizer 或模型。
默认自然案例是 cooking_onion、cooking_grill_control、headdress、
clothes_lengths_scope_control，来自两个独立来源。缺少的案例报告 pending，
数值或身份不一致报告 invalid，不填零冒充机制未出现。

输出：`mechanism_review.json`（逐词、逐层判读）及 `mechanism_review.md`（索引）。
原来的 case.json、effects.csv、mediation.csv、NPZ 都保持不变。

## 为什么“关注到事实”仍然会错

烹饪例中数字10–12来自较早的烹饪步骤；生成却把它用于取出香肠后的洋葱/啤酒步骤。
材料只说小火继续烹饪洋葱，没有给这个步骤唯一时长。1–2分钟属于黄油中炒洋葱，
不能作为啤酒继续烹饪的正确答案。头饰的内容来自原文，但 Only the Inca 的主体限定
在泛指 They 的续写中被省去。这些是原文/回答的人工关系判断，不是QK给出的真值。

因此至少要区分：

1. **内容被读取与用于输出**：模型取到了哪个值或描述。
2. **适用条件参与控制**：对象、动作、阶段、主体范围是否限制这个值的使用。
3. **后续沿用什么前提**：低熵续写是符合当前前缀，不等于符合原文约束。

不同原因都能产生表面错绑：query 没带全条件、source key/address 绑定不充分，
条件已到达但被竞争路径压过，或限定信息经QK而非当前移除的V路径起作用。
本版不能只靠attention或V切断将这些原因彻底区分。

## 对同一个目标词计算，而不是拼接不同词上的作用

设 `f=log p(原回答词 | 原固定前缀)`，`C` 是限定条件消息，`V` 是载荷消息。
每个比较保持原回答词、完整词表分布与前缀对应不变：

```
delta_C = f(remove C) - f(base)
delta_V = f(remove V) - f(base)
specific_C = delta_C - delta_control_C
specific_V = delta_V - delta_control_V
```

- `delta_C > floor` 且 `specific_C > floor`：该词呈现针对性的限定消息反对作用。
- `delta_V < -floor` 且 `specific_V < -floor`：该词呈现针对性的载荷支持作用。
- **同一词**同时满足，才标记 `joint_sign_pattern`。
- 不同词上的相反效应不能被均值拼成“模型同时知道真假”。
- 两个对照匹配token数量，不是严格匹配距离、消息能量或语义；差值不是显著性检验。
- 条件作用小标为 `constraint_unresolved`，不能推断没有编码、没有读取、完全没用。

分数的 `floor` 取显式最小量（默认0.001 nats）和已观测数值误差界的较大者。
这只排除小于数值精度的符号，不能校正分布外干预或其他系统偏差，不是p值。
`finite_nonadditivity = f(base)-f(remove C)-f(remove V)+f(remove C,V)`
是有限干预非加性，不是语义互信息、率失真下界或模型内部知识bit数。

## 中继作用怎样检验

对每一层、每个目标词同时检查：

- cut 世界恢复原 carrier，输出偏好按相反于cut的方向且更接近base；
- 完整世界置入cut carrier，偏好沿原cut方向改变；
- carrier恢复比另一个也受cut影响的历史位置更接近base；
- 控制位置状态确实改变过，不能用恒等位置作为强对照。

`recovery_fraction` 仅在原cut效应不小于数值界时输出，允许超出[0,1]，不裁剪。
恢复的是原运行（它可能本来就错），因此不是“正确答案恢复率”。整状态干预不证明
纯指针存在，也不保证这个位置是唯一路线。各层保留，不挑最有利的一层作独立确认。

## 后续持续作用

对已经生成的history_seed做消息切断，从该token的key首次可见的查询开始比较；
之前的预测必须保持不变。只在原词概率下降且超过历史位置对照时记录支持性延续。
仍然teacher-force原回答，因此不自动确定错误span边界或自由生成的修复效果。
衣服长度例保留 `scope_ambiguous_not_gold_error`，不会因为符合相同符号模式就重标为幻觉。

## 当前证据与不能下的结论

历史 `docs/reanchor_mechanism_audit.md` 记录：错误onion的12在L24/H27对source12
的attention约0.89844，原始概率约0.99523。它支持“值来自材料但适用关系不对”的
现象描述，不能证明Only/阶段条件已到达但被压过；后者必须看本次有限干预。
受控16对pointer/payload扫描也只是机制标尺，不是这两个自然来源的机制验证。

本地仅运行CPU软件测试与构造输出接口测试；不具备用户服务器上的8B权重和原NPZ。
没有用构造输入或历史表格冒充新的自然实验结果。
