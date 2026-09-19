# 当前分工与收敛方向

2026-09-19；核对reanchor main `0d1092c`，graph main基线 `3f2b524`。
完整证据和方法设计统一在
[graph/DETECTION_CONVERGENCE_20260919.md](https://github.com/DwightEd/graph/blob/main/docs/DETECTION_CONVERGENCE_20260919.md)。

reanchor保留采样、原始token ID、attention/状态缓存及机制案例；graph负责
功能读出、检测原型与评价。无需重新采样4题×4seed，也不移动原始NPZ。

最重要的修正：

- 服饰中层head的差异主要是prompt内cap地址读取，错误侧prompt总质量仍高。
- L31H14的当前强效应主要由query self复现；近/远历史直接边效应弱。
  self的value已经上下文化，不等于纯句法，也不排除更早层历史写入。
- 条件位置删除弱，不证明语义条件缺失；两条直接边同向也不证明完整绑定。
- head角色依赖当前候选和前缀；统计head协方差、局部投影、最终干预必须分开。

在graph根目录更新main后运行：

```bash
python -u main.py flow --output outputs/evidence_target_flow_v2
```

它使用既有样本目录（可用`--samples`指定），保留原A·V·W_O算子，增加
共享局部梯度投影、self/近远历史拆分、单头/双头四世界及整段候选评分。
新输出单独保存并复用完成条件，不用新协议覆盖旧结果。

检测目标收敛为：从内部消息读出的对象—阶段—值联合绑定，是否符合原证据允许的绑定。
现有binding投影数学保留；自然适用关系提取和联合读出还未完成。
不再将一般聚类、熵种子或历史传播当成已经验证的幻觉机制。

本轮在graph完成132项针对性测试，并复算上传的73,545个词和72个实测head记录。
没有新运行用户8B模型或重训自然检测器；这里是分工/结论更新，不宣称reanchor新增实验结果。
