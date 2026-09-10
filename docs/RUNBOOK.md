# G0 运行与验证契约

## 实现范围

新 G0 已接入 main.py；旧审计在 audit 子命令。scripts/run_pilot.sh 转到 run_g0.sh；
旧 run_pilot 的四个位置参数迁至 run_audit.sh。模型/数据路径仅从参数或环境变量读取，
不下载模型，不读取 SSH 私钥，不自动运行远端任务。

代码路径：

- binding_data.py：六条以上事实、A/B 交换、真实历史、显式/指代角色与 source split。
- features.py：Llama source-only/source-blind 因果重放、候选集合、冻结 cache。
- support.py：两层 candidate-conditioned source read，source-only value。
- training.py：直接 softmax(f) 训练、program dev 选模、冻结 test scores。
- calibration.py：未标注 mixed-reference rank，严格左侧 ties，非 FPR 保证。
- ragtruth.py / evaluation.py：原始文本归一化与最后单独 labels join。

未实现 G1、任意谓词/多跳关系生成、原生成器内部节点干预、严格 confirmatory 研究。
默认 64 程序 sources / 32 自然 sources 是探索性 pilot，不是 full benchmark。
全流指标将后续错误当“非首次错误目标”，不是当它们在事实意义上正常。

## Remote invocation

在已有 CUDA Python 环境和 reanchor 仓库内：

```bash
git pull --ff-only && bash scripts/run_g0.sh
```

不得在没有模型或依赖时声称远端环境 ready。脚本检查设备与 seeded matmul；
真实 Llama 权重加载和整个小任务运行后才能说明该平台的实际兼容性。
目前没有启动远端 GPU。

## 本地干净 CPU 环境

本机共享 research conda 环境导入 Llama 时打印了 pyarrow/pandas/sklearn DLL
access-violation 诊断；该次测试最终仍以 24 passed / exit 0 结束。
为消除这项环境噪声，不改共享 conda，改用独立 .venv-g0-test 做 CPU 测试。

在 D:/projects/research/reanchor 下按原样运行以下 PowerShell：

```powershell
$env:PYTHONPATH = 'D:/projects/research/reanchor/src'
& '.venv-g0-test/Scripts/python.exe' main.py preflight --device cpu
& '.venv-g0-test/Scripts/python.exe' -m pytest -q
& '.venv-g0-test/Scripts/python.exe' -m ruff check src tests main.py
& 'D:/Apps/Research/Tools/Git/bin/bash.exe' -n scripts/run_g0.sh
& 'D:/Apps/Research/Tools/Git/bin/bash.exe' -n scripts/run_pilot.sh
& 'D:/Apps/Research/Tools/Git/bin/bash.exe' -n scripts/run_audit.sh
```

环境 spec 为 configs/environment.json；本地实际版本/失败在未提交的
.aris/compute/local.md 留存。测试随机 tiny Llama 的分数不作方法有效性证据。
