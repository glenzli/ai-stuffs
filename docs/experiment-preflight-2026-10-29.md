# 复现前检查：先确认环境和固定输入

实测导航与复现说明已提交为 `85cab85d14006692c998e206358cca28d6a59156`，
作者与提交者时间均为 `2026-10-29 04:32:26 +0800`，比父提交晚 34 分 45 秒。
GitHub `main` 已读回确认同一提交。本轮后续的预检工具与来源卡说明尚未提交或推送。

## 解决什么问题

复现命令启动前，读者需要知道 Python 在哪里、依赖是否能导入、版本是否符合固定参考、
输入文件有没有缺失或变化，以及输出目录是否会与旧记录冲突。
这些问题原先要从多个说明和运行错误中逐个拼出。

现在每组来源卡先给出预检，再给独立重跑，两条命令使用相同输出路径。
预检是本地命令，不在网页构建或浏览器里自动执行；不启动训练、不下载模型、不创建实验输出目录。

```sh
npm run check:experiment -- --only padding-inference --out-dir artifacts/padding-inference-verify-run
node scripts/run-padding-inference.mjs --out-dir artifacts/padding-inference-verify-run
```

如果改变输出位置，应同时修改两条命令中的 `--out-dir`。
预检不预留目录；真正执行时 runner 仍会独立拒绝已经存在的目录。

## 用法和结果

```sh
# 检查全部十三组的环境和来源，不指定输出目录
npm run check:experiment

# 只检查 NumPy 过滤实验及其固定输入
npm run check:experiment -- --only retrieval-filtering

# 直接运行，标准输出只包含 JSON，适合保存或其他工具读取
node scripts/check-experiment.mjs --only accumulation --json
```

输出包括来源组、Python 路径、固定依赖文件、逐项状态、观测值和失败后的下一步。
JSON schema 为 `experiment-preflight-v1`。

| 状态 | 含义 | 退出码 |
| --- | --- | --- |
| `ready` | 所列前置检查通过 | 0 |
| `ready-with-warnings` | 所列检查通过，但 Python 版本与原报告不同，需要核验差异 | 0 |
| `blocked` | 依赖、导入、固定输入或输出位置存在阻塞 | 1 |
| 参数错误 | 未知来源组、重复参数、路径不合格式或缺少值 | 2 |

`--out-dir` 必须同时指定单个 `--only`，格式限于 `artifacts/<新目录名>`。
已存在的目录、普通文件和悬空链接均判为冲突，不覆盖或删除它们。
参数错误写入标准错误；成功解析参数后的 `--json` 检查结果写入标准输出。

## 检查范围

- Python 能否启动，固定依赖是否已安装且能实际导入。依赖版本按仓库参考文件做精确比较，包括本地版本后缀；这用于固定环境核验，不是对其他版本兼容性的全面判断。
- 记录的 Python 版本有差异时给出警告，避免把启动成功当作数值轨迹已经复现。
- 所选来源组及其直接输入来源的报告、schema、状态、校验表和表内原始文件。
- 实际重跑检索时使用的 `src/experiments/retrieval-queries.json` 与固定报告中的问题哈希。
- 可选输出路径是否已存在，以及 `artifacts` 父目录的可写状态。
- DDP 累积实验使用的 PyTorch 是否具有 Gloo 构建支持。

依赖按实际运行入口分别选择：

| 运行范围 | Python 选择 | 固定依赖 |
| --- | --- | --- |
| 大部分训练与对照 | `FRAMEWORK_PYTHON`，默认 `.venv-framework/bin/python` | `scripts/framework-requirements.txt` |
| 初始化、宽度与 Batch | 同上 | `scripts/dynamics-requirements.txt` |
| 真实中文检索 | `RETRIEVAL_PYTHON`，默认 `.venv-retrieval/bin/python` | `scripts/retrieval-requirements.txt` |
| 固定向量来源过滤 | `FRAMEWORK_PYTHON`，允许指向只有 NumPy 的环境 | `scripts/retrieval-filtering-requirements.txt` |

环境安装仍按 README 的固定步骤进行，工具不自动安装或修改包。
Python 通过参数数组启动，不通过 shell；用 `-I -B` 隔离项目搜索路径并禁用字节码写入，
探测只导入指定包并读取版本，同时设置 Hugging Face 离线选项。
批量检查复用相同依赖组合的探测与相同来源的读取。

输入依赖有意写到对应实验：四项实战会读取原学习实验，
同预算训练还会读取原学习实验的初始 checkpoint；文本误差审核依赖文本项目和语言基线。
来源过滤使用已保存检索输入，不重新编码文档，因此不要求导入 PyTorch 或 Faiss。

## 不由预检证明的事

Gloo 构建可用不代表本机端口权限、双进程通信或多机网络已通过；本命令不绑定端口。
通过预检也不等于训练已经完成、生成质量达到要求、所有数值逐位相同或性能可比较。

本工具检查登记报告和原始清单，不重新解压下载工件包，不逐字节比较当前程序与历史源码，
也不枚举所有系统动态库。所选组和直接输入组按完整原始清单核对，
因此可能包含本次重跑不会直接读取的文件。
压缩包和页面投影的完整核验继续由 `npm run test:provenance` 负责。

本轮没有新训练、计时或模型下载，也没有修改 `src/experiment-sources.json`、固定测量、
数值实现或已登记的实验工件。它补充复现入口，不新增科学结果。

## 验证记录

- 构建：66 页、113 个脚本、43 份样式、143 个公式。
- 13 项新契约检查，加既有导航和来源检查，共 28 项通过。
- 在本机真实 Python 环境执行全部十三组预检，均为 `ready`。
- 命令行实测：目录冲突、缺失解释器、未知来源组分别返回 1、1、2。
- 测试隔离覆盖缺包、版本不符、导入失败、Python 版本差异、Gloo 缺失、
  原始文件改变或缺失、上游来源缺失、问题身份不一致、非法路径及输出冲突。
- 浏览器展开核对导航、故障回放与来源过滤三处的预检与重跑路径；
  三页面 × 三尺寸 × 明暗主题，共 18 组布局无整页横向溢出。
- 原始来源检查重新核对全部报告、下载工件与校验表；来源清单仍为
  `44afd76af506999af7ab58441de78ea5dab9001b5d0570c4ae8f061aee3948cd`。

验证采用本轮负责模块、共享来源卡和已有来源的一次集中检查。
提交导航时复用的 464 项测试对应未改动的上一轮快照；本轮新增以后以这里的 28 项相关检查为准，
未为展示和诊断改动重复训练实验。

证据保存在 [`artifacts/preflight-20261029121035Z/`](../artifacts/preflight-20261029121035Z/)：

- [摘要](../artifacts/preflight-20261029121035Z/verification-summary.json)
- [最终测试](../artifacts/preflight-20261029121035Z/node-final.log)
- [全部真实环境预检](../artifacts/preflight-20261029121035Z/all-local-environments.json)
- [命令行案例](../artifacts/preflight-20261029121035Z/cli-cases.json)
- [浏览器核验](../artifacts/preflight-20261029121035Z/browser-audit.json)
- [已发布提交](../artifacts/preflight-20261029121035Z/publication-receipt.json)
