# 初始化、宽度迁移与 batch 实施说明

日期：2026-10-28。负责 slice 3/4；遵循父级
`learning-dynamics-design-2026-10-28.md`。本文件仅由本 worker 维护。

## 已完成范围

两页、后端、紧凑默认数据与轻量测试已完成。深度/Norm/残差、固定参数梯度和 batch
训练实际复用 `learning-core-v1` 的 `make_task_splits`、`collate_records`、`make_model`、
`forward_loss`、`evaluate_records` 及恢复接口；已读 `docs/learning-core-api.md`。
μP 单独使用固定两层隐藏 MLP，未将 decoder 的普通缩放称为 μP。

固定参数梯度以完整独立序列为单位，序列内部先对监督位置平均，再对序列等权平均。
总体协方差分母 N：有放回 trace/B；无放回 trace*(N-B)/(B*(N-1))。
均值范数接近零时噪声比值未定义/病态，不写 Infinity、不假造 0。

## 报告、重分析与默认数据

原测量 `artifacts/dynamics-lab-20261028-01/report.json` 保持原字节，19,044,218 B，
SHA-256 `587611efd52b00d9b0209af948bc72a54eb92075eb3aebea6e2d14bf9bfff645`。
包含 90 个宽度扫描、6 个保留宽度运行、54 个 depth/Norm/residual 配置、17 个精确抽样法则、
18 个真实 batch 训练；种子 11/29/47。原测量命令为
`.venv-framework/bin/python scripts/training_dynamics.py --out-dir artifacts/dynamics-lab-20261028-01 --only all`。

本次遵照用户要求未重新完整训练。最终透明重分析为
`artifacts/dynamics-reanalysis-20261028-02/report.json`，19,080,761 B，
SHA-256 `07095b8c74525ab626fc7f02616b338edf39001422fbfdf54f92e49612a49177`。
`reanalysis-checks.json` 为 11,427 B，`analysis-environment.json` 为 389 B；目录还保存
当前四份分析源码与 `resume-recheck/`。前一次重分析目录 `dynamics-reanalysis-20261028-01`
也保留，默认页面引用最终 02。

原训练 `training_dynamics.py` SHA-256 为
`35d3f729d25c28f5e0d2f3a385133d2ee1f179be7775f56485504b2155fccf58`，当前分析为
`46817b0bbe9f4de608bd065e412ba93e6b0368a3ffa699212f1d673a840e3937`。
报告的 `source_sha256` 保留原训练来源，`analysis_source_sha256` 独立记录当前分析；
`original_measurement` 保留原报告路径、日期、命令和哈希，`no_full_retraining=true`。
共享 core 原/今哈希相同：`f3d8cdfc21a2ec13ab829787bf0662063d45667848df02f1811987574571bfc7`。
已逐项比较原学习曲线、坐标、depth 探针、初末评估及 optimizer 步骤日志不变；
当前变更仅重新计算 PSD/直接中心化统计、澄清边界/经过时间和复核保存 checkpoint。

保存的 6×25,984 float64 梯度重算误差和 Gram 差均为 0；所有抽样法则重新穷举通过。
18 个 batch checkpoint 复核初末梯度统计，全部曲线共享
`0,4096,8192,12288,16384,20480,24576,28672,32768` input-token 网格。
原 run 的恢复参考未保存终态 RNG。本次从原 mid checkpoint 实际连续回放四步，取得终态 RNG，
显式标为 `derived-from-saved-mid-checkpoint-replay`；随后新 Python 进程对照该新增参考，
model/optimizer/next_steps/evaluation/RNG 最大误差全部为 0。这不是原 run 已保存终态 RNG 的声明。

生成命令与结果：

```sh
node scripts/run-dynamics-lab.mjs --reanalyze artifacts/dynamics-lab-20261028-01/report.json --out-dir artifacts/dynamics-reanalysis-20261028-02 --write-page-data
.venv-framework/bin/python scripts/test-training-dynamics.py
node --test scripts/test-training-dynamics.mjs
```

重分析经过 1.59 秒；Python 8/8、Node 8/8 通过。轻量检查覆盖向量独立穷举参考、PSD/零均值、
实际 μP 参数组/readout/恢复缩放、共享 sequence-mean 目标、token stream/scheduler、来源与
报告拒绝覆盖、跨 seed token 网格、两页默认初始化和完整报告导入、错误清空/转义/竞态、数学渲染。
未运行全站 build。原实验经过 14.72 秒包含评估/探针/保存，不能当吞吐比较。

默认 `initialization-width-data.js` 为 **503,880 B**（492.1 KiB），
`batch-noise-data.js` 为 **289,984 B**（283.2 KiB），均小于 500 KiB，实际默认导入通过。
提取坐标/六参数账本、必要逐层统计和九个 token 测量点；每 split 默认仅八条预测，指标使用全量题目。
完整参数、checkpoint、optimizer 日志、全部预测与枚举记录保留在本地工件，不在页面初载中携带。
两页 provenance 同时显示原训练/当前分析哈希、原报告哈希和未重训说明。

结果边界：SP/μP 只由宽度16选择 LR=.03，是原网格上边界中的最佳候选，最优值可能在网格外；
32/64 只诊断，128 未参与选择。坐标稳定不是迁移定理。batch 每 seed/LR 使用相同 4,096 条流、
32,768 input token / 16,384 supervised token 和评估集，token endpoint scheduler；
LR=.003 时 B4 测试 exact match 三种子均值 69.23%，B16/B32 为 0，长度5全部为0。
这些结果来自保存训练，不由噪声方差公式合成，未为每个 batch 单独搜索最优 LR。

## 依赖与实际读取的资料

已在原 `.venv-framework` 验证 Python 3.12.14、PyTorch 2.9.0、NumPy 2.3.4。
固定安装 `mup==1.0.0`，仅追加其 runtime 必需的 `PyYAML==6.0.3`，使用 `--no-deps`。
PyPI JSON 的 `requires_dist` 为 null；安装后 sdist metadata 实际列出 numpy、pandas、torch、
torchvision、seaborn、tqdm、pyyaml。此实验不使用绘图/示例依赖。

已读 Microsoft 作者 README 的 Basic Usage、How mup Works、Limitations、Coord Check；
已读安装的 1.0.0 `__init__.py`、`shape.py`、`layer.py`、`optim.py`、`init.py`、`infshape.py` 全文；
tag v1.0.0 的 layer/optim 与安装源码字节核对一致，源码哈希保存在原报告。
readout 前向先缩放输入 `output_mult*x/width_mult`，bias 本身不乘此前向因子；
`set_base_shapes` 仅首次转换随机初始化，恢复后用 `rescale_params=False`。
MuAdam 对 ninf=2 的矩阵组 LR 除 width multiplier；ninf<=1 不变，矩阵组 decay
乘 width multiplier。本实验 decay=0，使用 MuAdam/Adam，不使用版本敏感的 AdamW 包装。

完整出处：

- Microsoft, **Maximal Update Parametrization (μP) and Hyperparameter Transfer (μTransfer)**,
  https://github.com/microsoft/mup ;
  https://raw.githubusercontent.com/microsoft/mup/main/README.md 。
- 固定实现： https://github.com/microsoft/mup/tree/v1.0.0/mup ;
  https://raw.githubusercontent.com/microsoft/mup/v1.0.0/mup/layer.py ;
  https://raw.githubusercontent.com/microsoft/mup/v1.0.0/mup/optim.py ;
  https://raw.githubusercontent.com/microsoft/mup/v1.0.0/mup/init.py ;
  https://raw.githubusercontent.com/microsoft/mup/v1.0.0/mup/shape.py 。
- 包元数据： https://pypi.org/pypi/mup/1.0.0/json ;
  sdist SHA256 `9639e3d19f90e754f985ed444542ed2f8a049f3c0488fcb6efe150f30922cf74`。
- Greg Yang, Edward J. Hu, Igor Babuschkin, Szymon Sidor, Xiaodong Liu, David Farhi,
  Nick Ryder, Jakub Pachocki, Weizhu Chen, Jianfeng Gao, **Tensor Programs V: Tuning
  Large Neural Networks via Zero-Shot Hyperparameter Transfer** (2022),
  https://arxiv.org/abs/2203.03466 ; https://arxiv.org/pdf/2203.03466 。
  实际读取附录 B 表 8/9 的参数分类、D.1 坐标检查和 D.2 初始输出暂态。
- Ruibin Xiong, Yunchang Yang, Di He, Kai Zheng, Shuxin Zheng, Chen Xing, Huishuai Zhang,
  Yanyan Lan, Liwei Wang, Tie-Yan Liu, **On Layer Normalization in the Transformer
  Architecture**, ICML 2020, PMLR 119:10524–10533,
  https://proceedings.mlr.press/v119/xiong20b.html ;
  https://proceedings.mlr.press/v119/xiong20b/xiong20b.pdf 。
  实际读取第 3/4 节架构与初始化假设；本 decoder 不满足论文全部简化假设。
- Sam McCandlish, Jared Kaplan, Dario Amodei, OpenAI Dota Team, **An Empirical Model of
  Large-Batch Training** (2018), https://arxiv.org/abs/1812.06162 ;
  https://arxiv.org/pdf/1812.06162 。实际读取第 2.2–2.4 节、式 2.2/2.8/2.9 和附录 A.1；
  简单 trace/mean² 与论文 Hessian 加权量分开，未测量 Hessian。
- PyYAML 作者项目 https://github.com/yaml/pyyaml ;
  固定包元数据 https://pypi.org/pypi/PyYAML/6.0.3/json 。

资料原文缓存位于 `/private/tmp/dynamics-primary-20261028/`：作者 README、两份固定源码、
PyPI JSON/sdist，以及三份论文 PDF/文本。无预训练下载；以上版本是固定参考，不宣称最新。

## 集成接口与所有改动路径

| 页面 / feature | 导出 | 样式依赖（按顺序） |
| --- | --- | --- |
| `initialization-width.html` / `initialization-width.js` | `initInitializationWidth`, `validateInitializationWidthReport`, `widthCoordinateRows`, `depthProbeRows`; data: `initializationWidthData` | `modern`, `exact-mechanics`, `decision-chains`, **`initialization-width`** |
| `batch-noise.html` / `batch-noise.js` | `initBatchNoise`, `validateBatchNoiseReport`, `validateGradientGram`, `gradientGramStatistics`, `batchNoiseVariance`, `enumerateGradientBatches`, `batchTrainingRows`; data: `batchNoiseData` | `modern`, `exact-mechanics`, `decision-chains`, **`initialization-width`, `batch-noise`** |

共享局部 helper 为 `initialization-width-utils.js`。特别是 batch 必须加载 `initialization-width.css`，
该文件对两页分别限定共同 chart 配色、provenance 换行和 controls grid；integrator 已收到依赖说明。
390px 缺依赖的横溢出由 integrator 接入该样式后复测，不使用整页裁切。本 owner 未修改共享注册。

完整 owner 路径（相对项目根目录）：

```text
scripts/training_dynamics.py
scripts/run-dynamics-lab.mjs
scripts/dynamics-requirements.txt
scripts/test-training-dynamics.py
scripts/test-training-dynamics.mjs
src/content/initialization-width.html
src/content/batch-noise.html
src/scripts/features/initialization-width.js
src/scripts/features/initialization-width-utils.js
src/scripts/features/initialization-width-data.js
src/scripts/features/batch-noise.js
src/scripts/features/batch-noise-data.js
src/styles/initialization-width.css
src/styles/batch-noise.css
docs/training-dynamics-implementation.md
```

无可用 parent-message 工具；交接通过当前聊天。报告使用新目录，旧报告/未跟踪文件保留；
未 commit/reset/clean，未改共享 core、注册/导航/build/package/README 或其他 owner 文件。
