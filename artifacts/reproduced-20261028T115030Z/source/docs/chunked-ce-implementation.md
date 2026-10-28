# 分块输出投影与精确交叉熵实施说明

本文件只记录第 5 项分工，依据 `learning-dynamics-design-2026-10-28.md` 与对应评审契约。
实施者只改专属文件；共享元数据、导航、build 映射、README 与 package.json 留给 integrator。
所有旧的 untracked 文件与实验目录保留，不执行 git commit/reset/clean，也不在页面未注册时运行站点 build。

## 当前接口与可审核后端

独立 `scripts/chunked_ce_reference.py` 已可导入，无 core 依赖。

```python
chunked_linear_cross_entropy(
    H, W, labels, bias=None, mask=None,
    token_block=2, vocab_block=3, ignore_index=-100,
    score_shift=0., fault="none",
)
```

H[N,D] / W[V,D] / optional bias[V] 为 CPU float64，labels[N] 为 int64，mask[N] 为 bool。
True 表示参与监督；标签必须是词表 ID 或 ignore_index，mask 不隐藏越界标签。
目标为有效位置上的标量平均 CE；已经完成位置对齐，不在此处做 causal shift。
零有效位置返回可微零哨兵与零梯度，平均 loss 未定义，调用者应跳过 optimizer。
H 与 W 独立求导；没有输入查表、权重绑定、class weights、label smoothing、二阶梯度、GPU、CCE 依赖或预训练模型。

`ChunkedLinearCE.forward/backward` 显式在 `torch.no_grad()` 中执行。
保存 H/W/bias/有效标签/row_max/row_logsum，只有两个 N 统计向量，没有完整 logits/probability，也不在 ctx 上藏 tensor。
score tile 原地转换为 exp/p/dZ，再计算 dH/dW/db。`d_loss` 参与所有梯度的全局 C 分母缩放。
`validate` 也使用 no_grad：有限性检查本身不会因 requires_grad 输入而让 hooks 记录额外检查张量。
`once_differentiable` 明确拒绝二阶反向。

允许 T≥N 且 B≥V 的完整大小单块边界对照；不能对全部块参数宣称没有完整形状。
no-full-score 审计只针对 [N,V,D]=[5,7,3] 的 [T,B]=[2,2]/[3,3]/[5,2]/[2,7] 真分块配置；后续测量也只测 sub-full 分块。
dispatch 观察 mm/exp 的形状与 grad_enabled，hooks 检查只有预期六类保存请求、没有 score；存储引用计数按 backing storage 去重。

## 命令与当前实际结果

已执行：

```sh
node scripts/run-output-lab.mjs --out-dir artifacts/output-lab-2026-10-28-correctness-01 --correctness-only
```

360 个精确配置、12 个故障配置，2,981 项比较通过。dense 最大误差 4.441e-16，JS 最大误差 4.441e-16，中央有限差分最大误差 4.648e-11。
输入包括 5×3 H、7×3 W；T=[1,2,3,4,5,8]、B=[1,2,3,4,7,9]，有/无 bias、mixed/empty/ignore-all。
额外有空 N、单位置、所有位置有效、所有目标类别、±1e4/±1e6 平移、λ=[0,.375,-2.25,3.5]。
故障：只改反向的 local-softmax 与 overwrite-dh；测试可定位多块错误，也验证全词表单块或 λ=0 时故障被隐藏。
比较 loss、dH、dW、db 分别记录；finite differences 对 H/W/b 全部元素独立扰动。

报告目录：`artifacts/output-lab-2026-10-28-correctness-01/`，含 reference.json、correctness.json、environment.json。
正确性与最终 timing-only 均已完成；默认 data 模块由成功保存的真实报告提取，保留全部测量原始样本与 provenance。

常规完整实验命令（总是新建目录，EEXIST 拒绝覆盖）：

```sh
node --test scripts/test-chunked-ce.mjs
.venv-framework/bin/python scripts/test-chunked-ce.py
node scripts/run-output-lab.mjs --out-dir artifacts/output-lab-new
```

先正确性、后独立 timing-only 的接口：

```sh
node scripts/run-output-lab.mjs \
  --out-dir artifacts/output-lab-2026-10-28-timing-01 \
  --timing-only \
  --correctness-report artifacts/output-lab-2026-10-28-correctness-01/correctness.json \
  --reference artifacts/output-lab-2026-10-28-correctness-01/reference.json \
  --contention-note 'Parent-directed final timing slot; no project training process observed immediately before sampling; ordinary desktop/background processes may remain.'
```

timing-only 必须复用相同 Python 源码和 fixture SHA256；源码改变后需新正确性报告。
3 次预热、12 次独立 wall samples，单线程 CPU float64，包含 validation scans、清空 .grad、前向反向。
计时不含 hooks、dispatch、profiler、RNG、输入复制或 I/O。hooks 另一次调用测唯一 storage 引用，包含 H/W/b/标签与统计，排除未保存 tile/gradient 与 allocator/RSS。
逻辑表分开列完整 logits、最大单块、统计、共享输入参数与梯度；没有把它们称为分配器峰值。
H/W/b 预生成，同形状各方案使用相同张量；固定 seed=20261028+N，benchmark 标签全有效。

## 页面与整合接口

入口 `chunked-cross-entropy.html`，标题建议：**分块交叉熵 · 模型工程 Wiki**。
描述建议：**分块输出投影、在线 LSE、完整 hidden/权重/bias 梯度、保存引用与独立 CPU 实测。**
nav 建议 `engineering.html`；知识目录归训练工程计算；bodyPage 可同 slug。
styles：`modern`、`exact-mechanics`、`decision-chains`、`chunked-cross-entropy`（shared/global/wiki/math 由站点现有流程加入）。
feature：`chunked-cross-entropy` → **`initChunkedCrossEntropy`**。

feature 导出：`chunkedCEInputs`、`denseCEReference`、`chunkedCrossEntropy`、`validateChunkedCEReport`、`chunkedCEReportRows`、`createChunkedCEReportLoader`、`initChunkedCrossEntropy`。
Node exporter 导出 `makeChunkedCEFixtures` / `extractChunkedCESnapshot`。
Python 导出 `chunked_linear_cross_entropy` / `dense_linear_cross_entropy`、`ChunkedLinearCE`、hooks/dispatch 审计辅助函数。

data 模块只在 successful report 保存后通过 exporter 提取；字段 `chunkedCEReportSnapshot`。
schema `output-lab-v1`，默认实测有 run/report/fixture/source hash、时间、环境与各方案原始 timing 样本。
本地报告导入≤5MiB，验证形状、版本、CPU/float64、原始 time samples 与汇总、账本、逐变量误差。
动态文字使用 textContent 或 escapeText；递增 request 防旧异步结果覆盖新文件；失败时清掉旧报告视图。

主要锚点：`ce-contract`、`ce-forward`、`ce-lab`、`ce-backward`、`ce-memory`、`ce-cpu-lab`、`ce-cce-scope`。
页面含前向/反向 stepping、块索引网格、per-position target/LSE/Σp、全部 dH/dW/db 与 dense 表、mask/bias/target/score shift/λ/fault 开关。
实测图与表使用同一 saved report；单块/行统计教学快照明确不计入 Python 保存量。

## 一手来源与实际读取范围

2026-10-28 直接 HTTPS 获取并保存源文件在 `artifacts/output-source-audit-2026-10-28/`；没有宣称仓库 redirect 或未经核验的 git commit。

- 作者仓库 `https://github.com/apple/ml-cross-entropy`；直接读 `https://raw.githubusercontent.com/apple/ml-cross-entropy/main/README.md`。
  读取 intro/requirements/basic interface/impl table/generalized bias。`cce_exact` 是取消 classifier 与 embedding 两侧梯度过滤的 alias；`cce_kahan_full_c` 只取消 classifier 过滤。
  保存文件 apple-cce-readme.md，SHA256 `cb36302c6b6c5ff1214da80fbad9f06e7b0d11f1bd3343d5a47e2aaa646ea7db`。
- Apple 作者研究页 `https://machinelearning.apple.com/research/cut-your-losses`：标题、作者、ICLR / February 2025 元数据、摘要与 GPU 特定性能描述。
  保存 apple-research.html，SHA256 `941c8b87efdc4d468fc74b46c6ee19edd9a340edccbf189ec455616ec3c2a8d0`。
- Wijmans / Huval / Hertzberg / Koltun / Krähenbühl，*Cut Your Losses in Large-Vocabulary Language Models*，ICLR 2025；`https://arxiv.org/html/2411.09009v2`，版本日期 2025-03-10。
  读 metadata/abstract、第 3、4.1–4.3、5.2 节、附录 B/E；kernel 性能表未自行复现。
  保存 paper.html，SHA256 `d7100a8704e2d884a0364536131dbbee42897ab20a9794f50dd16685e60f3ee4`。
- 已安装 PyTorch 2.9.0 的 saved_tensors_hooks、Function.forward/backward/save_for_backward 与 F.cross_entropy docstrings 已实际读取。
  对应参考链接 `https://docs.pytorch.org/docs/2.9/autograd.html`、`https://docs.pytorch.org/docs/2.9/generated/torch.nn.CrossEntropyLoss.html`。
  官方网页全文未单独获取；语义依据本机固定版本文档。

references.mjs 建议新增 `cutCrossEntropy` source：author 标为上述作者 / Apple，year=2025，title 为论文标题，href 作者仓库或 arXiv v2；why 说明在线 LSE、局部重算与 exact/no-filter 范围。
pageEvidence scope：**小型 CPU H/W/b 目标的一阶梯度；没有 tied lookup、过滤近似、GPU 性能或语言模型质量；教学快照排除于独立 Python 保存引用统计。**
refs 建议 `cutCrossEntropy` + 固定版本的 `torch29Autograd` + `torch29CrossEntropy`（后者可新增）。已核对既有 `ce` 与 `autogradMechanics` 当前链接 2.14，与本实验 2.9 不同；引用语义需跟随明确固定环境，不冒称已阅读 2.14 文档。

## 最终实际测量与验收

最终报告：`artifacts/output-lab-2026-10-28-timing-01/report.json`，UTC 2026-10-28 06:20:33（上海时间 14:20:33）。
环境：macOS 15.7.3 arm64、Python 3.12.14、PyTorch 2.9.0、NumPy 2.3.4、CPU float64、intra/inter-op 各 1 线程、确定性算法。
执行前只读 ps 未匹配到本项目 learning/training_dynamics/failure/batch/dynamics/output 实验进程；不保证桌面后台完全空闲，该限制已保存为 contention_note。
报告复用了正确性报告，未重跑 360 项 sweep；Python SHA256 与 fixture SHA256 均匹配。

| N×V×D | 方案 / T×B | 前向+反向中位 ms | hooks 唯一 storage 保存引用 B |
| --- | --- | ---: | ---: |
| 64×257×32 | dense | 0.198 | 214280 |
| 64×257×32 | 8×32 | 4.522 | 85768 |
| 64×257×32 | 16×64 | 1.490 | 85768 |
| 64×257×32 | 64×64 | 0.566 | 85768 |
| 128×1021×64 | dense | 1.301 | 1634824 |
| 128×1021×64 | 16×64 | 9.107 | 599528 |
| 128×1021×64 | 32×256 | 2.670 | 599528 |
| 128×1021×64 | 128×256 | 1.669 | 599528 |

这两个小型 CPU 问题中分块减少保存引用但比 dense 慢；最大 benchmark loss 误差 1.776e-15，各梯度误差≤1.735e-17。
较小块增加 Python/GEMM 调用数，三个同形状分块方案的 hooks 引用字节相同，因为临时 tile 不保存给反向。
保存引用量没有包括无保存 tile 或梯度缓冲区，不能作为端到端内存峰值；不推导 GPU 性能。

```sh
node scripts/export-chunked-ce-fixtures.mjs --snapshot \
  artifacts/output-lab-2026-10-28-timing-01/report.json \
  src/scripts/features/chunked-cross-entropy-data.js
node --test scripts/test-chunked-ce.mjs
.venv-framework/bin/python scripts/test-chunked-ce.py
node scripts/test-chunked-ce.mjs --ui --qa-dir artifacts/output-ui-2026-10-28-02
```

结果：Node 19/19 passed，0 skipped；Python 13/13 passed；浏览器 64/64 passed。
四个 `data-math` 公式已用现有 renderMath/KaTeX 严格模式渲染并检查 MathML。
Python 额外核验负/非单位 dLoss、gradcheck、非连续 H/W/b、单词表类别、custom ignore index、冻结 W 时 dH、共享底层 storage 去重、saved-input 修改报错与一阶范围。
浏览器在独立 loopback 服务器加载此页、现有样式、KaTeX、feature/data，检查 390/768/1440 暗色、全部 details 展开、无整页横向溢出、表格可键盘滚动、标签与 SVG 名称、故障、尾块、负上游、bias、shift、安全导入、错误清理与异步竞态。
UI JSON：`artifacts/output-ui-2026-10-28-02/interaction-results.json`；6 张截图 `tiles-*` / `measurements-*`；独立页面 `isolated-page.html`。
首次沙箱内 UI 尝试仅因 loopback listen EPERM 失败，目录 `output-ui-2026-10-28-01` 保留；明确授权的 sandbox escalation 后上述第 02 次成功。
整合后的完整站点 build/导航/browser audit 仍由 integrator 执行，本 worker 未运行共享 build。

最终报告 SHA256：`1cc3168ae5220100ad1fca69d3d21bd2ffa2f73c56bafa2e56ea2e1351cc404d`。
Python SHA256：`92b0d955a164e2cc47058fd174e52b99008074a8a90bc5428e8f126b867e4071`。
fixture SHA256：`8051e987d02571b2662f7ba9688b71a3f8a6165918cb6ed69cdd283d30d1de09`。

## 完整文件交付清单

- `src/content/chunked-cross-entropy.html`
- `src/scripts/features/chunked-cross-entropy.js`
- `src/scripts/features/chunked-cross-entropy-data.js`
- `src/styles/chunked-cross-entropy.css`
- `scripts/chunked_ce_reference.py`
- `scripts/export-chunked-ce-fixtures.mjs`
- `scripts/run-output-lab.mjs`
- `scripts/test-chunked-ce.mjs`
- `scripts/test-chunked-ce.py`
- `docs/chunked-ce-implementation.md`

运行工件新增在 `artifacts/output-source-audit-2026-10-28/`、`artifacts/output-lab-2026-10-28-correctness-01/`、`artifacts/output-lab-2026-10-28-timing-01/`、上述两个独立 UI 新目录；旧报告均保留。
