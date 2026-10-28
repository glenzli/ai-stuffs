# 小模型学习实验实施记录

2026-10-28，学习 worker 直接编辑共享 `/Users/glen/lab/ai-learn`。核心及 runner、Python 测试现已可审查；未使用 fork、未创建其他 agent、未改共享元数据或其他 worker 文件、未运行站点 build。

接口见 `docs/learning-core-api.md`。五个兼容函数已交付，默认 405/130/139/143 条 train/validation/test/length，101,120 参数、固定正弦位置、LayerNorm pre/post/none、标准初始化。生成统一上限 20，模型预测 EOS；生成 PAD 作为真实轨迹位置，collate padding 排除。

正式学习预算已在测试/OOD 前冻结为 seed 11/29/47 × 1,200 更新、batch 64、LR 0.003、dropout 0.1、CPU float32 单线程，最终 checkpoint 直接报告，无 early stop 或测试选择。

原始完整正式报告已完成且保留：`artifacts/learning-lab-20261028T034636.240515Z/report.json`，SHA-256 `72ea7521be51f362f26b3645983909242a5d0185d593357bf6c63537d4d65970`。seed 11/29 train/validation/test EM 为 100%；seed 47 分别 93.333%/93.077%/91.367%。所有 seed 的长度 6–8 外推 EM 为 0，143/143 题提前 EOS。`all_seeds_fit=false` 是预设 train EM ≥95% 的描述性阈值，不伪称全部拟合或外推成功。

原报告计算检查状态 `failed`：seed 11 learned cache 的旧绝对阈值 2e-5 未通过，最大 logits 差 2.804398536682129e-5；所有 seed 的新进程完整恢复与 full/cache 自由生成一致。独立复核同一 checkpoint 的 float32 与 float64，未重新训练或调整数据/预算。定位 `reverse-8-DBDADDBD`、chunk 1、位置 8/token B：float32 full=0.11679732799530029/cache=0.11676928400993347；float64 同元素差 1.31117339208231e-13。原始诊断在 `artifacts/learning-cache-analysis-20261028T034954.919131Z/diagnostic.json`。

复核完成的新完整报告：`artifacts/learning-revalidated-20261028T035357.553145Z/report.json`，计算状态 passed、`all_seeds_fit=false`。保留原训练 source hashes、原报告/hash/status、当前复核 source hashes、完全一致 checkpoint hashes；`original-report.json` 为原报告逐字复制。全部 817 题 × 4 chunk × initial/final × 3 seeds 分别以 float32 与 float64 检查，free generation 对全部题目核验并比对原预测。检查按元素采用 `abs(full-cache) <= atol+rtol*max(abs(full),abs(cache))`：float32 atol=1e-4、rtol=1e-5，float64 atol=1e-9、rtol=0；同时要求 top1 零分歧、实际 free generation 一致。此标准明确是诊断后修改，不是训练前预设；float32 容差是本机经验接受标准（绝对部分约 839×eps），不是任意 Transformer 的理论误差界。原始固定绝对阈值仍保留为 failed。float64 近机器精度、全题相同 argmax/生成和 softmax 最大误差提供独立解释依据。

命令：`node scripts/run-learning-lab.mjs --revalidate-report artifacts/learning-lab-20261028T034636.240515Z/report.json`，退出 0。未训练、未调参。默认 data 模块现仅由新完整通过复核报告提取，原失败报告可本地导入并清楚显示失败状态。

Python 12 项核心测试已通过：`.venv-framework/bin/python scripts/test-learning-core.py`。正式命令 `node scripts/run-learning-lab.mjs --snapshot-out src/scripts/features/learning-run-data.js` 确实运行三个 seed，并因上述原阈值检查以退出码 1 保存失败报告。原报告及第一次复核均保留。

最终 runner 与完整复核报告已经交付：`artifacts/learning-revalidated-20261028T035637.650035Z/report.json`，SHA-256 `8983ff9c28640c067b1d9476bdd44b01b864c4c2676866dc7e76dbad57f07238`，计算状态 passed、all_seeds_fit=false。此报告复核相同初始/最终 checkpoint，包含当前 runner SHA-256 `224723dd1762d555865db632c2e377532d1852af77dc0f19d9efa9bcae3c8971` 和核心 SHA-256 `f3d8cdfc21a2ec13ab829787bf0662063d45667848df02f1811987574571bfc7`，并保留原训练源码/hash、原 failed 报告/hash 与第一份复核。普通未来 `node scripts/run-learning-lab.mjs` 在 frozen-config 中预先声明新精度标准，调用 `validated_cache_checks` 检查 float32、float64 和实际生成；不会继续使用旧固定绝对阈值作为失败条件。版本在普通运行、复核与新进程恢复启动时强制检查。当前正式学习结果没有重训。

CLI wrapper `scripts/run-learning-lab.mjs` 最终 SHA-256 `c95e25b4b078929128c9c87adfdf3ae2aef64e986e0a5a4648970519b87fb790`。最终 core 12/12、Node 10/10 再核验通过，feature/CLI wrapper 的 `node --check` 退出 0；核对原失败报告 bytes/hash 未改变。前端源码与默认数据均已冻结，可进行 integrator 的浏览器验证。

前端 source 已完成，可供 parent 与 Helmholtz 集成审查：`src/content/learning-run.html` / `src/scripts/features/learning-run.js` / `src/scripts/features/learning-run-data.js` / `src/styles/learning-run.css`。默认 data 直接从最新 035637 完整通过复核报告提取；默认 provenance 同时保留原 failed 报告的 `72ea7521be51f362f26b3645983909242a5d0185d593357bf6c63537d4d65970`。不手工填写测量。页面展示三 seed 固定结果、逐 seed train/validation 曲线与数值、模型/拆分/基线切换、EOS/缺失/额外输出、完整本地报告导入、25 条逐题分页、失败题筛选、报告下载和配对导出。计算核验、95% 训练阈值、测试和外推分别说明。默认快照仅有实际逐题片段，完整配对导出要求导入完整 report.json。

Node 验证 `node --test scripts/test-learning-run.mjs`：10/10 passed。覆盖默认报告 path/hash、原报告/hash保留、训练/验证曲线逐点等于保存报告、全部真实逐题 Python/JS EOS计分一致、原 failed/新 passed 两种结构、范围/计数/状态检查、完整配对417题、分页与转义、seed47/OOD可见、导入失败清理、异步最新文件获胜、16MiB限制、selector完整、两式KaTeX/MathML以及键盘滚动表容器。无需构建 dist。

集成信息：slug/file 为 `learning-run` / `learning-run.html`；feature export `initLearningRun`；styles 为 `modern`、`exact-mechanics`、`decision-chains`、`learning-run`（专属 CSS 需 integrator 加入映射）。H2 anchors 为 `lr-task`、`lr-report`、`lr-scoring`、`lr-reproduce`；实验控制 prefix `data-lr-*`。JS exports：`LEARNING_REPORT_SCHEMA`、`LEARNING_REPORT_MAX_BYTES`、`scoreLearningPrediction`、`validateLearningReport`、`learningEvaluation`、`learningCurveSeries`、`learningCurveMarkup`、`learningSummaryRows`、`learningPredictionRows`、`learningPairedData`、`initLearningRun`；data export `learningRunSnapshot`。paired 下载名称 `learning-{split}-{initial|identity}-to-learned-{exact_match|token_accuracy}-paired.json`，schema=1，保留逐题/反转group/seed；默认快照下载 `learning-run-real-snapshot.json`，完整报告下载 `learning-run-report.json`。

owned source 路径完整列表：`scripts/learning_core.py`、`scripts/learning-run.py`、`scripts/run-learning-lab.mjs`、`scripts/test-learning-core.py`、`scripts/test-learning-run.mjs`、`src/content/learning-run.html`、`src/scripts/features/learning-run.js`、`src/scripts/features/learning-run-data.js`、`src/styles/learning-run.css`、`docs/learning-core-api.md`、`docs/learning-run-implementation.md`。所有实验目录为新增：learning-pilot-20261028T033840.052415Z、learning-lab-20261028T034636.240515Z、learning-cache-analysis-20261028T034954.919131Z、learning-revalidated-20261028T035357.553145Z、learning-revalidated-20261028T035637.650035Z。未修改 shared metadata、README、package.json、build 或其他 owner 文件，未 git commit/reset/clean，未覆盖既有报告；浏览器与完整站点 build 留给已授权 integrator。

可供引用元数据：Vaswani 等 `Attention Is All You Need`（2017，读取 HTML v7；sections 3.1/3.2.1/3.5，地址与 hash 在报告 sources）；PyTorch 官方安装版2.9.0 SDPA/CE/LayerNorm/AdamW/RNG/save/load文档（原报告 `primary-installed-docstrings.txt`）；官方 tag v2.9.0 Numerical accuracy（原诊断保存 `.rst`，非结合性和batch/slice段落，hash `383c98439629eb6b38b9a09099d9803f5e1f74e93caa45707c3cef90d41a8feb`）；Xiong等 `On Layer Normalization in the Transformer Architecture`（ICML2020/PMLR119，仅摘要和出版元数据）。本轮没有新增依赖或下载预训练模型。
