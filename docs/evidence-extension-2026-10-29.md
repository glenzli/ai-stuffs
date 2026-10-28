# 五项扩展与核验

本轮按已确认的五项内容扩展，保留六步学习主线及黑白灰阅读主题。网站为 66 页、
141 个公式。新证据放在 `artifacts/extension-20261029T022617Z/`；两轮旧证据目录与
原训练源码保持原字节。所有修改保留在本地，未提交或 push。

## 1. 执行、准备与采纳分开

`lab:practice` 与 `lab:evidence` 只写新的输出目录。运行开始记录环境、输入身份、
执行源码哈希和当时的来源清单身份；打包时再次核对实际执行的源码，不把打包机器的
环境冒充实验环境。完成后产生完整报告、页面投影、源码与原始工件归档、两级校验表
和 `candidate.json`。实际重跑了一次语言基线，确认当前五个已采纳源文件逐字节不变；
记录在 `workflow-replay.json`。

```bash
npm run lab:practice -- --out-dir artifacts/practice-my-run
npm run accept:experiment -- --candidate artifacts/practice-my-run/candidate.json
```

采纳入口核对完整工件、归档内容与源码清单，限制实验拥有的页面和投影路径，并拒绝
基于旧来源清单生成的候选。多个候选一起传入时一起采纳；相同候选重试会再次核验，
无需重复写入。临时独占目录和恢复日志包围页面投影、来源清单和 `.gitignore` 的写入，
构建遇到未完成的采纳会先停止，保留已有输出。

```bash
npm run accept:experiment -- --recover
```

该命令确认原写入进程已结束，再恢复旧状态或保留已完成的新状态。已验证注入写入异常、
进程中断后的恢复、失效候选、篡改工件、归档内容不符和打包失败。这里的边界是进程中断
恢复，没有声称实现断电持久性。真实失败记录随对应运行保留，不再写入固定的虚构失败史。

## 2. 同一预测单位的语言基线

页面：`text-training-project.html#tp-baselines`。沿用原文本模型的固定文档划分、387 类
byte BPE 和原始评估窗口，计数只来自训练标签。unigram 和 bigram 均使用加法平滑，
从预先固定的 α∈{0.01,0.1,1} 中只按验证 CE 选择；测试不参与拟合或选择。

独立测试集有 22,731 个有效标签，CE 的单位为 nats/token：

| 方法 | α | 测试 CE | 测试 top-1 |
| --- | ---: | ---: | ---: |
| unigram | 1 | 5.195481 | 1.76% |
| bigram | 0.1 | 3.506274 | 18.32% |
| 原 Transformer seed 11 | 不适用 | 3.171643 | 24.72% |
| 原 Transformer seed 29 | 不适用 | 3.170038 | 24.90% |
| 原 Transformer seed 47 | 不适用 | 3.149952 | 25.81% |

保存全部计数、每个 α 的验证分数、每个文档的损失和正确数、原始窗口和 tokenizer。
核验从独立概率矩阵复算所有位置及文档汇总。Transformer 的结果来自原保存权重与
完整评估，旧快照和生成样本未改。损失更低支持下一 token 预测改善，不能替代流畅性、
事实可靠性或指令遵循评估。

## 3. 同监督 token 预算的长度支持对照

页面：`generalization-analysis.html#ga-matched-budget`。三个 seed 各从对应的同一初值
运行短支持（2–5）和扩展支持（2–8），每次恰好消费 410,000 个监督标签。
学习率 0.003、AdamW、裁剪 1、dropout 0.1、batch 64 保持一致。
最后一批按已固定的行优先顺序保留剩余标签，其余标签设为忽略，并保存裁切记录。
训练和评估的反转等价组无交集。

短支持运行均为 1,198 次更新，扩展支持均为 918 次；输入长度和计算位置也不同。
所以这是同监督预算，不能称为相同 FLOPs，仍不能把全部差异归因于长度支持一个因素。

| 扩展支持 seed | 长度 6 EM | 长度 7 EM | 长度 8 EM | 长度 9 / 10 EM |
| --- | ---: | ---: | ---: | ---: |
| 11 | 100% | 100% | 100% | 0% / 0% |
| 29 | 0% | 0% | 0% | 0% / 0% |
| 47 | 100% | 95.74% | 95.83% | 0% / 0% |

短支持三 seed 在长度 6–10 均为 0%。失败 seed 29 与全部外推失败保留；结果不支持
普遍获得算法式外推。完整报告保留每次更新、消费标签与计算位置、输入顺序、优化器、
随机状态、六份最终权重和所有逐题生成。核验从六份权重重新生成全部答案。

## 4. 真实中文段落与检索链

页面：`retrieval-project.html`。从同一份冻结语料保留 38 篇文档的 877 个唯一段落。
本仓库在编码和排名前编写 36 个问题，18 个来源主题各有术语型和换说法提问；
按来源文档组分为 12 个验证问题和 24 个测试问题，无交集。每问指定一段直接证据，
原文和哈希可回读。标注未穷尽其他相关段落，也未做独立盲评；因此称为“指定证据 Hit”
和“指定证据 MRR”，不能解释为通用检索精确率。

真实编码器为 `BAAI/bge-small-zh-v1.5`，提交
`7999e1d3359715c523056ef9478215996d62a620`，23,953,920 参数。
保存完整 safetensors 权重、tokenizer、发布者 MIT 说明和逐文件清单；权重 SHA-256 为
`354763b9b1357bc9c44f62c6be2276321081ed2567773608c0d0785b61d5a026`，
与发布者 LFS 元数据一致。CPU float32、CLS、L2 归一化，查询使用发布者检索指令，
文档不加指令，最大 512 token，本轮没有截断。运行只读本地模型，不调用远端推理。

BM25 按中文单字、英文词与数字计数，k1=1.2、b=0.75。Faiss FlatIP 使用完整向量；
IVFFlat 固定 16 个列表和聚类种子，nprobe 为 1/4/16。
RRF 常数为 60，将向量名次与全局 BM25 名次用于 Flat 前十候选内的实际重排；
没有运行神经 cross-encoder。全部参数在查看排名前冻结。

24 个测试问题的结果：

| 方法 | Hit@1 | Hit@5 | Hit@10 | MRR@10 | Flat 近邻 Recall@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| BM25 | 70.83% | 91.67% | 91.67% | 0.7826 | 不适用 |
| BGE Flat | 50.00% | 66.67% | 70.83% | 0.5789 | 不适用 |
| BGE IVF nprobe 1 | 25.00% | 25.00% | 33.33% | 0.2601 | 40.83% |
| BGE IVF nprobe 4 | 45.83% | 54.17% | 54.17% | 0.4931 | 77.08% |
| BGE IVF nprobe 16 | 50.00% | 66.67% | 70.83% | 0.5789 | 100% |
| Flat 候选内 RRF | 66.67% | 70.83% | 70.83% | 0.6806 | 不适用 |

本次 BM25 优于当前 BGE 配置；RRF 改善部分首位排序，但无法找回候选外证据。
所有问题的候选 ID、原文、实际得分、BM25 矩阵、向量、token ID 和真实索引均保存。
核验独立复算 BM25、点积和所有切片指标，重新加载索引并核对 IVF；另在独立进程中
重新编码 4 个段落和 3 个问题，最大绝对误差容限 3e-6。
没有测生产速度或 RAG 答案质量，不能将小型教学集结论外推到通用检索。

首次在同一进程导入 PyTorch 与 Faiss 时发生 OpenMP 库冲突，未产生可用指标。
失败时的源码、日志和处置说明保存在 `retrieval/history/openmp/`；正式运行改为
编码与索引分进程，未使用允许重复 OpenMP 库的环境绕过。

## 5. 阅读节奏和可恢复实验链接

五个实战页共六道预测题，作答后给出依据，可重新作答；结果面板仍可直接阅读。
顶部说明所属学习阶段，页尾返回对应主线和下一项实战。六步主线未扩展为新的必修链。
各实验的 seed、划分、方案、问题等选项保存在 URL，刷新可恢复，复制按钮提供当前链接；
点击复制时同时显示可手动复制的文本框，便于核对地址或应对剪贴板限制。同页不同实验使用独立参数前缀，未知值回到默认
并显示说明。保持黑白灰阅读面板，蓝紫区分曲线，琥珀提示判断需重新核对。

## 重跑新增实验

沿用固定 `.venv-framework`；检索另装 `scripts/retrieval-requirements.txt` 到
`.venv-retrieval`。两份可选 Python 环境均不影响静态网站依赖。

```bash
npm run lab:evidence -- --out-dir artifacts/extension-my-run
```

默认使用已采纳检索来源中的模型，可离线复跑。可设置 `FRAMEWORK_PYTHON` /
`RETRIEVAL_PYTHON`，或用 `--model-directory` 指定相同固定版本的本地模型。
单项重跑加 `--only language-baselines`、`--only matched-budget` 或 `--only retrieval`。
整个输出目录必须尚不存在。需要重新下载固定模型时：

```bash
.venv-retrieval/bin/python scripts/download-retrieval-model.py --out-dir .cache/models/bge-fixed
npm run lab:evidence -- --out-dir artifacts/extension-with-model --only retrieval --model-directory .cache/models/bge-fixed
```

三个子目录的候选可以一次采纳：

```bash
npm run accept:experiment -- --candidate \
  artifacts/extension-my-run/language-baselines/candidate.json \
  artifacts/extension-my-run/matched-budget/candidate.json \
  artifacts/extension-my-run/retrieval/candidate.json
```

来源登记包含完整报告、源码/原始工件归档、校验表和页面投影身份。归档约为 0.2 / 8.0 /
59.2 MiB；检索包包含约 91.4 MiB 原始权重，单文件低于 GitHub 100 MiB 限制。
依赖、缓存、其他试跑和构建输出均不进入跟踪范围。

## 验证入口与记录

```bash
npm run check
npm run test:evidence
npm run test:practice
```

新增 19 项 Python 核验分别在框架、Faiss 和编码器进程执行；工作流与页面契约另有
Node 核验。验证记录、HTTP 下载哈希、真实浏览器的交互/布局检查和截图随本轮证据保存。
最终检查身份和日志索引见 `verification-summary.json`。
