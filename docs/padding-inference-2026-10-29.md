# 完整模型的左右填充、读取列与缓存续算

检索过滤的 53 个文件已提交为 `78ef25b`，作者和提交时间均为
2026-10-29 03:27:24 +0800，比父提交晚 36 分 32 秒，未 push。
随后追加本文对应的实验，完成验收时保持未提交。

入口为 `position-cache.html#padding-real`，推理流程页也连接到此处。
保持 66 页、143 个公式、六步主线，新增第十三组来源，旧十二组来源保持不变。
证据目录为 `artifacts/padding-20261029082218Z/`。

## 比较对象与边界

复用自然语言项目 seed 11 第 800 步的完整权重：2 层、4 头、width 96、298,176 参数、
387 类词表。原始输入、tokenizer、checkpoint 和源报告逐字节保留，权重与位置 buffer
统一转为 CPU float64，不重新计算位置表，也不进行训练或参数更新。

这份模型使用固定正弦位置；位置页前半部分的四维 RoPE 教学模型与这里是不同配置。
原 TextDecoder 接口按右填充使用，不直接承诺任意左填充。新增独立适配器显式提供
逻辑位置、key mask、读取列和缓存；参考在每个阶段逐条调用原始未填充前向，并通过
hooks 记录实际层 K/V，没有把适配器输出充当自己的正确参考。

从保存的前三个训练窗口截取前缀，随后喂入每条原窗口的四个真实后缀 token。
所有分支使用相同后续输入。没有执行自由采样、EOS 停止或上下文滑窗，不据此评估
自然语言生成质量，也不推断采样出 PAD 时的行为。空提示直接拒绝。

## 冻结条件

| 条件 | 三条前缀的有效长度 | 批宽 | 目的 |
| --- | --- | ---: | --- |
| 变长前缀 | 7 / 19 / 41 | 41 | 检查短输入的读取列、位置与填充 key |
| 等长无填充 | 19 / 19 / 19 | 19 | 保留错误规则在这个条件下也会通过的对照 |
| 额外填充 | 7 / 19 / 41 | 44 | 最长输入也不占满物理宽度 |

每种条件分别使用左、右填充。六种方案为：正确逻辑位置/有效 key/最后有效列；物理
列号充当位置；总取最后一列；总取有效长度减一；忽略填充 key；仅续算位置取缓存槽数。
共 36 条执行路径，每条有三个样本、prefill 加四次续算，形成 540 个观测。

布尔可见 mask 与带缓存的因果偏移显式构造，再调用固定 PyTorch 2.9 SDPA。它的布尔
True 表示参与 Attention；非方阵缓存路径没有直接套用无偏移的下三角。接口依据见
[PyTorch 2.9 SDPA](https://docs.pytorch.org/docs/2.9/generated/torch.nn.functional.scaled_dot_product_attention.html)。

## 实际观察与三种误差

正确路径的全部有效 logits、实际读取的 next logits 和每层有效 K/V，与逐条完整
参考的最大差为 7.99e−15，低于冻结容差 2e−10。代表性负例采用变长条件的第一条输入：

| 操作 | 首次可见差异 | 实际观察 |
| --- | --- | --- |
| 右填充却总取最后一列 | prefill 读取 logits | 正确末列 6，实际读列 40；next logits 最大差 12.8865，全部有效前向与 K/V 仍吻合 |
| 左填充却总取长度减一 | prefill 读取 logits | 正确末列 40，实际读列 6；next logits 最大差 11.3801，有效前向与 K/V 仍吻合 |
| 右填充、忽略填充 key | 第一次续算 | prefill 通过；续算 next logits 最大差 2.1917，有效 K/V 最大差 0.4125 |
| 右填充、续算位置取物理槽数 | 第一次续算 | 逻辑位置应为 7，实际取 41；next logits 最大差 6.4980 |

右填充的 PAD 在 prefill 中位于有效 query 之后，因果关系遮住它；追加 token 后这些
物理槽已在新 query 之前，必须继续使用有效 key mask。位置计数也必须区分每条输入
的有效长度与整批占用槽数。本实验保留物理 PAD 槽，没有实现缓存压缩或分页。

等长、无填充条件下，六种规则的 180 个观测都在容差内；左填充总取最后一列、右填充
按长度减一取列也各有成立条件。页面逐条件判定，不把“故障规则”标签当作预设误差。
另有 Top-1 相同但 logits 或 K/V 已明显不同的记录，首次数值偏差和首次 Top-1 变化分开报告。

四次续算使用固定原始 token，因此取错 prefill 列后，后续检查仍可恢复一致。这只说明
固定输入下的计算路径；自由生成会把选出的 token 喂回去，可能从此改变后续输入。

## 原始工件与独立核验

来源包约 27.40 MiB，包含原 checkpoint、窗口、tokenizer、报告、运行时源码与配置。
每条路径保存全部 prefill logits、各阶段实际位置/读取索引/next logits、最终完整 K/V
和有效 mask。缓存实际逐步追加，旧槽不改写；每阶段保存独立字节哈希，验证最终缓存
的相应前缀确实等于该阶段的实际输出，避免重复存储相同缓存前缀。

六项 Python 核验包含：输入与执行源码身份、45 次原始参考的独立 dense softmax 重算、
180 次缓存事件的带历史位置完整重算、540 条误差和物理/逻辑账本重建、条件负例，以及
空输入/上下文/缓存结构守卫。独立 dense 路径不调用 SDPA、适配器或 DecoderBlock.forward。

实际下载包也在仓库外解压，只用包内源码和原始输入、清空 PYTHONPATH 后独立重放。
结果记录保存在 `qa/packaged-replay.json`；浏览器、下载、来源和提交身份见
`verification-summary.json`。通过表示声明的正确路径与负例边界得到核对，不表示全部
执行规则都正确，也不包含 GPU kernel、性能、长上下文能力或生成质量结论。

## 重跑与采纳

```bash
npm run lab:padding-inference -- --out-dir artifacts/padding-my-run
npm run test:padding-inference
npm run accept:experiment -- --candidate artifacts/padding-my-run/candidate.json
```

沿用固定 PyTorch 2.9.0、NumPy 2.3.4 和 Python 3.12.13，runner 拒绝已有输出目录。
准备候选不改变已采纳页面；显式采纳才更新来源登记、投影和忽略规则，不执行 Git 提交或 push。

解压来源包后，可用安装了固定依赖的 Python 在解压目录运行：

```bash
python source/scripts/padding_inference_lab.py \
  --out-dir replay-result --text-directory padding-inference/inputs
```

`replay-result` 必须不存在。SHA256SUMS 核对实验工件，SOURCE-SHA256SUMS 核对执行源码。
