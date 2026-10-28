# 可核算的计算机制补充 · 2026-10-27

## 范围

根据用户的新标准，补充成熟算法、局部导数、数值规则与执行语义，不继续扩张不确定理论。
新增 6 个专题，共 46 页、96 条公式。目录增加“计算机制”，训练工程师指南提供主阅读顺序。
原有研究入口标为“研究选读”，保留原有内容与证据限制。

| 页面 | 计算对象 |
| --- | --- |
| backpropagation | 两位置、单头、hidden=2、FFN=3、词表=3 的 pre-RMSNorm block；13 个反向观察节点 |
| numerical-computing | FP16/BF16/FP32 最近偶数舍入、渐进下溢、上溢、逐次求和与稳定 softmax |
| attention-kernels | 单 query 六分数、二维 V 的在线最大值/分母/加权和 |
| distributed-execution | 两 rank 的列/行分片 MLP、梯度重建与四种集合通信 |
| activation-checkpointing | 六层标量 tanh 链的前向、保留、重算、反向和释放 |
| decoding-algorithms | 温度、mask、Top-k、Top-p、逆 CDF、有限语言约束及两步 Beam search |

## 数学与实现约定

- 反传例子包含两个 RMSNorm、因果 Attention、两个残差、SwiGLU 和输出 CE；
  无位置编码、bias、dropout、最终 Norm。仅末位置监督；其他位置可经 K/V 依赖得到梯度。
  所有输入和权重元素在三个目标类别下通过中心有限差分核验。
- 浮点舍入保留 subnormal 与有符号零，溢出返回无穷；FP32 与 Math.fround 对照。
  输入、每次加法都舍入，不模拟硬件混合累加、FMA 或 flush-to-zero。
- 在线 Attention 的首块旧状态缩放为零；中途只算已读前缀，完成后与整行参考比较。
  未声称改变密集注意力的交互数量，未运行 GPU kernel。
- TP 先列切 W1、再按相同中间维度行切 W2；前向 Y 和反向 X 梯度求和，
  参数梯度按分片重建。固定 G 的损失对所有参数和输入做有限差分核验。
- 激活内存统计唯一逻辑 ID，重算相同边界合并计数，未计物理副本、梯度和 workspace。
  6 层全部保存峰值为 7、前向层执行 6 次；段长 2 的重算峰值为 5、执行 12 次。
  页面峰值和次数随已展示事件累计；未完成反向时不提前显示梯度校验成功。
- 解码顺序固定为语法 mask、温度 softmax、Top-k、重归一化、Top-p、重归一化。
  有限语言仅含两个 JSON 文本；完整对象后允许 EOS，所有字面 token 正确 HTML 转义。
  Beam 示例路径等长，不含 EOS 和长度惩罚，宽度 1/2 分别保留 Ax/Bx。

## 资料

核对 PyTorch 2.14 的 autograd、numerical accuracy、checkpoint 文档，NCCL 集合通信文档，
FlashAttention、Megatron-LM 论文和 Transformers generation 文档。
XGrammar 文档站请求失败，改读其论文摘要确认结构化生成与 token/grammar 机制；引用论文。
硬件速度、模型质量和具体框架默认值没有从教学例子外推。

## 维护

六个页面、各自 feature 和一份 exact-mechanics.css，沿用现有静态构建与按页加载。
本轮未新增依赖。反传、浮点和在线 Attention 由主任务实现；两个子任务交付
分片/重算与解码模块，主任务审核全部代码、撰写正文/公式并完成浏览器验证。
数值模块的委派未交付，停止后由主任务实现，未留下未完成文件。

## 验证

- `npm run check`：46 页、54 脚本、30 样式、96 条公式。
- 全部 Node 测试 207 项通过，新增精确机制 44 项。
- 浏览器交互共 130 项通过，含十三反传节点、有限差分、在线归一化、空输入、
  求和顺序、分片重建、激活重算等价、语法约束和 Beam search。
- 展开公式全站审核：390、768 浅色以及 1440 深色均 46/46 页通过。
- 人工截图检查：反传手机/桌面、浮点手机、Attention 手机、TP 桌面、
  解码与激活手机。长小数使用更宽读数布局，数值短表完整显示，事件表限高可滚动。

本轮所有模拟均基于明确数值或有限规则，没有模型质量曲线或未经测量的加速结论。
