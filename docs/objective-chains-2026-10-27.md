# 监督、LoRA 与跨卡梯度补充验收 · 2026-10-27

本轮完成三个约定方向及独立 PyTorch 对照实验。新增两个完整专题，深化并行正确性页，
在整层反传页提供框架实验入口。站点现有 51 页、71 个前端脚本、33 份样式、109 条公式。
网站没有新增 JavaScript 依赖；可选框架实验使用独立 Python 环境。

| 内容 | 入口 | 交付范围 |
| --- | --- | --- |
| SFT 样本打包 | `packing-correctness.html` | 两条多轮记录、教学模板、token ID、目标 mask、shift、位置编号、query 可见 key；分别处理、padding、独立 packing、跨样本泄漏、重复 shift；截断、末轮监督与零标签。 |
| LoRA 完整计算 | `lora-mechanics.html` | A/B 初始化、前向与输入梯度、冻结与切断计算图、同步 SGD 更新、低位底座、同底座合并与再次量化。 |
| 跨卡对比学习 | `parallel-correctness.html#contrastive-global` | 全局候选概率矩阵、每个 query 对 document 的梯度贡献、远端 detach、单样本 microbatch、表示拥有者和 DDP 共享参数平均。 |
| 独立框架对照 | `backpropagation.html#framework-reference` | PyTorch 自动微分、参数更新、AdamW 八步轨迹、新进程恢复，以及真实两进程 Gloo/gather/DDP/SGD。 |

训练工程师指南加入四个入口；微调配方和后训练目标页连接新专题。目录、搜索、
页尾适用范围与引用同步更新，避免在配方页重复整段推导。

## 数值与边界约定

Packing 使用 12 类符号词表、二维单头 Attention、输入残差和 12 类输出投影；
无 Norm、FFN 或 dropout。A/B 分别为 11/13 token，默认监督所有 assistant 内容及
对应 EOS，共 9 个有效标签。位置项为 `[0.04p,-0.03p]`，每条记录从 0 开始。
教学模板不声称兼容真实 tokenizer；“只监督末轮”按原记录最后一个 assistant 轮次定义，
右截断后不自动选择更早轮次。

页面把 padding batch 按行展开，用逻辑块隔离执行；Python 参考使用真实 batch 维度。
只比较有效位置，padding query 显式定义零 Attention 输出，不提供标签。
所有记录按同一个全局标签数归一；零标签时 loss 未定义，跳过 optimizer。
独立样本隔离验证固定模型参数，仅替换 A 中一个 user token，观察 B 的 logits。
它不要求训练更新共享权重后 B 仍不变。

默认分别处理和独立 packing 的 loss 约为 2.69834348057872，参数梯度最大差
约为 5.55e-17。跨样本可见造成 logits 最大差约 0.180635；重复 shift 保持默认标签数
为 9、前向 logits 不变，但输出权重梯度最大差约 0.191831。

LoRA 固定二维线性层、rank=1、s=1、平方损失和学习率 0.1。W/x 固定，仅同时更新
A/B。默认 B=0 时 ∇A=0，∇B=`[2.25,-2.875]`，∇x=`[2.49,-1.25]`。
两矩阵同时为零时，本例普通梯度更新停在零增量。切断 Wx 只删除输入的底座梯度分支；
切断整个输出后没有可回传图，实验显式不执行更新。

低位开关使用逐张量 max-abs INT4、刻度 −7…7、半数远离零舍入；计算使用恢复值。
它只隔离低位冻结底座与 adapter 的机制，不实现 NF4、双重量化、分页优化器或低位 kernel。
合并先固定同一个计算底座，另列底座误差和再次量化误差。

对比学习使用 q=`[1,2]`、d=`[1,-1]`，单向点积、无归一化、温度为 0.5/1/2。
每卡计算本地 query loss，候选表示反向按拥有者求和，参数梯度再由 DDP 平均。
没有为 detach 故障加补偿。τ=1 时两个全局候选方案的 loss 均为 2.0725389694803913；
d0 的完整/遗漏远端梯度分别为 +0.9224123290268496 和 −0.05960146101105884。
这里每个 microbatch 只有一对样本，局部候选仅一个正例，loss/梯度为零；这个反例
说明累积本身不会补充负例，不外推所有 mini-batch 对比学习配方。

## 框架实验与结果

运行：

```bash
python3.12 -m venv .venv-framework
.venv-framework/bin/python -m pip install -r scripts/framework-requirements.txt
npm run lab:framework
```

本轮最终记录位于 `artifacts/framework-lab-2026-10-27-final/`。环境为 Python 3.12.14、
PyTorch 2.9.0、NumPy 2.3.4、macOS arm64；固定 CPU float64、单计算线程。
完整安装包版本见该次 `environment.json`。

| 检查范围 | 结果 |
| --- | --- |
| 100 种 packing 配置 | token 标签、前向、loss、参数梯度、一次 SGD 更新或零标签跳步通过。 |
| 48 种 LoRA 配置 | 初始化、图切断、梯度、更新、合并与再次量化通过。 |
| 9 种对比学习配置 | 全局/远端 detach/单样本目标的 loss 和表示梯度通过。 |
| 三个类别的完整小型 block | 全部输入与权重梯度及前向通过。 |
| 八步 AdamW | 与站内 JS 小模型使用同一输入、权重、优化器超参数与随机算法，轨迹在容差内一致。 |
| 新进程完整恢复 | 第三步保存后启动新 Python 解释器，后续 loss 与参数和 PyTorch 连续基线逐值一致。 |
| 遗漏 Adam 对照 | 第一恢复步出现约 0.0162446292 的最大参数差。 |
| 三种方案、各两个 Gloo 进程 | 实际可微 all-gather、远端 detach、本地候选、DDP 平均与一次 SGD 更新符合参考。 |

最终共 1,671 项数值比较通过，最大绝对误差为 8.881784197001252e-16。
跨实现比较采用 atol=1e-9、rtol=1e-8；完整 PyTorch 恢复对照采用零容差。
Node 的手写梯度仅作为比较目标，Python 使用张量算子与 autograd 独立求导。
数据、参数和实际实验输出分别保存，可从报告回溯到每个配置。

第一次 Gloo 运行受沙箱端口绑定限制中断；在允许本机 loopback 通信的执行环境重跑后，
真实双进程验证通过。最终报告是完整成功运行；早期未完成目录不计入验收。

## 一手资料核对

2026-10-27 直接读取 TRL SFT Trainer、PEFT LoRA、PyTorch 2.9 autograd/DDP、
Sentence Transformers losses 官方正文。核对 assistant mask 与模板要求、LoRA A/B
初始化和合并、冻结与 no-grad 区别、DDP 平均、跨设备候选和梯度缓存。
PEFT 原开发指南地址重定向到 v0.21.0 文档，按实际返回正文核对。
另查看本机 PyTorch 2.9.0 的可微 all-gather backward 实现，再以 Gloo 进程验证。

资料集中于 `src/references.mjs`；LoRA/QLoRA 原论文引用沿用项目现有条目。
本次没有把网页检索未返回内容当作成功核对，官方正文经联网命令行读取。
PyTorch 2.9.0 是实验固定版本，不声称最新。

## 网站验收

- `npm run check`：51 页、71 个脚本、33 份样式、109 条公式通过。
- 全部 Node 测试 297 项通过，新增 `test:objectives` 16 项。
- 有限差分核对 packing 全部参数、LoRA 的 A/B/x，以及多种温度的全局对比学习表示梯度。
- 浏览器交互回归 183 项通过，含新增故障开关、截断、量化、更新和目标切换；无未捕获异常。
- 全站 51 页展开公式审核：390px 浅色、768px 浅色、1440px 暗色，资源、链接、锚点、目录、标签和正文溢出均通过。
- 实际截图检查 packing 桌面、LoRA 手机和跨卡对比学习桌面。LoRA 小表已调整为手机直接显示数值与梯度，宽账本仍在独立滚动容器中。
- 最后样式与公式调整后，复核六个改动页在三种视口/主题下的 18 组状态，包含全部折叠内容展开与五步低位 LoRA；无运行错误或警告。记录在 `artifacts/objective-chains-qa/final-targeted-audit.json`。

静态页面和框架实验分别标注适用范围；本轮不运行预训练模型、CUDA/NCCL，
不把小例子输出当作真实模型质量、GPU 吞吐或跨硬件复现结论。
