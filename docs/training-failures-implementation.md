# 真实训练故障与确定性回放实施说明

状态：**实现、实际实验与切片测试完成，可整合**。2026-10-28 父级已审阅实际报告、最终 prose 和 feature；确认默认／完整报告 validation、Node 8/8、Python 4/4，无阻断性发现。共享构建和三尺寸浏览器验收由整合者执行；本 worker 未运行 build。仅编辑本切片文件，保留原有未跟踪文件与全部报告，未修改 core、共享元数据／导航／package／README，未 commit、reset、clean，未启动其他 agent。

## 接口与诊断约定

已读取 README、父级设计、内容审查、`docs/learning-core-api.md` 与共享 core。导入 `learning-core-v1` 的任务／collation／model／loss、`RecordBatchStream`、RNG 和环境接口；模型前向为 `model(input_ids, attention_mask=...)`。两层 width=32、heads=4、Pre-LayerNorm、dropout=0.1、17,664 参数，未绑定输出头。block 返回 tuple，hook 递归记录 hidden／KV；参数 gradient hooks 按实际 autograd 回调顺序记录。未复制模型／任务或修改核心。

固定同一更新前 checkpoint 和同一输入，分别运行 baseline、faulty、repaired，再运行正常 batch 对照。三个故障只改变一个环节：不稳定 exp 归一化、optimizer LR、有效位置的标签。指数案例使用声明的共同 logit 平移刺激；稳定参考与修复接受同一平移，数学目标仍是原 token CE。loss 始终从本次实际 logits 和标签求出，保存张量后独立重算。

“首个异常”指监控集合中最早观察到的非有限张量；“首个分歧”由与 baseline 的实际张量比较得出。模块 hooks 只覆盖模块边界，损失中的 exp/归一化/目标选择另外显式记录；不声称监控未记录的所有底层 kernel。高 LR 的相对参数更新阈值是预先声明的诊断启发式，有限的异常更新不改写成 NaN。`detect_anomaly` 另行定向回放，只报告实际反向行为，不检测标签语义。

标签故障为 A→B→C→D→A 的内容符号 ID 轮换，**不是位置移位**。EOS／−100／监督 mask 不变。high-LR anomaly probe **仅更新前前向／反向，未执行 optimizer step 或下一前向**；实际 high-LR 后果由主 replay 的更新与下一前向记录。快照保存于完成更新 8 后，待执行步骤 9，二者明确区分。

## 实际报告与测量

最终用于页面的来源：`artifacts/failure-lab-2026-10-28T03-56-27-850Z/report.json`，`passed`，**115 checks**；生成于 `2026-10-28T03:56:48.006363+00:00`。环境为 Python 3.12.14、PyTorch 2.9.0、NumPy 2.3.4、macOS 15.7.3 arm64、CPU float32、单线程与 deterministic algorithms。无新增依赖、预训练下载或 GPU 执行。

- 报告 SHA-256：`9b13bc3acdae593f57fddfc478d58934b10aa15eacea8a675da8b619458a89e9`。
- checkpoint 文件 SHA-256：`83637605b49b034dd6ab67e6b8dc195fd0e378cb648252adfac0c5f81a0fe7c1`。
- 更新前内容 digest：`9f9b41433f6660e4723402c3f1b5b2f3432d4d9dd5f31aa50bc3f13b1f8987fe`。
- 保留最初已审的 `artifacts/failure-lab-2026-10-28T03-50-43-761Z/report.json`（97 checks），未覆盖。

| 案例 | 实际观察 | 修复核验 |
| --- | --- | --- |
| 指数溢出 | 三个 target 分支共同 +128；`loss.exp` 首先出现 768 个 +∞；loss／梯度随后非有限，按共同安全规则跳步。稳定 loss=1.7739410400390625 nats/token。 | 稳定 log_softmax；相同 checkpoint／input／RNG；全部保存张量与最终状态一致。 |
| 高 LR | 0.003→0.6，loss／梯度相同且所有张量有限；相对参数更新 2.898247694023571，超过声明阈值 0.5；下一 loss=5.481266975402832，基线=1.7554837465286255。 | 恢复约定 LR；张量及完整状态相同。 |
| 错误标签 | 36 个内容标签轮换，监督数均为 44；所有张量有限。错误 loss=1.7548508644104004，正确=1.7739406824111938；梯度差 L2=0.9608148442550288。 | 恢复正确标签，从完全相同更新前快照运行；全部张量与最终状态一致。 |

三个案例均有 target/control × baseline/faulty/repaired，共 **18 个独立 Python 进程**。工件包括 `pre-step.pt`、`checkpoint-manifest.json`、每分支 `trace.json / tensors.pt / final-state.pt`、每案例 `anomaly-probe.json`、`environment.json`、完整 `report.json`、`snapshot.json` 和 `primary-installed-docstrings.txt`。非有限标量使用 `value:null + status`；张量统计使用 null 和有限／NaN／±∞ 计数，JSON 写入 `allow_nan=False`。

## 页面与默认数据

- 页面：`training-failures.html`；feature：`training-failures` / `initTrainingFailures`。
- 专属样式：`training-failures.css`；沿用 `modern`、`exact-mechanics`、`decision-chains` 和全站数学排版。
- 实验入口：`node scripts/run-failure-lab.mjs`，沿用 `.venv-framework`，拒绝覆盖已有报告目录。
- 本地导入 schema：`failure-replay-v1`，支持完整 `report.json` 与精简 `snapshot.json`；文件上限 12 MiB。验证矩阵、监督数、finite counts、原始事件顺序、首个异常与 repair/control 证明；安全 textContent／escapeText，选择竞态保护，空／坏文件清空旧状态。
- 默认 `training-failures-data.js` **514,313 bytes（约 502 KiB）**。从上述已保存 115-check 报告确定性提取；原始事件 index 与实际测量保留，不产生模拟数据。完整传播仍可导入；正常对照精简逐参数表。
- 控件覆盖三故障、target/control、baseline/faulty/repaired、事件阶段、上／下记录、首个证据、默认恢复；显示实际 loss、梯度、相对更新、下一 loss、层激活、逐参数梯度／位移、mask／标签和证明。四个宽表有局部滚动、命名 region 与 tabindex。图使用保存的下一 batch loss；下载名为 `training-failures-saved-evidence.json`。
- 导出 API：`initTrainingFailures`、`validateTrainingFailuresReport`、`failureReplayView`、`failureEventRows`、`failureLossChart`、`formatFailureValue`；data 导出 `trainingFailuresSnapshot`。相邻模块 imports 使用构建器支持的双引号形式。
- 页面 H2 anchors：`tf-contract`、`tf-lab`、`tf-numerics`、`tf-update`、`tf-labels`、`tf-limits`、`tf-run`。共享元数据与 stylesheet 注册由 integrator 所有。

## 已执行命令与验证

```sh
# 实际实验，最终报告115 checks；之后没有重训18分支
node scripts/run-failure-lab.mjs

# 只提取已保存报告，不执行训练
.venv-framework/bin/python scripts/failure_replay.py \
  --snapshot-from artifacts/failure-lab-2026-10-28T03-56-27-850Z/report.json \
  --snapshot-output src/scripts/features/training-failures-data.js

# 最终8/8
node --test scripts/test-failure-replay.mjs

# 最终4/4；只读保存的18组张量，无训练／子进程
.venv-framework/bin/python scripts/test-failure-replay.py

# 全部通过
node --check src/scripts/features/training-failures.js
node --check src/scripts/features/training-failures-data.js
node --check scripts/test-failure-replay.mjs
node --check scripts/run-failure-lab.mjs
```

Node 核验 default/full report、报告哈希与默认投影逐项一致、3 类实际证据、repair/control 证明、非法非有限编码／矩阵／计数／首个异常／状态、转义、全部18分支交互、导入错误／空／过大／异步竞态、4条公式独立 KaTeX 渲染、4个宽表 region 与重复 ID。Python 从18组保存 binary tensors 独立重算 loss、逐事件 finite/RMS/max/L2，核验18个不同 PID；修复／正常对照完整 tensor 与 model/optimizer/RNG/cursor/config state 内容哈希相等。父级另独立运行默认／完整报告验证、Node8/8与Python4/4并确认通过。

页面测试可用 `FAILURE_REPORT` 指定其他完整报告路径；Python 用 `FAILURE_REPORT_DIR`。默认指向该115-check实报，不做隐含训练。报告来源保留运行时的后端 hash；运行结束后后端仅进一步精简 snapshot 提取逻辑，未改写旧报告或真实测量。

## 一手资料实际读取与引用元数据

已读**本机已安装 PyTorch 2.9.0 的完整 docstrings**：`detect_anomaly`、`Module.register_forward_hook`、`Tensor.register_hook`、`F.log_softmax`、`F.cross_entropy`、`use_deterministic_algorithms`、`torch.random.get_rng_state`、`AdamW`；还核对 `Module.state_dict` 与 `Optimizer.load_state_dict` 的状态语义。官方网络 fetch 未得到正文，不把网络 URL 视为已读正文。所读版本正文保存在最终工件 `primary-installed-docstrings.txt`，report/data 的 references 记录 read_scope、version 和 docstring SHA-256。

| reference ID | 官方定位 URL | 实际核对范围 |
| --- | --- | --- |
| `torch-autograd-2.9` | `https://docs.pytorch.org/docs/2.9/autograd.html` | 失败反向的关联前向 traceback、反向NaN检查及诊断开销。 |
| `torch-hooks-2.9` | `https://docs.pytorch.org/docs/2.9/generated/torch.nn.Module.html` | module hook 在前向后执行、tensor梯度回调、返回None不改张量。 |
| `torch-logsoftmax-2.9` | `https://docs.pytorch.org/docs/2.9/generated/torch.nn.functional.log_softmax.html` | 稳定log_softmax、CE logits输入、ignore_index及归一化。 |
| `torch-randomness-2.9` | `https://docs.pytorch.org/docs/2.9/notes/randomness.html` | 同软硬件确定性边界、单flag不足、CPU RNG状态。 |
| `torch-adamw-2.9` | `https://docs.pytorch.org/docs/2.9/generated/torch.optim.AdamW.html` | LR乘以解耦decay／动量归一化更新，完整optimizer状态。 |

版本为复现实验固定值，不代表最新依赖推荐。前向“首个”只覆盖监控集合；更新后参数观察按 named-parameter 顺序，未声称优化器内部kernel首个操作；恢复逐位比较只在记录的CPU／软件环境。实验不提供模型泛化或性能结论。

## 最终文件清单（整合者可直接接入）

所有路径均位于 `/Users/glen/lab/ai-learn/`：

- `scripts/failure_replay.py`
- `scripts/run-failure-lab.mjs`
- `scripts/test-failure-replay.py`
- `scripts/test-failure-replay.mjs`
- `src/content/training-failures.html`
- `src/scripts/features/training-failures.js`
- `src/scripts/features/training-failures-data.js`
- `src/styles/training-failures.css`
- `docs/training-failures-implementation.md`

切片无剩余实现阻断。整合者可加载 `training-failures` → `initTrainingFailures`，追加专属 stylesheet 后统一运行共享 build／浏览器验收。本切片不会并发运行 build 或继续重训。
