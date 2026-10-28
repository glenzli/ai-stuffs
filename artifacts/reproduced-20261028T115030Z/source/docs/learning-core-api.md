# 共享学习核心 API v1

文件：`scripts/learning_core.py`。2026-10-28 首次接口交付；导入没有训练、下载、线程或 RNG 副作用。所有 worker 从共享 `/Users/glen/lab/ai-learn` 导入，未使用 fork。

```python
import sys
sys.path.insert(0, "scripts")
from learning_core import (make_task_splits, collate_records, make_model,
                           forward_loss, evaluate_records, configure_cpu)
configure_cpu(threads=1)
splits = make_task_splits()
batch = collate_records(splits["train"][:8])
model = make_model(seed=11, width=64, depth=2, heads=4,
                   dropout=0.1, norm="pre", residual_scale=1.0)
loss = forward_loss(model, batch)
loss.backward()
```

- `CORE_API_VERSION="learning-core-v1"`；兼容命名 `make_task_splits`、`collate_records`、`make_model`、`forward_loss`、`evaluate_records` 保持不变。
- 词表 `PAD=0, BOS=1, SEP=2, EOS=3`，`CONTENT_IDS=(4,5,6,7)` 对应 A/B/C/D，`VOCAB_SIZE=8`，`IGNORE_INDEX=-100`。
- `make_task_splits(data_seed=20261028, *, train_lengths=(2,3,4,5), length_lengths=(6,7,8), train_fraction=.6, validation_fraction=.2, max_groups_per_length=180, length_groups_per_length=24)` 返回 `train/validation/test/length` 的记录列表。记录包含稳定 `id/group_id/content/target/length`；content/target 均为内容 token 的整数列表，target 不含 EOS。先对输入和反向等价的输入分组，再按长度分层拆分；组完整保留，回文只有一条记录。`assert_split_integrity` 验证目标、exact input、ID、组与长度支持；`split_digest` 为完整数据生成 SHA-256。
- `collate_records(records)` 返回 CPU Tensor `input_ids/labels: int64 [B,T]`，`attention_mask: bool [B,T]`（True=有效）、`record_ids/ids/group_ids/records`、整数 `supervised_tokens/input_tokens` 与 `supervised_per_record: int64 [B]`。完整串为 BOS + input + SEP + reversed input + EOS，input_ids 去掉最后 EOS。labels 已移位：仅 SEP 开始预测的答案与 EOS 受监督，提示词与 padding 为 -100。调用者不要再次移位。
- `make_model(seed=11, **config)` 返回 CPU `SmallDecoder`，仅构造时使用隔离的 torch RNG。配置为 `ModelConfig` 的 width/depth/heads/dropout/norm/residual_scale/ff_multiplier/max_seq_len/norm_eps。norm 为 `pre/post/none`，pre-ln/post-ln 是别名。Norm 为 LayerNorm；固定正弦位置编码，标准 0.02 正态线性/Embedding 初始化，GELU MLP，未绑定词表输出头。没有 μP 声明。
- `model(input_ids, attention_mask=...) -> logits [B,T,8]`。普通 forward 总是 logits，不返回 loss/tuple。模型有 `token_embedding/embedding_dropout/blocks/final_norm/lm_head`；各 block 有 `norm1/norm2/attention/mlp/dropout`；Attention 有 `qkv/output`，可直接注册 PyTorch forward hooks。`activation_summary` 与 `parameter_count/model_config` 可用于监控。
- `model.forward_cached(new_input_ids, attention_mask=None, cache=None) -> (new_logits, DecoderCache)` 仅 eval 模式。attention_mask 只描述新增 chunk；cache 包含每层 K/V 与历史有效 mask，`.length` 为物理缓存长度。仅投影新增位置，使用显式偏移因果 mask；固定正弦位置由累计有效位置得到。支持 prompt prefill 与逐 token 或多 token 追加，full/cache 浮点误差与生成一致性会在实验报告单独记录。
- `forward_loss(model,batch,reduction="token") -> scalar Tensor`；`loss_from_logits` 可复用已取得的 logits。token 为全部有效监督 token 加权平均；sequence 为先逐序列平均、再对有监督序列平均，sum 为总和。全无标签抛出 ValueError，调用者必须跳步，不能执行 AdamW 的假零 loss 更新。
- `evaluate_records(model,records, *, batch_size=64, max_new_tokens=20, include_predictions=True, use_cache=True)` 返回 `{metrics,by_length,predictions}`。metrics 区分 teacher_forced_loss（nats/监督 token）、teacher_forced_token_accuracy、free_token_accuracy（micro）、free_token_accuracy_macro 和 exact_match。恢复所有模块原有 train/eval 状态与 Python/NumPy/torch RNG。包含逐题 teacher_forced_predicted 与真实 free predicted，不能互换。
- 所有模型题目使用同一 `DEFAULT_MAX_NEW_TOKENS=20`；prompt 由输入构造，逐步 greedy，首次 EOS 或统一 cap 才结束。不会根据 gold 长度截断。生成中的 PAD/BOS/SEP 均保留为真实轨迹位置、显式传入全 True attention_mask，按错误输出计分；仅 collate 的训练 padding 排除。`score_generation(predicted,target)` 的 target 不含 EOS，predicted 包含实际 EOS（如有）；位置对齐分母为 `max(len(predicted), len(target)+1)`，缺失/额外位置错误，EOS 也是一个计分 token。exact match 必须全部内容与 EOS 的位置正确。termination 为 eos、确实达到上限的 cap，或独立计分器中短而未结束的 incomplete；EOS 后 token 被拒绝。`summarize_predictions` 分别汇总 cap_count/incomplete_count。
- `baseline_predictions(records,kind="identity|constant|oracle",constant_token=4)` 为独立规则结果，不伪造 teacher-forced loss。复制和常量基线利用可观察输入长度并发出 EOS；constant_token 用 `most_common_training_symbol(train)` 仅从训练目标统计。oracle 显式执行反转规则，标为算法上界而非学习结果。
- `RecordBatchStream(records,batch_size=64,seed=11)` 提供 `next_records/state_dict/load_state_dict`，保存专用数据 RNG、完整 permutation、cursor、epoch 与有序记录 IDs。跨 epoch 保持 batch_size；`capture_rng_state/restore_rng_state` 保存 Python、NumPy、CPU torch RNG。学习 runner 会再提供完整 model/optimizer/counters/config checkpoint 与新进程恢复对照。

CLI/报告的 owner 为学习 worker；其他 worker 只需以上核心，勿修改核心文件。初始化和 batch 实验可独立选择配置、优化器与监督归一化，报告须记录实际选择。固定依赖 `.venv-framework`：PyTorch 2.9.0 / NumPy 2.3.4；本核心未新增依赖。
