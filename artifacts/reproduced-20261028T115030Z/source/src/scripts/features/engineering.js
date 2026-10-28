import { $, $$ } from "../core.js";

export function initEngineering() {
    const modeButtons = $$("[data-eng-mode]");
    if (modeButtons.length) {
      const engineeringModes = {
        pretrain: {
          label: "CAUSAL PRETRAINING",
          steps: [
            {
              title: "从 token shard 形成可训练的序列",
              body: "从经过清洗、去重和 tokenizer 处理的二进制 shard 中读取 token。真实系统会并行预取下一批数据，避免 GPU 等磁盘；数据顺序与 shard 本身都是实验版本的一部分。",
              input: "token shard",
              action: "shuffle + read",
              output: "token arrays",
              cpu: "解压、索引、预取",
              gpu: "通常空闲或算上一步",
              network: "共享存储 / 节点传输",
              code: `for shard in shuffled(train_shards):
    for batch in read_sequences(shard):
        train_step(batch)`,
              check: "shard 是否与 tokenizer 版本匹配；dataloader 是否真的没有等 GPU。",
              risk: "GPU 利用率周期性掉到零；step time 抖动，但 kernel 本身没有变慢。",
            },
            {
              title: "组成 [B,T] 的 microbatch，并构造 labels",
              body: "本例从 T+1 个 token 取前 T 个输入、后 T 个标签；由数据侧只错开一次。若模型内部负责 shift，就改传与输入对齐的原标签。变长数据还需 packing、bucket 或 padding/mask。",
              input: "token arrays",
              action: "pack / shift labels",
              output: "ids [B,T]; labels [B,T]",
              cpu: "collate、mask、pin memory",
              gpu: "H2D copy 后等待 forward",
              network: "无梯度通信",
              code: `raw = stack(sequences)  # [B,T+1]
ids = raw[:, :-1]      # [B,T]
labels = raw[:, 1:]    # [B,T]
labels[invalid_target_positions] = -100`,
              check: "packing 是否产生跨样本 attention；padding 是否只占形状不产生有效 loss。",
              risk: "loss 看起来正常但序列边界泄漏；有效 token 统计把 padding 也算进去。",
            },
            {
              title: "每卡做 forward，长序列的主要计算在这里",
              body: "token ID 查表，经过多层 attention / MLP，再由输出投影得到词表 logits。计算量主要随 token 数、模型参数量和序列注意力结构增长。",
              input: "ids [B,T]",
              action: "embed → blocks → logits",
              output: "logits [B,T,V]",
              cpu: "dataloader 并行准备下一批",
              gpu: "矩阵乘、attention、kernel 调度",
              network: "TP / PP 时层内或层间通信",
              code: `x = E[ids]
h = Transformer(x)
logits = h @ W_vocab.T`,
              check: "记录显存峰值与 tokens/s；确认 attention mask 与位置编码没有串样本。",
              risk: "序列长度翻倍可能让激活与 attention 成本显著上升；只看参数量会低估显存。",
            },
            {
              title: "把有效位置的下一 token 概率变成 loss",
              body: "交叉熵在每个有效预测位置计算，再按 token 或样本归一化。softmax 分母包含整个词表，输出层本身就是大矩阵计算。",
              input: "logits + shifted labels",
              action: "stable log-softmax + CE",
              output: "scalar loss",
              cpu: "通常只做日志聚合",
              gpu: "logsumexp、CE、mask reduce",
              network: "一般没有全模型通信",
              code: `loss = cross_entropy(
    logits.reshape(-1, V), labels.reshape(-1),
    ignore_index=-100)`,
              check: "有效 token 数、loss normalization、label shift 是否正确。",
              risk: "loss 下降只说明训练分布更可预测；不代表事实性、指令遵循或检索质量。",
            },
            {
              title: "反向传播为每个 microbatch 累积梯度",
              body: "反向模式自动微分沿计算图求导。为了扩大有效 batch，训练器可能连续处理多个 microstep 后才同步和更新，梯度暂时存在参数/分片缓冲区。",
              input: "scalar loss",
              action: "backward + accumulate",
              output: "dP accumulated",
              cpu: "记录日志、检查异常",
              gpu: "反向 kernel、activation 释放",
              network: "取决于并行策略",
              code: `(loss / accum_steps).backward()
# 仅适用于各 microbatch 有效 token 数相等`,
              check: "变长 microbatch 应按全局有效 token 总数归一化；各卡归约求和/平均系数是否匹配；grad norm 在 unscale 后检查。",
              risk: "loss 与 gradient 的缩放不一致，导致不同 accumulation 配置“数值不等价”。",
            },
            {
              title: "DP/ZeRO 同步或分片规约梯度",
              body: "普通 DP 每卡持有完整模型，不同卡算不同样本，然后 all-reduce 同位置梯度。ZeRO/FSDP 会把 reduce-scatter、all-gather 穿插进反向和 forward。",
              input: "local gradients",
              action: "all-reduce / reduce-scatter",
              output: "global gradients",
              cpu: "配置通信 bucket",
              gpu: "等待 collective",
              network: "主要带宽/延迟瓶颈",
              code: `all_reduce(gradients, op=SUM)
gradients /= dp_world_size
# 框架已同步/平均时，不要重复执行`,
              check: "通信量、overlap、bucket 大小与拓扑是否匹配；多节点还要看跨节点带宽。",
              risk: "GPU 计算利用率高但 step time 仍长，常见原因是 collective 等待或 bucket 太小。",
            },
            {
              title: "优化器更新参数与学习率",
              body: "AdamW 使用一阶/二阶矩估计更新参数，学习率通常带 warmup 和 decay。反向结束后先裁剪梯度，再执行 optimizer.step() 与 scheduler.step()。",
              input: "global grads + optimizer state",
              action: "clip → AdamW → schedule",
              output: "updated weights",
              cpu: "写日志与 checkpoint 元数据",
              gpu: "elementwise update / reductions",
              network: "ZeRO 时可能有参数 all-gather",
              code: `clip_grad_norm_(model.parameters(), max_norm)
optimizer.step()
scheduler.step()
optimizer.zero_grad()`,
              check: "优化器状态是否正确恢复；学习率与 accumulation 是否改变了有效训练预算。",
              risk: "checkpoint 只存模型不存 optimizer/scheduler，恢复后训练轨迹会改变。",
            },
            {
              title: "用 held-out 与生成样例决定是否继续",
              body: "训练 loss 只监测拟合。评估集、生成样例、grad norm、tokens/s、数据游标与 step 时间共同决定是否保留 checkpoint、回滚或调整数据。",
              input: "candidate checkpoint",
              action: "eval + save + gate",
              output: "accepted / rejected",
              cpu: "tokenize eval prompts、汇总指标",
              gpu: "generation / validation forward",
              network: "读评估集、写 checkpoint",
              code: `metrics = evaluate(model, heldout)
if passes_gate(metrics):
    save_checkpoint(model, optimizer, sampler)`,
              check: "checkpoint 能否完整恢复；验证集是否与训练 shard 去重隔离。",
              risk: "只看 training loss 一路下降，最后把严重过拟合或格式退化的模型推进线上。",
            },
          ],
        },
        embedding: {
          label: "CONTRASTIVE EMBEDDING TRAINING",
          steps: [
            {
              title: "从真实检索分布抽 query、正块与难负样本",
              body: "每条训练记录包含 query、正例与负例之间的关系。离线管线要做长度 bucketing、难负样本刷新、false-negative 检查，并留出独立的检索评估集。",
              input: "query / passage shards",
              action: "pair + hard-negative sampler",
              output: "grouped examples",
              cpu: "读取、分组、动态 mask",
              gpu: "通常等待 forward",
              network: "读候选与样本元数据",
              code: `sample = {
  query, positive,
  hard_negatives,
}`,
              check: "难负例是否来自当前模型的真实候选；测试 query 是否在训练数据中泄漏。",
              risk: "训练集只含模板化短问句，线上真实问法召回大幅下降。",
            },
            {
              title: "把一组文本 tokenize 成 query/doc 张量",
              body: "同一 batch 里有不同长度文本，通常做长度分桶减少 padding。query 和 document 可能使用不同 instruction template，但共享同一个 encoder。",
              input: "query [B,Tq]; docs [B,K+1,Td]",
              action: "tokenize + bucket + masks",
              output: "IDs + attention masks",
              cpu: "通常占主要预处理时间",
              gpu: "开始 gather embedding",
              network: "无",
              code: `q_ids = tokenize(query)
d_ids = tokenize([positive, *negatives])`,
              check: "query 与 doc 的 instruction 是否一致复现模型训练协议。",
              risk: "训练时没有 prefix、部署时加了 prefix，导致分布错位。",
            },
            {
              title: "共享编码器为全部文本做 forward",
              body: "query 和 candidate 分别过 backbone，得到 token 状态。为了省算力，正负文档可以批量化；同一份文档被多个 query 命中时也可能复用编码结果。",
              input: "IDs + masks",
              action: "shared Transformer encoder",
              output: "h_q [B,Tq,d]; h_d [B,K,Td,d]",
              cpu: "准备下一组",
              gpu: " Transformer forward",
              network: "无或 DP/TP 通信",
              code: `hq = Encoder(q_ids)
hd = Encoder(d_ids)` ,
              check: "document 编码数量与 batch 组织是否会导致吞吐与显存超预期。",
              risk: "只把 query 数量当 batch，忽略 K 个负文档，导致实际算力翻数倍。",
            },
            {
              title: "池化、归一化并计算候选矩阵",
              body: "从 [B,T,d] 的 token 状态得到句向量，L2 normalize 后用点积得到 [B,C] 相似度。这里的 pooling、温度与 normalize 都必须与部署一致。",
              input: "token states",
              action: "pool → normalize → matmul",
              output: "scores [B,C]",
              cpu: "通常不计分",
              gpu: "dot / softmax / loss",
              network: "DP 时可聚合候选",
              code: `q = normalize(mean_pool(hq))
d = normalize(mean_pool(hd))
scores = q @ d.T / temperature`,
              check: "score 是否包含 in-batch negatives；对角线是否一定对应正样本。",
              risk: "batch 组成或采样顺序错误，把别的 query 的正样本标成负例。",
            },
            {
              title: "用 InfoNCE 对每个 query 行求匹配损失",
              body: "正文档是当前 query 的正确类别。负例分数导数 p/τ 为正，梯度下降会压低分数；高概率负例的这项导数更大。负例的分数导数为正，实际更新方向取其相反方向。",
              input: "scores + positive index",
              action: "row-wise cross entropy",
              output: "mean contrastive loss",
              cpu: "记录难例",
              gpu: "softmax / CE",
              network: "in-batch gather 可能跨卡",
              code: `loss = cross_entropy(
  scores, positive_index)`,
              check: "temperature、loss 归一化和候选 include/exclude 规则是否一致。",
              risk: "增大 batch 后 false negatives 增加，loss 不降或模型把真实答案推远。",
            },
            {
              title: "共享编码器的梯度从所有位置累加",
              body: "query 和 document 共用 backbone，因此两边误差都进入同一组参数。若冻结大部分权重，只训练 projection / LoRA，梯度范围会缩小。",
              input: "contrastive loss",
              action: "backward through shared encoder",
              output: "encoder + pooler grads",
              cpu: "准备下一次难负候选",
              gpu: "backward",
              network: "DP 通信",
              code: `loss.backward()
# shared encoder accumulates q/d gradients`,
              check: "freeze 参数是否真的 requires_grad=False；gradient checkpointing 是否正常。",
              risk: "名义上只训练 projection，但某个 encoder 模块也未冻结，导致表示大幅漂移。",
            },
            {
              title: "跨卡同步梯度并更新编码器",
              body: "不同 GPU 可能分到不同 query 分布。all-reduce 让一次 optimizer step 看到全局匹配关系；有效 batch 与 in-batch negatives 数量会一起变化。",
              input: "local grads",
              action: "sync + optimizer",
              output: "updated encoder",
              cpu: "记录实验配置",
              gpu: "update weights",
              network: "all-reduce / sharded states",
              code: `sync_gradients()
optimizer.step()
scheduler.step()`,
              check: "全局 batch 改变后是否重新检查学习率与训练步数。",
              risk: "显存仅支持小 batch，却沿用了大 batch 训练时的学习率与 temperature。",
            },
            {
              title: "评估完整检索流程的召回与排序",
              body: "评估需要完整语料库：编码文档、建索引、对 held-out query 召回。看 Recall@k、nDCG/ MRR、跨领域与长度分组；上线前用新模型重建文档向量。",
              input: "candidate encoder",
              action: "index docs → evaluate retrieval",
              output: "retrieval report",
              cpu: "索引/评估，可能是瓶颈",
              gpu: "批量编码 documents",
              network: "分布式编码/索引通信",
              code: `index = build_index(encode(docs))
report = eval_retrieval(index, queries)`,
              check: "query 和 document 的接口、normalize、维度、索引版本是否全部一致。",
              risk: "只测句子相似，线上却做长文档问答检索；两者目标分布不同。",
            },
          ],
        },
        sft: {
          label: "INSTRUCTION SUPERVISED FINE-TUNING",
          steps: [
            {
              title: "把对话记录套成确定模板，并生成 assistant mask",
              body: "原始对话先经过角色、system prompt、工具调用与 EOS 约定，再转成 token 序列。通常只保留 assistant / response token 的 label，并对超长样本截断或拆分。",
              input: "conversation records",
              action: "chat template + response mask",
              output: "tokenized conversations",
              cpu: "模板化、mask、长度过滤",
              gpu: "尚未开始",
              network: "读数据",
              code: `tokens = apply_chat_template(messages)
labels = assistant_only_labels(tokens)`,
              check: "训练模板必须与推理请求格式、EOS 和停止条件一致。",
              risk: "模板多出/少掉角色 token，loss 正常但生成边界异常。",
            },
            {
              title: "batch 中真实形状是 IDs + 只对回答有效的 labels",
              body: "若做 packing，多个对话会共享定长张量，但必须隔离 attention 或提供 sample mask，避免模型从一个对话看到另一个对话的目标。",
              input: "tokenized conversations",
              action: "dynamic batch / packing",
              output: "ids [B,T]; labels [B,T]",
              cpu: "长度排序与 collate",
              gpu: "接收 H2D batch",
              network: "无",
              code: `labels[prompt_positions] = -100
labels[padding_positions] = -100`,
              check: "有效 response token 数与长回答占比；是否误训练了 user/system token。",
              risk: "packing 造成跨样本 attention 或下一条 prompt 被当成上一条回答。",
            },
            {
              title: "对 prompt + response 一起做 causal forward",
              body: "模型仍按因果注意力生成每个位置的 hidden state。输入包含 prompt，是为了给回答提供条件；可训练参数可以是全模型、LoRA 或部分层。",
              input: "ids [B,T]",
              action: "causal Transformer",
              output: "logits [B,T,V]",
              cpu: "准备验证 prompt",
              gpu: "forward + attention",
              network: "并行训练时通信",
              code: `logits = model(input_ids=ids,
    attention_mask=mask).logits`,
              check: "训练最大长度、RoPE/位置设置与推理一致；是否启用 activation checkpointing。",
              risk: "上下文被静默截断，长对话后半段从未成为有效训练目标。",
            },
            {
              title: "只对 assistant token 计算 next-token CE",
              body: "prompt token 的 label 被 ignore。更深的配置问题是按 token、按回答还是按样本归一化；长回答多时，按 token 汇总会给它们更高权重。",
              input: "logits + masked labels",
              action: "response-token CE",
              output: "scalar loss",
              cpu: "记录有效 token 数",
              gpu: "log-softmax + CE",
              network: "无",
              code: `loss = cross_entropy(
  logits[:, :-1].reshape(-1, V),
  labels[:, 1:].reshape(-1),
  ignore_index=-100)`,
              check: "loss 下降是否来自复述 prompt、学会立即 EOS，或只拟合少数长样本。",
              risk: "模板或 mask 错误导致模型训练“看到问题就生成固定 token”。",
            },
            {
              title: "反向只沿保留的 token 路径累积",
              body: "梯度经过回答位置的模型计算，prompt 位置也可能通过 attention 影响后续预测，但它们的输出 loss 为零。最终参数更新来自整个条件计算图。",
              input: "masked CE loss",
              action: "backward",
              output: "model / LoRA gradients",
              cpu: "日志",
              gpu: "backward",
              network: "并行时同步",
              code: `loss.backward()
# prompt-only positions do not add CE loss`,
              check: "LoRA 作用于哪些模块、是否意外训练 embedding 与 lm_head。",
              risk: "以为只训 LoRA 就不会遗忘；数据分布与学习率仍能造成明显漂移。",
            },
            {
              title: "同步参数或 adapter 梯度",
              body: "数据并行下不同 GPU 分到不同对话。全参数训练同步全部梯度，LoRA 只需同步 adapter 梯度，通信量显著更小。",
              input: "local grads",
              action: "all-reduce adapter / model",
              output: "global grads",
              cpu: "配置 bucket",
              gpu: "等待 collective",
              network: "显著少于全参数",
              code: `all_reduce(trainable_gradients)`,
              check: "固有权重是否被优化器创建状态；冻结参数是否仍在 autograd 图中。",
              risk: "广告说 LoRA，但实际 optimizer 仍对全参数建状态，显存没省下来。",
            },
            {
              title: "更新，并在固定步数检查生成行为",
              body: "SFT 的 token loss 不能完整代表回答质量。训练中应定期用 held-out prompts 生成，检查格式、EOS、长度、事实错误与拒答是否退化。",
              input: "candidate weights",
              action: "update + generation eval",
              output: "candidate checkpoint",
              cpu: "prompt 集合与指标",
              gpu: "生成 + 评分",
              network: "服务模型时可能需要导出",
              code: `optimizer.step()
samples = generate(heldout_prompts)
report = eval_samples(samples)`,
              check: "评估要固定 decoding 参数；对比新旧模型时用同一 prompt 集。",
              risk: "训练 loss 极低，但模型学会泛化到固定模板，开放输入变差。",
            },
            {
              title: "决定保存全权重、LoRA adapter 还是合并模型",
              body: "保存格式决定后续部署。LoRA 需要同时记录 base model revision、adapter、tokenizer 与模板；部署如果要合并权重，必须验证数值和回滚路径。",
              input: "checkpoint + config",
              action: "export + regression",
              output: "deployable revision",
              cpu: "导出/上传",
              gpu: "merge / evaluation",
              network: "权重读写",
              code: `save_adapter(adapter, base_revision)
merge_and_verify_if_needed()`,
              check: "base 模型版本可追溯；合并后再跑一次 smoke test 与完整评估。",
              risk: "训练时用了一个 base revision，部署时加载了另一个，行为不可复现。",
            },
          ],
        },
        dpo: {
          label: "DIRECT PREFERENCE OPTIMIZATION",
          steps: [
            {
              title: "从同一 prompt 构造 chosen / rejected 偏好对",
              body: "DPO 数据提供同一问题下的回答偏好。要检查 chosen/rejected 是否只差在目标维度，避免长度偏差或把事实正确与风格偏好混在一起。",
              input: "preference shards",
              action: "pair + template + length audit",
              output: "prompt / winner / loser",
              cpu: "模板、长度与重复检查",
              gpu: "尚未开始",
              network: "读数据",
              code: `pair = {
  prompt,
  chosen, rejected,
}`,
              check: "同一 prompt 的两条回答是否来自可比较的生成条件；是否有长度偏置。",
              risk: "长度偏好可能使回答变长，正确率却没有相应改善。",
            },
            {
              title: "把偏好对整理成 policy 与 reference 的输入",
              body: "同一 batch 要保留 prompt 对齐与响应 token 边界。reference 模型通常冻结，但仍需要 forward 得到 reference log-probs。",
              input: "prompt + pair",
              action: "tokenize + response mask",
              output: "chosen/rejected IDs; same ref inputs",
              cpu: "打包、mask、padding",
              gpu: "H2D",
              network: "无",
              code: `inputs = [tokens(prompt, chosen),
          tokens(prompt, rejected)]`,
              check: "chosen/rejected 边界与 response-only log-prob 的定义是否一致。",
              risk: "对包含 prompt 的整段概率做差，prompt 相同但 token 化长度不同引入偏差。",
            },
            {
              title: "policy 与 reference 分别 forward",
              body: "engine 常对 chosen/rejected 各做 policy 和 reference 前向，并只对 response token 求和 log-prob。reference 不计算梯度或使用单独推理路径。",
              input: "chosen/rejected IDs",
              action: "policy logits + frozen reference logits",
              output: "log πθ, log πref",
              cpu: "调度与日志",
              gpu: "多次 sequence forward",
              network: "参考模型分片/复制配置",
              code: `policy_logp = logprob(policy(ids), mask)
ref_logp = no_grad(
  logprob(reference(ids), mask))`,
              check: "参考模型 revision、precision 与 tokenization 是否固定；显存是否容纳双 forward。",
              risk: "只算一条 answer 或 reference cache 复用到错误 prompt，偏好 margin 失真。",
            },
            {
              title: "计算 reference-corrected margin",
              body: "DPO 比较 chosen 相对 rejected 的优势，再减去 reference 的同一优势。该目标优化的是相对 reference 的概率差。",
              input: "log πθ; log πref",
              action: "pairwise margin",
              output: "m per preference pair",
              cpu: "记录 margin 分布",
              gpu: "sum / subtract",
              network: "一般无",
              code: `delta = logp_chosen - logp_rejected
margin = beta * (delta_theta - delta_ref)`,
              check: "长度是否导致 log-prob 和更大；是否需要长度归一化作为数据处理选择。",
              risk: "只优化长度、格式或某种措辞差异，偏好收益不可迁移。",
            },
            {
              title: "把 margin 送进 pairwise logistic loss",
              body: "loss = -log σ(margin)。margin 已经正确时梯度较小；相反时梯度更大。β 同时缩放隐式 reward 与目标锐度。",
              input: "margin",
              action: "log-sigmoid",
              output: "scalar DPO loss",
              cpu: "记录 chosen/rejected 样本",
              gpu: "sigmoid / log / reduce",
              network: "无",
              code: `loss = -logsigmoid(margin)`,
              check: "β、reference revision 与长度聚合是否一致；训练/评估 margin 分布是否异常。",
              risk: "β 同时缩放梯度与改变 sigmoid 饱和程度；错误偏好、学习率与训练时长共同影响偏移，需实测 KL 与回归能力。",
            },
            {
              title: "反向只更新 policy（或 adapter）",
              body: "reference 不接收梯度。需 policy forward/backward 和 reference log-prob；固定数据可预计算 reference，LoRA 也可复用冻结底座，所以不一定每步驻留两套完整模型。",
              input: "DPO loss",
              action: "policy backward",
              output: "policy gradients",
              cpu: "调度",
              gpu: "backward policy path",
              network: "并行时通信",
              code: `loss.backward()
# reference parameters remain frozen`,
              check: "reference 是否真的无梯度；adapter 以外的参数是否被误更新。",
              risk: "reference 也参与更新，DPO 的相对约束基准逐渐移动。",
            },
            {
              title: "跨卡同步并按偏好预算更新",
              body: "全局 batch 决定一次更新看到多少偏好对。数据并行同步 policy 梯度；多种并行方案还要考虑两个模型的参数和 activation 放置。",
              input: "policy gradients",
              action: "sync + optimizer",
              output: "updated policy",
              cpu: "记录 steps / pairs",
              gpu: "collective + update",
              network: "梯度与参数通信",
              code: `sync_gradients()
optimizer.step()
clip / schedule as configured`,
              check: "每步偏好对数量、β、学习率与参考模型版本一起记录。",
              risk: "只记录全局 pair 数，忽略同一 prompt 多 pair 导致的重复权重。",
            },
            {
              title: "评估偏好胜率、能力和回归",
              body: "目标内看 held-out pair 胜率、隐式 reward/margin；还必须测事实性、安全、拒答、指令遵循、长度分布和基础能力。DPO 可能同时改善偏好与损害其他能力。",
              input: "candidate policy",
              action: "pair eval + regression suite",
              output: "release decision",
              cpu: "对比与汇报",
              gpu: "生成候选回答",
              network: "评估/部署",
              code: `report = eval_preferences(policy)
report += regression_suite(policy)`,
              check: "评估 judge 是否与新偏好同源；有没有人工抽检和过拟合验证集。",
              risk: "用同一个 judge 既造训练偏好又评估，形成闭环自我强化。",
            },
          ],
        },
      };

      let engineeringMode = "pretrain";
      let engineeringStep = 0;
      const fields = {
        label: $("[data-eng-label]"),
        title: $("[data-eng-title]"),
        step: $("[data-eng-step]"),
        body: $("[data-eng-body]"),
        input: $("[data-eng-input]"),
        action: $("[data-eng-action]"),
        output: $("[data-eng-output]"),
        cpu: $("[data-eng-cpu]"),
        gpu: $("[data-eng-gpu]"),
        network: $("[data-eng-network]"),
        code: $("[data-eng-code]"),
        check: $("[data-eng-check]"),
        risk: $("[data-eng-risk]"),
        prev: $("[data-eng-prev]"),
        next: $("[data-eng-next]"),
      };

      const renderEngineeringStep = () => {
        const mode = engineeringModes[engineeringMode];
        const step = mode.steps[engineeringStep];
        fields.label.textContent = mode.label;
        fields.title.textContent = step.title;
        fields.step.textContent = String(engineeringStep + 1).padStart(2, "0");
        fields.body.textContent = step.body;
        fields.input.textContent = step.input;
        fields.action.textContent = step.action;
        fields.output.textContent = step.output;
        fields.cpu.textContent = step.cpu;
        fields.gpu.textContent = step.gpu;
        fields.network.textContent = step.network;
        fields.code.textContent = step.code;
        fields.check.textContent = step.check;
        fields.risk.textContent = step.risk;
        $$("[data-eng-step-index]").forEach((button, index) => {
          button.classList.toggle("is-current", index === engineeringStep);
        });
        fields.prev.disabled = engineeringStep === 0;
        fields.next.disabled = engineeringStep === mode.steps.length - 1;
      };

      modeButtons.forEach((button) => {
        button.addEventListener("click", () => {
          engineeringMode = button.dataset.engMode;
          engineeringStep = 0;
          modeButtons.forEach((item) => {
            item.setAttribute("aria-selected", String(item === button));
          });
          renderEngineeringStep();
        });
      });
      $$("[data-eng-step-index]").forEach((button) => {
        button.addEventListener("click", () => {
          engineeringStep = Number(button.dataset.engStepIndex);
          renderEngineeringStep();
        });
      });
      fields.prev.addEventListener("click", () => {
        engineeringStep = Math.max(0, engineeringStep - 1);
        renderEngineeringStep();
      });
      fields.next.addEventListener("click", () => {
        engineeringStep = Math.min(
          engineeringModes[engineeringMode].steps.length - 1,
          engineeringStep + 1,
        );
        renderEngineeringStep();
      });
      renderEngineeringStep();
    }

    const budgetGpus = $("[data-budget-gpus]");
    if (budgetGpus) {
      const budgetMicro = $("[data-budget-micro]");
      const budgetSeq = $("[data-budget-seq]");
      const budgetAccum = $("[data-budget-accum]");

      const formatTokens = (value) => {
        if (value >= 1e9) return `${(value / 1e9).toFixed(1)}B`;
        if (value >= 1e6) return `${(value / 1e6).toFixed(1)}M`;
        if (value >= 1e3) return `${(value / 1e3).toFixed(1)}K`;
        return String(value);
      };

      const renderBudget = () => {
        const gpus = Number(budgetGpus.value);
        const micro = Number(budgetMicro.value);
        const seq = Number(budgetSeq.value);
        const accum = Number(budgetAccum.value);
        $("[data-budget-gpus-value]").textContent = String(gpus);
        $("[data-budget-micro-value]").textContent = String(micro);
        $("[data-budget-seq-value]").textContent = String(seq);
        $("[data-budget-accum-value]").textContent = String(accum);
        $("[data-budget-forward]").textContent = formatTokens(micro * seq);
        $("[data-budget-global-micro]").textContent = formatTokens(gpus * micro * seq);
        $("[data-budget-tokens]").textContent = formatTokens(gpus * micro * seq * accum);
        $("[data-budget-samples]").textContent = String(gpus * micro * accum);
      };
      [budgetGpus, budgetMicro, budgetSeq, budgetAccum].forEach((input) => {
        input.addEventListener("input", renderBudget);
      });
      renderBudget();
    }

    const memoryModel = $("[data-memory-model]");
    if (memoryModel) {
      const optimizer = $("[data-memory-optimizer]");
      const gpuCount = $("[data-memory-gpus]");
      const stageButtons = $$("[data-zero-stage]");
      let zeroStage = "dp";

      const formatGb = (value) => `${value.toFixed(value >= 10 ? 1 : 2)} GB`;

      const renderMemory = () => {
        const paramsB = Number(memoryModel.value);
        const bytesPerParam = Number(optimizer.value);
        const gpus = Number(gpuCount.value);
        const weightsPerGpu = 2 * paramsB;
        const gradsPerGpu = 2 * paramsB;
        const optimizerPerGpuBase = (bytesPerParam - 4) * paramsB;
        let perGpu;
        if (zeroStage === "dp") {
          perGpu = { weights: weightsPerGpu, grads: gradsPerGpu, optimizer: optimizerPerGpuBase };
        } else if (zeroStage === "1") {
          perGpu = { weights: weightsPerGpu, grads: gradsPerGpu, optimizer: optimizerPerGpuBase / gpus };
        } else if (zeroStage === "2") {
          perGpu = { weights: weightsPerGpu, grads: gradsPerGpu / gpus, optimizer: optimizerPerGpuBase / gpus };
        } else {
          perGpu = {
            weights: weightsPerGpu / gpus,
            grads: gradsPerGpu / gpus,
            optimizer: optimizerPerGpuBase / gpus,
          };
        }
        const perGpuTotal = perGpu.weights + perGpu.grads + perGpu.optimizer;
        const baseline = bytesPerParam * paramsB;
        const saved = Math.max(0, (1 - perGpuTotal / baseline) * 100);
        const stageName = zeroStage === "dp" ? "DP" : `ZeRO-${zeroStage}`;
        const optimizerName = bytesPerParam === 16 ? "AdamW" : "SGD+momentum";
        $("[data-memory-summary]").textContent = `${paramsB}B · ${optimizerName} · ${gpus} GPU · ${stageName}`;
        $("[data-memory-per-gpu]").textContent = formatGb(perGpuTotal);
        $("[data-memory-total]").textContent = formatGb(perGpuTotal * gpus);
        $("[data-memory-saved]").textContent = `${Math.round(saved)}%`;

        const segments = [
          { label: "权重 BF16", value: perGpu.weights, color: "var(--purple)" },
          { label: "梯度 BF16", value: perGpu.grads, color: "var(--blue)" },
          { label: "优化器状态", value: perGpu.optimizer, color: "var(--coral)" },
        ].filter((segment) => segment.value > 0.0001);
        $("[data-memory-stack]").innerHTML = segments.map((segment) => `
          <div class="memory-segment" style="flex:${segment.value}">
            <i style="background:${segment.color}"></i>
            <span>${segment.label}</span>
            <strong>${formatGb(segment.value)}</strong>
          </div>
        `).join("");

        const notes = {
          dp: "普通数据并行中，每张卡都保留完整参数、梯度和优化器状态。模型本身放不下一张卡时，不能靠增加 accumulation 解决。",
          "1": "ZeRO-1 分片优化器状态；权重仍复制。本图按未分片梯度估算静态内存，理想通信量可与 DP 同阶，实际看规约、参数同步与重叠方式。",
          "2": "ZeRO-2 再切梯度。每卡仍持有完整参数，但 optimizer step 前的通信与内存占用更复杂。",
          "3": "ZeRO-3 连参数也分片，计算某层前临时 all-gather。静态状态变小，通信和调度更多；仍不含 activation。",
        };
        $("[data-memory-note]").textContent = notes[zeroStage];
        stageButtons.forEach((button) => {
          button.setAttribute("aria-pressed", String(button.dataset.zeroStage === zeroStage));
        });
      };

      stageButtons.forEach((button) => {
        button.addEventListener("click", () => {
          zeroStage = button.dataset.zeroStage;
          renderMemory();
        });
      });
      [memoryModel, optimizer, gpuCount].forEach((input) => {
        input.addEventListener("change", renderMemory);
      });
      $("[data-memory-reset]").addEventListener("click", () => {
        zeroStage = "dp";
        renderMemory();
      });
      renderMemory();
    }
}
