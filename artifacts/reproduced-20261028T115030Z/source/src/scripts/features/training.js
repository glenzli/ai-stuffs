import { $, $$ } from "../core.js";

export function initTraining() {
    const modes = {
      word2vec: {
        label: "静态词向量 / WORD2VEC SGNS",
        steps: [
          {
            title: "先采样“中心词 → 真实上下文”与噪声词",
            body: "从语料滑窗构造训练对。中心词与真实上下文是正样本；再按词频的 3/4 次方分布抽 K 个噪声词作为负样本。该目标学习局部词与上下文的搭配关系。",
            input: "[B] center ID",
            op: "positive + K negatives",
            output: "[B, 1+K] IDs",
            note: "噪声分布通常比原始词频更平，降低高频虚词垄断负样本的概率，采样依据是词频，语义难度需另行判断。",
            code: `for center, context in windows:
    pos = context
    negs = sample(noise_dist, k=K)
    update(center, pos, negs)`,
          },
          {
            title: "中心词和上下文各查一张表",
            body: "经典实现有输入表 E 和输出/上下文表 W。每个 ID 取出一行；一个中心词配一个正上下文与 K 个负上下文，于是形成 1+K 个向量。",
            input: "[B, 1+K] IDs",
            op: "E[ids], W[ids]",
            output: "[B,d], [B,1+K,d]",
            note: "有些现代实现共用或单独定义输出投影，但“离散 ID 查向量的行”这一核心不变。",
            code: `center = E[center_ids]     # [B, d]
context = W[context_ids]   # [B, 1+K, d]`,
          },
          {
            title: "这一步没有 Transformer",
            body: "Word2Vec 的上下文混合很轻：中心词向量直接与每个上下文向量做点积。它不从整句动态生成新状态，所以学到的仍是静态词向量。",
            input: "center [B,d]",
            op: "broadcast dot",
            output: "scores [B,1+K]",
            note: "与 BERT/LLM 对照：后者会先用注意力让每个位置吸收上下文，再产生预测。",
            code: `scores = einsum('bd,bkd->bk',
                  center, context)`,
          },
          {
            title: "正样本应得高分，负样本应得低分",
            body: "对每个点积取 sigmoid，得到“这一对像真实搭配”的概率。真实上下文希望接近 1；噪声词希望接近 0。",
            input: "scores [B,1+K]",
            op: "sigmoid",
            output: "probs [B,1+K]",
            note: "SGNS 定义了独立的二分类目标；它不是完整词表 softmax 交叉熵的无偏近似。负样本数 K 和噪声分布会改变学到的几何。",
            code: "p_pair = sigmoid(scores)  # [B, 1+K]",
          },
          {
            title: "把正负两类错误合成一个标量",
            body: "正样本用 -log σ(score)；负样本用 -log σ(-score)。模型如果既让正样本高、又让负样本低，损失才会下降。",
            input: "scores + labels",
            op: "logistic loss",
            output: "scalar loss",
            note: "其他未抽到的词并没有显式变成负样本；它们只是这一轮没有参与。训练上的近似会带来方差与采样偏差。",
            code: `loss = -log(sigmoid(s_pos))
     - sum(log(sigmoid(-s_neg)))`,
          },
          {
            title: "梯度只回到这一批被取出的行",
            body: "SGNS 反传只给采样到的中心/上下文行当前损失梯度，重复出现会累加。无动量、无衰减的 SGD 下，未访问行保持原样；其他优化器要另查历史状态。",
            input: "loss + comp graph",
            op: "autograd + SGD/Adam",
            output: "updated E / W rows",
            note: "高频词会更新很多次，低频词更新少；这也是训练分布直接影响向量质量的原因。",
            code: `loss.backward()
optimizer.step()  # 一次更新两张表`,
          },
        ],
      },
      masked: {
        label: "掩码语言模型 / BERT-STYLE MLM",
        steps: [
          {
            title: "随机遮住一部分 token，让模型填空",
            body: "原始 BERT 选约 15% 的位置计 MLM loss：其中 80% 换成 [MASK]，10% 换成随机 token，10% 保持原样。标签始终是替换前的 ID。",
            input: "token IDs [B,T]",
            op: "mask positions",
            output: "corrupted IDs + labels",
            note: "直接计 loss 的位置更少，但其上下文位置仍可收到反传梯度；不能按监督位置数直接比较模型能力或总信息量。",
            code: `corrupted, labels = random_mask(ids)
# 未选中位置的 label = -100`,
          },
          {
            title: "输入 embedding 表与位置编码叠加",
            body: "查表用替换后的 corrupted_ids：被换成 [MASK] 就查 [MASK] 行，随机替换就查随机 token 行，保留原样才查原 ID。再加入位置与 token-type embedding。原始标签不能泄漏回输入。",
            input: "IDs [B,T]",
            op: "E[ids] + position",
            output: "x [B,T,d]",
            note: "查出的输入行只是起点；最终用于填空的是经过多层双向 Transformer 的隐藏状态。",
            code: "x = E[corrupted_ids] + pos + token_type",
          },
          {
            title: "双向注意力让每个位置看左右两边",
            body: "与 causal LM 不同，MLM 编码器通常没有未来 token 遮罩，因此左右语义都能进入表示。这有利于理解，但不能直接按因果顺序逐词生成。",
            input: "x [B,T,d]",
            op: "bidirectional attention",
            output: "h [B,T,d]",
            note: "用于句向量时，需要选择 pooling，并在句级任务上训练或评估该接口。",
            code: "h = BertEncoder(x, attention_mask)",
          },
          {
            title: "把被遮位置的隐藏状态投到词表",
            body: "每个遮住的位置都输出一个词表分布。你只关心它被遮掉的那个真实 token 在词表里的概率。",
            input: "h [B,T,d]",
            op: "linear head → vocab",
            output: "logits [B,T,V]",
            note: "先选监督位置再过词表 head 可省输出投影计算；若先算所有 logits 再 mask loss 则不会省这部分。Encoder 仍需处理上下文，梯度也会经过上下文位置。",
            code: "logits = h @ W_vocab.T",
          },
          {
            title: "交叉熵让真实词的分数相对变高",
            body: "在 mask 位置，正确 token 的 softmax 概率应该提高；同一词表里的其他 token 不需要显式标负，它们的概率在归一化后相对下降。",
            input: "logits + labels",
            op: "masked cross entropy",
            output: "scalar loss",
            note: "这属于自监督信号：监督来自文本中被遮住的真实 token，不需要人类标注。",
            code: `loss = CE(logits[mask],
             labels[mask])`,
          },
          {
            title: "共享权重沿所有使用它的位置得到梯度",
            body: "一个 mask 位置的误差会经输出头、Transformer 和多层网络回到对应 embedding 行；同一 token 在别处被访问时还会累加。Encoder 与输出头通常都更新。",
            input: "loss + comp graph",
            op: "autograd",
            output: "updated params",
            note: "如果拿 BERT 输出直接做句向量，常见问题是没有专门优化“句对距离”；后训练才会更有针对性地改变几何。",
            code: `loss.backward()
optimizer.step()`,
          },
        ],
      },
      causal: {
        label: "下一词预测 / CAUSAL LM",
        steps: [
          {
            title: "错开一格：输入前 T 个，标签后 T 个",
            body: "每段 token 都能产生很多预测位置。模型看到 t 及以前的内容，目标是预测 t+1；输入与目标来自同一段文本，因此是自监督。",
            input: "原始 ids [B,T+1] int64",
            op: "labels = ids[:,1:]",
            output: "inputs / labels 均为 [B,T]",
            note: "训练时使用 teacher forcing，直接用真实历史；推理时却要拿模型刚生成的 token 继续喂入。",
            code: `x = ids[:, :-1]
y = ids[:, 1:]`,
          },
          {
            title: "每个 ID 从 embedding 表取一行",
            body: "E 的行数等于 tokenizer 词表大小，列数是隐藏宽度 d。初始参数通常随机小值初始化，之后通过梯度逐步形成可用的表示。",
            input: "ids [B,T]",
            op: "gather rows",
            output: "x [B,T,d]",
            note: "位置信息通常另加位置编码或由 RoPE 等机制在注意力中引入，让网络利用序列顺序。",
            code: "x = E[input_ids]  # RoPE 在各层 Q/K 上应用",
          },
          {
            title: "因果注意力让位置只看左边",
            body: "第 t 个位置的隐藏状态融合前文，却不能偷看 t+1。否则预测下一词时模型直接看到答案，loss 会虚假地下降。",
            input: "x [B,T,d]",
            op: "causal attention stack",
            output: "h [B,T,d]",
            note: "每个位置都有自己的上下文向量。要得到整句向量，还需按任务选择 pooling 或读出位置。",
            code: `h = Transformer(x,
    causal_mask=True)`,
          },
          {
            title: "每个位置产生整个词表上的 logits",
            body: "隐藏状态h_t乘输出投影，得到V个下一 token 候选分数。softmax 后形成概率分布，训练与推理都在这个分布上工作。",
            input: "h [B,T,d]",
            op: "h @ W_vocab.T",
            output: "logits [B,T,V]",
            note: "logits 是未归一化分数；softmax 才把同一位置的所有候选变成和为 1 的概率。",
            code: "logits = h @ W_vocab.T",
          },
          {
            title: "真实下一词是正类，其他词被相对压低",
            body: "交叉熵最大化真实 token 的对数概率。对所有位置求平均后得到标量 loss；模型不需要人工准备“负例”，词表中其他候选自然承担概率分母。",
            input: "logits, next IDs",
            op: "next-token CE",
            output: "scalar loss",
            note: "如果文本里有重复/模板，模型会利用这些统计规律；预训练目标并不自动保证事实性或有用性。",
            code: `loss = cross_entropy(
    logits, next_ids)`,
          },
          {
            title: "梯度贯穿输出头、Transformer 与 embedding 表",
            body: "输入 lookup 梯度来自 token 出现的位置；若 E 与输出头绑定，还要加上完整词表预测的输出梯度。Transformer 各层同样沿计算图收到梯度。",
            input: "loss + graph",
            op: "backprop + optimizer",
            output: "updated shared params",
            note: "冻结 embedding 后查表行保持固定，其他可训练层仍继续更新。",
            code: `loss.backward()
optimizer.step()
# E may be frozen separately`,
          },
        ],
      },
      contrastive: {
        label: "句向量 / CONTRASTIVE InfoNCE",
        steps: [
          {
            title: "构造 query、正文档和难负文档",
            body: "样本按 query 与文档之间的成对或成组关系组织。正样本是应命中的文档；“表面像但不该答”的段落作为难负样本，才能细调边界。",
            input: "query [B,Tq]",
            op: "pair + hard negatives",
            output: "docs [C,Td]；C=B+K",
            note: "本例把 B 个正文档排在前 B 列，再追加 K 个难负文档，query i 的正例索引就是 i。真实负例必须符合任务的相关性标准。",
            code: `queries = [q0, ..., q_last]
documents = positives + extra_negatives
# B 个正例 + K 个额外候选；正例索引为 0..B-1`,
          },
          {
            title: "query 和 document 共用编码器",
            body: "同一个 Transformer backbone 接收两边文本，各自查 token embedding 并经过上下文层；先输出逐 token 状态，后续再聚合为句向量。",
            input: "token IDs",
            op: "shared encoder",
            output: "hq [B,Tq,d]；hd [C,Td,d]",
            note: "共享编码器迫使 query 与文档进入同一个可比空间；也可以使用不同编码器，但训练和对齐更复杂。",
            code: `hq = Encoder(query_ids)
hd = Encoder(doc_ids)`,
          },
          {
            title: "池化成一个向量并做归一化",
            body: "句向量必须决定如何从 T 个 token 状态压成一个 d 维向量：CLS、mean、last-token 或学习式 pooler。归一化后点积等于余弦相似度。",
            input: "hq [B,Tq,d]；hd [C,Td,d]",
            op: "pool + L2 normalize",
            output: "q [B,d]；documents [C,d]",
            note: "池化是模型接口的一部分；换 pooling 或漏掉训练时的 normalize，都可能让索引结果漂移。",
            code: `q = normalize(pool(hq))
documents = normalize(pool(hd))`,
          },
          {
            title: "所有配对组成相似度矩阵",
            body: "query 与 batch 中每个正/负文档算点积，得到矩阵。batch 内其他 query 的正文档有时自然成为 in-batch negatives。",
            input: "q [B,d]；documents [C,d]",
            op: "q @ dᵀ",
            output: "scores [B,C]",
            note: "相似度数值本身不可跨模型直接比较；在不同模型、不同提示前缀下，score 标尺会变了。",
            code: `scores = q @ documents.T
# diagonal = positives`,
          },
          {
            title: "让正样本在候选中排得更靠前",
            body: "InfoNCE 对每个 query 的候选行做分类。正文档相对其他候选更占优，loss 就能下降；不要求每个正分数绝对上升、每个负分数都下降。共享参数更新还会耦合不同样本。",
            input: "scores + labels",
            op: "InfoNCE / CE",
            output: "scalar loss",
            note: "难负样本太少，模型容易学会“泛泛相关”；负样本标错，模型会被迫把真正答案推远。",
            code: `loss = CE(scores / temperature,
           positive_index)`,
          },
          {
            title: "梯度改变整段文本的几何",
            body: "梯度穿过归一化、池化与编码器，更新其中可训练的参数；mean pooling 本身没有参数。可以全参数训练，也可以只训练 LoRA 或部分层，需按数据、成本与评估决定。",
            input: "loss + graph",
            op: "backprop + mining",
            output: "new embedding geometry",
            note: "后训练可能改善特定检索，也可能损伤其他语言或领域，因此需做跨任务回归评估。",
            code: `loss.backward()
optimizer.step()
# optionally freeze backbone`,
          },
        ],
      },
    };

    const modeButtons = $$("[data-training-mode]");
    const stepButtons = $$("[data-training-step]");
    if (!modeButtons.length || !stepButtons.length) return;

    let activeMode = "causal";
    let activeStep = 0;
    const fields = {
      mode: $("[data-walker-mode]"),
      title: $("[data-walker-title]"),
      counter: $("[data-walker-step]"),
      body: $("[data-walker-body]"),
      input: $("[data-walker-in]"),
      op: $("[data-walker-op]"),
      output: $("[data-walker-out]"),
      note: $("[data-walker-note]"),
      code: $("[data-walker-code]"),
      prev: $("[data-walker-prev]"),
      next: $("[data-walker-next]"),
    };

    const renderTrainingStep = () => {
      const mode = modes[activeMode];
      const step = mode.steps[activeStep];
      fields.mode.textContent = mode.label;
      fields.title.textContent = step.title;
      fields.counter.textContent = String(activeStep + 1).padStart(2, "0");
      fields.body.textContent = step.body;
      fields.input.textContent = step.input;
      fields.op.textContent = step.op;
      fields.output.textContent = step.output;
      fields.note.textContent = step.note;
      fields.code.textContent = step.code;
      stepButtons.forEach((button, index) => {
        button.classList.toggle("is-current", index === activeStep);
      });
      fields.prev.disabled = activeStep === 0;
      fields.next.disabled = activeStep === mode.steps.length - 1;
    };

    modeButtons.forEach((button) => {
      button.addEventListener("click", () => {
        activeMode = button.dataset.trainingMode;
        activeStep = 0;
        modeButtons.forEach((item) => {
          item.setAttribute("aria-selected", String(item === button));
        });
        renderTrainingStep();
      });
    });

    stepButtons.forEach((button) => {
      button.addEventListener("click", () => {
        activeStep = Number(button.dataset.trainingStep);
        renderTrainingStep();
      });
    });
    fields.prev.addEventListener("click", () => {
      activeStep = Math.max(0, activeStep - 1);
      renderTrainingStep();
    });
    fields.next.addEventListener("click", () => {
      activeStep = Math.min(modes[activeMode].steps.length - 1, activeStep + 1);
      renderTrainingStep();
    });
    renderTrainingStep();

    const labSvg = $("[data-lab-svg]");
    if (!labSvg) return;

    const query = [0.84, 0.54];
    const initialContexts = {
      pos: [0.7, 0.2],
      neg1: [0.6, 0.35],
      neg2: [0.2, 0.9],
    };
    let contexts = structuredClone(initialContexts);
    let updates = 0;

    const dot = (a, b) => a[0] * b[0] + a[1] * b[1];
    const sigmoid = (x) => 1 / (1 + Math.exp(-x));
    const softplus = (x) => Math.log1p(Math.exp(x));
    const lossFor = (vectors) => {
      const posLoss = softplus(-dot(query, vectors.pos));
      const negLoss1 = softplus(dot(query, vectors.neg1));
      const negLoss2 = softplus(dot(query, vectors.neg2));
      return posLoss + negLoss1 + negLoss2;
    };

    const mapPoint = ([x, y]) => [210 + x * 145, 150 - y * 120];

    const renderLab = () => {
      const [qx, qy] = mapPoint(query);
      const points = [
        { key: "q", label: "Q", point: [qx, qy], color: "var(--coral)" },
        { key: "p", label: "P", point: mapPoint(contexts.pos), color: "var(--purple)" },
        { key: "n1", label: "N1", point: mapPoint(contexts.neg1), color: "var(--blue)" },
        { key: "n2", label: "N2", point: mapPoint(contexts.neg2), color: "var(--blue)" },
      ];
      const lines = points.slice(1).map(({ point, color }) =>
        `<line x1="${qx}" y1="${qy}" x2="${point[0]}" y2="${point[1]}" stroke="${color}" stroke-width="1.5" opacity=".28" />`,
      ).join("");
      const circles = points.map(({ point, color, label }) =>
        `<circle cx="${point[0]}" cy="${point[1]}" r="8" fill="${color}" />
         <text x="${point[0] + 12}" y="${point[1] + 4}" fill="var(--ink)" font-family="var(--mono)" font-size="10">${label}</text>`,
      ).join("");
      labSvg.innerHTML = `
        <title>二维负采样梯度演示</title>
        <desc>query 与一个正样本、两个负样本的点积关系；点之间连线只表示与 query 的连接。</desc>
        <rect x="0.5" y="0.5" width="419" height="299" rx="7" fill="none" stroke="var(--line)" />
        <line x1="65" y1="150" x2="395" y2="150" stroke="var(--line)" stroke-width="1" />
        <line x1="210" y1="15" x2="210" y2="280" stroke="var(--line)" stroke-width="1" />
        <text x="380" y="143" fill="var(--ink-faint)" font-family="var(--mono)" font-size="8">dim 1</text>
        <text x="218" y="23" fill="var(--ink-faint)" font-family="var(--mono)" font-size="8">dim 2</text>
        ${lines}
        ${circles}
      `;

      const scorePos = dot(query, contexts.pos);
      const scoreNeg1 = dot(query, contexts.neg1);
      const scoreNeg2 = dot(query, contexts.neg2);
      $("[data-score-pos]").textContent = scorePos.toFixed(2);
      $("[data-score-neg1]").textContent = scoreNeg1.toFixed(2);
      $("[data-score-neg2]").textContent = scoreNeg2.toFixed(2);
      $("[data-lab-loss]").textContent = lossFor(contexts).toFixed(3);
      $("[data-lab-updates]").textContent = String(updates);

      const explanation = $("[data-lab-explanation]");
      if (updates === 0) {
        explanation.textContent = "对正例 p，梯度为 (σ(q·p)−1)q，更新方向 −梯度朝向 q；对负例 n，梯度为 σ(q·n)q，更新方向背离 q。这里优化点积，并不保证欧氏距离减小。";
      } else if (lossFor(contexts) > 0.35) {
        explanation.textContent = "每次 SGD 都按当前 sigmoid 概率决定步长：正样本越“没把握”，把它拉近的梯度越大；负样本越容易被误判成正样本，把它推远的力度也越大。";
      } else {
        explanation.textContent = "正例点积变大，负例点积变小。数值到达 [-1,1] 边界后投影会限制继续更新；该示例是固定 query 的投影 SGD，不能把边界平台当成模型训练收敛。";
      }
    };

    $("[data-lab-step]").addEventListener("click", () => {
      const learningRate = 0.45;
      const gradPos = sigmoid(dot(query, contexts.pos)) - 1;
      const gradNeg1 = sigmoid(dot(query, contexts.neg1));
      const gradNeg2 = sigmoid(dot(query, contexts.neg2));
      ["pos", "neg1", "neg2"].forEach((key) => {
        const grad = key === "pos" ? gradPos : key === "neg1" ? gradNeg1 : gradNeg2;
        contexts[key] = contexts[key].map((value, index) =>
          Math.max(-1, Math.min(1, value - learningRate * grad * query[index])),
        );
      });
      updates += 1;
      renderLab();
    });

    $("[data-lab-reset]").addEventListener("click", () => {
      contexts = structuredClone(initialContexts);
      updates = 0;
      renderLab();
    });
    renderLab();
}
