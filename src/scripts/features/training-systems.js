import { $, $$ } from "../core.js";

export function initTrainingSystems() {
    const buttons = $$("[data-par-mode]");
    if (buttons.length) {
      const modes = {
        dp: {
          kicker: "DATA PARALLEL",
          title: "复制完整模型，让不同 GPU 算不同 batch",
          copy: "每卡有完整权重、梯度和优化器状态，适合单卡放得下训练状态的情况。梯度用 all-reduce 同步；DDP 可在梯度桶就绪后发起通信，与反向计算重叠。",
          shard: "batch",
          comm: "all-reduce gradients",
          limit: "每卡完整训练状态放不下",
          note: "DP 的每个 GPU 都持有完整参数与 optimizer state；不同 GPU 分担 batch 计算，每卡仍需保存完整训练状态。",
          cards: [
            ["GPU0", "完整权重 / grads / opt", "batch A"],
            ["GPU1", "完整权重 / grads / opt", "batch B"],
            ["GPU2", "完整权重 / grads / opt", "batch C"],
            ["GPU3", "完整权重 / grads / opt", "batch D"],
          ],
        },
        zero: {
          kicker: "ZERO-3 / FULL-SHARD FSDP",
          title: "把参数、梯度和优化器状态分片保存",
          copy: "这里展示 ZeRO-3 / FSDP 全分片：计算前 gather 所需参数，按策略 reshard；梯度可 reduce-scatter。静态状态减少，额外通信取决于 gather 粒度与调度。",
          shard: "parameters / gradients / optimizer states",
          comm: "all-gather + reduce-scatter",
          limit: "通信与 gather 调度复杂",
          note: "ZeRO-1 分片优化器状态，ZeRO-2 再分片梯度，ZeRO-3 才进一步分片参数。图中的按需参数聚合对应第三阶段或 FSDP 全分片，不能套用到所有 ZeRO 策略。",
          cards: [
            ["GPU0", "1/4 states + layer shard", "gather → compute"],
            ["GPU1", "1/4 states + layer shard", "gather → compute"],
            ["GPU2", "1/4 states + layer shard", "gather → compute"],
            ["GPU3", "1/4 states + layer shard", "gather → compute"],
          ],
        },
        tp: {
          kicker: "TENSOR PARALLEL",
          title: "把单层矩阵乘切到多张卡",
          copy: "Q/K/V、MLP 与输出投影按 hidden/head/FFN 维度切分。每层内部需要通信，适合高速互联；通常与 DP 组合。",
          shard: "layer matrices / heads / FFN width",
          comm: "all-reduce / all-gather per layer",
          limit: "高频层内通信",
          note: "通信位置由分片布局决定。复制输入的列并行 → 逐元素激活 → 行并行 MLP 可让中间激活保持本地，只在前向合并输出、反向合并输入梯度；序列并行等策略会改变通信方式与次数。吞吐仍需结合互联和重叠调度实测。",
          cards: [
            ["GPU0", "heads / FFN shard 0", "layer-local compute"],
            ["GPU1", "heads / FFN shard 1", "layer-local compute"],
            ["GPU2", "heads / FFN shard 2", "layer-local compute"],
            ["GPU3", "heads / FFN shard 3", "layer-local compute"],
          ],
        },
        pp: {
          kicker: "PIPELINE PARALLEL",
          title: "不同 GPU 负责不同层，microbatch 流水",
          copy: "stage 之间传 activation 与梯度。需要处理 pipeline bubble、microbatch 调度和激活缓存；适合层数多的超大模型。",
          shard: "layers / stages",
          comm: "stage-to-stage activations",
          limit: "bubble 与激活缓存",
          note: "PP 将各层权重与计算分配给不同设备。增加每次更新的 microbatch 数量可降低空泡比例；增大单个 microbatch 的样本数是另一件事。激活峰值还取决于具体调度与重算策略。",
          cards: [
            ["GPU0", "layers 0–7", "microbatch 1–4"],
            ["GPU1", "layers 8–15", "microbatch 1–4"],
            ["GPU2", "layers 16–23", "microbatch 1–4"],
            ["GPU3", "layers 24–31", "microbatch 1–4"],
          ],
        },
        ep: {
          kicker: "EXPERT PARALLEL",
          title: "不同 expert 放在不同设备",
          copy: "router 将 token 发给指定 experts，可能需要 all-to-all。总权重远大于每 token 激活，但必须处理专家负载均衡与跨设备通信。",
          shard: "experts",
          comm: "all-to-all token dispatch",
          limit: "负载不均与热点",
          note: "EP 将 token 发送到拥有目标 expert 的设备，再收回输出。负载不均会增加等待并降低吞吐。",
          cards: [
            ["GPU0", "experts 0–1", "hot expert risk"],
            ["GPU1", "experts 2–3", "token all-to-all"],
            ["GPU2", "experts 4–5", "capacity / drop"],
            ["GPU3", "experts 6–7", "router sync"],
          ],
        },
        cp: {
          kicker: "CONTEXT PARALLEL",
          title: "把长序列拆给多卡计算",
          copy: "每卡存一段 token 的 activation，但 attention 需要跨序列段交换 K/V。适合超长上下文，通信模式与 TP/PP 不同。",
          shard: "sequence tokens",
          comm: "K/V exchange across context blocks",
          limit: "长序列通信与 attention kernel 支持",
          note: "CP 不改变参数总量，主要解决超长序列的激活与单序列计算；会引入 K/V 交换与不规则 attention 访问。",
          cards: [
            ["GPU0", "tokens 0–K", "local attention"],
            ["GPU1", "tokens K+1–2K", "exchange KV"],
            ["GPU2", "tokens 2K+1–3K", "exchange KV"],
            ["GPU3", "tokens 3K+1–4K", "merge outputs"],
          ],
        },
      };
      const render = (key) => {
        const data = modes[key];
        buttons.forEach((button) => {
          button.setAttribute("aria-pressed", String(button.dataset.parMode === key));
        });
        $("[data-par-kicker]").textContent = data.kicker;
        $("[data-par-title]").textContent = data.title;
        $("[data-par-copy]").textContent = data.copy;
        $("[data-par-shard]").textContent = data.shard;
        $("[data-par-comm]").textContent = data.comm;
        $("[data-par-limit]").textContent = data.limit;
        $("[data-par-note]").textContent = data.note;
        $("[data-par-grid]").innerHTML = data.cards.map(([name, role, detail], index) => `
          <div class="par-gpu-card">
            <span>${name}</span>
            <strong>${role}</strong>
            <p>${detail}</p>
            <i class="${index % 2 ? "is-local" : "is-replicated"}"></i>
          </div>
        `).join("");
      };
      buttons.forEach((button) => {
        button.addEventListener("click", () => render(button.dataset.parMode));
      });
      render("dp");
    }

    const activationB = $("[data-act-b]");
    if (!activationB) return;
    const activationT = $("[data-act-t]");
    const activationH = $("[data-act-h]");
    const activationD = $("[data-act-d]");
    const activationF = $("[data-act-f]");
    const activationBytes = $("[data-act-bytes]");
    const formatBytes = (value) => {
      if (value >= 1024 ** 3) return `${(value / 1024 ** 3).toFixed(2)} GiB`;
      if (value >= 1024 ** 2) return `${(value / 1024 ** 2).toFixed(1)} MiB`;
      if (value >= 1024) return `${(value / 1024).toFixed(1)} KiB`;
      return `${Math.round(value)} B`;
    };
    const renderActivation = () => {
      const b = Number(activationB.value);
      const t = Number(activationT.value);
      const h = Number(activationH.value);
      const d = Number(activationD.value);
      const f = Number(activationF.value);
      const bytes = Number(activationBytes.value);
      $("[data-act-b-value]").textContent = String(b);
      $("[data-act-t-value]").textContent = t.toLocaleString();
      $("[data-act-h-value]").textContent = String(h);
      $("[data-act-d-value]").textContent = String(d);
      $("[data-act-f-value]").textContent = f.toLocaleString();
      const attention = b * h * t * t * bytes;
      const qkv = 3 * b * t * h * d * bytes;
      const mlp = 2 * b * t * f * bytes;
      const hidden = b * t * h * d * bytes;
      $("[data-act-attention]").textContent = formatBytes(attention);
      $("[data-act-qkv]").textContent = formatBytes(qkv);
      $("[data-act-mlp]").textContent = formatBytes(mlp);
      $("[data-act-hidden]").textContent = formatBytes(hidden);
    };
    [activationB, activationT, activationH, activationD, activationF, activationBytes].forEach((input) => {
      input.addEventListener("input", renderActivation);
      input.addEventListener("change", renderActivation);
    });
    renderActivation();
}
