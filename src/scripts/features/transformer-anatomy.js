import { $, $$ } from "../core.js";

export function initTransformerAnatomy() {
    const fields = $$("[data-arch-field]");
    const table = $("[data-weight-table]");
    if (!fields.length || !table) return;

    const presets = {
      dense13: { d: 2048, layers: 24, heads: 16, kvHeads: 8, headDim: 128, ffn: 5632, vocab: 32000, experts: 1, topK: 1, tied: false },
      dense7: { d: 4096, layers: 32, heads: 32, kvHeads: 8, headDim: 128, ffn: 14336, vocab: 32000, experts: 1, topK: 1, tied: false },
      moe: { d: 2048, layers: 24, heads: 16, kvHeads: 8, headDim: 128, ffn: 5632, vocab: 32000, experts: 8, topK: 2, tied: false },
      llama3: { d: 4096, layers: 32, heads: 32, kvHeads: 8, headDim: 128, ffn: 14336, vocab: 128256, experts: 1, topK: 1, tied: false },
      mixtral: { d: 4096, layers: 32, heads: 32, kvHeads: 8, headDim: 128, ffn: 14336, vocab: 32000, experts: 8, topK: 2, tied: false },
    };
    const fieldMap = Object.fromEntries(fields.map((field) => [field.dataset.archField, field]));
    let routeSeed = 7;

    const formatCount = (value) => {
      if (value >= 1e9) return `${(value / 1e9).toFixed(2)}B`;
      if (value >= 1e6) return `${(value / 1e6).toFixed(1)}M`;
      if (value >= 1e3) return `${(value / 1e3).toFixed(1)}K`;
      return String(Math.round(value));
    };
    const readConfig = () => ({
      d: Number(fieldMap.d.value),
      layers: Number(fieldMap.layers.value),
      heads: Number(fieldMap.heads.value),
      kvHeads: Number(fieldMap.kvHeads.value),
      headDim: Number(fieldMap.headDim.value),
      ffn: Number(fieldMap.ffn.value),
      vocab: Number(fieldMap.vocab.value),
      experts: Number(fieldMap.experts.value),
      topK: Number(fieldMap.topK.value),
      tied: fieldMap.tied.value === "true",
    });

    const renderMoe = (config) => {
      const tokenCount = 8;
      const rng = (() => {
        let value = routeSeed;
        return () => {
          value = (value * 1664525 + 1013904223) >>> 0;
          return value / 4294967296;
        };
      })();
      const tokenRoutes = Array.from({ length: tokenCount }, (_, tokenIndex) => {
        const scores = Array.from({ length: config.experts }, (_, expertIndex) => {
          const base = ((tokenIndex + 1) * (expertIndex + 3) + routeSeed * 7) % 17;
          return (base / 17) + rng() * 0.08;
        });
        const selected = scores
          .map((score, expertIndex) => ({ score, expertIndex }))
          .sort((a, b) => b.score - a.score)
          .slice(0, Math.min(config.topK, config.experts));
        return { tokenIndex, selected };
      });
      const loads = Array(config.experts).fill(0);
      tokenRoutes.forEach((route) => route.selected.forEach((item) => {
        loads[item.expertIndex] += 1;
      }));
      $("[data-moe-experts]").textContent = String(config.experts);
      $("[data-moe-topk]").textContent = String(Math.min(config.topK, config.experts));
      $("[data-moe-tokens]").innerHTML = tokenRoutes.map((route) => `
        <div class="moe-token">
          <strong>T${route.tokenIndex + 1}</strong>
          <span>${route.selected.map((item) => `E${item.expertIndex + 1} · ${item.score.toFixed(2)}`).join(" / ")}</span>
        </div>
      `).join("");
      $("[data-moe-network]").innerHTML = loads.map((load, expertIndex) => `
        <div class="moe-expert">
          <strong>Expert ${expertIndex + 1}</strong>
          <div><i style="width:${config.experts === 1 ? 100 : Math.min(100, (load / Math.max(1, tokenCount)) * 100)}%"></i></div>
          <span>${load} token assignments</span>
        </div>
      `).join("");
    };

    const render = () => {
      let config = readConfig();
      if (config.kvHeads > config.heads) {
        config.kvHeads = config.heads;
        fieldMap.kvHeads.value = String(config.kvHeads);
      }
      if (config.topK > config.experts) {
        config.topK = config.experts;
        fieldMap.topK.value = String(config.experts);
      }
      const queryWidth = config.heads * config.headDim;
      const kvWidth = config.kvHeads * config.headDim;
      const attentionLayer = config.d * queryWidth + 2 * config.d * kvWidth + queryWidth * config.d;
      const expertMlp = 3 * config.d * config.ffn;
      const router = config.experts > 1 ? config.d * config.experts : 0;
      const mlpLayer = config.experts * expertMlp + router;
      const activeMlpLayer = config.topK * expertMlp + router;
      const normPerLayer = 2 * config.d;
      const embedding = config.vocab * config.d;
      const head = config.tied ? 0 : config.vocab * config.d;
      const attention = config.layers * attentionLayer;
      const mlp = config.layers * mlpLayer;
      const norm = config.layers * normPerLayer + config.d;
      const total = embedding + attention + mlp + norm + head;
      const active = embedding
        + config.layers * (attentionLayer + activeMlpLayer + normPerLayer)
        + config.d
        + head;
      const ratio = active / total;
      const memory = (total * 2) / 1e9;

      $("[data-arch-total]").textContent = formatCount(total);
      $("[data-arch-active]").textContent = formatCount(active);
      $("[data-arch-ratio]").textContent = `${Math.round(ratio * 100)}%`;
      $("[data-arch-memory]").textContent = `${memory.toFixed(2)} GB`;
      $("[data-arch-embedding]").textContent = formatCount(embedding);
      $("[data-arch-attention]").textContent = formatCount(attention);
      $("[data-arch-mlp]").textContent = formatCount(mlp);
      $("[data-arch-norm]").textContent = formatCount(norm);
      $("[data-arch-head]").textContent = config.tied ? "0 · tied" : formatCount(head);
      const maxPart = Math.max(embedding, attention, mlp, norm, head, 1);
      [["embedding", embedding], ["attention", attention], ["mlp", mlp], ["norm", norm], ["head", head]].forEach(([key, value]) => {
        $(`[data-arch-bar="${key}"]`).style.width = `${Math.max(2, (value / maxPart) * 100)}%`;
      });
      $("[data-arch-formula]").textContent = config.experts > 1
        ? "Vd + L[Q+K+V+O + router + E·3d·d_ff + 2d] + d_final + head"
        : "Vd + L[Q+K+V+O + 3d·d_ff + 2d] + d_final + head";
      $("[data-arch-caption]").textContent = config.experts > 1
        ? `当前 MoE 有 ${config.experts} 个 experts、每 token 激活 top-${config.topK}。总参数包含全部专家；每 token 计算只走被选中的 ${config.topK} 个 MLP，但权重仍需全部驻留或被路由到其他设备。`
        : config.tied
          ? "当前是 tied dense：输出头复用 embedding 矩阵，总参数不重复计算 Vd；每个 token 经过全部 MLP 参数。"
          : "当前是 untied dense：embedding 与输出头各占一份矩阵。每个 token 都经过全部 MLP 参数。";

      const rows = [
        ["Embedding E", `[${config.vocab}, ${config.d}]`, embedding, "token ID 查表"],
        ["Q projection", `[${config.d}, ${queryWidth}]`, config.layers * config.d * queryWidth, "query heads"],
        ["K projection", `[${config.d}, ${kvWidth}]`, config.layers * config.d * kvWidth, `${config.kvHeads} 个 KV heads`],
        ["V projection", `[${config.d}, ${kvWidth}]`, config.layers * config.d * kvWidth, `${config.kvHeads} 个 KV heads`],
        ["O projection", `[${queryWidth}, ${config.d}]`, config.layers * queryWidth * config.d, "合并 heads"],
        ["SwiGLU gate", `[${config.d}, ${config.ffn}] × ${config.experts}`, config.layers * config.experts * config.d * config.ffn, "门控分支"],
        ["SwiGLU up", `[${config.d}, ${config.ffn}] × ${config.experts}`, config.layers * config.experts * config.d * config.ffn, "值分支"],
        ["SwiGLU down", `[${config.ffn}, ${config.d}] × ${config.experts}`, config.layers * config.experts * config.ffn * config.d, "回到 d 维"],
        ["Router", config.experts > 1 ? `[${config.d}, ${config.experts}]` : "—", config.layers * router, "expert 门控分数"],
        ["RMSNorm", `[${config.d}] × 2 / layer`, norm, "两处缩放 + final"],
        ["LM head", config.tied ? "tied to E" : `[${config.d}, ${config.vocab}]`, head, "hidden → vocab logits"],
      ];
      table.innerHTML = rows.map((row) => `
        <tr><td>${row[0]}</td><td><code>${row[1]}</code></td><td><strong>${formatCount(row[2])}</strong></td><td>${row[3]}</td></tr>
      `).join("");
      renderMoe(config);
    };

    const applyPreset = (preset) => {
      Object.entries(preset).forEach(([key, value]) => {
        fieldMap[key].value = String(value);
      });
      $$("[data-arch-preset]").forEach((button) => {
        button.setAttribute("aria-pressed", String(button.dataset.archPreset === preset));
      });
      render();
    };

    $$("[data-arch-preset]").forEach((button) => {
      button.addEventListener("click", () => {
        const key = button.dataset.archPreset;
        Object.entries(presets[key]).forEach(([field, value]) => {
          fieldMap[field].value = String(value);
        });
        $$("[data-arch-preset]").forEach((item) => {
          item.setAttribute("aria-pressed", String(item === button));
        });
        render();
      });
    });
    fields.forEach((field) => {
      field.addEventListener("change", () => {
        $$("[data-arch-preset]").forEach((button) => button.setAttribute("aria-pressed", "false"));
        render();
      });
    });
    $("[data-moe-route]").addEventListener("click", () => {
      routeSeed += 13;
      render();
    });
    fieldMap.experts.value = "1";
    render();
}
