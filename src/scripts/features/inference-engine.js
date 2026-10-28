import { $, $$ } from "../core.js";

export function inferenceState(promptLength, batchSize, generated) {
  return {
    cacheTokens: generated > 0 ? promptLength + generated - 1 : 0,
    computedTokens: generated === 0 ? 0 : generated === 1 ? promptLength * batchSize : batchSize,
  };
}

export function initInferenceEngine() {
    const promptSelect = $("[data-infer-prompt]");
    if (!promptSelect) return;
    const batchSelect = $("[data-infer-batch]");
    let prefilled = false;
    let generated = 0;

    const renderSequence = (promptLength, batchSize) => {
      const shownGenerated = Math.min(generated, 16);
      $("[data-infer-sequence]").innerHTML = `
        <div class="seq-prompt" style="flex:${Math.max(1, Math.sqrt(promptLength))}">
          <strong>prompt</strong><span>${promptLength.toLocaleString()} tokens</span>
        </div>
        ${Array.from({ length: shownGenerated }, (_, index) => `
          <i class="seq-token" title="generated token ${index + 1}"></i>
        `).join("")}
        ${generated > shownGenerated ? `<b>+${generated - shownGenerated}</b>` : ""}
      `;
      const { cacheTokens, computedTokens } = inferenceState(promptLength, batchSize, generated);
      $("[data-infer-phase]").textContent = !prefilled
        ? "等待 prefill"
        : generated > 1
          ? `DECODE · 已生成 ${generated}`
          : "PREFILL 完成 · 已采样首 token";
      $("[data-infer-generated]").textContent = String(generated);
      $("[data-infer-compute]").textContent = computedTokens.toLocaleString();
      $("[data-infer-cache-tokens]").textContent = cacheTokens.toLocaleString();
      $("[data-infer-batch-cache]").textContent = (cacheTokens * batchSize).toLocaleString();
      $("[data-infer-note]").textContent = !prefilled
        ? "点击“执行 prefill”：prompt 的所有 token 一次并行进入模型，并写入所有层的 KV。"
        : generated === 1
          ? "prefill 的最后一个位置给出首 token 的分布。已采样 1 个输出，但它尚未再次输入模型，因此 KV 仍只有 prompt 的位置。"
          : `已完成 ${generated - 1} 次 decode forward，缓存 ${cacheTokens.toLocaleString()} 个位置。刚采样的第 ${generated} 个输出尚无 K/V；下一轮输入它，才会写入缓存并预测第 ${generated + 1} 个。`;
      $("[data-infer-one]").disabled = !prefilled || generated >= 40;
      $("[data-infer-ten]").disabled = !prefilled || generated >= 40;
    };

    const render = () => {
      renderSequence(Number(promptSelect.value), Number(batchSelect.value));
    };
    $("[data-infer-prefill]").addEventListener("click", () => {
      prefilled = true;
      generated = 1;
      render();
    });
    $("[data-infer-one]").addEventListener("click", () => {
      if (!prefilled) return;
      generated = Math.min(40, generated + 1);
      render();
    });
    $("[data-infer-ten]").addEventListener("click", () => {
      if (!prefilled) return;
      generated = Math.min(40, generated + 10);
      render();
    });
    $("[data-infer-reset]").addEventListener("click", () => {
      prefilled = false;
      generated = 0;
      render();
    });
    promptSelect.addEventListener("change", () => {
      prefilled = false;
      generated = 0;
      render();
    });
    batchSelect.addEventListener("change", render);
    render();

    const layers = $("[data-kv-layers]");
    const kvHeads = $("[data-kv-heads]");
    const headDim = $("[data-kv-dim]");
    const batch = $("[data-kv-batch]");
    const seq = $("[data-kv-seq]");
    const bytes = $("[data-kv-bytes]");
    if (!layers) return;

    const formatBytes = (value) => {
      const clean = (number) => number.toFixed(2).replace(/\.?0+$/, "");
      if (value >= 1024 ** 3) return `${clean(value / 1024 ** 3)} GiB`;
      if (value >= 1024 ** 2) return `${clean(value / 1024 ** 2)} MiB`;
      if (value >= 1024) return `${clean(value / 1024)} KiB`;
      return `${Math.round(value)} B`;
    };

    const renderKv = () => {
      const layerCount = Number(layers.value);
      const heads = Number(kvHeads.value);
      const dimension = Number(headDim.value);
      const batchSize = Number(batch.value);
      const sequence = Number(seq.value);
      const cacheBytes = Number(bytes.value);
      const perToken = 2 * layerCount * heads * dimension * cacheBytes;
      const perRequest = perToken * sequence;
      const total = perRequest * batchSize;
      $("[data-kv-per-token]").textContent = formatBytes(perToken);
      $("[data-kv-request]").textContent = formatBytes(perRequest);
      $("[data-kv-total]").textContent = formatBytes(total);
      $("[data-kv-at-max]").textContent = `${formatBytes(total)} @ ${sequence >= 1024 ? `${Math.round(sequence / 1024)}K` : sequence}`;
      const kind = heads === 1 ? "MQA" : heads === 32 ? "MHA" : "GQA";
      $("[data-kv-note]").textContent = `本例固定 32 个 query heads，${heads} 个 KV heads：${kind}。每组 KV 被 ${32 / heads} 个 query heads 使用。计算按完整历史缓存；不含分页碎片、量化 scale、共享前缀与临时 workspace。`;
      $$("[data-kv-preset]").forEach((button) => {
        button.setAttribute("aria-pressed", String(presets[button.dataset.kvPreset] === String(heads)));
      });

      const points = 12;
      const values = Array.from({ length: points + 1 }, (_, index) =>
        perToken * (sequence * index / points) * batchSize,
      );
      const maxValue = Math.max(...values, 1);
      const width = 500;
      const height = 180;
      const pad = 25;
      const path = values.map((value, index) => {
        const x = pad + (index / points) * (width - pad * 2);
        const y = height - pad - (value / maxValue) * (height - pad * 2);
        return `${index === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`;
      }).join(" ");
      $("[data-kv-chart]").innerHTML = `
        <title>KV Cache 随上下文长度线性增长</title>
        <rect x="0.5" y="0.5" width="499" height="179" rx="6" fill="none" stroke="var(--line)" />
        <line x1="${pad}" y1="${height - pad}" x2="${width - pad}" y2="${height - pad}" stroke="var(--line)" />
        <line x1="${pad}" y1="${pad}" x2="${pad}" y2="${height - pad}" stroke="var(--line)" />
        <path d="${path}" fill="none" stroke="var(--accent)" stroke-width="2" />
        <text x="${pad}" y="15" fill="var(--ink-faint)" font-size="8">${formatBytes(maxValue)}</text>
        <text x="${width - pad - 25}" y="${height - 7}" fill="var(--ink-faint)" font-size="8">${sequence >= 1024 ? `${Math.round(sequence / 1024)}K` : sequence} tokens</text>
      `;
    };
    const presets = { mha: "32", gqa: "8", mqa: "1" };
    $$("[data-kv-preset]").forEach((button) => {
      button.addEventListener("click", () => {
        kvHeads.value = presets[button.dataset.kvPreset];
        $$("[data-kv-preset]").forEach((item) => {
          item.setAttribute("aria-pressed", String(item === button));
        });
        renderKv();
      });
    });
    [layers, kvHeads, headDim, batch, seq, bytes].forEach((input) => {
      input.addEventListener("change", renderKv);
    });
    renderKv();
}
