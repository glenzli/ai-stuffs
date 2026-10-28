import { $, $$ } from "../core.js";

export function initRepresentationDeep() {
    const poolingButtons = $$("[data-pooling]");
    if (poolingButtons.length) {
      const tokenVectors = [
        { label: "负", value: [0.92, 0.28] },
        { label: "让", value: [0.18, 0.34] },
        { label: "错误", value: [0.84, 0.42] },
        { label: "候选", value: [0.66, 0.58] },
        { label: "远离", value: [0.38, 0.92] },
      ];
      const query = [0.92, 0.28];
      const clsState = [0.78, 0.46];
      const weights = [0.35, 0.05, 0.3, 0.15, 0.15];
      const poolingData = {
        mean: {
          kicker: "TOKEN AVERAGING",
          title: "把所有 token 状态平均",
          body: "每个位置等权。稳定、常见，也容易把虚词、标点和无关 token 一起平均进去；对短句与长句的方差也不同。",
          train: "训练目标应作用于最终使用的 pooled 向量。",
        },
        cls: {
          kicker: "LEARNED AGGREGATION SLOT",
          title: "读取专门的 CLS 位置",
          body: "读取专门 [CLS] token 的上下文状态。这里另设一个教学 CLS 状态；在双向 encoder 中它可汇总全句，在因果序列开头则看不到后文。质量仍取决于目标与训练。",
          train: "需要 CLS 位置在训练中承担匹配或分类任务；句向量质量需要在目标任务上验证。",
        },
        max: {
          kicker: "COORDINATE-WISE MAX",
          title: "每个维度保留最强响应",
          body: "在 T 个 token 的每个坐标上取最大值。能抓住显著特征，但会丢掉位置、顺序和“负向证据”，且输出分布不容易解释。",
          train: "常与特定任务头一起出现；用于检索时要看模型是否按这种池化训练过。",
        },
        last: {
          kicker: "LAST TOKEN STATE",
          title: "读取最后一个位置",
          body: "因果语言模型里常用 last token，因为它在因果注意力中看过整个前文；编码器中的读出位置应按训练目标和接口约定选择。",
          train: "可能需要专门的 EOS / last-token pooling 和对比目标。",
        },
        weighted: {
          kicker: "WEIGHTED AGGREGATION",
          title: "按学到的重要性加权",
          body: "权重可以来自注意力、另一个 scorer，或训练出的 pooler。表达力更强，也更容易过拟合或把权重集中在少数高频 token。",
          train: "权重本身也需要目标函数与数据训练，不能凭直觉手工固定后预期跨领域有效。",
        },
      };

      const cosine = (a, b) => {
        const dot = a.reduce((sum, value, index) => sum + value * b[index], 0);
        const normA = Math.sqrt(a.reduce((sum, value) => sum + value * value, 0));
        const normB = Math.sqrt(b.reduce((sum, value) => sum + value * value, 0));
        return dot / (normA * normB);
      };

      const getPooled = (mode) => {
        const values = tokenVectors.map((token) => token.value);
        if (mode === "cls") return clsState;
        if (mode === "last") return values[values.length - 1];
        if (mode === "max") return [0, 1].map((axis) => Math.max(...values.map((value) => value[axis])));
        if (mode === "weighted") {
          return [0, 1].map((axis) =>
            values.reduce((sum, value, index) => sum + value[axis] * weights[index], 0),
          );
        }
        return [0, 1].map((axis) =>
          values.reduce((sum, value) => sum + value[axis], 0) / values.length,
        );
      };

      const map = ([x, y]) => [55 + x * 230, 220 - y * 185];
      const renderPooling = (mode) => {
        poolingButtons.forEach((button) => {
          button.setAttribute("aria-pressed", String(button.dataset.pooling === mode));
        });
        const pooled = getPooled(mode);
        const data = poolingData[mode];
        $("[data-pooling-kicker]").textContent = data.kicker;
        $("[data-pooling-title]").textContent = data.title;
        $("[data-pooling-body]").textContent = data.body;
        $("[data-pooling-train]").textContent = data.train;
        $("[data-pooling-score]").textContent = cosine(pooled, query).toFixed(3);
        $("[data-pooling-norm]").textContent = Math.sqrt(pooled[0] ** 2 + pooled[1] ** 2).toFixed(3);

        const [qx, qy] = map(query);
        const [px, py] = map(pooled);
        const plottedTokens = mode === "cls" ? [...tokenVectors, { label: "[CLS]", value: clsState }] : tokenVectors;
        const tokenPoints = plottedTokens.map((token, index) => {
          const [x, y] = map(token.value);
          return `<circle cx="${x}" cy="${y}" r="5" fill="var(--blue)" opacity=".65" />
            <text x="${x + 8}" y="${y + 4}" fill="var(--ink-faint)" font-size="9">${token.label}</text>
            <text x="${x + 8}" y="${y + 15}" fill="var(--ink-faint)" font-family="var(--mono)" font-size="7">${index < tokenVectors.length ? `t${index + 1}` : "special"}</text>`;
        }).join("");
        $("[data-pooling-svg]").innerHTML = `
          <title>池化向量与 query 的二维方向示意</title>
          <desc>五个 token 状态和不同池化规则产生的句向量。</desc>
          <rect x="0.5" y="0.5" width="359" height="259" rx="6" fill="none" stroke="var(--line)" />
          <line x1="50" y1="225" x2="325" y2="225" stroke="var(--line)" />
          <line x1="50" y1="225" x2="50" y2="20" stroke="var(--line)" />
          <line x1="50" y1="225" x2="${qx}" y2="${qy}" stroke="var(--ink-faint)" stroke-width="1.5" stroke-dasharray="4 3" />
          <line x1="50" y1="225" x2="${px}" y2="${py}" stroke="var(--purple)" stroke-width="2.5" />
          ${tokenPoints}
          <circle cx="${px}" cy="${py}" r="8" fill="var(--purple)" />
          <circle cx="${qx}" cy="${qy}" r="6" fill="var(--ink)" opacity=".45" />
          <text x="${qx + 9}" y="${qy - 8}" fill="var(--ink-soft)" font-size="9">query / 参照方向</text>
          <text x="${px + 10}" y="${py + 17}" fill="var(--purple)" font-size="10">pooled v</text>
        `;
      };
      poolingButtons.forEach((button) => {
        button.addEventListener("click", () => renderPooling(button.dataset.pooling));
      });
      renderPooling("mean");
    }

    const anisoButtons = $$("[data-aniso-mode]");
    if (anisoButtons.length) {
      const raw = [
        [0.91, 0.82],
        [0.83, 0.94],
        [0.66, 0.78],
        [0.78, 0.62],
        [0.59, 0.88],
        [0.72, 0.71],
        [0.86, 0.67],
      ];
      const mean = [
        raw.reduce((sum, point) => sum + point[0], 0) / raw.length,
        raw.reduce((sum, point) => sum + point[1], 0) / raw.length,
      ];
      const centered = raw.map(([x, y]) => [x - mean[0], y - mean[1]]);
      const cosine = (a, b) => {
        const dot = a[0] * b[0] + a[1] * b[1];
        return dot / (Math.hypot(...a) * Math.hypot(...b));
      };
      const state = (mode) => mode === "center" ? centered : raw;
      const renderAniso = (mode) => {
        anisoButtons.forEach((button) => {
          button.setAttribute("aria-pressed", String(button.dataset.anisoMode === mode));
        });
        const points = state(mode);
        const pointsMean = [
          points.reduce((sum, point) => sum + point[0], 0) / points.length,
          points.reduce((sum, point) => sum + point[1], 0) / points.length,
        ];
        const averageNorm = points.reduce((sum, point) => sum + Math.hypot(...point), 0) / points.length;
        const share = Math.hypot(...pointsMean) / averageNorm;
        $("[data-aniso-cos]").textContent = cosine(points[0], points[1]).toFixed(3);
        $("[data-aniso-share]").textContent = `${Math.round(share * 100)}%`;
        $("[data-aniso-copy]").textContent = mode === "raw"
          ? "原始向量共享一个很强的公共方向，即使内容不同，cosine 也可能很高。判断细粒度语义时，还需结合训练目标和任务评估。"
          : "去掉公共均值后，A 与 B 不再主要靠“大家都在同一方向”获得高分。whitening 会进一步重标各方向方差，但也可能放大噪声。";

        const range = mode === "raw" ? 1.05 : 0.3;
        const map = ([x, y]) => [210 + (x / range) * 160, 150 - (y / range) * 115];
        const circles = points.map((point, index) => {
          const [x, y] = map(point);
          const color = index === 0 || index === 1 ? (index === 0 ? "var(--purple)" : "var(--coral)") : "var(--blue)";
          return `<circle cx="${x}" cy="${y}" r="${index < 2 ? 7 : 4}" fill="${color}" opacity="${index < 2 ? 1 : 0.5}" />
            ${index < 2 ? `<text x="${x + 10}" y="${y + 4}" fill="${color}" font-size="10">${index === 0 ? "A" : "B"}</text>` : ""}`;
        }).join("");
        const [mx, my] = map(pointsMean);
        $("[data-aniso-svg]").innerHTML = `
          <title>各向异性向量空间</title>
          <desc>公共方向导致余弦相近，去均值后样本沿不同方向展开。</desc>
          <rect x="0.5" y="0.5" width="419" height="299" rx="7" fill="none" stroke="var(--line)" />
          <line x1="45" y1="150" x2="385" y2="150" stroke="var(--line)" />
          <line x1="210" y1="20" x2="210" y2="280" stroke="var(--line)" />
          <line x1="210" y1="150" x2="${mx}" y2="${my}" stroke="var(--yellow)" stroke-width="2" stroke-dasharray="5 3" />
          ${circles}
          <text x="${mx + 8}" y="${my - 8}" fill="var(--yellow)" font-size="9">mean direction</text>
        `;
      };
      anisoButtons.forEach((button) => {
        button.addEventListener("click", () => renderAniso(button.dataset.anisoMode));
      });
      renderAniso("raw");
    }

    const dimButtons = $$("[data-dim]");
    if (dimButtons.length) {
      const query = [0.8, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.01];
      const candidates = [
        { name: "A", label: "可回答块", vector: [0.82, 0.48, 0.18, 0.1, 0.02, 0.01, 0, 0] },
        { name: "B", label: "短维假近邻", vector: [0.95, 0.62, 0.02, 0.02, 0.02, 0.01, 0, 0] },
        { name: "C", label: "上下文相关块", vector: [0.2, 0.1, 0.85, 0.4, 0.1, 0.02, 0.01, 0.01] },
        { name: "D", label: "关键词难负例", vector: [0.42, 0.3, 0.5, 0.6, 0.1, 0.02, 0.01, 0.01] },
        { name: "E", label: "无关块", vector: [0.05, 0.02, 0.01, 0.01, 0.02, 0.01, 0, 0] },
      ];
      const prefixCosine = (a, b, dim) => {
        const left = a.slice(0, dim);
        const right = b.slice(0, dim);
        const dot = left.reduce((sum, value, index) => sum + value * right[index], 0);
        return dot / (Math.hypot(...left) * Math.hypot(...right));
      };
      const renderMrl = (dim) => {
        dimButtons.forEach((button) => {
          button.setAttribute("aria-pressed", String(Number(button.dataset.dim) === dim));
        });
        $("[data-dim-value]").textContent = String(dim);
        const ranked = candidates.map((candidate) => ({
          ...candidate,
          score: prefixCosine(query, candidate.vector, dim),
        })).sort((a, b) => b.score - a.score);
        $("[data-mrl-list]").innerHTML = ranked.map((candidate, index) => `
          <div class="mrl-row">
            <div><span class="mrl-rank">${String(index + 1).padStart(2, "0")}</span><strong>${candidate.name}</strong><small>${candidate.label}</small></div>
            <div class="mrl-bar"><i style="width:${candidate.score * 100}%"></i><code>${candidate.score.toFixed(4)}</code></div>
            <span class="${index === 0 ? "tag tag-teal" : "tag"}">${index === 0 ? "Top 1" : "候选"}</span>
          </div>
        `).join("");
      };
      dimButtons.forEach((button) => {
        button.addEventListener("click", () => renderMrl(Number(button.dataset.dim)));
      });
      renderMrl(4);
    }
}
