import { $, $$ } from "../core.js";

export function negativePool(docs, classIndex, positiveIndex) {
  return docs.map((doc, index) => ({ doc, index }))
    .filter(({ doc, index }) => index !== positiveIndex && doc.classIndex !== classIndex);
}

export function initSimulator() {
    const map = $("[data-sim-map]");
    if (!map) return;

    const controls = {
      lr: $("[data-sim-lr]"),
      temp: $("[data-sim-temp]"),
      hard: $("[data-sim-hard]"),
      noise: $("[data-sim-noise]"),
      batch: $("[data-sim-batch]"),
    };
    const buttons = {
      run: $("[data-sim-run]"),
      step: $("[data-sim-step]"),
      reset: $("[data-sim-reset]"),
    };
    const classInfo = [
      { name: "检索", color: "var(--purple)" },
      { name: "生成", color: "var(--coral)" },
      { name: "偏好", color: "var(--blue)" },
    ];
    const centers = [
      [-0.7, 0.1],
      [0.05, 0.85],
      [0.75, -0.18],
    ];
    const maxSteps = 200;
    let running = false;
    let timer = null;
    let state = null;

    const makeRng = (seed) => {
      let value = seed >>> 0;
      return () => {
        value += 0x6d2b79f5;
        let result = value;
        result = Math.imul(result ^ (result >>> 15), result | 1);
        result ^= result + Math.imul(result ^ (result >>> 7), result | 61);
        return ((result ^ (result >>> 14)) >>> 0) / 4294967296;
      };
    };

    const gaussian = (rng) => {
      const u = Math.max(rng(), 1e-9);
      const v = Math.max(rng(), 1e-9);
      return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
    };

    const dot = (a, b) => a[0] * b[0] + a[1] * b[1];
    const cosine = (a, b) => dot(a, b) / (Math.hypot(...a) * Math.hypot(...b) || 1);
    const softmax = (values) => {
      const max = Math.max(...values);
      const exps = values.map((value) => Math.exp(value - max));
      const sum = exps.reduce((total, value) => total + value, 0);
      return exps.map((value) => value / sum);
    };
    const add = (a, b) => [a[0] + b[0], a[1] + b[1]];
    const scale = (a, k) => [a[0] * k, a[1] * k];
    const clipVector = (vector, maxNorm = 2.5) => {
      const norm = Math.hypot(...vector);
      return norm > maxNorm ? scale(vector, maxNorm / norm) : vector;
    };

    const createInitialState = () => {
      const dataRng = makeRng(20261026);
      const docs = [];
      const queries = [];
      for (let classIndex = 0; classIndex < classInfo.length; classIndex += 1) {
        for (let index = 0; index < 5; index += 1) {
          docs.push({
            id: `D${docs.length + 1}`,
            classIndex,
            vector: [
              centers[classIndex][0] + gaussian(dataRng) * 0.42,
              centers[classIndex][1] + gaussian(dataRng) * 0.42,
            ],
          });
        }
        for (let index = 0; index < 3; index += 1) {
          queries.push({
            id: `Q${queries.length + 1}`,
            classIndex,
            vector: [
              centers[classIndex][0] + gaussian(dataRng) * 0.42,
              centers[classIndex][1] + gaussian(dataRng) * 0.42,
            ],
          });
        }
      }
      return {
        docs,
        queries,
        rng: makeRng(77123),
        steps: 0,
        lossHistory: [],
        currentLoss: null,
        labelErrors: 0,
        lastBatch: [],
      };
    };

    const evaluate = () => {
      let hits = 0;
      let sameSum = 0;
      let sameCount = 0;
      let diffSum = 0;
      let diffCount = 0;
      state.queries.forEach((query) => {
        const ranked = [...state.docs].sort((a, b) => dot(query.vector, b.vector) - dot(query.vector, a.vector));
        if (ranked[0].classIndex === query.classIndex) hits += 1;
        state.docs.forEach((doc) => {
          const score = cosine(query.vector, doc.vector);
          if (doc.classIndex === query.classIndex) {
            sameSum += score;
            sameCount += 1;
          } else {
            diffSum += score;
            diffCount += 1;
          }
        });
      });
      return {
        recall: hits / state.queries.length,
        same: sameSum / sameCount,
        diff: diffSum / diffCount,
        gap: sameSum / sameCount - diffSum / diffCount,
      };
    };

    const stopRunning = () => {
      running = false;
      if (timer) clearInterval(timer);
      timer = null;
      buttons.run.textContent = "自动运行";
    };

    const renderMap = () => {
      const width = 620;
      const height = 390;
      const pad = 38;
      const min = -2.6;
      const max = 2.6;
      const mapPoint = ([x, y]) => [
        pad + ((x - min) / (max - min)) * (width - pad * 2),
        height - pad - ((y - min) / (max - min)) * (height - pad * 2),
      ];
      const lines = state.lastBatch.map((item) => {
        const query = state.queries[item.queryIndex];
        const [qx, qy] = mapPoint(query.vector);
        const [px, py] = mapPoint(state.docs[item.positiveIndex].vector);
        const negatives = item.negativeIndices.map((docIndex) => {
          const [nx, ny] = mapPoint(state.docs[docIndex].vector);
          return `<line x1="${qx}" y1="${qy}" x2="${nx}" y2="${ny}" stroke="var(--line-strong)" stroke-width="1" opacity=".45" />`;
        }).join("");
        return `${negatives}<line x1="${qx}" y1="${qy}" x2="${px}" y2="${py}" stroke="var(--purple)" stroke-width="1.5" opacity=".65" />`;
      }).join("");
      const docs = state.docs.map((doc) => {
        const [x, y] = mapPoint(doc.vector);
        return `<rect x="${x - 4}" y="${y - 4}" width="8" height="8" fill="${classInfo[doc.classIndex].color}" opacity=".78">
          <title>${doc.id} · ${classInfo[doc.classIndex].name}</title>
        </rect>`;
      }).join("");
      const queries = state.queries.map((query) => {
        const [x, y] = mapPoint(query.vector);
        return `<polygon points="${x},${y - 6} ${x + 6},${y + 5} ${x - 6},${y + 5}" fill="${classInfo[query.classIndex].color}">
          <title>${query.id} · ${classInfo[query.classIndex].name}</title>
        </polygon>`;
      }).join("");
      map.innerHTML = `
        <title>训练中的 query 与 document embedding</title>
        <desc>三角形是 query，方形是 document，颜色代表真实主题。连线显示最近一轮训练使用的正负样本。</desc>
        <rect x="0.5" y="0.5" width="619" height="389" rx="7" fill="none" stroke="var(--line)" />
        <line x1="${pad}" y1="${height - pad}" x2="${width - pad}" y2="${height - pad}" stroke="var(--line)" />
        <line x1="${pad}" y1="${height - pad}" x2="${pad}" y2="${pad}" stroke="var(--line)" />
        <line x1="${width / 2}" y1="${pad}" x2="${width / 2}" y2="${height - pad}" stroke="var(--line)" opacity=".5" />
        <line x1="${pad}" y1="${height / 2}" x2="${width - pad}" y2="${height / 2}" stroke="var(--line)" opacity=".5" />
        <text x="${width - pad - 25}" y="${height - pad - 6}" fill="var(--ink-faint)" font-size="8">dim 1</text>
        <text x="${pad + 8}" y="${pad + 10}" fill="var(--ink-faint)" font-size="8">dim 2</text>
        ${lines}
        ${docs}
        ${queries}
      `;
    };

    const renderChart = () => {
      const svg = $("[data-sim-chart]");
      if (!state.lossHistory.length) {
        svg.innerHTML = `
          <rect x="0.5" y="0.5" width="419" height="189" rx="6" fill="none" stroke="var(--line)" />
          <text x="210" y="102" fill="var(--ink-faint)" font-size="10" text-anchor="middle">运行一步后开始记录</text>
        `;
        return;
      }
      const width = 420;
      const height = 190;
      const pad = 24;
      const values = state.lossHistory;
      const min = Math.min(...values);
      const max = Math.max(...values);
      const range = Math.max(0.1, max - min);
      const point = (value, index) => [
        pad + (index / Math.max(1, values.length - 1)) * (width - pad * 2),
        height - pad - ((value - min) / range) * (height - pad * 2),
      ];
      const path = values.map((value, index) => {
        const [x, y] = point(value, index);
        return `${index === 0 ? "M" : "L"}${x.toFixed(2)},${y.toFixed(2)}`;
      }).join(" ");
      const last = values[values.length - 1];
      const [lastX, lastY] = point(last, values.length - 1);
      svg.innerHTML = `
        <title>InfoNCE loss 曲线</title>
        <rect x="0.5" y="0.5" width="419" height="189" rx="6" fill="none" stroke="var(--line)" />
        <line x1="${pad}" y1="${height - pad}" x2="${width - pad}" y2="${height - pad}" stroke="var(--line)" />
        <line x1="${pad}" y1="${pad}" x2="${pad}" y2="${height - pad}" stroke="var(--line)" />
        <path d="${path}" fill="none" stroke="var(--accent)" stroke-width="2" />
        <circle cx="${lastX}" cy="${lastY}" r="3.5" fill="var(--accent)" />
        <text x="${pad}" y="15" fill="var(--ink-faint)" font-size="8">${max.toFixed(2)}</text>
        <text x="${pad}" y="${height - 6}" fill="var(--ink-faint)" font-size="8">${min.toFixed(2)}</text>
      `;
    };

    const renderBatch = () => {
      const list = $("[data-sim-batch-list]");
      if (!state.lastBatch.length) {
        list.innerHTML = `<p class="sim-empty">还没有 batch。点击“单步”或“自动运行”。</p>`;
        $("[data-sim-batch-note]").textContent = "还没有训练样本";
        return;
      }
      $("[data-sim-batch-note]").textContent = `${state.lastBatch.length} 个 query · 每个 1 正 + 3 负 · 概率取更新前`;
      list.innerHTML = state.lastBatch.map((item) => {
        const query = state.queries[item.queryIndex];
        const candidates = [
          { index: item.positiveIndex, role: item.labelError ? "错误正例" : "正样本", probability: item.probabilities[0] },
          ...item.negativeIndices.map((index, negativeIndex) => ({
            index,
            role: "负样本",
            probability: item.probabilities[negativeIndex + 1],
          })),
        ];
        return `
          <div class="sim-query-row">
            <div><strong>${query.id}</strong><span>${classInfo[query.classIndex].name}</span>${item.labelError ? '<em>label noise</em>' : ""}</div>
            <div class="sim-candidates">
              ${candidates.map((candidate, index) => `
                <span class="${index === 0 ? "is-positive" : ""}">
                  ${candidate.role} · ${classInfo[state.docs[candidate.index].classIndex].name}
                  <code>${candidate.probability.toFixed(3)}</code>
                </span>
              `).join("")}
            </div>
          </div>
        `;
      }).join("");
    };

    const render = () => {
      const metrics = evaluate();
      const lr = Number(controls.lr.value) / 1000;
      const temp = Number(controls.temp.value) / 100;
      const hard = Number(controls.hard.value);
      const noise = Number(controls.noise.value);
      $("[data-sim-lr-value]").textContent = lr.toFixed(3);
      $("[data-sim-temp-value]").textContent = temp.toFixed(2);
      $("[data-sim-hard-value]").textContent = `${hard}%`;
      $("[data-sim-noise-value]").textContent = `${noise}%`;
      $("[data-sim-steps]").textContent = String(state.steps);
      $("[data-sim-loss]").textContent = state.currentLoss === null ? "—" : state.currentLoss.toFixed(3);
      $("[data-sim-last-loss]").textContent = state.currentLoss === null ? "尚未训练" : `step ${state.steps} 更新前 · ${state.currentLoss.toFixed(3)}`;
      $("[data-sim-recall]").textContent = `${Math.round(metrics.recall * 100)}%`;
      $("[data-sim-same]").textContent = metrics.same.toFixed(3);
      $("[data-sim-gap]").textContent = metrics.gap.toFixed(3);
      $("[data-sim-status]").textContent = running
        ? `训练中 · ${state.steps} steps`
        : state.steps ? `已暂停 · ${state.steps} steps` : "准备开始";
      $("[data-sim-error-note]").textContent = state.labelErrors
        ? `本轮已累计注入 ${state.labelErrors} 个错误正样本。它们也参与了真实反传，评估却不使用这些错误标签。`
        : noise > 0
          ? `当前设为 ${noise}% 错误正样本，但尚未实际抽到错误标签；继续运行会按概率触发。`
          : "当前没有错误正样本。训练目标与评估目标一致，指标仍可能受采样、学习率和温度影响。";
      renderMap();
      renderChart();
      renderBatch();
    };

    const trainStep = () => {
      if (state.steps >= maxSteps) {
        stopRunning();
        return;
      }
      const lr = Number(controls.lr.value) / 1000;
      const temp = Number(controls.temp.value) / 100;
      const hardRatio = Number(controls.hard.value) / 100;
      const noiseRate = Number(controls.noise.value) / 100;
      const batchSize = Math.min(Number(controls.batch.value), state.queries.length);
      const queryGrads = new Map();
      const docGrads = new Map();
      let stepLoss = 0;
      const batch = [];

      for (let batchIndex = 0; batchIndex < batchSize; batchIndex += 1) {
        const queryIndex = Math.floor(state.rng() * state.queries.length);
        const query = state.queries[queryIndex];
        const labeledWrong = state.rng() < noiseRate;
        let positiveIndex;
        if (labeledWrong) {
          const wrongDocs = state.docs
            .map((doc, index) => ({ doc, index }))
            .filter(({ doc }) => doc.classIndex !== query.classIndex);
          positiveIndex = wrongDocs[Math.floor(state.rng() * wrongDocs.length)].index;
          state.labelErrors += 1;
        } else {
          const sameDocs = state.docs
            .map((doc, index) => ({ doc, index }))
            .filter(({ doc }) => doc.classIndex === query.classIndex);
          positiveIndex = sameDocs[Math.floor(state.rng() * sameDocs.length)].index;
        }

        const wrongIndices = negativePool(state.docs, query.classIndex, positiveIndex)
          .sort((a, b) => dot(query.vector, b.doc.vector) - dot(query.vector, a.doc.vector));
        const negativeIndices = [];
        while (negativeIndices.length < 3) {
          let candidate;
          if (state.rng() < hardRatio) {
            const hardPool = wrongIndices.slice(0, Math.min(6, wrongIndices.length));
            candidate = hardPool[Math.floor(state.rng() * hardPool.length)];
          } else {
            candidate = wrongIndices[Math.floor(state.rng() * wrongIndices.length)];
          }
          if (!negativeIndices.includes(candidate.index)) {
            negativeIndices.push(candidate.index);
          }
        }

        const candidateIndices = [positiveIndex, ...negativeIndices];
        const scores = candidateIndices.map((index) => dot(query.vector, state.docs[index].vector) / temp);
        const probabilities = softmax(scores);
        const maxScore = Math.max(...scores);
        const loss = maxScore - scores[0] + Math.log(scores.reduce((sum, score) => sum + Math.exp(score - maxScore), 0));
        stepLoss += loss;

        const queryGrad = [0, 0];
        candidateIndices.forEach((docIndex, candidateIndex) => {
          const grad = probabilities[candidateIndex] - (candidateIndex === 0 ? 1 : 0);
          queryGrad[0] += (grad * state.docs[docIndex].vector[0]) / temp;
          queryGrad[1] += (grad * state.docs[docIndex].vector[1]) / temp;
          const docGrad = docGrads.get(docIndex) || [0, 0];
          docGrad[0] += (grad * query.vector[0]) / temp;
          docGrad[1] += (grad * query.vector[1]) / temp;
          docGrads.set(docIndex, docGrad);
        });
        queryGrads.set(queryIndex, add(queryGrads.get(queryIndex) || [0, 0], queryGrad));
        batch.push({
          queryIndex,
          positiveIndex,
          negativeIndices,
          probabilities,
          labelError: labeledWrong,
        });
      }

      const clipGradient = (grad) => {
        const norm = Math.hypot(...grad);
        return norm > 4 ? scale(grad, 4 / norm) : grad;
      };
      queryGrads.forEach((grad, index) => {
        const clipped = clipGradient(scale(grad, 1 / batchSize));
        state.queries[index].vector = clipVector([
          state.queries[index].vector[0] - lr * clipped[0],
          state.queries[index].vector[1] - lr * clipped[1],
        ]);
      });
      docGrads.forEach((grad, index) => {
        const clipped = clipGradient(scale(grad, 1 / batchSize));
        state.docs[index].vector = clipVector([
          state.docs[index].vector[0] - lr * clipped[0],
          state.docs[index].vector[1] - lr * clipped[1],
        ]);
      });

      state.steps += 1;
      state.currentLoss = stepLoss / batchSize;
      state.lossHistory.push(state.currentLoss);
      if (state.lossHistory.length > maxSteps) state.lossHistory.shift();
      state.lastBatch = batch;
      render();
      if (state.steps >= maxSteps) stopRunning();
    };

    const resetSimulation = () => {
      stopRunning();
      state = createInitialState();
      render();
    };

    buttons.step.addEventListener("click", () => {
      stopRunning();
      trainStep();
    });
    buttons.run.addEventListener("click", () => {
      if (running) {
        stopRunning();
        render();
        return;
      }
      running = true;
      buttons.run.textContent = "暂停";
      render();
      timer = setInterval(trainStep, 110);
    });
    buttons.reset.addEventListener("click", resetSimulation);
    [controls.lr, controls.temp, controls.hard, controls.noise].forEach((input) => {
      input.addEventListener("input", render);
    });
    controls.batch.addEventListener("change", render);

    const presets = {
      clean: { lr: 5, temp: 20, hard: 70, noise: 0 },
      nohard: { lr: 5, temp: 20, hard: 0, noise: 0 },
      noisy: { lr: 5, temp: 20, hard: 70, noise: 20 },
    };
    $$("[data-sim-preset]").forEach((button) => {
      button.addEventListener("click", () => {
        const preset = presets[button.dataset.simPreset];
        controls.lr.value = String(preset.lr);
        controls.temp.value = String(preset.temp);
        controls.hard.value = String(preset.hard);
        controls.noise.value = String(preset.noise);
        $$("[data-sim-preset]").forEach((item) => {
          item.setAttribute("aria-pressed", String(item === button));
        });
        resetSimulation();
      });
    });

    state = createInitialState();
    render();
}
