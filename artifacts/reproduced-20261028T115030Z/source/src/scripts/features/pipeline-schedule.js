import { $ } from "../core.js";

export function pipelineSchedule({ stages = 3, microbatches = 4, mode = "gpipe" } = {}) {
  if (!Number.isInteger(stages) || stages < 2 || stages > 4) {
    throw new RangeError("阶段数必须为 2 至 4 的整数。");
  }
  if (!Number.isInteger(microbatches) || microbatches < 1 || microbatches > 8) {
    throw new RangeError("微批次数必须为 1 至 8 的整数。");
  }
  if (mode !== "gpipe" && mode !== "1f1b") {
    throw new RangeError("排程模式必须为 gpipe 或 1f1b。");
  }

  const key = (stage, kind, micro) => `${stage}:${kind}:${micro}`;
  const orders = Array.from({ length: stages }, (_, stage) => {
    const order = [];
    const append = (kind, micro) => order.push({ stage, kind, micro });
    if (mode === "gpipe") {
      for (let micro = 0; micro < microbatches; micro++) append("F", micro);
      for (let micro = microbatches - 1; micro >= 0; micro--) append("B", micro);
    } else {
      const warmup = Math.min(microbatches, stages - stage - 1);
      for (let micro = 0; micro < warmup; micro++) append("F", micro);
      for (let micro = 0; micro < microbatches - warmup; micro++) {
        append("F", warmup + micro);
        append("B", micro);
      }
      for (let micro = microbatches - warmup; micro < microbatches; micro++) append("B", micro);
    }
    return order;
  });
  const forwardKeys = orders.flat().filter(op => op.kind === "F")
    .map(op => key(op.stage, op.kind, op.micro));
  const nodes = orders.map((order, stage) => order.map((op, index) => {
    const requires = new Set();
    if (index > 0) {
      const previous = order[index - 1];
      requires.add(key(stage, previous.kind, previous.micro));
    }
    if (op.kind === "F") {
      if (stage > 0) requires.add(key(stage - 1, "F", op.micro));
    } else {
      requires.add(key(stage, "F", op.micro));
      if (stage < stages - 1) requires.add(key(stage + 1, "B", op.micro));
      if (mode === "gpipe") forwardKeys.forEach(id => requires.add(id));
    }
    return { op, id: key(stage, op.kind, op.micro), requires: [...requires] };
  }));

  const next = Array(stages).fill(0);
  const live = Array(stages).fill(0);
  const peakByStage = Array(stages).fill(0);
  const completed = new Set();
  const timeline = [], operations = [];
  const totalWork = 2 * stages * microbatches;
  while (operations.length < totalWork) {
    const tick = timeline.length;
    // Every stage reads the same prior-tick snapshot before any completion is committed.
    const ready = nodes.flatMap((stageNodes, stage) => {
      const node = stageNodes[next[stage]];
      return node && node.requires.every(id => completed.has(id)) ? [node] : [];
    });
    if (!ready.length) throw new Error("流水线依赖与本地顺序发生死锁。");
    for (const node of ready) {
      const { stage, kind } = node.op;
      completed.add(node.id);
      next[stage]++;
      live[stage] += kind === "F" ? 1 : -1;
      peakByStage[stage] = Math.max(peakByStage[stage], live[stage]);
      operations.push({ ...node.op, start: tick, end: tick + 1 });
    }
    timeline.push({ tick, ops: ready.map(node => ({ ...node.op })), live: [...live] });
  }

  const finish = timeline.length;
  const capacity = finish * stages;
  const idle = capacity - totalWork;
  return {
    stages, microbatches, mode, unit: 1, unitLabel: "示意时间片（非毫秒）",
    timeline, operations, finish, totalWork, capacity, idle, idleFraction: idle / capacity,
    peakByStage,
    sequenceByStage: orders.map(order => order.map(op => `${op.kind}${op.micro}`)),
    optimizerUpdate: { tick: finish, count: 1 },
  };
}

export function initPipelineSchedule() {
  if (typeof document === "undefined") return;
  const stagesInput = $("[data-pp-stages]");
  if (!stagesInput) return;
  const microInput = $("[data-pp-micro]"), modeInput = $("[data-pp-mode]");
  if (!microInput || !modeInput) return;
  const grid = $("[data-pp-grid]"), ledger = $("[data-pp-ledger]");
  const setText = (selector, value) => {
    const output = $(selector);
    if (output) output.textContent = String(value);
  };
  const writeTable = (target, caption, header, rows) => {
    if (!target) return;
    if (target.tagName === "TBODY") {
      target.innerHTML = rows;
      return;
    }
    const markup = `<caption>${caption}</caption><thead><tr>${header}</tr></thead><tbody>${rows}</tbody>`;
    target.innerHTML = target.tagName === "TABLE" ? markup : `<table class="pp-table">${markup}</table>`;
  };
  const render = () => {
    const result = pipelineSchedule({
      stages: Number(stagesInput.value), microbatches: Number(microInput.value), mode: modeInput.value,
    });
    setText("[data-pp-micro-value]", result.microbatches);
    setText("[data-pp-finish]", result.finish);
    setText("[data-pp-idle]", `${result.idle} / ${result.capacity}（${(100 * result.idleFraction).toFixed(1)}%）`);
    setText("[data-pp-peak]", result.peakByStage.join(" / "));
    const description = result.mode === "gpipe"
      ? "GPipe：每阶段按编号递增执行全部前向，等待所有阶段的前向完成后，按编号递减执行反向。"
      : `1F1B：阶段 r 的预热数 w=min(${result.microbatches},${result.stages}-r-1)，先预热 w 次前向，再依次执行 F(w+i)、B(i)，最后按编号递增完成剩余反向。`;
    setText("[data-pp-caption]", `${description}阶段与微批次编号从 0 开始。同构模型中每次前向或反向占 1 个示意时间片（非毫秒），通信耗时为 0；每阶段每片最多执行一次操作，依赖完成后在下一片可用。全部微批次使用同一版本权重，全部反向完成后在边界 ${result.finish} 统一更新一次参数，更新不另占时间片。激活在前向完成时保留，在反向完成时释放；空闲按阶段时间片计数。`);

    const header = '<th scope="col">阶段 / 时间片</th>' + result.timeline.map(step =>
      `<th scope="col" class="pp-tick">${step.tick}</th>`
    ).join("");
    const rows = Array.from({ length: result.stages }, (_, stage) => {
      const cells = result.timeline.map(step => {
        const op = step.ops.find(candidate => candidate.stage === stage);
        const label = op ? `${op.kind}${op.micro}` : "-";
        const phase = op ? (op.kind === "F" ? "前向" : "反向") : "空闲";
        const state = op ? `pp-${op.kind === "F" ? "forward" : "backward"}` : "pp-idle";
        const title = `阶段 ${stage}，时间片 ${step.tick}：${phase}${op ? `微批次 ${op.micro}` : ""}；完成后保留 ${step.live[stage]} 份激活。`;
        return `<td class="pp-cell ${state}" title="${title}">${label}</td>`;
      }).join("");
      return `<tr><th scope="row">阶段 ${stage}</th>${cells}</tr>`;
    }).join("");
    writeTable(grid, "流水线事件时间表（示意时间片）", header, rows);

    const ledgerHeader = ["阶段", "前向次数", "反向次数", "激活峰值"]
      .map(label => `<th scope="col">${label}</th>`).join("");
    const ledgerRows = Array.from({ length: result.stages }, (_, stage) => {
      const stageOps = result.operations.filter(op => op.stage === stage);
      const forward = stageOps.filter(op => op.kind === "F").length;
      const backward = stageOps.filter(op => op.kind === "B").length;
      return `<tr><th scope="row">阶段 ${stage}</th><td>${forward}</td><td>${backward}</td><td>${result.peakByStage[stage]}</td></tr>`;
    }).join("");
    writeTable(ledger, "每阶段操作与激活账本", ledgerHeader, ledgerRows);
  };
  for (const control of [stagesInput, microInput, modeInput]) {
    control.addEventListener("input", render);
    control.addEventListener("change", render);
  }
  render();
}
