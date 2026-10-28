import assert from "node:assert/strict";
import { test } from "node:test";
import { parseFragment } from "parse5";
import { pipelineSchedule, initPipelineSchedule } from "../src/scripts/features/pipeline-schedule.js";

const key = op => `${op.stage}:${op.kind}:${op.micro}`;
const label = op => `${op.kind}${op.micro}`;

function declaredOrder(stages, microbatches, mode, stage) {
  const micros = Array.from({ length: microbatches }, (_, micro) => micro);
  const forwards = micros.map(micro => ({ stage, kind: "F", micro }));
  const backwards = micros.map(micro => ({ stage, kind: "B", micro }));
  if (mode === "gpipe") return [...forwards, ...backwards.reverse()];
  const warmup = Math.min(microbatches, stages - stage - 1);
  const pairs = backwards.slice(0, microbatches - warmup)
    .flatMap((backward, index) => [forwards[warmup + index], backward]);
  return [...forwards.slice(0, warmup), ...pairs, ...backwards.slice(microbatches - warmup)];
}

function verifySchedule(options) {
  const { stages, microbatches, mode } = options;
  const result = pipelineSchedule(options);
  const orders = Array.from({ length: stages }, (_, stage) => declaredOrder(stages, microbatches, mode, stage));
  const byKey = new Map(result.operations.map(op => [key(op), op]));
  const expectedWork = 2 * stages * microbatches;
  assert.equal(result.unit, 1);
  assert.match(result.unitLabel, /示意时间片.*非毫秒/);
  assert.equal(result.operations.length, expectedWork);
  assert.equal(byKey.size, expectedWork);
  assert.equal(result.finish, result.timeline.length);
  assert.ok(result.finish > 0 && result.finish <= expectedWork);
  assert.equal(result.totalWork, expectedWork);
  assert.equal(result.capacity, stages * result.finish);
  assert.equal(result.idle, result.capacity - expectedWork);
  assert.equal(result.idleFraction, result.idle / result.capacity);
  assert.deepEqual(result.sequenceByStage, orders.map(order => order.map(label)));
  assert.deepEqual(result.optimizerUpdate, { tick: result.finish, count: 1 });

  const lastForwardEnd = Math.max(...result.operations.filter(op => op.kind === "F").map(op => op.end));
  for (const [stage, order] of orders.entries()) {
    const actual = result.operations.filter(op => op.stage === stage).sort((a, b) => a.start - b.start);
    assert.deepEqual(actual.map(label), order.map(label));
    assert.equal(actual.filter(op => op.kind === "F").length, microbatches);
    assert.equal(actual.filter(op => op.kind === "B").length, microbatches);
    actual.forEach((op, index) => {
      assert.equal(op.end, op.start + 1);
      assert.ok(Number.isInteger(op.start) && op.start >= 0 && op.end <= result.finish);
      if (index) assert.ok(actual[index - 1].end <= op.start);
      if (op.kind === "F" && stage > 0) {
        assert.ok(byKey.get(`${stage - 1}:F:${op.micro}`).end <= op.start);
      }
      if (op.kind === "B") {
        assert.ok(byKey.get(`${stage}:F:${op.micro}`).end <= op.start);
        if (stage < stages - 1) assert.ok(byKey.get(`${stage + 1}:B:${op.micro}`).end <= op.start);
        if (mode === "gpipe") assert.ok(lastForwardEnd <= op.start);
        assert.ok(op.end <= result.optimizerUpdate.tick);
      }
    });
  }

  const completed = new Set(), flattened = [];
  const positions = Array(stages).fill(0);
  const activations = Array.from({ length: stages }, () => new Set());
  const peaks = Array(stages).fill(0);
  let completedForwards = 0, idleSlots = 0;
  for (const [tick, step] of result.timeline.entries()) {
    assert.equal(step.tick, tick);
    assert.ok(step.ops.length > 0 && step.ops.length <= stages);
    assert.equal(new Set(step.ops.map(op => op.stage)).size, step.ops.length);
    const expectedReady = [];
    for (let stage = 0; stage < stages; stage++) {
      const op = orders[stage][positions[stage]];
      if (!op) continue;
      const prior = orders[stage][positions[stage] - 1];
      if (prior && !completed.has(key(prior))) continue;
      if (op.kind === "F" && stage > 0 && !completed.has(`${stage - 1}:F:${op.micro}`)) continue;
      if (op.kind === "B") {
        if (!completed.has(`${stage}:F:${op.micro}`)) continue;
        if (stage < stages - 1 && !completed.has(`${stage + 1}:B:${op.micro}`)) continue;
        if (mode === "gpipe" && completedForwards !== stages * microbatches) continue;
      }
      expectedReady.push(op);
    }
    assert.deepEqual(step.ops, expectedReady, `时间片 ${tick} 必须同时执行所有就绪的本地下一操作`);
    for (const op of step.ops) {
      const timed = { ...op, start: tick, end: tick + 1 };
      assert.deepEqual(byKey.get(key(op)), timed);
      assert.ok(!completed.has(key(op)));
      flattened.push(timed);
      const live = activations[op.stage];
      if (op.kind === "F") {
        assert.ok(!live.has(op.micro));
        live.add(op.micro);
        peaks[op.stage] = Math.max(peaks[op.stage], live.size);
      } else {
        assert.ok(live.has(op.micro));
        live.delete(op.micro);
      }
      positions[op.stage]++;
    }
    // Commit only after checking every selected operation against the prior snapshot.
    for (const op of step.ops) {
      completed.add(key(op));
      if (op.kind === "F") completedForwards++;
    }
    assert.deepEqual(step.live, activations.map(live => live.size));
    idleSlots += stages - step.ops.length;
  }
  assert.deepEqual(result.operations, flattened);
  assert.equal(completed.size, expectedWork);
  assert.deepEqual(positions, Array(stages).fill(2 * microbatches));
  assert.deepEqual(activations.map(live => live.size), Array(stages).fill(0));
  assert.deepEqual(result.peakByStage, peaks);
  assert.deepEqual(peaks, Array.from({ length: stages }, (_, stage) =>
    mode === "gpipe" ? microbatches : Math.min(microbatches, stages - stage)
  ));
  assert.equal(result.idle, idleSlots);
  assert.deepEqual(result, pipelineSchedule(options));
  return result;
}

const tickLabels = result => result.timeline.map(step => Array.from({ length: result.stages }, (_, stage) => {
  const op = step.ops.find(candidate => candidate.stage === stage);
  return op ? label(op) : "-";
}).join("|"));

test("全部 48 种组合独立验证本地顺序、跨阶段依赖、并行提交、计数和无死锁", () => {
  for (let stages = 2; stages <= 4; stages++) {
    for (let microbatches = 1; microbatches <= 8; microbatches++) {
      for (const mode of ["gpipe", "1f1b"]) verifySchedule({ stages, microbatches, mode });
    }
  }
});

test("默认 GPipe 的精确事件表包含全部前向屏障和逆序反向", () => {
  const result = pipelineSchedule();
  assert.deepEqual(result, pipelineSchedule({ stages: 3, microbatches: 4, mode: "gpipe" }));
  assert.deepEqual(result, pipelineSchedule({}));
  assert.deepEqual(tickLabels(result), [
    "F0|-|-", "F1|F0|-", "F2|F1|F0", "F3|F2|F1", "-|F3|F2", "-|-|F3",
    "-|-|B3", "-|B3|B2", "B3|B2|B1", "B2|B1|B0", "B1|B0|-", "B0|-|-",
  ]);
  assert.equal(result.finish, 12);
  assert.equal(result.totalWork, 24);
  assert.equal(result.capacity, 36);
  assert.equal(result.idle, 12);
  assert.equal(result.idleFraction, 1 / 3);
  assert.deepEqual(result.peakByStage, [4, 4, 4]);
});

test("三阶段四微批次 1F1B 严格保留预热、交替和冷却顺序", () => {
  const result = pipelineSchedule({ mode: "1f1b" });
  assert.deepEqual(result.sequenceByStage, [
    ["F0", "F1", "F2", "B0", "F3", "B1", "B2", "B3"],
    ["F0", "F1", "B0", "F2", "B1", "F3", "B2", "B3"],
    ["F0", "B0", "F1", "B1", "F2", "B2", "F3", "B3"],
  ]);
  assert.deepEqual(tickLabels(result), [
    "F0|-|-", "F1|F0|-", "F2|F1|F0", "-|-|B0", "-|B0|F1", "B0|F2|B1",
    "F3|B1|F2", "B1|F3|B2", "-|B2|F3", "B2|-|B3", "-|B3|-", "B3|-|-",
  ]);
  assert.equal(result.finish, 12);
  assert.deepEqual(result.timeline[2].live, [3, 2, 1]);
  assert.deepEqual(result.peakByStage, [3, 2, 1]);
  const find = (stage, kind, micro) => result.operations.find(op => key(op) === `${stage}:${kind}:${micro}`);
  assert.ok(find(2, "B", 0).start < find(1, "B", 0).start);
  assert.ok(find(1, "B", 0).start < find(0, "B", 0).start);
  assert.ok(find(2, "B", 0).end <= find(0, "F", 3).start);
});

test("单微批次与预热覆盖全部微批次的边界不改变算法", () => {
  for (let stages = 2; stages <= 4; stages++) {
    const gpipe = pipelineSchedule({ stages, microbatches: 1 });
    const one = pipelineSchedule({ stages, microbatches: 1, mode: "1f1b" });
    assert.equal(one.finish, 2 * stages);
    assert.deepEqual(one.timeline, gpipe.timeline);
    assert.deepEqual(one.peakByStage, Array(stages).fill(1));
  }
  assert.deepEqual(pipelineSchedule({ stages: 4, microbatches: 2, mode: "1f1b" }).sequenceByStage, [
    ["F0", "F1", "B0", "B1"], ["F0", "F1", "B0", "B1"],
    ["F0", "F1", "B0", "B1"], ["F0", "B0", "F1", "B1"],
  ]);
});

test("常见配置中 1F1B 降低每阶段激活峰值且操作总量相同", () => {
  for (const [stages, microbatches] of [[2, 4], [3, 4], [4, 8]]) {
    const gpipe = pipelineSchedule({ stages, microbatches });
    const one = pipelineSchedule({ stages, microbatches, mode: "1f1b" });
    assert.equal(one.totalWork, gpipe.totalWork);
    assert.equal(one.finish, gpipe.finish);
    one.peakByStage.forEach((peak, stage) => assert.ok(peak < gpipe.peakByStage[stage]));
  }
});

test("非法范围、非整数和未知模式明确拒绝且不静默修正", () => {
  for (const stages of [0, 1, 5, -1, 2.5, NaN, Infinity, "3", null, true]) {
    assert.throws(() => pipelineSchedule({ stages }), RangeError);
  }
  for (const microbatches of [0, 9, -1, 1.5, NaN, Infinity, "4", null, false]) {
    assert.throws(() => pipelineSchedule({ microbatches }), RangeError);
  }
  for (const mode of ["", "GPipe", "1F1B", "flush", "__proto__", "constructor", null, 0, {}, []]) {
    assert.throws(() => pipelineSchedule({ mode }), RangeError);
  }
  assert.throws(() => pipelineSchedule(null), TypeError);
  assert.deepEqual(pipelineSchedule({ stages: undefined, microbatches: undefined, mode: undefined }), pipelineSchedule());
});

const descendants = (node, tag) => (node.tagName === tag ? [node] : [])
  .concat((node.childNodes || []).flatMap(child => descendants(child, tag)));
const content = node => node.nodeName === "#text" ? node.value : (node.childNodes || []).map(content).join("");
const attr = (node, name) => node.attrs.find(attribute => attribute.name === name)?.value;
const cells = row => (row.childNodes || []).filter(node => ["th", "td"].includes(node.tagName));

function mount(t, values, tags = {}) {
  const nodes = new Map(Object.entries(values).map(([name, value]) => {
    const listeners = new Map();
    return [`[data-${name}]`, {
      tagName: tags[name] || "DIV", value: String(value), textContent: "", innerHTML: "",
      addEventListener(type, listener) {
        if (!listeners.has(type)) listeners.set(type, []);
        listeners.get(type).push(listener);
      },
      emit(type) { for (const listener of listeners.get(type) || []) listener(); },
    }];
  }));
  const previous = Object.getOwnPropertyDescriptor(globalThis, "document");
  Object.defineProperty(globalThis, "document", { configurable: true, value: { querySelector: selector => nodes.get(selector) || null } });
  t.after(() => {
    if (previous) Object.defineProperty(globalThis, "document", previous);
    else delete globalThis.document;
  });
  return name => nodes.get(`[data-${name}]`);
}

const uiValues = {
  "pp-stages": 3, "pp-micro": 4, "pp-mode": "gpipe", "pp-micro-value": "",
  "pp-finish": "", "pp-idle": "", "pp-peak": "", "pp-caption": "", "pp-grid": "", "pp-ledger": "",
};

function verifyRendered(node) {
  const result = pipelineSchedule({
    stages: Number(node("pp-stages").value), microbatches: Number(node("pp-micro").value), mode: node("pp-mode").value,
  });
  assert.equal(node("pp-micro-value").textContent, String(result.microbatches));
  assert.equal(node("pp-finish").textContent, String(result.finish));
  assert.equal(node("pp-idle").textContent, `${result.idle} / ${result.capacity}（${(100 * result.idleFraction).toFixed(1)}%）`);
  assert.equal(node("pp-peak").textContent, result.peakByStage.join(" / "));
  assert.match(node("pp-caption").textContent, /1 个示意时间片（非毫秒）/);
  assert.match(node("pp-caption").textContent, /同一版本权重/);
  assert.match(node("pp-caption").textContent, /统一更新一次参数/);
  assert.doesNotMatch(node("pp-caption").textContent, /吞吐|tokens\/s|\bms\b/i);

  const gridHTML = node("pp-grid").tagName === "TABLE" ? `<table>${node("pp-grid").innerHTML}</table>` : node("pp-grid").innerHTML;
  const tables = descendants(parseFragment(gridHTML), "table");
  assert.equal(tables.length, 1);
  const table = tables[0];
  assert.match(content(descendants(table, "caption")[0]), /示意时间片/);
  const headings = descendants(descendants(table, "thead")[0], "th");
  assert.equal(headings.length, result.finish + 1);
  headings.forEach(heading => assert.equal(attr(heading, "scope"), "col"));
  assert.deepEqual(headings.slice(1).map(content), result.timeline.map(step => String(step.tick)));
  const rows = descendants(descendants(table, "tbody")[0], "tr");
  assert.equal(rows.length, result.stages);
  rows.forEach((row, stage) => {
    const rowCells = cells(row);
    assert.equal(rowCells.length, result.finish + 1);
    assert.equal(content(rowCells[0]), `阶段 ${stage}`);
    assert.equal(attr(rowCells[0], "scope"), "row");
    rowCells.slice(1).forEach((cell, tick) => {
      const op = result.timeline[tick].ops.find(candidate => candidate.stage === stage);
      assert.equal(cell.tagName, "td");
      assert.equal(content(cell), op ? label(op) : "-");
      assert.equal(attr(cell, "class"), `pp-cell pp-${op ? (op.kind === "F" ? "forward" : "backward") : "idle"}`);
      assert.match(attr(cell, "title"), /阶段.*时间片.*保留.*份激活/);
      assert.ok(attr(cell, "title").includes(op ? (op.kind === "F" ? "前向" : "反向") : "空闲"));
      assert.equal(attr(cell, "style"), undefined);
    });
  });

  const ledgerHTML = node("pp-ledger").innerHTML;
  const ledger = descendants(parseFragment(node("pp-ledger").tagName === "TABLE" ? `<table>${ledgerHTML}</table>` : ledgerHTML), "table")[0];
  assert.deepEqual(descendants(descendants(ledger, "thead")[0], "th").map(content), ["阶段", "前向次数", "反向次数", "激活峰值"]);
  const ledgerRows = descendants(descendants(ledger, "tbody")[0], "tr");
  assert.equal(ledgerRows.length, result.stages);
  ledgerRows.forEach((row, stage) => {
    assert.deepEqual(cells(row).map(content), [`阶段 ${stage}`, String(result.microbatches), String(result.microbatches), String(result.peakByStage[stage])]);
    assert.equal(attr(cells(row)[0], "scope"), "row");
  });
}

test("初始化守卫兼容无页面控件与不完整挂载", t => {
  assert.doesNotThrow(() => initPipelineSchedule());
  mount(t, {});
  assert.doesNotThrow(() => initPipelineSchedule());
});

test("语义时间表和四列账本随全部控件更新，保留主页面尺寸类名", t => {
  const node = mount(t, uiValues, { "pp-ledger": "TABLE" });
  initPipelineSchedule();
  verifyRendered(node);
  for (const [name, value, event] of [
    ["pp-mode", "1f1b", "change"], ["pp-micro", "8", "input"],
    ["pp-stages", "4", "change"], ["pp-stages", "2", "input"],
    ["pp-micro", "1", "change"], ["pp-mode", "gpipe", "input"],
  ]) {
    node(name).value = value;
    node(name).emit(event);
    verifyRendered(node);
  }
});

test("已有表格和账本容器保持有效结构且不嵌套表格", t => {
  const node = mount(t, uiValues, { "pp-grid": "TABLE" });
  initPipelineSchedule();
  verifyRendered(node);
});

test("父页面提供账本表体时只填充四列阶段行", t => {
  const node = mount(t, uiValues, { "pp-ledger": "TBODY" });
  initPipelineSchedule();
  assert.doesNotMatch(node("pp-ledger").innerHTML, /<table|<thead|<tbody/);
  const rows = descendants(parseFragment(`<table><tbody>${node("pp-ledger").innerHTML}</tbody></table>`), "tr");
  assert.equal(rows.length, 3);
  rows.forEach((row, stage) => assert.deepEqual(cells(row).map(content), [`阶段 ${stage}`, "4", "4", "4"]));
});
