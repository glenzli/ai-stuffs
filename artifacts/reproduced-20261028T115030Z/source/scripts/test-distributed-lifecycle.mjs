import assert from "node:assert/strict";
import { test } from "node:test";
import { parseFragment } from "parse5";
import { tensorParallelTrace, collectiveExample, initDistributedExecution } from "../src/scripts/features/distributed-execution.js";
import { activationTrace, initActivationCheckpointing } from "../src/scripts/features/activation-checkpointing.js";

const X = [[1, 2], [3, -1]];
const W1 = [[0.2, -0.3, 0.5, 0.1], [0.4, 0.6, -0.2, 0.3]];
const W2 = [[0.7, -0.2], [0.1, 0.5], [-0.4, 0.8], [0.3, 0.2]];
const G = [[0.2, -0.4], [0.6, 0.1]];
const weights = [0.9, 1.1, 0.8, 1.2, 0.7, 1.05];
const close = (actual, expected, tolerance = 1e-10) => {
  assert.ok(Math.abs(actual - expected) <= tolerance, `实际值 ${actual}，预期值 ${expected}`);
};
const vectorClose = (actual, expected, tolerance) => {
  assert.equal(actual.length, expected.length);
  actual.forEach((value, index) => close(value, expected[index], tolerance));
};
const matrixClose = (actual, expected, tolerance) => {
  assert.equal(actual.length, expected.length);
  actual.forEach((row, index) => vectorClose(row, expected[index], tolerance));
};
const copyMatrix = matrix => matrix.map(row => row.slice());
const addMatrices = (left, right) => left.map((row, i) => row.map((value, j) => value + right[i][j]));

function denseOutput(input, first, second) {
  return input.map(row => {
    const output = Array(second[0].length).fill(0);
    for (let hidden = 0; hidden < second.length; hidden++) {
      let z = 0;
      for (let feature = 0; feature < row.length; feature++) z += row[feature] * first[feature][hidden];
      for (let column = 0; column < output.length; column++) output[column] += Math.tanh(z) * second[hidden][column];
    }
    return output;
  });
}

function denseLoss(input, first, second) {
  const output = denseOutput(input, first, second);
  let loss = 0;
  for (let row = 0; row < G.length; row++) {
    for (let column = 0; column < G[row].length; column++) loss += output[row][column] * G[row][column];
  }
  return loss;
}

function finiteMatrix(matrix, loss) {
  const epsilon = 1e-6;
  return matrix.map((row, i) => row.map((_, j) => {
    const positive = copyMatrix(matrix), negative = copyMatrix(matrix);
    positive[i][j] += epsilon;
    negative[i][j] -= epsilon;
    return (loss(positive) - loss(negative)) / (2 * epsilon);
  }));
}

const scalarLoss = parameters => 0.5 * parameters.reduce((hidden, weight) => Math.tanh(weight * hidden), 0.8) ** 2;
function scalarFiniteDifferences() {
  const epsilon = 1e-6;
  return weights.map((_, index) => {
    const positive = weights.slice(), negative = weights.slice();
    positive[index] += epsilon;
    negative[index] -= epsilon;
    return (scalarLoss(positive) - scalarLoss(negative)) / (2 * epsilon);
  });
}

const descendants = (node, tag) => (node.tagName === tag ? [node] : []).concat((node.childNodes || []).flatMap(child => descendants(child, tag)));
const content = node => node.nodeName === "#text" ? node.value : (node.childNodes || []).map(content).join("");
const tableRows = html => descendants(parseFragment(`<table><tbody>${html}</tbody></table>`), "tr");
const cells = row => (row.childNodes || []).filter(node => ["th", "td"].includes(node.tagName)).map(content);
const readMatrix = value => JSON.parse(`[${value.replaceAll("；", ",")}]`);

function mount(t, values = {}) {
  const nodes = new Map(Object.entries(values).map(([key, value]) => {
    const listeners = new Map();
    return [`[data-${key}]`, {
      value: String(value), textContent: "", innerHTML: "", disabled: false,
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
  return key => nodes.get(`[data-${key}]`);
}

const dxValues = {
  "dx-stage": "forward", "dx-collective": "allreduce", "dx-ranks": "",
  "dx-reference": "", "dx-combined": "", "dx-error": "", "dx-note": "", "dx-collective-rows": "",
};
const acValues = {
  "ac-mode": "store", "ac-segment": 2, "ac-next": "", "ac-reset": "", "ac-step": "",
  "ac-peak": "", "ac-forward": "", "ac-loss": "", "ac-error": "", "ac-live": "", "ac-events": "", "ac-note": "",
};
const expectedCollectives = {
  allreduce: [[11, 22, 33, 44], [11, 22, 33, 44]],
  allgather: [[1, 2, 3, 4, 10, 20, 30, 40], [1, 2, 3, 4, 10, 20, 30, 40]],
  reducescatter: [[11, 22], [33, 44]],
  alltoall: [[1, 2, 10, 20], [3, 4, 30, 40]],
};

test("两进程严格采用指定的列分片与行分片，重建全部输出和梯度", () => {
  const trace = tensorParallelTrace();
  assert.deepEqual({ X: trace.X, W1: trace.W1, W2: trace.W2, G: trace.G }, { X, W1, W2, G });
  assert.equal(trace.ranks.length, 2);
  trace.ranks.forEach((rank, index) => {
    assert.equal(rank.rank, index);
    assert.deepEqual(rank.W1, W1.map(row => row.slice(index * 2, index * 2 + 2)));
    assert.deepEqual(rank.W2, W2.slice(index * 2, index * 2 + 2));
    const z = X.map(row => [0, 1].map(column => row[0] * rank.W1[0][column] + row[1] * rank.W1[1][column]));
    matrixClose(rank.Z, z);
    matrixClose(rank.H, z.map(row => row.map(value => Math.tanh(value))));
    matrixClose(rank.Y, denseOutput(X, rank.W1, rank.W2));
    const dH = G.map(row => rank.W2.map(weight => row[0] * weight[0] + row[1] * weight[1]));
    matrixClose(rank.dH, dH);
    matrixClose(rank.dZ, dH.map((row, i) => row.map((value, j) => value * (1 - rank.H[i][j] ** 2))));
  });
  matrixClose(trace.Y, denseOutput(X, W1, W2));
  matrixClose(trace.Y, addMatrices(trace.ranks[0].Y, trace.ranks[1].Y));
  matrixClose(trace.dX, addMatrices(trace.ranks[0].dX, trace.ranks[1].dX));
  const reconstructedW1 = W1.map((_, row) => trace.ranks.flatMap(rank => rank.dW1[row]));
  const reconstructedW2 = trace.ranks.flatMap(rank => rank.dW2);
  matrixClose(trace.Y, trace.reference.Y);
  matrixClose(trace.dX, trace.reference.dX);
  matrixClose(reconstructedW1, trace.reference.dW1);
  matrixClose(reconstructedW2, trace.reference.dW2);
  const errors = [[trace.Y, trace.reference.Y], [trace.dX, trace.reference.dX], [reconstructedW1, trace.reference.dW1], [reconstructedW2, trace.reference.dW2]];
  const maximum = Math.max(...errors.flatMap(([actual, expected]) => actual.flat().map((value, index) => Math.abs(value - expected.flat()[index]))));
  assert.equal(trace.error, maximum);
  assert.ok(trace.error < 1e-12);
});

test("独立损失有限差分验证输入、两层权重及每个分片的真实梯度", () => {
  const trace = tensorParallelTrace();
  matrixClose(trace.reference.dX, finiteMatrix(X, input => denseLoss(input, W1, W2)), 1e-9);
  matrixClose(trace.reference.dW1, finiteMatrix(W1, first => denseLoss(X, first, W2)), 1e-9);
  matrixClose(trace.reference.dW2, finiteMatrix(W2, second => denseLoss(X, W1, second)), 1e-9);
  for (const rank of trace.ranks) {
    matrixClose(rank.dX, finiteMatrix(X, input => denseLoss(input, rank.W1, rank.W2)), 1e-9);
    matrixClose(rank.dW1, finiteMatrix(rank.W1, first => denseLoss(X, first, rank.W2)), 1e-9);
    matrixClose(rank.dW2, finiteMatrix(rank.W2, second => denseLoss(X, rank.W1, second)), 1e-9);
  }
});

test("四种集合通信显式保留输入和正确的输出分布", () => {
  for (const [mode, expected] of Object.entries(expectedCollectives)) {
    const result = collectiveExample(mode);
    assert.deepEqual(result.input, [[1, 2, 3, 4], [10, 20, 30, 40]]);
    assert.deepEqual(result.output, expected);
    result.output[0][0] = -100;
    assert.deepEqual(result.input, [[1, 2, 3, 4], [10, 20, 30, 40]]);
    assert.deepEqual(result.output[1], expected[1]);
    assert.deepEqual(collectiveExample(mode).output, expected);
  }
  for (const mode of [undefined, null, "", "sum", "__proto__", "constructor", {}]) {
    assert.throws(() => collectiveExample(mode), RangeError);
  }
});

test("六层标量损失及全部梯度通过独立有限差分验证", () => {
  const baseline = activationTrace();
  close(baseline.loss, scalarLoss(weights), 1e-14);
  vectorClose(baseline.gradients, scalarFiniteDifferences(), 1e-10);
  vectorClose(baseline.referenceGradients, scalarFiniteDifferences(), 1e-10);
  assert.equal(baseline.totalForward, 6);
  assert.equal(baseline.peak, 7);
  assert.deepEqual(baseline.gradients, baseline.referenceGradients);
});

test("所有分段大小与全部保留模式具有相同损失和梯度，完整重算恰好增加六次前向", () => {
  const baseline = activationTrace();
  for (let segment = 1; segment <= 6; segment++) {
    for (const mode of ["store", "checkpoint"]) {
      const trace = activationTrace(mode, segment);
      assert.equal(trace.loss, baseline.loss);
      assert.deepEqual(trace.gradients, baseline.gradients);
      assert.deepEqual(trace.referenceGradients, baseline.gradients);
      assert.equal(trace.totalForward, mode === "store" ? 6 : 12);
      assert.equal(trace.events.at(-1).backwardCount, 6);
      assert.deepEqual(trace.events.at(-1).live, []);
    }
  }
  assert.deepEqual(activationTrace(), activationTrace("store", 2));
});

test("事件账本验证依赖、释放时机、边界复用和唯一逻辑激活峰值", () => {
  for (const mode of ["store", "checkpoint"]) {
    for (let segment = 1; segment <= 6; segment++) {
      const trace = activationTrace(mode, segment);
      const boundaries = [0];
      for (let layer = segment; layer < 6; layer += segment) boundaries.push(layer);
      boundaries.push(6);
      const recomputedLayers = [];
      if (mode === "checkpoint") {
        for (let index = boundaries.length - 1; index > 0; index--) {
          for (let layer = boundaries[index - 1] + 1; layer <= boundaries[index]; layer++) recomputedLayers.push(layer);
        }
      }
      const initial = trace.events[0];
      assert.deepEqual(initial, { phase: "initial", label: "保留初始激活 h0", live: ["h0"], count: 1, forwardCount: 0, backwardCount: 0 });
      let forward = 0, backward = 0, recomputed = 0, original = 0, losses = 0;
      for (let index = 1; index < trace.events.length; index++) {
        const event = trace.events[index], previous = trace.events[index - 1];
        assert.equal(event.count, event.live.length);
        assert.equal(new Set(event.live).size, event.live.length);
        assert.deepEqual(event.live, event.live.slice().sort());
        assert.ok(event.live.every(id => /^h[0-6]$/.test(id)));
        assert.match(event.label, /[\u4e00-\u9fff]/);
        if (event.phase === "forward" || event.phase === "recompute") {
          const layer = event.phase === "forward" ? ++original : recomputedLayers[recomputed++];
          assert.ok(layer >= 1 && layer <= 6);
          assert.ok(previous.live.includes(`h${layer - 1}`));
          assert.ok(event.live.includes(`h${layer - 1}`));
          assert.ok(event.live.includes(`h${layer}`));
          assert.deepEqual(event.live, Array.from(new Set([...previous.live, `h${layer}`])).sort());
          if (previous.live.includes(`h${layer}`)) {
            assert.equal(event.phase, "recompute");
            assert.ok(boundaries.includes(layer));
            assert.equal(event.count, previous.count);
          }
          forward++;
        } else if (event.phase === "backward") {
          const layer = 6 - backward;
          assert.deepEqual(event.live, previous.live);
          assert.ok(event.live.includes(`h${layer - 1}`) && event.live.includes(`h${layer}`));
          const release = trace.events[index + 1];
          assert.equal(release.phase, "release");
          assert.deepEqual(release.live, event.live.filter(id => id !== `h${layer}`));
          backward++;
        } else if (event.phase === "release") {
          assert.equal(event.count, previous.count - 1);
          assert.ok(event.live.every(id => previous.live.includes(id)));
          if (!event.live.includes("h0")) assert.equal(backward, 6);
        } else if (event.phase === "loss") {
          losses++;
          assert.equal(original, 6);
          assert.equal(backward, 0);
          assert.deepEqual(event.live, (mode === "store" ? [0, 1, 2, 3, 4, 5, 6] : boundaries).map(layer => `h${layer}`));
        } else {
          assert.fail(`未知事件阶段：${event.phase}`);
        }
        assert.equal(event.forwardCount, forward);
        assert.equal(event.backwardCount, backward);
        if (backward < 6) assert.ok(event.live.includes("h0"));
      }
      assert.equal(original, 6);
      assert.equal(recomputed, recomputedLayers.length);
      assert.equal(backward, 6);
      assert.equal(losses, 1);
      assert.equal(trace.peak, Math.max(...trace.events.map(event => event.count)));
      const finalForward = trace.events.findIndex(event => event.phase === "forward" && event.forwardCount === 6);
      const firstBackward = trace.events.findIndex(event => event.phase === "backward");
      assert.ok(trace.events.slice(finalForward, firstBackward + 1).every(event => event.live.includes("h6")));
    }
  }
});

test("瞬时分配计入峰值，检查点并不保证降低逻辑激活峰值", () => {
  const expectedPeaks = [7, 5, 5, 5, 6, 7];
  expectedPeaks.forEach((peak, index) => assert.equal(activationTrace("checkpoint", index + 1).peak, peak));
  const trace = activationTrace("checkpoint", 2);
  assert.deepEqual(trace.events.filter(event => event.phase === "forward").map(event => event.count), [2, 3, 3, 4, 4, 5]);
  const finalForward = trace.events.findIndex(event => event.phase === "forward" && event.forwardCount === 6);
  assert.deepEqual(trace.events[finalForward].live, ["h0", "h2", "h4", "h5", "h6"]);
  assert.deepEqual(trace.events[finalForward + 1].live, ["h0", "h2", "h4", "h6"]);
  const single = activationTrace("checkpoint", 1), baseline = activationTrace();
  assert.equal(single.peak, baseline.peak);
  assert.ok(single.totalForward > baseline.totalForward);
});

test("激活轨迹拒绝无效模式和分段大小", () => {
  for (const mode of [null, "", "unknown", "__proto__", "constructor", {}]) {
    assert.throws(() => activationTrace(mode), RangeError);
  }
  for (const segment of [0, -1, 7, 1.5, NaN, Infinity, "2", null, {}]) {
    assert.throws(() => activationTrace("checkpoint", segment), RangeError);
  }
});

test("页面标记缺失时初始化函数安全返回", t => {
  mount(t);
  assert.doesNotThrow(initDistributedExecution);
  assert.doesNotThrow(initActivationCheckpointing);
});

test("分布式界面在前向和反向之间更新四列表格与完整结果", t => {
  const get = mount(t, dxValues), trace = tensorParallelTrace();
  initDistributedExecution();
  for (const stage of ["forward", "backward", "forward"]) {
    get("dx-stage").value = stage;
    get("dx-stage").emit(stage === "backward" ? "change" : "input");
    const key = stage === "backward" ? "dX" : "Y";
    const rows = tableRows(get("dx-ranks").innerHTML);
    assert.equal(rows.length, 2);
    rows.forEach((row, index) => {
      const values = cells(row), rank = trace.ranks[index];
      assert.equal(values.length, 4);
      assert.equal(values[0], `进程 ${index}`);
      matrixClose(readMatrix(values[1]), rank.W1, 0.00005);
      matrixClose(readMatrix(values[2]), stage === "backward" ? rank.dZ : rank.H, 0.00005);
      matrixClose(readMatrix(values[3]), rank[key], 0.00005);
    });
    matrixClose(readMatrix(get("dx-reference").textContent), trace.reference[key], 0.00005);
    matrixClose(readMatrix(get("dx-combined").textContent), trace[key], 0.00005);
    assert.ok(Number(get("dx-error").textContent) < 1e-12);
    assert.match(get("dx-note").textContent, stage === "backward" ? /链式法则.*按列拼接.*按行拼接/ : /分列.*分行/);
  }
});

test("集合通信选择器渲染三列真实输入和输出", t => {
  const get = mount(t, dxValues);
  initDistributedExecution();
  for (const [index, [mode, output]] of Object.entries(expectedCollectives).entries()) {
    get("dx-collective").value = mode;
    get("dx-collective").emit(index % 2 ? "change" : "input");
    const rows = tableRows(get("dx-collective-rows").innerHTML);
    assert.equal(rows.length, 2);
    rows.forEach((row, rank) => {
      const values = cells(row);
      assert.equal(values.length, 3);
      assert.equal(values[0], `进程 ${rank}`);
      assert.deepEqual(JSON.parse(values[1]), rank === 0 ? [1, 2, 3, 4] : [10, 20, 30, 40]);
      assert.deepEqual(JSON.parse(values[2]), output[rank]);
    });
  }
});

test("激活界面从零步开始，逐事件更新计数并正确结束和复位", t => {
  const get = mount(t, acValues), trace = activationTrace();
  initActivationCheckpointing();
  const checkInitial = () => {
    assert.equal(get("ac-step").textContent, `0/${trace.events.length - 1}`);
    assert.equal(get("ac-forward").textContent, "0");
    assert.equal(get("ac-peak").textContent, "1");
    assert.equal(get("ac-loss").textContent, "尚未计算");
    assert.equal(get("ac-error").textContent, "尚未完成反向");
    assert.deepEqual(descendants(parseFragment(get("ac-live").innerHTML), "span").map(content), ["h0"]);
    assert.equal(tableRows(get("ac-events").innerHTML).length, 1);
    assert.equal(get("ac-next").disabled, false);
    assert.equal(get("ac-segment").disabled, true);
  };
  checkInitial();
  for (let step = 1; step < trace.events.length; step++) {
    get("ac-next").emit("click");
    const event = trace.events[step], shown = trace.events.slice(0, step + 1);
    assert.equal(get("ac-step").textContent, `${step}/${trace.events.length - 1}`);
    assert.equal(Number(get("ac-forward").textContent), event.forwardCount);
    assert.equal(Number(get("ac-peak").textContent), Math.max(...shown.map(item => item.count)));
    assert.deepEqual(descendants(parseFragment(get("ac-live").innerHTML), "span").map(content), event.live);
    const rows = tableRows(get("ac-events").innerHTML);
    assert.equal(rows.length, step + 1);
    assert.ok(rows.every(row => cells(row).length === 5));
    assert.equal(Number(cells(rows.at(-1))[3]), event.count);
    assert.equal(Number(cells(rows.at(-1))[4]), event.forwardCount);
    if (!shown.some(item => item.phase === "loss")) assert.equal(get("ac-loss").textContent, "尚未计算");
    if (event.backwardCount < 6) assert.equal(get("ac-error").textContent, "尚未完成反向");
  }
  assert.equal(get("ac-next").disabled, true);
  assert.equal(get("ac-live").innerHTML, "无保留激活");
  close(Number(get("ac-loss").textContent), trace.loss, 5e-9);
  assert.equal(get("ac-error").textContent, "0");
  get("ac-next").emit("click");
  assert.equal(get("ac-step").textContent, `${trace.events.length - 1}/${trace.events.length - 1}`);
  get("ac-reset").emit("click");
  checkInitial();
});

test("模式和分段选择均重置进度，所有界面分段达到同一最终损失和梯度", t => {
  const get = mount(t, acValues);
  initActivationCheckpointing();
  get("ac-next").emit("click");
  get("ac-mode").value = "checkpoint";
  get("ac-mode").emit("change");
  assert.equal(get("ac-segment").disabled, false);
  assert.match(get("ac-step").textContent, /^0\//);
  for (const segment of [1, 2, 3, 6]) {
    get("ac-segment").value = String(segment);
    get("ac-segment").emit(segment === 2 ? "input" : "change");
    const trace = activationTrace("checkpoint", segment);
    assert.equal(get("ac-step").textContent, `0/${trace.events.length - 1}`);
    assert.equal(get("ac-forward").textContent, "0");
    for (let step = 1; step < trace.events.length; step++) get("ac-next").emit("click");
    assert.equal(Number(get("ac-peak").textContent), trace.peak);
    assert.equal(Number(get("ac-forward").textContent), 12);
    close(Number(get("ac-loss").textContent), scalarLoss(weights), 5e-9);
    assert.equal(get("ac-error").textContent, "0");
    assert.equal(get("ac-next").disabled, true);
    assert.match(get("ac-note").textContent, /完整重算.*简化.*复用.*逻辑激活.*不代表脚本实际内存/);
  }
  get("ac-mode").value = "store";
  get("ac-mode").emit("input");
  assert.match(get("ac-step").textContent, /^0\//);
  assert.equal(get("ac-segment").disabled, true);
  assert.equal(get("ac-next").disabled, false);
});
