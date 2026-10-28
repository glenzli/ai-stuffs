import assert from "node:assert/strict";
import { test } from "node:test";
import { parseFragment } from "parse5";
import { interferenceStep, projectConflict, initContinualLearning } from "../src/scripts/features/continual-learning.js";
import { matrixDirections, initMatrixOptimizer } from "../src/scripts/features/matrix-optimizer.js";

const close = (actual, expected, tolerance = 1e-10) => assert.ok(Math.abs(actual - expected) <= tolerance, `${actual} != ${expected}`);
const vectorClose = (actual, expected, tolerance) => {
  assert.equal(actual.length, expected.length);
  actual.forEach((value, index) => close(value, expected[index], tolerance));
};
const matrixClose = (actual, expected, tolerance) => {
  assert.equal(actual.length, expected.length);
  actual.forEach((row, index) => vectorClose(row, expected[index], tolerance));
};
const dot = (a, b) => a.reduce((sum, value, index) => sum + value * b[index], 0);
const loss = (theta, target) => 0.5 * theta.reduce((sum, value, index) => sum + (value - target[index]) ** 2, 0);
const norm = matrix => Math.hypot(...matrix.flat());
const invalidNumbers = [NaN, Infinity, -Infinity, "1", null, true, {}, [], 1n];
const descendants = (node, tag) => (node.tagName === tag ? [node] : []).concat((node.childNodes || []).flatMap(child => descendants(child, tag)));
const textContent = node => node.nodeName === "#text" ? node.value : (node.childNodes || []).map(textContent).join("");
const tableRows = html => descendants(parseFragment(`<table><tbody>${html}</tbody></table>`), "tr");
const rowCells = row => (row.childNodes || []).filter(node => ["th", "td"].includes(node.tagName)).map(textContent);

function mount(t, values = {}, { canvas = false } = {}) {
  const nodes = new Map(Object.entries(values).map(([key, value]) => {
    const listeners = new Map();
    return [`[data-${key}]`, {
      value: String(value), textContent: "", innerHTML: "",
      addEventListener(type, listener) {
        if (!listeners.has(type)) listeners.set(type, []);
        listeners.get(type).push(listener);
      },
      emit(type) { for (const listener of listeners.get(type) || []) listener(); },
    }];
  }));
  const calls = [], observers = [], root = {}, style = new Map([
    ["--surface-strong", "#ffffff"], ["--ink", "#20272d"], ["--ink-faint", "#6c7882"],
    ["--line", "#dce2e6"], ["--line-strong", "#aab6bf"], ["--purple", "#7452ad"],
    ["--blue", "#355bc2"], ["--coral", "#d95f36"], ["--sans", "sans-serif"],
  ]);
  const ctx = {};
  for (const method of ["save", "restore", "fillRect", "fillText", "beginPath", "moveTo", "lineTo", "stroke", "arc", "fill", "rect", "clip", "setLineDash", "closePath"]) {
    ctx[method] = (...args) => calls.push({ method, args, fill: ctx.fillStyle, stroke: ctx.strokeStyle });
  }
  if (canvas) nodes.set("[data-cl-canvas]", { getContext: type => type === "2d" ? ctx : null });
  const globals = {
    document: { documentElement: root, querySelector: selector => nodes.get(selector) || null },
    getComputedStyle: () => ({ getPropertyValue: name => style.get(name) || "" }),
    MutationObserver: class {
      constructor(callback) { this.callback = callback; observers.push(this); }
      observe(target, options) { this.target = target; this.options = options; }
    },
  };
  for (const [key, value] of Object.entries(globals)) {
    const previous = Object.getOwnPropertyDescriptor(globalThis, key);
    Object.defineProperty(globalThis, key, { configurable: true, value });
    t.after(() => {
      if (previous) Object.defineProperty(globalThis, key, previous);
      else delete globalThis[key];
    });
  }
  return { get: key => nodes.get(`[data-${key}]`), calls, observers, root, style };
}

const clValues = {
  "cl-angle": 120, "cl-weight": 50, "cl-lr": 50, "cl-mode": "joint",
  "cl-angle-value": "", "cl-weight-value": "", "cl-lr-value": "", "cl-cos": "",
  "cl-loss-a": "", "cl-loss-b": "", "cl-rows": "", "cl-note": "",
};
const muValues = { "mu-ratio": 8, "mu-ratio-value": "", "mu-rows": "", "mu-note": "" };

test("default quadratics return original gradients, the weighted update, and actual losses", () => {
  const result = interferenceStep();
  assert.deepEqual(interferenceStep({}), result);
  assert.deepEqual(result.gA, [-1, 0]);
  assert.deepEqual(result.targetA, [1, 0]);
  vectorClose(result.targetB, [-0.5, Math.sqrt(3) / 2]);
  vectorClose(result.gB, [0.5, -Math.sqrt(3) / 2]);
  vectorClose(result.g, [-0.25, -Math.sqrt(3) / 4]);
  vectorClose(result.theta, [0.125, Math.sqrt(3) / 8]);
  close(result.cosine, -0.5);
  for (const task of ["A", "B"]) {
    assert.equal(result[`old${task}`], 0.5);
    close(result[`linear${task}`], -0.125);
    close(result[`new${task}`], 0.40625);
    close(result[`delta${task}`], -0.09375);
    assert.equal(result[`delta${task}`], result[`new${task}`] - result[`old${task}`]);
  }
});

test("finite differences recover both initial gradients and first-order loss changes", () => {
  const h = 1e-5;
  for (const angle of [0, 30, 90, 120, 180]) {
    for (const mode of ["joint", "projected"]) {
      const result = interferenceStep({ angle, weight: 0.8, lr: 0.4, mode });
      for (const task of ["A", "B"]) {
        const target = result[`target${task}`], gradient = result[`g${task}`];
        for (let axis = 0; axis < 2; axis++) {
          const positive = [0, 0], negative = [0, 0];
          positive[axis] = h; negative[axis] = -h;
          close((loss(positive, target) - loss(negative, target)) / (2 * h), gradient[axis], 1e-9);
        }
        const positive = result.g.map(value => -h * value), negative = result.g.map(value => h * value);
        const directional = (loss(positive, target) - loss(negative, target)) / (2 * h);
        close(result[`linear${task}`], 0.4 * directional, 1e-9);
      }
    }
  }
});

test("the actual quadratic remainder is exact across modes, weights, angles, and step sizes", () => {
  for (const angle of [0, 25.5, 90, 120, 175, 180]) {
    for (const weight of [0, 0.2, 0.5, 0.8, 1]) {
      for (const lr of [0, 0.01, 0.5, 2.5, 3]) {
        for (const mode of ["joint", "projected"]) {
          const result = interferenceStep({ angle, weight, lr, mode });
          const remainder = 0.5 * lr ** 2 * dot(result.g, result.g);
          for (const task of ["A", "B"]) {
            close(result[`new${task}`], loss(result.theta, result[`target${task}`]));
            close(result[`delta${task}`] - result[`linear${task}`], remainder);
            close(result[`delta${task}`], lr * dot(result[`target${task}`], result.g) + remainder);
            assert.ok(result[`new${task}`] >= 0);
          }
          close(result.cosine, dot(result.gA, result.gB));
          if (mode === "joint") vectorClose(result.g, result.gA.map((value, i) => (1 - weight) * value + weight * result.gB[i]));
        }
      }
    }
  }
});

test("opposite gradients cancel exactly and zero learning rate preserves the origin", () => {
  for (const mode of ["joint", "projected"]) {
    const result = interferenceStep({ angle: 180, mode });
    assert.deepEqual(result.g, [0, 0]);
    assert.deepEqual(result.theta, [0, 0]);
    assert.equal(result.cosine, -1);
    assert.equal(result.newA, 0.5); assert.equal(result.newB, 0.5);
    assert.equal(result.deltaA, 0); assert.equal(result.deltaB, 0);
  }
  for (const weight of [0, 0.5, 1]) {
    assert.deepEqual(interferenceStep({ angle: 180, weight, mode: "projected" }).g, [0, 0]);
    assert.deepEqual(interferenceStep({ angle: 90, weight, lr: 0 }).theta, [0, 0]);
  }
  assert.deepEqual(interferenceStep({ angle: 90 }).gB, [0, -1]);
  assert.equal(interferenceStep({ angle: 90 }).cosine, 0);
});

test("symmetric conflict projection uses both original vectors and their own denominators", () => {
  const gA = [2, 0], gB = [-1, 1];
  const result = projectConflict(gA, gB);
  assert.deepEqual(result, { a: [1, 1], b: [0, 1] });
  assert.deepEqual(gA, [2, 0]); assert.deepEqual(gB, [-1, 1]);
  close(dot(result.a, gB), 0); close(dot(result.b, gA), 0);
  const scaled = projectConflict(gA.map(value => value * 8), gB.map(value => value * 0.25));
  vectorClose(scaled.a, result.a.map(value => value * 8));
  vectorClose(scaled.b, result.b.map(value => value * 0.25));
  assert.deepEqual(projectConflict([3, 4], [-6, -8]), { a: [0, 0], b: [0, 0] });
});

test("zero and nonconflicting vectors are preserved without mutation or shared arrays", () => {
  for (const [gA, gB] of [[[0, 0], [3, -4]], [[3, -4], [0, 0]], [[0, 0], [0, 0]], [[1, 0], [0, 2]], [[1, 2], [2, 1]]]) {
    const result = projectConflict(gA, gB);
    assert.deepEqual(result, { a: gA, b: gB });
    assert.notEqual(result.a, gA); assert.notEqual(result.b, gB);
  }
});

test("projected mode keeps original cosine while removing conflicting first-order changes", () => {
  for (const angle of [100, 120, 150, 180]) {
    for (const weight of [0, 0.2, 0.5, 0.8, 1]) {
      const joint = interferenceStep({ angle, weight, lr: 0.01 });
      const projected = interferenceStep({ angle, weight, lr: 0.01, mode: "projected" });
      assert.equal(projected.cosine, joint.cosine);
      assert.deepEqual(projected.gA, joint.gA); assert.deepEqual(projected.gB, joint.gB);
      assert.ok(projected.linearA <= 1e-12 && projected.linearB <= 1e-12);
    }
  }
  const joint = interferenceStep({ angle: 120, weight: 1, lr: 0.1 });
  const projected = interferenceStep({ angle: 120, weight: 1, lr: 0.1, mode: "projected" });
  assert.ok(joint.deltaA > 0 && joint.deltaB < 0);
  close(projected.linearA, 0); assert.ok(projected.deltaA > 0);
  for (const angle of [0, 60, 90]) assert.deepEqual(interferenceStep({ angle, mode: "projected" }), interferenceStep({ angle }));
});

test("large learning rates can increase actual loss despite a negative linear prediction", () => {
  for (const mode of ["joint", "projected"]) {
    const result = interferenceStep({ angle: 0, lr: 3, mode });
    assert.deepEqual(result.theta, [3, 0]);
    assert.equal(result.linearA, -3); assert.equal(result.linearB, -3);
    assert.equal(result.newA, 2); assert.equal(result.newB, 2);
    assert.equal(result.deltaA, 1.5); assert.equal(result.deltaB, 1.5);
  }
  const projected = interferenceStep({ angle: 120, weight: 0, lr: 3, mode: "projected" });
  assert.ok(projected.linearA < 0 && projected.deltaA > 0);
});

test("interference inputs and mode reject invalid values while accepting exact boundaries", () => {
  for (const [key, max] of [["angle", 180], ["weight", 1], ["lr", 3]]) {
    for (const value of [...invalidNumbers, -0.01, max + 0.01]) assert.throws(() => interferenceStep({ [key]: value }), RangeError);
    for (const value of [0, max]) assert.doesNotThrow(() => interferenceStep({ [key]: value }));
  }
  for (const mode of ["", "pcgrad", "muon", null, false, 1, {}]) assert.throws(() => interferenceStep({ mode }), RangeError);
  for (const [a, b] of [[[], []], [[1], [1, 2]], [null, [1]], [[1], "1"], [[NaN], [1]], [[Infinity], [1]], [["1"], [1]], [Array(2), [1, 2]], [[1e308], [-1e308]]]) {
    assert.throws(() => projectConflict(a, b), RangeError);
  }
});

test("known rotations reconstruct the matrix and its singular axes exactly", () => {
  const alpha = Math.PI / 6, beta = Math.PI / 9;
  const ca = Math.cos(alpha), sa = Math.sin(alpha), cb = Math.cos(beta), sb = Math.sin(beta);
  for (const ratio of [1, 1.5, 8, 32]) {
    const result = matrixDirections(ratio);
    assert.deepEqual(result.singulars, [ratio, 1]);
    matrixClose(result.raw, [[ratio * ca * cb + sa * sb, ratio * ca * sb - sa * cb], [ratio * sa * cb - ca * sb, ratio * sa * sb + ca * cb]]);
    const apply = vector => result.raw.map(row => dot(row, vector));
    vectorClose(apply([cb, sb]), [ratio * ca, ratio * sa]);
    vectorClose(apply([-sb, cb]), [-sa, ca]);
    close(norm(result.raw), Math.hypot(ratio, 1));
    close(result.raw[0][0] * result.raw[1][1] - result.raw[0][1] * result.raw[1][0], ratio);
  }
});

test("the raw polar factor is orthogonal and independent of the positive singular ratio", () => {
  const angle = Math.PI / 18, c = Math.cos(angle), s = Math.sin(angle);
  const expected = [[c, -s], [s, c]];
  for (const ratio of [1, 2, 8, 32]) {
    const result = matrixDirections(ratio);
    matrixClose(result.polar, expected);
    close(norm(result.polar), Math.sqrt(2));
    for (let i = 0; i < 2; i++) {
      for (let j = 0; j < 2; j++) close(dot(result.polar[i], result.polar[j]), i === j ? 1 : 0);
    }
    close(result.orthogonalityError, 0, 1e-14);
    matrixClose(result.polarDirection, expected.map(row => row.map(value => value / Math.sqrt(2))));
  }
  matrixClose(matrixDirections(1).sgdDirection, matrixDirections(1).polarDirection);
});

test("all directions have unit Frobenius norm while preserving distinct scales and entry signs", () => {
  assert.deepEqual(matrixDirections(), matrixDirections(8));
  for (const ratio of [1, 1.5, 8, 32]) {
    const result = matrixDirections(ratio);
    for (const key of ["sgdDirection", "adamDirection", "polarDirection"]) {
      close(norm(result[key]), 1);
      assert.ok(result[key].flat().every(Number.isFinite));
    }
    matrixClose(result.sgdDirection, result.raw.map(row => row.map(value => value / Math.hypot(ratio, 1))));
    const elementwise = result.raw.map(row => row.map(value => value / (Math.abs(value) + 1e-8)));
    matrixClose(result.adamDirection, elementwise.map(row => row.map(value => value / norm(elementwise))));
    result.raw.flat().forEach((value, index) => assert.equal(Math.sign(result.adamDirection.flat()[index]), Math.sign(value)));
  }
  const result = matrixDirections(8);
  result.adamDirection.flat().forEach(value => close(value, 0.5, 1e-8));
  assert.ok(norm(result.sgdDirection.map((row, i) => row.map((value, j) => value - result.polarDirection[i][j]))) > 0.5);
});

test("first-step elementwise Adam includes epsilon near a zero entry", () => {
  const zeroRatio = Math.tan(Math.PI / 6) / Math.tan(Math.PI / 9);
  const result = matrixDirections(zeroRatio + 1e-9);
  const entry = result.raw[0][1];
  assert.ok(entry > 0 && entry < 1e-8);
  const first = entry / (entry + 1e-8);
  const elementwise = result.raw.flat().map(value => value / (Math.abs(value) + 1e-8));
  close(result.adamDirection[0][1], first / Math.hypot(...elementwise), 1e-12);
  assert.ok(result.adamDirection[0][1] < 0.1);
});

test("matrix ratios reject invalid values and accept positive boundaries", () => {
  for (const ratio of [...invalidNumbers, 0, 0.99, -1, 32.01]) assert.throws(() => matrixDirections(ratio), RangeError);
  for (const ratio of [1, 1.25, 32]) assert.doesNotThrow(() => matrixDirections(ratio));
});

test("both initializers return when their page markers are absent", t => {
  mount(t);
  assert.doesNotThrow(initContinualLearning);
  assert.doesNotThrow(initMatrixOptimizer);
});

test("Chinese continual-learning controls render four loss columns and update both modes", t => {
  const { get } = mount(t, clValues);
  initContinualLearning();
  assert.equal(get("cl-angle-value").textContent, "120°");
  assert.equal(get("cl-weight-value").textContent, "50%");
  assert.equal(get("cl-lr-value").textContent, "0.50");
  assert.equal(get("cl-cos").textContent, "-0.5000");
  assert.equal(get("cl-loss-a").textContent, "0.4063");
  assert.equal(get("cl-loss-b").textContent, "0.4063");
  const rows = tableRows(get("cl-rows").innerHTML);
  assert.deepEqual(rows.map(rowCells), [["任务 A", "0.5000", "0.3750", "0.4063"], ["任务 B", "0.5000", "0.3750", "0.4063"]]);
  assert.ok(descendants(rows[0], "th").every(node => node.attrs.some(attr => attr.name === "scope" && attr.value === "row")));
  get("cl-weight").value = "100"; get("cl-weight").emit("input");
  assert.equal(get("cl-weight-value").textContent, "100%");
  assert.equal(get("cl-loss-a").textContent, "0.8750");
  get("cl-mode").value = "projected"; get("cl-mode").emit("change");
  assert.equal(get("cl-cos").textContent, "-0.5000");
  assert.equal(get("cl-loss-a").textContent, "0.5938");
  assert.equal(get("cl-loss-b").textContent, "0.2188");
  assert.match(get("cl-note").textContent, /PCGrad 双任务对称局部示意/);
  get("cl-angle").value = "150"; get("cl-weight").value = "0"; get("cl-angle").emit("input");
  assert.equal(get("cl-loss-b").textContent, "0.5313");
  assert.doesNotMatch(get("cl-note").textContent, /步长过大/);
  get("cl-angle").value = "180"; get("cl-angle").emit("input");
  assert.equal(get("cl-loss-a").textContent, "0.5000");
  assert.match(get("cl-note").textContent, /合成梯度为零/);
  get("cl-angle").value = "0"; get("cl-lr").value = "250"; get("cl-lr").emit("input");
  assert.equal(get("cl-lr-value").textContent, "2.50");
  assert.equal(get("cl-loss-a").textContent, "1.1250");
  assert.match(get("cl-note").textContent, /线性预测下降.*实际损失上升/);
  get("cl-lr").value = "1"; get("cl-lr").emit("change");
  assert.equal(get("cl-lr-value").textContent, "0.01");
});

test("the continual-learning mode selector is optional and defaults to joint", t => {
  const values = { ...clValues }; delete values["cl-mode"];
  const { get } = mount(t, values);
  assert.doesNotThrow(initContinualLearning);
  assert.equal(get("cl-loss-a").textContent, "0.4063");
});

test("canvas coordinates remain isotropic, draw the update, and read colors again after theme changes", t => {
  const { get, calls, observers, root, style } = mount(t, clValues, { canvas: true });
  initContinualLearning();
  assert.equal(get("cl-canvas").width, 640); assert.equal(get("cl-canvas").height, 440);
  const arcs = calls.filter(call => call.method === "arc");
  assert.equal(arcs.length, 10);
  vectorClose(arcs.slice(0, 3).map(call => call.args[2]), [35.2, 70.4, 105.6]);
  vectorClose(arcs.slice(3, 6).map(call => call.args[2]), [35.2, 70.4, 105.6]);
  vectorClose(arcs[6].args.slice(0, 2), [320, 220]);
  vectorClose(arcs[7].args.slice(0, 2), [390.4, 220]);
  vectorClose(arcs[8].args.slice(0, 2), [284.8, 220 - Math.sqrt(3) / 2 * 70.4]);
  vectorClose(arcs[9].args.slice(0, 2), [328.8, 220 - Math.sqrt(3) / 8 * 70.4]);
  assert.ok(calls.some(call => call.method === "stroke" && call.stroke === "#d95f36"));
  assert.ok(calls.some(call => call.method === "setLineDash" && call.args[0].length === 2));
  assert.equal(observers.length, 1); assert.equal(observers[0].target, root);
  assert.ok(observers[0].options.attributeFilter.includes("data-theme"));
  const before = calls.length;
  style.set("--surface-strong", "#25292d"); style.set("--coral", "#ff8f68");
  observers[0].callback();
  assert.ok(calls.slice(before).some(call => call.method === "fillRect" && call.fill === "#25292d"));
  assert.ok(calls.slice(before).some(call => call.method === "stroke" && call.stroke === "#ff8f68"));
  get("cl-angle").value = "180"; get("cl-angle").emit("input");
  const last = calls.filter(call => call.method === "arc").at(-1);
  vectorClose(last.args.slice(0, 2), [320, 220]);
  for (const call of calls) assert.ok(call.args.filter(value => typeof value === "number").every(Number.isFinite));
});

test("Chinese matrix controls render five columns in row-major order and recompute on input", t => {
  const { get } = mount(t, muValues);
  initMatrixOptimizer();
  assert.equal(get("mu-ratio-value").textContent, "8");
  const rows = tableRows(get("mu-rows").innerHTML);
  assert.equal(rows.length, 4); assert.ok(rows.every(row => rowCells(row).length === 5));
  assert.deepEqual(rows.map(row => rowCells(row)[0]), ["(1, 1)", "(1, 2)", "(2, 1)", "(2, 2)"]);
  const result = matrixDirections();
  rows.forEach((row, index) => {
    const i = Math.floor(index / 2), j = index % 2;
    assert.deepEqual(rowCells(row).slice(1), [result.raw[i][j], result.sgdDirection[i][j], result.adamDirection[i][j], result.polarDirection[i][j]].map(value => value.toFixed(4)));
  });
  assert.match(get("mu-note").textContent, /首步 AdamW.*1e-8.*忽略权重衰减/);
  assert.match(get("mu-note").textContent, /仅比较方向/);
  const polarColumn = rows.map(row => rowCells(row)[4]);
  get("mu-ratio").value = "1"; get("mu-ratio").emit("input");
  const equalRows = tableRows(get("mu-rows").innerHTML);
  assert.equal(get("mu-ratio-value").textContent, "1");
  assert.deepEqual(equalRows.map(row => rowCells(row)[4]), polarColumn);
  equalRows.forEach(row => assert.equal(rowCells(row)[2], rowCells(row)[4]));
  assert.ok(Number(rowCells(equalRows[1])[3]) < 0);
  get("mu-ratio").value = "32"; get("mu-ratio").emit("change");
  assert.equal(get("mu-ratio-value").textContent, "32");
});
