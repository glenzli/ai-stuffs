import assert from "node:assert/strict";
import { test } from "node:test";
import { parseFragment } from "parse5";
import { layoutReadout, initTensorLayout } from "../src/scripts/features/tensor-layout.js";
import { vocabParallel, initVocabParallel } from "../src/scripts/features/vocab-parallel.js";

const close = (actual, expected, tolerance = 1e-12) => {
  assert.ok(Number.isFinite(actual) && Number.isFinite(expected));
  assert.ok(Math.abs(actual - expected) <= tolerance, `${actual} versus ${expected}`);
};
const vectorClose = (actual, expected, tolerance) => {
  assert.equal(actual.length, expected.length);
  actual.forEach((value, index) => close(value, expected[index], tolerance));
};
const initialStorage = Array.from({ length: 12 }, (_, index) => index);
const layoutCases = {
  base: {
    shape: [3, 4], strides: [4, 1], offset: 0, contiguous: true, flatStride: 1,
    addresses: [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]],
  },
  transpose: {
    shape: [4, 3], strides: [1, 4], offset: 0, contiguous: false, flatStride: null,
    addresses: [[0, 4, 8], [1, 5, 9], [2, 6, 10], [3, 7, 11]],
  },
  slice: {
    shape: [3, 2], strides: [4, 2], offset: 0, contiguous: false, flatStride: 2,
    addresses: [[0, 2], [4, 6], [8, 10]],
  },
  offset: {
    shape: [2, 4], strides: [4, 1], offset: 4, contiguous: true, flatStride: 1,
    addresses: [[4, 5, 6, 7], [8, 9, 10, 11]],
  },
};
const logits = [1.2, -0.5, 2, 0.1, 1.4, -1];
const shifts = [-1000, -123.5, 0, 123.5, 1000];

function stableLoss(scores, target) {
  const maximum = Math.max(...scores);
  const total = scores.reduce((sum, score) => sum + Math.exp(score - maximum), 0);
  return Math.log(total) - (scores[target] - maximum);
}

function finiteGradients(scores, target) {
  const epsilon = 1e-5;
  return scores.map((_, token) => {
    const positive = scores.slice(), negative = scores.slice();
    positive[token] += epsilon;
    negative[token] -= epsilon;
    return (stableLoss(positive, target) - stableLoss(negative, target)) / (2 * epsilon);
  });
}

const descendants = (node, tag) => (node.tagName === tag ? [node] : []).concat(
  (node.childNodes || []).flatMap(child => descendants(child, tag))
);
const content = node => node.nodeName === "#text" ? node.value : (node.childNodes || []).map(content).join("");
const attribute = (node, name) => node.attrs?.find(attr => attr.name === name)?.value ?? "";
const tableCells = html => descendants(parseFragment(`<table><tbody>${html}</tbody></table>`), "tr").map(row =>
  (row.childNodes || []).filter(node => node.tagName === "td").map(content)
);
const spans = html => parseFragment(html).childNodes.filter(node => node.tagName === "span");

function mount(t, values = {}) {
  const nodes = new Map(Object.entries(values).map(([key, value]) => {
    const listeners = new Map();
    return [`[data-${key}]`, {
      value: String(value), checked: value === true, textContent: "", innerHTML: "", style: {},
      addEventListener(type, listener) {
        if (!listeners.has(type)) listeners.set(type, []);
        listeners.get(type).push(listener);
      },
      emit(type) { for (const listener of listeners.get(type) || []) listener(); },
    }];
  }));
  const previous = Object.getOwnPropertyDescriptor(globalThis, "document");
  Object.defineProperty(globalThis, "document", {
    configurable: true, value: { querySelector: selector => nodes.get(selector) || null },
  });
  t.after(() => {
    if (previous) Object.defineProperty(globalThis, "document", previous);
    else delete globalThis.document;
  });
  return key => nodes.get(`[data-${key}]`);
}

const tlValues = {
  "tl-mode": "transpose", "tl-materialize": false, "tl-write": false,
  "tl-shape": "", "tl-strides": "", "tl-offset": "", "tl-shared": "", "tl-view": "", "tl-note": "",
  "tl-storage": "", "tl-rows": "", "tl-matrix": "",
};
const vpValues = {
  "vp-shards": 2, "vp-target": 4, "vp-offset": 0, "vp-offset-value": "", "vp-max": "",
  "vp-denom": "", "vp-loss": "", "vp-target-owner": "", "vp-error": "", "vp-note": "",
  "vp-shards-rows": "", "vp-grad-rows": "",
};

test("四种固定布局保留逻辑顺序、物理地址与原始存储共享", () => {
  assert.deepEqual(layoutReadout(), layoutReadout({ mode: "transpose" }));
  for (const [mode, expected] of Object.entries(layoutCases)) {
    const result = layoutReadout({ mode });
    assert.deepEqual(result.shape, expected.shape);
    assert.deepEqual(result.strides, expected.strides);
    assert.equal(result.offset, expected.offset);
    assert.deepEqual(result.addresses, expected.addresses);
    assert.deepEqual(result.values, expected.addresses);
    assert.deepEqual(result.baseStorage, initialStorage);
    assert.strictEqual(result.tensorStorage, result.baseStorage);
    assert.equal(result.sharesStorage, true);
    assert.equal(result.contiguous, expected.contiguous);
    assert.equal(result.flatViewPossible, expected.flatStride !== null);
    assert.equal(result.flatStride, expected.flatStride);
    assert.deepEqual(result.flatValues, expected.addresses.flat());
  }
});

test("非连续切片仍可零复制展平，非零起始偏移不影响连续性", () => {
  const slice = layoutReadout({ mode: "slice" });
  assert.equal(slice.contiguous, false);
  assert.equal(slice.flatViewPossible, true);
  assert.equal(slice.flatStride, 2);
  assert.deepEqual(slice.addresses.flat(), [0, 2, 4, 6, 8, 10]);
  const offset = layoutReadout({ mode: "offset" });
  assert.equal(offset.offset, 4);
  assert.equal(offset.contiguous, true);
  assert.equal(offset.flatStride, 1);
  const transpose = layoutReadout();
  assert.equal(transpose.flatViewPossible, false);
  assert.equal(transpose.flatStride, null);
});

test("每种共享视图写入逻辑首元素时只改变对应原始地址", () => {
  for (const [mode, expected] of Object.entries(layoutCases)) {
    const result = layoutReadout({ mode, write: true });
    const storage = initialStorage.slice();
    storage[expected.offset] = 99;
    assert.deepEqual(result.baseStorage, storage);
    assert.strictEqual(result.tensorStorage, result.baseStorage);
    assert.equal(result.sharesStorage, true);
    assert.deepEqual(result.addresses, expected.addresses);
    assert.deepEqual(result.values, expected.addresses.map(row => row.map(address => storage[address])));
    assert.deepEqual(result.flatValues, result.values.flat());
    assert.equal(result.flatStride, expected.flatStride);
  }
});

test("所有布局的显式复制按逻辑顺序生成独立连续存储，写入不改变原始存储", () => {
  for (const [mode, expected] of Object.entries(layoutCases)) {
    for (const write of [false, true]) {
      const result = layoutReadout({ mode, materialize: true, write });
      const values = expected.addresses.map(row => row.slice());
      if (write) values[0][0] = 99;
      assert.deepEqual(result.shape, expected.shape);
      assert.deepEqual(result.strides, [expected.shape[1], 1]);
      assert.equal(result.offset, 0);
      assert.deepEqual(result.addresses, Array.from({ length: expected.shape[0] }, (_, i) =>
        Array.from({ length: expected.shape[1] }, (_, j) => i * expected.shape[1] + j)
      ));
      assert.deepEqual(result.values, values);
      assert.deepEqual(result.tensorStorage, values.flat());
      assert.deepEqual(result.baseStorage, initialStorage);
      assert.notStrictEqual(result.tensorStorage, result.baseStorage);
      assert.equal(result.sharesStorage, false);
      assert.equal(result.contiguous, true);
      assert.equal(result.flatViewPossible, true);
      assert.equal(result.flatStride, 1);
      assert.deepEqual(result.flatValues, values.flat());
    }
  }
});

test("布局结果与写入不污染下次调用，模式及布尔选项有明确输入契约", () => {
  for (const mode of Object.keys(layoutCases)) {
    const result = layoutReadout({ mode, write: true });
    result.shape[0] = 99;
    result.strides[0] = 99;
    result.baseStorage[11] = 99;
    assert.deepEqual(layoutReadout({ mode }).shape, layoutCases[mode].shape);
    assert.deepEqual(layoutReadout({ mode }).strides, layoutCases[mode].strides);
    assert.deepEqual(layoutReadout({ mode }).baseStorage, initialStorage);
  }
  for (const mode of [null, "", "unknown", "__proto__", "constructor", {}, 0]) {
    assert.throws(() => layoutReadout({ mode }), RangeError);
  }
  for (const key of ["materialize", "write"]) {
    for (const value of [null, 0, 1, "true", {}]) {
      assert.throws(() => layoutReadout({ [key]: value }), TypeError);
    }
  }
});

test("全部分片数、目标与偏移重建稳定交叉熵、归一化概率及梯度", () => {
  assert.deepEqual(vocabParallel(), vocabParallel({ shards: 2, target: 4, offset: 0 }));
  for (const shards of [1, 2, 3]) {
    for (let target = 0; target < 6; target++) {
      const baseline = vocabParallel({ shards, target });
      for (const offset of shifts) {
        const scores = logits.map(value => value + offset);
        const result = vocabParallel({ shards, target, offset });
        const width = 6 / shards;
        assert.equal(result.rows.length, shards);
        assert.equal(result.globalMax, offset + 2);
        close(result.targetShifted, scores[target] - result.globalMax);
        close(result.loss, stableLoss(scores, target));
        close(result.reference, stableLoss(scores, target));
        assert.equal(result.error, Math.abs(result.loss - result.reference));
        assert.ok(result.error < 1e-12);
        close(result.loss, baseline.loss);
        close(result.denominator, baseline.denominator);
        vectorClose(result.probabilities, baseline.probabilities);
        vectorClose(result.gradients, baseline.gradients);
        close(result.probabilities.reduce((sum, value) => sum + value, 0), 1);
        close(result.gradients.reduce((sum, value) => sum + value, 0), 0);
        assert.ok(result.probabilities.every(value => value > 0 && value < 1));
        result.rows.forEach((row, rank) => {
          assert.equal(row.rank, rank);
          assert.equal(row.start, rank * width);
          assert.equal(row.end, (rank + 1) * width);
          assert.deepEqual(row.logits, scores.slice(row.start, row.end));
          assert.equal(row.localMax, Math.max(...row.logits));
          close(row.localSum, row.logits.reduce((sum, value) => sum + Math.exp(value - result.globalMax), 0));
          assert.equal(row.targetContribution, target >= row.start && target < row.end ? scores[target] - result.globalMax : 0);
          vectorClose(row.probabilities, row.logits.map((_, index) => Math.exp(-stableLoss(scores, row.start + index))));
          vectorClose(row.gradients, row.probabilities.map((value, index) => value - (row.start + index === target ? 1 : 0)));
        });
        close(result.denominator, result.rows.reduce((sum, row) => sum + row.localSum, 0));
        close(result.targetShifted, result.rows.reduce((sum, row) => sum + row.targetContribution, 0));
        assert.deepEqual(result.rows.flatMap(row => row.probabilities), result.probabilities);
        assert.deepEqual(result.rows.flatMap(row => row.gradients), result.gradients);
      }
    }
  }
});

test("每个目标、每种分片数及全部偏移的六个梯度通过独立有限差分", () => {
  for (const shards of [1, 2, 3]) {
    for (let target = 0; target < 6; target++) {
      for (const offset of shifts) {
        const result = vocabParallel({ shards, target, offset });
        const finite = finiteGradients(logits.map(value => value + offset), target);
        vectorClose(result.gradients, finite, 1e-8);
        for (const row of result.rows) vectorClose(row.gradients, finite.slice(row.start, row.end), 1e-8);
      }
    }
  }
});

test("最大值由不同分片提供时使用全局平移，目标等于最大值时归约目标项为零", () => {
  const result = vocabParallel({ shards: 3, target: 5 });
  assert.deepEqual(result.rows.map(row => row.localMax), [1.2, 2, 1.4]);
  close(result.rows[0].localSum, Math.exp(-0.8) + Math.exp(-2.5));
  close(result.rows[2].localSum, Math.exp(-0.6) + Math.exp(-3));
  assert.deepEqual(result.rows.map(row => row.targetContribution), [0, 0, -3]);
  for (const shards of [1, 2, 3]) {
    const maximumTarget = vocabParallel({ shards, target: 2 });
    assert.equal(maximumTarget.targetShifted, 0);
    assert.ok(maximumTarget.rows.every(row => row.targetContribution === 0));
    close(maximumTarget.loss, Math.log(maximumTarget.denominator));
  }
});

test("词表并行拒绝无效分片、目标与超范围或非有限偏移", () => {
  for (const shards of [0, -1, 4, 1.5, NaN, Infinity, "2", null, {}]) {
    assert.throws(() => vocabParallel({ shards }), RangeError);
  }
  for (const target of [-1, 6, 1.5, NaN, Infinity, "4", null, {}]) {
    assert.throws(() => vocabParallel({ target }), RangeError);
  }
  for (const offset of [-1001, 1001, NaN, Infinity, -Infinity, "0", null, {}]) {
    assert.throws(() => vocabParallel({ offset }), RangeError);
  }
});

test("页面控件缺失时两个初始化函数安全返回", t => {
  mount(t);
  assert.doesNotThrow(initTensorLayout);
  assert.doesNotThrow(initVocabParallel);
});

test("布局界面更新十二个原始存储格、四列表格、矩阵列数和共享写入说明", t => {
  const get = mount(t, tlValues);
  const check = () => {
    const result = layoutReadout({ mode: get("tl-mode").value, materialize: get("tl-materialize").checked, write: get("tl-write").checked });
    assert.equal(get("tl-shape").textContent, `[${result.shape.join(", ")}]`);
    assert.equal(get("tl-strides").textContent, `[${result.strides.join(", ")}]`);
    assert.equal(get("tl-offset").textContent, String(result.offset));
    assert.equal(get("tl-shared").textContent, result.sharesStorage ? "共享" : "独立副本");
    assert.equal(get("tl-view").textContent, result.flatViewPossible ? `可行（步幅 ${result.flatStride}）` : "不可行（需复制）");
    const storage = spans(get("tl-storage").innerHTML);
    assert.equal(storage.length, 12);
    storage.forEach((cell, index) => {
      assert.equal(content(cell), `地址 ${index}值 ${result.baseStorage[index]}`);
      assert.ok(attribute(cell, "title").includes(`原始存储地址 ${index}，值 ${result.baseStorage[index]}`));
      const classes = attribute(cell, "class").split(" ");
      assert.ok(classes.includes("storage-cell"));
      assert.equal(classes.includes("is-addressed"), result.sharesStorage && result.addresses.flat().includes(index));
    });
    const rows = tableCells(get("tl-rows").innerHTML);
    assert.equal(rows.length, result.shape[0] * result.shape[1]);
    rows.forEach((cells, index) => {
      const i = Math.floor(index / result.shape[1]), j = index % result.shape[1], address = result.addresses[i][j];
      assert.deepEqual(cells, [`[${i}, ${j}]`, String(address - result.offset), String(address), String(result.values[i][j])]);
    });
    assert.equal(get("tl-matrix").style.gridTemplateColumns, `repeat(${result.shape[1]}, minmax(0, 1fr))`);
    assert.deepEqual(spans(get("tl-matrix").innerHTML).map(cell => Number(content(cell))), result.flatValues);
    assert.match(get("tl-note").textContent, /[\u4e00-\u9fff]/);
    if (get("tl-mode").value === "slice" && !get("tl-materialize").checked) {
      assert.match(get("tl-note").textContent, /非连续.*步幅 2/);
    }
    if (get("tl-materialize").checked) assert.match(get("tl-note").textContent, /原始存储保持不变/);
    if (get("tl-write").checked && !get("tl-materialize").checked) {
      assert.ok(get("tl-note").textContent.includes(`原始存储地址 ${result.addresses[0][0]} 同步改变`));
    }
  };
  initTensorLayout();
  check();
  for (const [index, mode] of Object.keys(layoutCases).entries()) {
    get("tl-mode").value = mode;
    get("tl-mode").emit(index % 2 ? "input" : "change");
    check();
    get("tl-materialize").checked = true;
    get("tl-materialize").emit("change");
    check();
    get("tl-write").checked = true;
    get("tl-write").emit("input");
    check();
    get("tl-materialize").checked = false;
    get("tl-materialize").emit("input");
    check();
    get("tl-write").checked = false;
    get("tl-write").emit("change");
    check();
  }
});

test("词表界面全部控件更新六列分片表、四列梯度表与目标所属进程", t => {
  const get = mount(t, vpValues);
  const check = () => {
    const shards = Number(get("vp-shards").value), target = Number(get("vp-target").value), offset = Number(get("vp-offset").value);
    const result = vocabParallel({ shards, target, offset }), width = 6 / shards;
    assert.equal(get("vp-offset-value").textContent, String(offset));
    close(Number(get("vp-max").textContent), result.globalMax, 5e-7);
    close(Number(get("vp-denom").textContent), result.denominator, 5e-7);
    close(Number(get("vp-loss").textContent), result.loss, 5e-7);
    assert.equal(get("vp-target-owner").textContent, `进程 ${Math.floor(target / width)}`);
    assert.ok(Number(get("vp-error").textContent) < 1e-12);
    const rows = tableCells(get("vp-shards-rows").innerHTML);
    assert.equal(rows.length, shards);
    rows.forEach((cells, rank) => {
      const row = result.rows[rank];
      assert.equal(cells.length, 6);
      assert.equal(cells[0], `进程 ${rank}`);
      assert.equal(cells[1], `${row.start}-${row.end - 1}`);
      vectorClose(JSON.parse(cells[2]), row.logits, 5e-7);
      close(Number(cells[3]), row.localMax, 5e-7);
      close(Number(cells[4]), row.localSum, 5e-7);
      close(Number(cells[5]), row.targetContribution, 5e-7);
    });
    const gradients = tableCells(get("vp-grad-rows").innerHTML);
    assert.equal(gradients.length, 6);
    gradients.forEach((cells, token) => {
      assert.equal(cells.length, 4);
      assert.equal(cells[0], String(token));
      assert.equal(cells[1], `进程 ${Math.floor(token / width)}`);
      close(Number(cells[2]), result.probabilities[token], 5e-7);
      close(Number(cells[3]), result.gradients[token], 5e-7);
    });
    assert.match(get("vp-note").textContent, /全局最大值.*局部指数和.*目标项.*概率和梯度/);
    assert.match(get("vp-note").textContent, /未执行 NCCL 通信或训练/);
  };
  initVocabParallel();
  check();
  for (const shards of [1, 2, 3]) {
    get("vp-shards").value = String(shards);
    get("vp-shards").emit(shards % 2 ? "change" : "input");
    for (let target = 0; target < 6; target++) {
      get("vp-target").value = String(target);
      get("vp-target").emit(target % 2 ? "input" : "change");
      for (const [index, offset] of shifts.entries()) {
        get("vp-offset").value = String(offset);
        get("vp-offset").emit(index % 2 ? "change" : "input");
        check();
      }
    }
  }
});
