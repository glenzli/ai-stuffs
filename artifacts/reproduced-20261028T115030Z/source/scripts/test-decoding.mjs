import assert from "node:assert/strict";
import { test } from "node:test";
import { parseFragment } from "parse5";
import { decodingDistribution, sampleFrom, beamExample, initDecodingAlgorithms } from "../src/scripts/features/decoding-algorithms.js";

const tokens = ['{"ok":', "true", "false", "}", "<eos>", "junk"];
const logits = [2, 1.4, 0.8, 0.4, 0, -0.5];
const validStrings = ['{"ok":true}', '{"ok":false}'];
const uiPrefixes = ["", '{"ok":', '{"ok":true', '{"ok":true}'];
const reachablePrefixes = [...uiPrefixes, '{"ok":false', '{"ok":false}'];
const close = (actual, expected) => assert.ok(Math.abs(actual - expected) < 1e-12, `${actual} 与 ${expected} 不符`);
const keys = object => Object.keys(object).sort();
const positive = result => result.rows.filter(row => row.probability > 0);
const sum = values => values.reduce((total, value) => total + value, 0);

function referenceProbabilities(temperature, indices = [0, 1, 2, 3, 4, 5]) {
  const weights = indices.map(index => Math.exp(logits[index] / temperature));
  const total = sum(weights);
  return indices.map((index, position) => weights[position] / total);
}

test("固定 token、logits 与返回字段明确且每次独立", () => {
  const result = decodingDistribution();
  assert.deepEqual(keys(result), ["candidates", "mass", "rows", "selected"]);
  assert.deepEqual(result.rows.map(row => row.token), tokens);
  assert.deepEqual(result.rows.map(row => row.logit), logits);
  assert.deepEqual(result.candidates, tokens);
  assert.equal(result.selected, tokens[0]);
  close(result.mass, 1);
  const expected = referenceProbabilities(1);
  for (const [index, row] of result.rows.entries()) {
    assert.deepEqual(keys(row), ["allowed", "keptK", "keptP", "logit", "probBeforeTopk", "probability", "token"]);
    assert.equal(row.allowed, true);
    assert.equal(row.keptK, true);
    assert.equal(row.keptP, true);
    close(row.probBeforeTopk, expected[index]);
    close(row.probability, expected[index]);
  }
  result.rows[0].logit = 999;
  result.rows[0].probability = 0;
  result.candidates.length = 0;
  assert.equal(decodingDistribution().rows[0].logit, 2);
  assert.deepEqual(decodingDistribution().candidates, tokens);
});

test("全部温度、top-k、top-p 滑块组合与语法下拉状态保持归一化", () => {
  const states = [{ grammar: false, prefix: "" }, ...uiPrefixes.map(prefix => ({ grammar: true, prefix }))];
  for (let temperatureValue = 10; temperatureValue <= 300; temperatureValue++) {
    for (let topk = 1; topk <= 6; topk++) {
      for (let pValue = 1; pValue <= 100; pValue++) {
        for (const state of states) {
          const result = decodingDistribution({ temperature: temperatureValue / 100, topk, topp: pValue / 100, ...state });
          const mass = sum(result.rows.map(row => row.probability));
          assert.ok(Math.abs(mass - 1) < 1e-12);
          assert.ok(Math.abs(sum(result.rows.map(row => row.probBeforeTopk)) - 1) < 1e-12);
          assert.equal(result.mass, mass);
          assert.ok(result.rows.filter(row => row.keptK).length <= topk);
          assert.ok(result.rows.every(row => Number.isFinite(row.probability) && row.probability >= 0 &&
            (!row.keptP || row.keptK) && (!row.keptK || row.allowed) && (row.probability > 0) === row.keptP &&
            (row.allowed || row.probBeforeTopk === 0)));
          assert.deepEqual(result.candidates, positive(result).map(row => row.token));
          assert.equal(result.selected, positive(result).reduce((best, row) => row.probability > best.probability ? row : best).token);
        }
      }
    }
  }
});

test("温度缩放保持排序，升温降低最大概率并提高熵", () => {
  let lastMaximum = Infinity;
  let lastEntropy = -Infinity;
  for (let value = 10; value <= 300; value++) {
    const temperature = value / 100;
    const result = decodingDistribution({ temperature });
    const expected = referenceProbabilities(temperature);
    result.rows.forEach((row, index) => close(row.probBeforeTopk, expected[index]));
    assert.ok(result.rows.slice(1).every((row, index) => result.rows[index].probability > row.probability));
    const maximum = result.rows[0].probability;
    const entropy = -sum(result.rows.map(row => row.probability * Math.log(row.probability)));
    assert.ok(maximum < lastMaximum);
    assert.ok(entropy > lastEntropy);
    lastMaximum = maximum;
    lastEntropy = entropy;
  }
});

test("top-k 先重归一化，top-p 在精确等号处停止且取最小累计前缀", () => {
  for (const temperature of [0.1, 1, 3]) {
    for (let topk = 1; topk <= 6; topk++) {
      const base = decodingDistribution({ temperature, topk });
      const kept = base.rows.filter(row => row.keptK);
      const massK = sum(kept.map(row => row.probBeforeTopk));
      let boundary = 0;
      for (let count = 1; count < kept.length; count++) {
        boundary += kept[count - 1].probBeforeTopk / massK;
        const at = decodingDistribution({ temperature, topk, topp: boundary });
        assert.equal(at.rows.filter(row => row.keptP).length, count);
        const above = decodingDistribution({ temperature, topk, topp: boundary + Number.EPSILON });
        assert.equal(above.rows.filter(row => row.keptP).length, count + 1);
      }
      const all = decodingDistribution({ temperature, topk, topp: 1 });
      assert.deepEqual(all.rows.map(row => row.keptP), all.rows.map(row => row.keptK));
      for (let value = 1; value <= 100; value++) {
        const topp = value / 100;
        const result = decodingDistribution({ temperature, topk, topp });
        const nucleus = result.rows.filter(row => row.keptP);
        const nucleusMass = sum(nucleus.map(row => row.probBeforeTopk / massK));
        assert.ok(nucleusMass >= topp - 1e-12);
        if (nucleus.length > 1) assert.ok(sum(nucleus.slice(0, -1).map(row => row.probBeforeTopk / massK)) < topp);
        for (const row of nucleus) close(row.probability, (row.probBeforeTopk / massK) / nucleusMass);
      }
    }
  }
  assert.equal(decodingDistribution({ topk: 2, topp: 0.6 }).candidates.length, 1);
  assert.equal(decodingDistribution({ topk: 6, topp: 0.6 }).candidates.length, 2);
});

test("语法在截断前屏蔽，字符拼接只能保留两个固定合法字符串的前缀", () => {
  const prefixes = new Set(validStrings.flatMap(value => Array.from({ length: value.length + 1 }, (_, index) => value.slice(0, index))));
  for (const prefix of prefixes) {
    const expected = tokens.map(token => token === "<eos>"
      ? validStrings.includes(prefix)
      : validStrings.some(value => value.startsWith(prefix + token)));
    for (let topk = 1; topk <= 6; topk++) {
      for (const topp of [Number.MIN_VALUE, 0.5, 1]) {
        const result = decodingDistribution({ grammar: true, prefix, topk, topp });
        assert.deepEqual(result.rows.map(row => row.allowed), expected);
        for (const row of result.rows) {
          if (!row.allowed) {
            assert.equal(row.probBeforeTopk, 0);
            assert.equal(row.probability, 0);
            assert.equal(row.keptK, false);
            assert.equal(row.keptP, false);
          }
          if (row.token === "<eos>" && !validStrings.includes(prefix)) assert.equal(row.probability, 0);
        }
        if (expected.some(Boolean)) close(result.mass, 1);
        else {
          assert.equal(result.mass, 0);
          assert.equal(result.selected, null);
          assert.deepEqual(result.candidates, []);
          assert.throws(() => sampleFrom(result, 0), RangeError);
        }
      }
    }
  }
  assert.deepEqual(decodingDistribution({ grammar: true, prefix: '{"ok":', topk: 1 }).candidates, ["true"]);
  assert.deepEqual(decodingDistribution({ grammar: true, prefix: '{"ok":true', topk: 1 }).candidates, ["}"]);
  assert.deepEqual(decodingDistribution({ grammar: true, prefix: '{"ok":true}', topk: 1 }).candidates, ["<eos>"]);
  const values = decodingDistribution({ grammar: true, prefix: '{"ok":' });
  const expected = referenceProbabilities(1, [1, 2]);
  close(values.rows[1].probBeforeTopk, expected[0]);
  close(values.rows[2].probBeforeTopk, expected[1]);
  assert.equal(values.rows[5].allowed, false);
});

test("关闭语法时完全忽略前缀，开启时拒绝语言外前缀与非字符串", () => {
  for (const prefix of ["", "{", ...reachablePrefixes, "junk", '<script>alert(1)</script>', null, 1, false, {}, [], Symbol("prefix")]) {
    assert.deepEqual(decodingDistribution({ prefix }), decodingDistribution());
  }
  for (const prefix of ["junk", '{"ok":null}', '{"ok": true}', '{"ok":true} ', '{"ok":true}{', '{"other":false}', "<eos>", null, 1, false, {}, []]) {
    assert.throws(() => decodingDistribution({ grammar: true, prefix }), RangeError);
  }
});

test("逆累积分布采样使用原 token 顺序、左闭右开边界并排除零概率", () => {
  const rows = [
    { token: "zero-before", probability: 0 },
    { token: "first", probability: 0.25 },
    { token: "zero-middle", probability: 0 },
    { token: "second", probability: 0.5 },
    { token: "last", probability: 0.25 },
    { token: "zero-after", probability: 0 },
  ];
  const cases = [[0, "first"], [0.25 - Number.EPSILON, "first"], [0.25, "second"], [0.75 - Number.EPSILON, "second"], [0.75, "last"], [1 - Number.EPSILON, "last"]];
  for (const [u, token] of cases) {
    assert.equal(sampleFrom({ rows }, u), token);
    assert.equal(sampleFrom(rows, u), token);
    assert.equal(sampleFrom({ rows }, u), sampleFrom({ rows }, u));
  }
  const rounding = [{ token: "first", probability: 0.1 }, { token: "last", probability: 0.8999999999999998 }, { token: "zero", probability: 0 }];
  assert.equal(sampleFrom(rounding, 1 - Number.EPSILON), "last");
  const constrained = decodingDistribution({ grammar: true, prefix: '{"ok":true}' });
  for (let value = 0; value <= 999; value++) assert.equal(sampleFrom(constrained, value / 1000), "<eos>");
  const base = decodingDistribution();
  let cumulative = 0;
  for (const row of base.rows.slice(0, -1)) {
    cumulative += row.probability;
    const next = base.rows[base.rows.indexOf(row) + 1];
    assert.equal(sampleFrom(base, cumulative), next.token);
  }
});

test("结构化采样能完成两个替代字符串，结束前后均通过 JSON.parse", () => {
  for (const [u, expected] of [[0, validStrings[0]], [0.999, validStrings[1]]]) {
    let prefix = "";
    const emitted = [];
    for (let step = 0; step < 5; step++) {
      const result = decodingDistribution({ grammar: true, prefix });
      const token = sampleFrom(result, u);
      emitted.push(token);
      if (token === "<eos>") {
        assert.equal(prefix, expected);
        assert.deepEqual(JSON.parse(prefix), { ok: expected === validStrings[0] });
        break;
      }
      prefix += token;
      assert.ok(validStrings.some(value => value.startsWith(prefix)));
    }
    assert.equal(emitted.at(-1), "<eos>");
    assert.equal(prefix, expected);
    assert.deepEqual(JSON.parse(prefix), { ok: expected === validStrings[0] });
  }
});

test("解码参数拒绝越界、非有限数、错误类型和隐式转换", () => {
  for (const temperature of [0, 0.099, 3.01, NaN, Infinity, -Infinity, "1", null, {}, [], true]) {
    assert.throws(() => decodingDistribution({ temperature }), RangeError);
  }
  for (const topk of [0, 7, 1.5, NaN, Infinity, "2", null, {}, [], true]) {
    assert.throws(() => decodingDistribution({ topk }), RangeError);
  }
  for (const topp of [0, -0.1, 1.01, NaN, Infinity, -Infinity, "0.5", null, {}, [], true]) {
    assert.throws(() => decodingDistribution({ topp }), RangeError);
  }
  for (const grammar of [0, 1, "false", null, {}, []]) {
    assert.throws(() => decodingDistribution({ grammar }), RangeError);
  }
  assert.throws(() => decodingDistribution(null), TypeError);
  assert.deepEqual(decodingDistribution({ temperature: undefined, topk: undefined, topp: undefined, grammar: undefined, prefix: undefined }), decodingDistribution());
});

test("采样拒绝非法位置、无候选、未归一化分布和非法概率", () => {
  for (const u of [undefined, -0.01, 1, 1.1, NaN, Infinity, -Infinity, "0", null, {}, [], true]) {
    assert.throws(() => sampleFrom(decodingDistribution(), u), RangeError);
  }
  for (const distribution of [undefined, null, {}, { rows: {} }, [], [{ token: "A", probability: 0 }], [{ token: "A", probability: 0.5 }], [{ token: "A", probability: 0.6 }, { token: "B", probability: 0.6 }]]) {
    assert.throws(() => sampleFrom(distribution, 0), RangeError);
  }
  for (const probability of [-0.1, 1.1, NaN, Infinity, -Infinity, "1", null, undefined]) {
    assert.throws(() => sampleFrom([{ token: "A", probability }], 0), RangeError);
  }
  for (const row of [null, {}, { token: 1, probability: 1 }]) {
    assert.throws(() => sampleFrom([row], 0), RangeError);
  }
});

test("束宽一为贪心 Ax，束宽二保留 B 并选中有限树最优 Bx", () => {
  const greedy = beamExample(1);
  const beam = beamExample(2);
  assert.deepEqual(beamExample(), beam);
  assert.equal(greedy.selected, "Ax");
  close(greedy.probability, 0.306);
  assert.equal(beam.selected, "Bx");
  close(beam.probability, 0.396);
  assert.deepEqual(greedy.nodes.map(node => [node.step, node.path, node.kept]), [[1, "A", true], [1, "B", false], [2, "Ax", true], [2, "Ay", false]]);
  assert.deepEqual(beam.nodes.map(node => [node.step, node.path, node.kept]), [[1, "A", true], [1, "B", true], [2, "Ax", true], [2, "Ay", false], [2, "Bx", true], [2, "By", false]]);
  for (const result of [greedy, beam]) {
    assert.deepEqual(keys(result), ["logProbability", "nodes", "optimum", "probability", "selected"]);
    assert.deepEqual(keys(result.optimum), ["logProbability", "path", "probability"]);
    assert.equal(result.optimum.path, "Bx");
    close(result.optimum.probability, 0.396);
    close(result.logProbability, Math.log(result.probability));
    close(result.optimum.logProbability, Math.log(0.4) + Math.log(0.99));
    for (const node of result.nodes) {
      assert.deepEqual(keys(node), ["kept", "logProbability", "path", "probability", "step"]);
      close(node.logProbability, Math.log(node.probability));
      assert.equal(node.path.length, node.step);
      assert.ok(!node.path.includes("<eos>"));
    }
  }
});

test("束搜索逐步累计对数概率并做全局截断，等长叶子无需长度归一化", () => {
  const probabilities = { A: 0.6, B: 0.4, Ax: 0.6 * 0.51, Ay: 0.6 * 0.49, Bx: 0.4 * 0.99, By: 0.4 * 0.01 };
  for (const width of [1, 2]) {
    const result = beamExample(width);
    for (const node of result.nodes) close(node.probability, probabilities[node.path]);
    for (const step of [1, 2]) {
      const candidates = result.nodes.filter(node => node.step === step);
      const ranked = candidates.map((node, index) => index).sort((a, b) => candidates[b].logProbability - candidates[a].logProbability || a - b);
      assert.deepEqual(candidates.filter(node => node.kept).map(node => node.path).sort(), ranked.slice(0, width).map(index => candidates[index].path).sort());
      if (step === 2) {
        const parents = result.nodes.filter(node => node.step === 1 && node.kept);
        assert.equal(candidates.length, parents.length * 2);
        assert.ok(candidates.every(node => parents.some(parent => node.path.startsWith(parent.path))));
      }
    }
  }
  const leaves = beamExample(2).nodes.filter(node => node.step === 2);
  assert.ok(leaves.every(node => node.path.length === 2));
  close(sum(leaves.map(node => node.probability)), 1);
  assert.equal(leaves.reduce((best, node) => node.probability > best.probability ? node : best).path, "Bx");
  const result = beamExample(2);
  result.nodes[0].probability = 0;
  result.optimum.path = "changed";
  assert.equal(beamExample(2).nodes[0].probability, 0.6);
  assert.equal(beamExample(2).optimum.path, "Bx");
  for (const width of [0, 3, 1.5, NaN, Infinity, "1", null, {}, [], true]) assert.throws(() => beamExample(width), RangeError);
});

const descendants = (node, tag) => (node.tagName === tag ? [node] : []).concat((node.childNodes || []).flatMap(child => descendants(child, tag)));
const textContent = node => node.nodeName === "#text" ? node.value : (node.childNodes || []).map(textContent).join("");
const tableTree = html => parseFragment(`<table><tbody>${html}</tbody></table>`);
const tableRows = html => descendants(tableTree(html), "tr").map(row =>
  (row.childNodes || []).filter(node => node.tagName === "th" || node.tagName === "td").map(textContent)
);

function mount(t, values = {}) {
  const nodes = new Map(Object.entries(values).map(([key, initial]) => {
    const listeners = new Map();
    return [`[data-dec-${key}]`, {
      value: String(initial),
      checked: initial === true,
      textContent: "",
      innerHTML: "",
      addEventListener(type, listener) {
        if (!listeners.has(type)) listeners.set(type, []);
        listeners.get(type).push(listener);
      },
      emit(type) {
        for (const listener of listeners.get(type) || []) listener();
      },
    }];
  }));
  const previous = Object.getOwnPropertyDescriptor(globalThis, "document");
  Object.defineProperty(globalThis, "document", { configurable: true, value: { querySelector: selector => nodes.get(selector) || null } });
  t.after(() => {
    if (previous) Object.defineProperty(globalThis, "document", previous);
    else delete globalThis.document;
  });
  return key => nodes.get(`[data-dec-${key}]`);
}

test("主控件缺失时安全返回，解码与束搜索可以分别初始化", t => {
  mount(t);
  assert.doesNotThrow(initDecodingAlgorithms);
});

test("只有温度控件时使用默认值且输出可缺失", t => {
  const get = mount(t, { temperature: 100 });
  assert.doesNotThrow(initDecodingAlgorithms);
  get("temperature").value = "10";
  assert.doesNotThrow(() => get("temperature").emit("input"));
});

test("只有束宽控件时也可渲染四列表格并响应选择", t => {
  const get = mount(t, { beam: 1, "beam-result": "", "beam-rows": "" });
  initDecodingAlgorithms();
  assert.match(get("beam-result").textContent, /束宽 1：Ax，联合概率 0\.306/);
  assert.equal(tableRows(get("beam-rows").innerHTML).length, 4);
  get("beam").value = "2";
  get("beam").emit("change");
  assert.match(get("beam-result").textContent, /束宽 2：Bx，联合概率 0\.396/);
  const rows = tableRows(get("beam-rows").innerHTML);
  assert.equal(rows.length, 6);
  assert.ok(rows.every(row => row.length === 4));
  assert.deepEqual(rows[4], ["2", "Bx", "0.396", "保留"]);
  assert.match(get("beam-result").textContent, /两步.*累计对数概率.*无 EOS 或长度归一化/);
});

test("解码六列表格保留字面 token，所有控件更新中文状态且束搜索独立", t => {
  const get = mount(t, {
    temperature: 100, k: 6, p: 100, grammar: false, prefix: "", u: 500,
    "temp-value": "", "p-value": "", "u-value": "", sampled: "", greedy: "", mass: "", rows: "", note: "",
    beam: 2, "beam-result": "", "beam-rows": "",
  });
  initDecodingAlgorithms();
  assert.equal(get("temp-value").textContent, "1.00");
  assert.equal(get("p-value").textContent, "1.00");
  assert.equal(get("u-value").textContent, "0.500");
  assert.equal(get("mass").textContent, "100.00%");
  assert.equal(get("greedy").textContent, tokens[0]);
  assert.equal(get("sampled").textContent, sampleFrom(decodingDistribution(), 0.5));
  const initial = tableRows(get("rows").innerHTML);
  assert.equal(initial.length, 6);
  assert.ok(initial.every(row => row.length === 6));
  assert.deepEqual(initial.map(row => row[0]), tokens);
  assert.deepEqual(initial.map(row => row[1]), ["2.0", "1.4", "0.8", "0.4", "0.0", "-0.5"]);
  assert.equal(descendants(tableTree(get("rows").innerHTML), "eos").length, 0);
  assert.ok(descendants(tableTree(get("rows").innerHTML), "th").every(node => node.attrs.some(attr => attr.name === "scope" && attr.value === "row")));
  assert.match(get("note").textContent, /顺序为语法屏蔽、温度 softmax、top-k、重归一化、top-p、重归一化/);
  assert.match(get("note").textContent, /最小前缀.*原索引较小者/);
  assert.match(get("note").textContent, /前缀不参与/);
  const originalBeamRows = get("beam-rows").innerHTML;

  get("temperature").value = "300";
  get("temperature").emit("input");
  assert.equal(get("temp-value").textContent, "3.00");
  assert.notEqual(tableRows(get("rows").innerHTML)[0][3], initial[0][3]);
  get("k").value = "1";
  get("k").emit("change");
  assert.equal(get("sampled").textContent, tokens[0]);
  assert.equal(tableRows(get("rows").innerHTML)[0][5], "100.00%");
  get("k").value = "6";
  get("k").emit("change");
  get("p").value = "1";
  get("p").emit("input");
  assert.equal(get("p-value").textContent, "0.01");
  assert.equal(tableRows(get("rows").innerHTML)[1][4], "top-k：保留；top-p：剔除");
  get("p").value = "100";
  get("p").emit("input");
  get("grammar").checked = true;
  get("grammar").emit("change");
  assert.equal(tableRows(get("rows").innerHTML)[5][2], "屏蔽");
  get("prefix").value = '{"ok":';
  get("prefix").emit("change");
  assert.equal(get("greedy").textContent, "true");
  assert.match(get("note").textContent, /只检查.*字符前缀.*JSON\.parse/);
  get("u").value = "999";
  get("u").emit("input");
  assert.equal(get("u-value").textContent, "0.999");
  assert.equal(get("sampled").textContent, "false");
  get("u").value = "0";
  get("u").emit("input");
  assert.equal(get("sampled").textContent, "true");
  get("prefix").value = '{"ok":true';
  get("prefix").emit("change");
  assert.equal(get("greedy").textContent, "}");
  get("prefix").value = '{"ok":true}';
  get("prefix").emit("change");
  assert.equal(get("sampled").textContent, "<eos>");
  assert.equal(get("greedy").textContent, "<eos>");
  const completed = tableRows(get("rows").innerHTML);
  assert.deepEqual(completed[4], ["<eos>", "0.0", "允许", "100.00%", "top-k：保留；top-p：保留", "100.00%"]);
  assert.equal(get("beam-rows").innerHTML, originalBeamRows);
  const decodingRows = get("rows").innerHTML;
  get("beam").value = "1";
  get("beam").emit("change");
  assert.equal(get("rows").innerHTML, decodingRows);

  get("prefix").value = "{";
  get("prefix").emit("change");
  assert.equal(get("sampled").textContent, "无候选");
  assert.equal(get("greedy").textContent, "无候选");
  assert.equal(get("mass").textContent, "0.00%");
  assert.match(get("note").textContent, /没有可接续/);
});
