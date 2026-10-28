import assert from "node:assert/strict";
import { test } from "node:test";
import { parseFragment } from "parse5";
import { selectiveReadout, semanticReadout, initUncertaintyCalibration } from "../src/scripts/features/uncertainty-calibration.js";
import { pathEvidence, initReasoningEvidence } from "../src/scripts/features/reasoning-evidence.js";

const close = (actual, expected) => assert.ok(Math.abs(actual - expected) < 1e-12, `${actual} 与 ${expected} 不符`);
const keys = object => Object.keys(object).sort();
const descendants = (node, tag) => (node.tagName === tag ? [node] : []).concat((node.childNodes || []).flatMap(child => descendants(child, tag)));
const textContent = node => node.nodeName === "#text" ? node.value : (node.childNodes || []).map(textContent).join("");
const attribute = (node, name) => node.attrs?.find(item => item.name === name)?.value;
const tableTree = html => parseFragment(`<table><tbody>${html}</tbody></table>`);
const tableRows = html => descendants(tableTree(html), "tr").map(row =>
  (row.childNodes || []).filter(node => node.tagName === "th" || node.tagName === "td").map(textContent)
);

function mount(t, values = {}) {
  const nodes = new Map(Object.entries(values).map(([key, initial]) => {
    const listeners = new Map();
    return [`[data-${key}]`, {
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
  return key => nodes.get(`[data-${key}]`);
}

test("选择性作答使用固定十条记录与明确字段", () => {
  const result = selectiveReadout();
  assert.deepEqual(keys(result), ["answered", "bins", "brier", "coverage", "ece", "errors", "risk", "rows"]);
  assert.deepEqual(result.rows.map(row => row.id), [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
  assert.deepEqual(result.rows.map(row => row.score), [0.99, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.55, 0.40]);
  assert.deepEqual(result.rows.map(row => row.correct), [true, true, false, true, true, false, true, false, true, false]);
  assert.ok(result.rows.every(row => JSON.stringify(keys(row)) === JSON.stringify(["accepted", "correct", "id", "score"])));
  assert.equal(result.answered, 7);
  assert.equal(result.errors, 2);
  close(result.coverage, 0.7);
  close(result.risk, 2 / 7);
  close(result.brier, 0.23126);
  close(result.ece, 0.174);
});

test("阈值包含等号且无人作答时风险为 null", () => {
  const all = selectiveReadout(0);
  assert.equal(all.answered, 10);
  assert.equal(all.errors, 4);
  close(all.risk, 0.4);
  const none = selectiveReadout(1);
  assert.equal(none.answered, 0);
  assert.equal(none.errors, 0);
  assert.equal(none.coverage, 0);
  assert.equal(none.risk, null);
  for (const row of all.rows) {
    const atScore = selectiveReadout(row.score);
    assert.equal(atScore.rows[row.id - 1].accepted, true);
    assert.equal(atScore.answered, row.id);
  }
  assert.ok(selectiveReadout(0.75).risk > selectiveReadout(0.7).risk);
});

test("高分错答仅改变首条真值，不改变分数与接受集合", () => {
  for (const threshold of [0, 0.7, 0.99, 1]) {
    const base = selectiveReadout(threshold);
    const wrong = selectiveReadout(threshold, true);
    assert.deepEqual(wrong.rows.map(row => row.score), base.rows.map(row => row.score));
    assert.deepEqual(wrong.rows.map(row => row.accepted), base.rows.map(row => row.accepted));
    assert.equal(wrong.rows[0].correct, false);
    assert.deepEqual(wrong.rows.slice(1), base.rows.slice(1));
    assert.equal(wrong.coverage, base.coverage);
    assert.equal(wrong.errors, base.errors + Number(base.rows[0].accepted));
    close(wrong.brier, 0.32926);
    close(wrong.ece, 0.274);
  }
  assert.equal(selectiveReadout(0.99, true).risk, 1);
  assert.equal(selectiveReadout(1, true).risk, null);
});

test("校准分箱覆盖全部记录，0.75 归入最后一箱", () => {
  const { bins } = selectiveReadout();
  assert.deepEqual(bins.map(bin => bin.label), ["[0, 0.5)", "[0.5, 0.75)", "[0.75, 1]"]);
  assert.deepEqual(bins.map(bin => bin.n), [1, 3, 6]);
  assert.equal(bins.reduce((sum, bin) => sum + bin.n, 0), 10);
  for (const [index, [confidence, accuracy]] of [[0.4, 0], [1.9 / 3, 2 / 3], [5.24 / 6, 4 / 6]].entries()) {
    assert.deepEqual(keys(bins[index]), ["accuracy", "confidence", "label", "n"]);
    close(bins[index].confidence, confidence);
    close(bins[index].accuracy, accuracy);
  }
});

test("所有滑块阈值均保持全样本校准统计，覆盖率随阈值不增", () => {
  for (const stableError of [false, true]) {
    const baseline = selectiveReadout(0, stableError);
    let previousCoverage = 1;
    for (let value = 0; value <= 100; value++) {
      const result = selectiveReadout(value / 100, stableError);
      assert.ok(result.coverage <= previousCoverage);
      assert.equal(result.coverage, result.answered / 10);
      assert.equal(result.risk, result.answered ? result.errors / result.answered : null);
      assert.deepEqual(result.bins, baseline.bins);
      assert.equal(result.brier, baseline.brier);
      assert.equal(result.ece, baseline.ece);
      previousCoverage = result.coverage;
    }
  }
});

test("阈值拒绝越界、非有限数和隐式类型转换", () => {
  for (const threshold of [-0.01, 1.01, NaN, Infinity, -Infinity, "0.7", null, {}, [], true]) {
    assert.throws(() => selectiveReadout(threshold), RangeError);
  }
});

test("语义概率按六次输出的簇频数归一化", () => {
  for (const mode of ["paraphrase", "conflict", "stablewrong"]) {
    const result = semanticReadout(mode);
    assert.deepEqual(keys(result), ["accuracy", "clusters", "entropy", "samples"]);
    assert.equal(result.samples.reduce((sum, sample) => sum + sample.count, 0), 6);
    assert.equal(result.clusters.reduce((sum, cluster) => sum + cluster.count, 0), 6);
    close(result.clusters.reduce((sum, cluster) => sum + cluster.probability, 0), 1);
    for (const sample of result.samples) {
      assert.deepEqual(keys(sample), ["cluster", "correct", "count", "text"]);
      assert.equal(typeof sample.correct, "boolean");
      assert.ok(Number.isInteger(sample.count) && sample.count > 0);
    }
    for (const cluster of result.clusters) {
      assert.deepEqual(keys(cluster), ["count", "id", "probability"]);
      const members = result.samples.filter(sample => sample.cluster === cluster.id);
      assert.equal(cluster.count, members.reduce((sum, sample) => sum + sample.count, 0));
      close(cluster.probability, cluster.count / 6);
    }
    close(result.accuracy, result.samples.reduce((sum, sample) => sum + Number(sample.correct) * sample.count, 0) / 6);
  }
});

test("三种北京释义合为一簇，零熵无法区分全对与全错", () => {
  const paraphrase = semanticReadout("paraphrase");
  const wrong = semanticReadout("stablewrong");
  assert.deepEqual(paraphrase.samples.map(sample => sample.count), [3, 2, 1]);
  assert.equal(new Set(paraphrase.samples.map(sample => sample.text)).size, 3);
  assert.ok(paraphrase.samples.every(sample => sample.cluster === "A" && sample.text.includes("北京")));
  assert.deepEqual(paraphrase.clusters, [{ id: "A", count: 6, probability: 1 }]);
  assert.equal(paraphrase.entropy, 0);
  assert.equal(paraphrase.accuracy, 1);
  assert.deepEqual(wrong.clusters, [{ id: "B", count: 6, probability: 1 }]);
  assert.ok(wrong.samples.every(sample => !sample.correct && sample.text.includes("上海")));
  assert.equal(wrong.entropy, 0);
  assert.equal(wrong.accuracy, 0);
});

test("冲突语义簇频数为三、二、一，熵使用自然对数", () => {
  const result = semanticReadout("conflict");
  assert.deepEqual(result.clusters.map(cluster => cluster.count), [3, 2, 1]);
  close(result.entropy, 1.0114042647073516);
  close(result.entropy, [0.5, 1 / 3, 1 / 6].reduce((sum, p) => sum - p * Math.log(p), 0));
  close(result.accuracy, 0.5);
});

test("语义模式拒绝无效值与对象原型属性", () => {
  for (const mode of [undefined, null, "", "unknown", "constructor", "__proto__", "toString", {}, []]) {
    assert.throws(() => semanticReadout(mode), RangeError);
  }
});

const expectedDistributions = {
  base: [[0.2, 0.1, 0.7], [0.04, 0.01, 0.95]],
  efficient: [[0.6, 0.1, 0.3], [0.009, 0.001, 0.99]],
  process: [[0.29, 0.01, 0.7], [0.049, 0.001, 0.95]],
};

test("路径证据保留两题三类路径的正概率与相同支持", () => {
  assert.deepEqual(pathEvidence(), pathEvidence("base", 16));
  for (const [mode, expected] of Object.entries(expectedDistributions)) {
    const result = pathEvidence(mode);
    assert.deepEqual(keys(result), ["pass1", "passk", "rows", "sound", "soundGivenCorrect"]);
    assert.deepEqual(result.rows.map(row => row.task), ["A", "B"]);
    for (const [index, row] of result.rows.entries()) {
      assert.deepEqual(keys(row), ["answer", "pLucky", "pSound", "pWrong", "passk", "process", "task"]);
      assert.deepEqual([row.pSound, row.pLucky, row.pWrong], expected[index]);
      assert.ok([row.pSound, row.pLucky, row.pWrong].every(p => p > 0 && p < 1));
      close(row.pSound + row.pLucky + row.pWrong, 1);
      close(row.answer, row.pSound + row.pLucky);
      assert.equal(row.process, row.pSound);
    }
  }
});

test("指标按两题等权汇总，条件可靠率是总可靠率除以总正确率", () => {
  const result = pathEvidence();
  close(result.pass1, 0.175);
  close(result.sound, 0.12);
  close(result.soundGivenCorrect, 0.12 / 0.175);
  close(result.passk, (1 - 0.7 ** 16 + 1 - 0.95 ** 16) / 2);
  const meanConditional = result.rows.reduce((sum, row) => sum + row.pSound / row.answer, 0) / 2;
  assert.ok(Math.abs(result.soundGivenCorrect - meanConditional) > 0.01);
  const pooledCoverage = 1 - (1 - result.pass1) ** 16;
  assert.ok(Math.abs(result.passk - pooledCoverage) > 0.1);
});

test("各题独立采样覆盖随有限 k 不减，k 为一时等于单次正确率", () => {
  for (const mode of Object.keys(expectedDistributions)) {
    const first = pathEvidence(mode, 1);
    close(first.passk, first.pass1);
    let previous = first;
    for (let k = 1; k <= 256; k++) {
      const result = pathEvidence(mode, k);
      close(result.pass1, first.pass1);
      close(result.sound, first.sound);
      close(result.soundGivenCorrect, first.soundGivenCorrect);
      assert.ok(result.passk >= previous.passk && result.passk <= 1);
      for (const [index, row] of result.rows.entries()) {
        close(row.passk, 1 - (1 - row.answer) ** k);
        assert.ok(row.passk >= previous.rows[index].passk && row.passk <= 1);
      }
      previous = result;
    }
  }
});

test("效率分配提高单次均值，却降低较大 k 的平均覆盖", () => {
  const base = pathEvidence("base", 1);
  const efficient = pathEvidence("efficient", 1);
  close(efficient.pass1, 0.355);
  assert.ok(efficient.pass1 > base.pass1);
  assert.ok(efficient.rows[0].answer > base.rows[0].answer);
  assert.ok(efficient.rows[1].answer < base.rows[1].answer);
  for (const k of [16, 64, 128, 256]) {
    assert.ok(pathEvidence("efficient", k).passk < pathEvidence("base", k).passk);
  }
});

test("过程分配保持每题答案正确率与覆盖，提高可靠路径占比", () => {
  for (const k of [1, 16, 64, 256]) {
    const base = pathEvidence("base", k);
    const process = pathEvidence("process", k);
    close(process.pass1, base.pass1);
    close(process.passk, base.passk);
    close(process.sound, 0.1695);
    assert.ok(process.sound > base.sound);
    assert.ok(process.soundGivenCorrect > base.soundGivenCorrect);
    for (const [index, row] of process.rows.entries()) {
      close(row.answer, base.rows[index].answer);
      close(row.passk, base.rows[index].passk);
      assert.equal(row.pWrong, base.rows[index].pWrong);
      assert.ok(row.pSound > base.rows[index].pSound);
      assert.ok(row.pLucky < base.rows[index].pLucky);
    }
  }
});

test("路径模式和采样次数拒绝无效输入", () => {
  for (const mode of [null, "", "unknown", "constructor", "__proto__", "toString", {}, []]) {
    assert.throws(() => pathEvidence(mode), RangeError);
  }
  for (const k of [0, -1, 1.5, 257, NaN, Infinity, -Infinity, "16", null, {}, [], true]) {
    assert.throws(() => pathEvidence("base", k), RangeError);
  }
});

test("调用结果可独立修改，不污染后续固定教学数据", () => {
  const selective = selectiveReadout();
  selective.rows[0].score = 0;
  selective.rows[0].correct = false;
  selective.bins[0].n = 100;
  assert.equal(selectiveReadout().rows[0].score, 0.99);
  assert.equal(selectiveReadout().rows[0].correct, true);
  assert.equal(selectiveReadout().bins[0].n, 1);
  const semantic = semanticReadout("paraphrase");
  semantic.samples[0].count = 100;
  semantic.clusters[0].probability = 0;
  assert.equal(semanticReadout("paraphrase").samples[0].count, 3);
  assert.equal(semanticReadout("paraphrase").clusters[0].probability, 1);
  const paths = pathEvidence();
  paths.rows[0].pSound = 0;
  assert.equal(pathEvidence().rows[0].pSound, 0.2);
});

test("页面控件缺失时初始化安全返回", t => {
  mount(t);
  assert.doesNotThrow(initUncertaintyCalibration);
  assert.doesNotThrow(initReasoningEvidence);
});

test("仅主控件存在时允许其余控件与输出缺失", t => {
  const get = mount(t, { "uc-threshold": 70, "re-mode": "base" });
  assert.doesNotThrow(initUncertaintyCalibration);
  assert.doesNotThrow(initReasoningEvidence);
  get("uc-threshold").value = "100";
  assert.doesNotThrow(() => get("uc-threshold").emit("input"));
  get("re-mode").value = "process";
  assert.doesNotThrow(() => get("re-mode").emit("change"));
});

test("不确定性控件渲染指标与四、四、三列表格，并响应滑块与勾选", t => {
  const get = mount(t, {
    "uc-threshold": 70, "uc-stable": false, "uc-semantic": "paraphrase",
    "uc-threshold-value": "", "uc-coverage": "", "uc-risk": "", "uc-answered": "",
    "uc-brier": "", "uc-ece": "", "uc-rows": "", "uc-bins": "", "uc-entropy": "",
    "uc-semantic-accuracy": "", "uc-meanings": "", "uc-note": "",
  });
  initUncertaintyCalibration();
  assert.equal(get("uc-threshold-value").textContent, "70%");
  assert.equal(get("uc-coverage").textContent, "70.00%");
  assert.equal(get("uc-risk").textContent, "28.57%");
  assert.equal(get("uc-answered").textContent, "7");
  assert.equal(get("uc-brier").textContent, "0.2313");
  assert.equal(get("uc-ece").textContent, "0.1740");
  const rows = tableRows(get("uc-rows").innerHTML);
  assert.equal(rows.length, 10);
  assert.ok(rows.every(row => row.length === 4));
  assert.deepEqual(rows[0], ["1", "0.99", "正确", "作答"]);
  assert.deepEqual(rows[6], ["7", "0.70", "正确", "作答"]);
  assert.deepEqual(rows[7], ["8", "0.65", "错误", "弃答"]);
  assert.ok(descendants(tableTree(get("uc-rows").innerHTML), "th").every(node => attribute(node, "scope") === "row"));
  assert.deepEqual(tableRows(get("uc-bins").innerHTML), [
    ["[0, 0.5)", "1", "40.00%", "0.00%"],
    ["[0.5, 0.75)", "3", "63.33%", "66.67%"],
    ["[0.75, 1]", "6", "87.33%", "66.67%"],
  ]);
  assert.equal(get("uc-entropy").textContent, "0.0000");
  assert.equal(get("uc-semantic-accuracy").textContent, "100.00%");
  assert.deepEqual(tableRows(get("uc-meanings").innerHTML).map(row => row.slice(1)), [["簇 A", "3"], ["簇 A", "2"], ["簇 A", "1"]]);
  get("uc-stable").checked = true;
  get("uc-stable").emit("change");
  assert.equal(get("uc-risk").textContent, "42.86%");
  assert.equal(get("uc-brier").textContent, "0.3293");
  assert.equal(get("uc-ece").textContent, "0.2740");
  assert.deepEqual(tableRows(get("uc-rows").innerHTML)[0], ["1", "0.99", "错误", "作答"]);
  get("uc-threshold").value = "99";
  get("uc-threshold").emit("input");
  assert.equal(get("uc-risk").textContent, "100.00%");
  get("uc-threshold").value = "100";
  get("uc-threshold").emit("input");
  assert.equal(get("uc-threshold-value").textContent, "100%");
  assert.equal(get("uc-coverage").textContent, "0.00%");
  assert.equal(get("uc-risk").textContent, "无作答");
  assert.equal(get("uc-answered").textContent, "0");
  assert.equal(get("uc-brier").textContent, "0.3293");
  assert.equal(get("uc-ece").textContent, "0.2740");
  get("uc-semantic").value = "conflict";
  get("uc-semantic").emit("change");
  assert.equal(get("uc-entropy").textContent, "1.0114");
  assert.equal(get("uc-semantic-accuracy").textContent, "50.00%");
  assert.deepEqual(tableRows(get("uc-meanings").innerHTML).map(row => row.slice(1)), [["簇 A", "3"], ["簇 B", "2"], ["簇 C", "1"]]);
  get("uc-semantic").value = "stablewrong";
  get("uc-semantic").emit("change");
  assert.equal(get("uc-entropy").textContent, "0.0000");
  assert.equal(get("uc-semantic-accuracy").textContent, "0.00%");
  assert.deepEqual(tableRows(get("uc-meanings").innerHTML), [["中国的首都是上海。", "簇 B", "6"]]);
  assert.match(get("uc-note").textContent, /人工标注/);
  assert.match(get("uc-note").textContent, /概率来自频数/);
  assert.match(get("uc-note").textContent, /零熵不保证正确/);
});

test("证据控件渲染五列等权指标并响应模式与采样次数", t => {
  const get = mount(t, {
    "re-mode": "base", "re-k": 16, "re-k-value": "", "re-pass1": "", "re-passk": "",
    "re-sound": "", "re-conditional": "", "re-rows": "", "re-note": "",
  });
  initReasoningEvidence();
  assert.equal(get("re-k-value").textContent, "16");
  assert.equal(get("re-pass1").textContent, "17.50%");
  assert.equal(get("re-passk").textContent, "77.83%");
  assert.equal(get("re-sound").textContent, "12.00%");
  assert.equal(get("re-conditional").textContent, "68.57%");
  assert.deepEqual(tableRows(get("re-rows").innerHTML), [
    ["任务甲", "20.00%", "10.00%", "70.00%", "99.67%"],
    ["任务乙", "4.00%", "1.00%", "95.00%", "55.99%"],
  ]);
  assert.match(get("re-note").textContent, /固定教学分布/);
  assert.match(get("re-note").textContent, /等权/);
  assert.match(get("re-note").textContent, /独立同分布/);
  assert.match(get("re-note").textContent, /正概率/);
  get("re-mode").value = "efficient";
  get("re-mode").emit("change");
  assert.equal(get("re-pass1").textContent, "35.50%");
  assert.equal(get("re-passk").textContent, "57.43%");
  assert.equal(get("re-sound").textContent, "30.45%");
  assert.match(get("re-note").textContent, /平均覆盖更低/);
  const rows = tableRows(get("re-rows").innerHTML);
  assert.equal(rows.length, 2);
  assert.ok(rows.every(row => row.length === 5));
  assert.deepEqual(rows[1].slice(1, 4), ["0.90%", "0.10%", "99.00%"]);
  get("re-mode").value = "process";
  get("re-mode").emit("change");
  assert.equal(get("re-pass1").textContent, "17.50%");
  assert.equal(get("re-passk").textContent, "77.83%");
  assert.equal(get("re-sound").textContent, "16.95%");
  assert.equal(get("re-conditional").textContent, "96.86%");
  get("re-k").value = "1";
  get("re-k").emit("input");
  assert.equal(get("re-k-value").textContent, "1");
  assert.equal(get("re-passk").textContent, "17.50%");
  get("re-k").value = "256";
  get("re-k").emit("input");
  assert.equal(get("re-k-value").textContent, "256");
  assert.equal(get("re-passk").textContent, "100.00%");
});
