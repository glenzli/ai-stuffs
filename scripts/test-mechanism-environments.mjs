import assert from "node:assert/strict";
import { test } from "node:test";
import { inductionTrace, initLearningMechanisms } from "../src/scripts/features/learning-mechanisms.js";
import { environmentReadout, initTrainingEnvironments } from "../src/scripts/features/training-environments.js";

const close = (actual, expected) => assert.ok(Math.abs(actual - expected) < 1e-12, `${actual} != ${expected}`);
const favored = Math.exp(2) / (Math.exp(2) + 1);
const outcomes = {
  good: { state: { stock: 8, reserved: 2, otherChanged: false }, rewards: [1, 1, 1], checks: [true, true, true, true] },
  claim: { state: { stock: 10, reserved: 0, otherChanged: false }, rewards: [1, 0, 0], checks: [true, false, false, true] },
  overreserve: { state: { stock: 7, reserved: 3, otherChanged: false }, rewards: [1, 0, 0], checks: [true, false, false, true] },
  sideeffect: { state: { stock: 8, reserved: 2, otherChanged: true }, rewards: [1, 1, 0], checks: [true, true, true, false] },
};
const graders = ["reply", "state", "strict"];

for (const example of ["B", "D"]) {
  for (let mask = 0; mask < 8; mask++) {
    const previous = Boolean(mask & 4);
    const matching = Boolean(mask & 2);
    const copy = Boolean(mask & 1);
    test(`induction ${example}: previous=${previous}, matching=${matching}, copy=${copy}`, () => {
      const result = inductionTrace({ example, previous, matching, copy });
      assert.deepEqual(result.tokens, ["A", example, "C", "A"]);
      assert.deepEqual(result.previousTokens, previous ? [null, "A", example, "C"] : [null, null, null, null]);
      assert.equal(result.matchIndex, previous && matching ? 1 : null);
      assert.equal(result.copied, mask === 7 ? example : null);
      assert.equal(result.answer, mask === 7 ? example : "不区分");
      assert.deepEqual(result.stages.map(stage => stage.id), ["previous", "matching", "copy"]);
      assert.deepEqual(result.stages.map(stage => stage.enabled), [previous, matching, copy]);
      assert.deepEqual(result.stages.map(stage => stage.active), [previous, previous && matching, mask === 7]);
      assert.ok(result.stages.every(stage => typeof stage.description === "string" && stage.description.length > 0));
      for (const candidate of ["B", "D"]) {
        const probability = result.probabilities[candidate];
        assert.ok(Number.isFinite(probability) && probability >= 0 && probability <= 1);
        close(probability, mask === 7 ? candidate === example ? favored : 1 - favored : 0.5);
      }
      close(result.probabilities.B + result.probabilities.D, 1);
      if (mask !== 7) assert.deepEqual(result.probabilities, { B: 0.5, D: 0.5 });
    });
  }
}

test("induction defaults and swapping the historical value preserve the softmax rule", () => {
  const b = inductionTrace();
  assert.deepEqual(b, inductionTrace({}));
  const d = inductionTrace({ example: "D" });
  assert.equal(b.copied, "B");
  assert.equal(d.copied, "D");
  assert.equal(b.matchIndex, 1);
  assert.equal(d.matchIndex, 1);
  close(b.probabilities.B, d.probabilities.D);
  close(b.probabilities.D, d.probabilities.B);
  close(b.probabilities.B / b.probabilities.D, Math.exp(2));
});

test("induction neither mutates options nor shares mutable trace results", () => {
  const options = Object.freeze({ example: "D", previous: true, matching: true, copy: true });
  const expected = inductionTrace(options);
  const changed = inductionTrace(options);
  changed.tokens[1] = "B";
  changed.previousTokens[1] = null;
  changed.probabilities.D = 0;
  changed.stages[0].active = false;
  assert.deepEqual(inductionTrace(options), expected);
});

test("induction rejects values outside the fixed token and boolean controls", () => {
  for (const example of ["A", "C", "<script>", "", null, 1]) {
    assert.throws(() => inductionTrace({ example }), RangeError);
  }
  for (const name of ["previous", "matching", "copy"]) {
    for (const value of [0, 1, "false", null, NaN]) {
      assert.throws(() => inductionTrace({ [name]: value }), TypeError);
    }
  }
});

for (const [scenario, expected] of Object.entries(outcomes)) {
  for (const [index, grader] of graders.entries()) {
    test(`environment ${scenario} graded by ${grader}`, () => {
      const result = environmentReadout(scenario, grader);
      assert.deepEqual(result.state, expected.state);
      assert.deepEqual(result.checks, Object.fromEntries(["claimed", "reservation", "stock", "sideEffects"].map((key, i) => [key, expected.checks[i]])));
      assert.equal(result.reward, expected.rewards[index]);
      assert.ok(Number.isFinite(result.reward));
      assert.ok(Number.isFinite(result.state.stock) && Number.isFinite(result.state.reserved));
      assert.equal(result.claimed, true);
      assert.equal(result.format, true);
      assert.equal(result.done, true);
      assert.ok(result.description.length > 0 && result.reason.length > 0);
      assert.ok(result.trace.length >= 5 && result.trace.every(step => typeof step === "string"));
      assert.match(result.trace[0], /初始快照.*库存 10.*预留 0/);
      assert.ok(result.trace.some(step => step.includes(`模拟终态：库存 ${expected.state.stock}，预留 ${expected.state.reserved}`)));
      assert.ok(result.trace.some(step => step.includes("未发送")));
    });
  }
}

test("hidden side effects pass a quantity-only grader and fail the strict grader", () => {
  const quantity = environmentReadout("sideeffect", "state");
  const strict = environmentReadout("sideeffect", "strict");
  assert.equal(quantity.reward, 1);
  assert.equal(strict.reward, 0);
  assert.deepEqual(quantity.state, strict.state);
  assert.deepEqual(quantity.checks, strict.checks);
  assert.equal(quantity.checks.reservation, true);
  assert.equal(quantity.checks.stock, true);
  assert.equal(quantity.checks.sideEffects, false);
  assert.match(quantity.reason, /其他项目改动未计入/);
  assert.match(strict.reason, /其他项目有改动/);
});

test("environment calls reset from a frozen snapshot and isolate all returned mutable data", () => {
  const baseline = environmentReadout();
  assert.deepEqual(baseline, environmentReadout("good", "state"));
  assert.ok(Object.isFrozen(baseline.initialState));
  assert.deepEqual(baseline.initialState, { stock: 10, reserved: 0, otherChanged: false });
  assert.throws(() => { baseline.initialState.stock = 0; }, TypeError);
  for (const scenario of ["sideeffect", "overreserve", "claim", "good", "sideeffect", "good"]) {
    const altered = environmentReadout(scenario);
    altered.state.stock = -1;
    altered.state.reserved = 99;
    altered.state.otherChanged = !altered.state.otherChanged;
    altered.checks.claimed = false;
    altered.trace.push("外部修改");
    altered.trace[2] = "外部修改动作";
    assert.deepEqual(environmentReadout(scenario).state, outcomes[scenario].state);
    assert.deepEqual(environmentReadout(), baseline);
  }
});

test("environment rejects unknown scenarios and graders", () => {
  for (const scenario of ["broken", "__proto__", "toString", "", null, 0, {}]) {
    assert.throws(() => environmentReadout(scenario), RangeError);
  }
  for (const grader of ["other", "__proto__", "", null, 0, {}]) {
    assert.throws(() => environmentReadout("good", grader), RangeError);
  }
});

function mount(t, values = {}) {
  const elements = new Map(Object.entries(values).map(([name, value]) => {
    const listeners = new Map();
    return [`[data-${name}]`, {
      value: typeof value === "boolean" ? "on" : String(value),
      checked: value === true,
      textContent: "", innerHTML: "",
      addEventListener(event, listener) {
        if (!listeners.has(event)) listeners.set(event, []);
        listeners.get(event).push(listener);
      },
      emit(event) { for (const listener of listeners.get(event) || []) listener(); },
    }];
  }));
  const previous = Object.getOwnPropertyDescriptor(globalThis, "document");
  Object.defineProperty(globalThis, "document", { configurable: true, value: { querySelector: selector => elements.get(selector) || null } });
  t.after(() => {
    if (previous) Object.defineProperty(globalThis, "document", previous);
    else delete globalThis.document;
  });
  return name => elements.get(`[data-${name}]`);
}

function assertTable(html, expectedRows) {
  const rows = Array.from(html.matchAll(/<tr>(.*?)<\/tr>/gs));
  assert.equal(rows.length, expectedRows);
  for (const [, cells] of rows) assert.match(cells, /^(?:<td>[^<]*<\/td>){3}$/);
}

test("initializers skip absent guard selectors and tolerate partial markup", t => {
  const get = mount(t, { "icl-example": "B", "env-scenario": "good" });
  assert.doesNotThrow(() => initLearningMechanisms());
  assert.doesNotThrow(() => initTrainingEnvironments());
  get("icl-example").value = "D";
  get("icl-example").emit("change");
  get("env-scenario").value = "sideeffect";
  get("env-scenario").emit("input");
  Object.defineProperty(globalThis, "document", { configurable: true, value: { querySelector: () => null } });
  assert.doesNotThrow(() => initLearningMechanisms());
  assert.doesNotThrow(() => initTrainingEnvironments());
});

test("ICL hooks render all ablations and keep the copy checkbox separate from its result", t => {
  const values = { "icl-example": "B", "icl-previous": true, "icl-matching": true, "icl-copy": true };
  for (const name of ["tokens", "rows", "copy-result", "prob-b", "prob-d", "note"]) values[`icl-${name}`] = "";
  const get = mount(t, values);
  initLearningMechanisms();
  assert.equal(get("icl-copy-result").textContent, "B");
  assert.equal(get("icl-prob-b").textContent, "88.1%");
  assert.equal(get("icl-prob-d").textContent, "11.9%");
  assert.equal(get("icl-copy").checked, true);
  assert.equal(get("icl-copy").value, "on");
  assert.equal(get("icl-copy").textContent, "");
  assert.equal(get("icl-copy").innerHTML, "");
  for (const example of ["B", "D"]) {
    get("icl-example").value = example;
    get("icl-example").emit("change");
    for (let mask = 0; mask < 8; mask++) {
      for (const [name, bit] of [["previous", 4], ["matching", 2], ["copy", 1]]) {
        get(`icl-${name}`).checked = Boolean(mask & bit);
        get(`icl-${name}`).emit(mask % 2 ? "input" : "change");
      }
      assert.equal(get("icl-copy-result").textContent, mask === 7 ? example : "未复制");
      assert.equal(get("icl-prob-b").textContent, mask === 7 ? example === "B" ? "88.1%" : "11.9%" : "50.0%");
      assert.equal(get("icl-prob-d").textContent, mask === 7 ? example === "D" ? "88.1%" : "11.9%" : "50.0%");
      assertTable(get("icl-rows").innerHTML, 4);
      const spans = Array.from(get("icl-tokens").innerHTML.matchAll(/<span ([^>]+)>([^<]+)<\/span>/g));
      assert.deepEqual(spans.map(([, , token]) => token), ["A", example, "C", "A"]);
      assert.match(spans[3][1], /class="pending".*title="[^"]*当前查询/);
      if ((mask & 6) === 6) assert.match(spans[1][1], /class="accepted".*title="[^"]*历史匹配位置/);
      else assert.ok(spans.every(([, attrs]) => !attrs.includes('class="accepted"')));
      assert.match(get("icl-note").textContent, /固定规则.*因果示意.*未加载或训练 Transformer.*手工构造/);
      if (mask !== 7) assert.match(get("icl-note").textContent, /判定：不区分/);
    }
  }
  assert.match(get("icl-rows").innerHTML, /<tr><td>1<\/td><td>D<\/td><td>A<\/td><\/tr>/);
});

test("environment hooks grade every selection and show actual failures even when rewarded", t => {
  const values = { "env-scenario": "good", "env-grader": "state" };
  for (const name of ["reward", "stock", "reserved", "note", "trace", "checks"]) values[`env-${name}`] = "";
  const get = mount(t, values);
  initTrainingEnvironments();
  assert.equal(get("env-reward").textContent, "1");
  assert.equal(get("env-stock").textContent, "8");
  assert.equal(get("env-reserved").textContent, "2");
  for (const scenario of ["sideeffect", "claim", "overreserve", "good", "sideeffect", "good"]) {
    const expected = outcomes[scenario];
    get("env-scenario").value = scenario;
    get("env-scenario").emit("change");
    for (const [index, grader] of graders.entries()) {
      get("env-grader").value = grader;
      get("env-grader").emit(index % 2 ? "input" : "change");
      assert.equal(get("env-reward").textContent, String(expected.rewards[index]));
      assert.equal(get("env-stock").textContent, String(expected.state.stock));
      assert.equal(get("env-reserved").textContent, String(expected.state.reserved));
      const rows = get("env-checks").innerHTML;
      assertTable(rows, 4);
      assert.equal((rows.match(/<td>不满足<\/td>/g) || []).length, expected.checks.filter(pass => !pass).length);
      if (scenario === "sideeffect") assert.match(rows, /<td>其他项目无改动<\/td><td>有改动<\/td><td>不满足<\/td>/);
      if (scenario === "good") assert.doesNotMatch(rows, /不满足/);
      if (scenario === "claim") assert.match(rows, /<td>预留恰为 2 件<\/td><td>0 件<\/td><td>不满足<\/td>/);
      if (scenario === "overreserve") assert.match(rows, /<td>预留恰为 2 件<\/td><td>3 件<\/td><td>不满足<\/td>/);
      const trace = get("env-trace").innerHTML;
      assert.match(trace, /^<li>人工初始快照：商品 A 库存 10，预留 0/);
      assert.match(trace, /预设动作（仅模拟）/);
      assert.match(trace, /预设最终回复（未发送）/);
      assert.equal((trace.match(/<li>/g) || []).length, scenario === "sideeffect" ? 6 : 5);
      assert.match(get("env-note").textContent, /冻结的人工初始快照.*固定轨迹.*无自适应智能体.*没有真实库存操作或交易/);
    }
  }
});
