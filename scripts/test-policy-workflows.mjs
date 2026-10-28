import assert from "node:assert/strict";
import { test } from "node:test";
import { parseFragment } from "parse5";
import { groupAdvantages, clippedSurrogate, initRlDynamics } from "../src/scripts/features/rl-dynamics.js";
import { prefixDistillation, initOnPolicyDistillation } from "../src/scripts/features/on-policy-distillation.js";

const close = (actual, expected, tolerance = 1e-12) => {
  assert.ok(Math.abs(actual - expected) <= tolerance, `${actual} != ${expected}`);
};
const sum = values => values.reduce((total, value) => total + value, 0);

test("mixed rewards use the centered population standard deviation", () => {
  const rewards = [0, 0, 1, 1];
  assert.deepEqual(groupAdvantages(rewards), { mean: 0.5, std: 0.5, advantages: [-1, -1, 1, 1], signal: true });
  assert.deepEqual(rewards, [0, 0, 1, 1]);
});

test("equal rewards and singleton groups have exactly zero advantages", () => {
  for (const rewards of [[0, 0, 0, 0], [1, 1, 1, 1], [0.3, 0.3, 0.3], [-7]]) {
    assert.deepEqual(groupAdvantages(rewards), {
      mean: rewards[0], std: 0, advantages: rewards.map(() => 0), signal: false,
    });
  }
});

test("graded advantages are centered with unit population variance", () => {
  const result = groupAdvantages([0, 0.2, 0.7, 1]);
  close(result.mean, 0.475);
  close(result.std, Math.sqrt(0.156875));
  close(sum(result.advantages), 0);
  close(sum(result.advantages.map(value => value ** 2)) / 4, 1);
  assert.equal(result.signal, true);
  assert.ok(result.advantages[0] < result.advantages[1]);
  assert.ok(result.advantages[1] < 0 && result.advantages[2] > 0);
  assert.ok(result.advantages[2] < result.advantages[3]);
});

test("normalization handles negative, large and small finite reward scales", () => {
  const rewards = [-4, -1, 1, 4];
  const baseline = groupAdvantages(rewards);
  close(baseline.mean, 0);
  close(baseline.std, Math.sqrt(8.5));
  const shifted = groupAdvantages(rewards.map(value => 3 * value + 7));
  shifted.advantages.forEach((value, index) => close(value, baseline.advantages[index]));
  for (const scale of [1e200, 1e-200]) {
    const result = groupAdvantages([-scale, scale]);
    assert.equal(result.mean, 0);
    assert.equal(result.std, scale);
    assert.deepEqual(result.advantages, [-1, 1]);
  }
  const offset = groupAdvantages([1e20, 1e20 + 16384]);
  assert.equal(offset.std, 8192);
  assert.deepEqual(offset.advantages, [-1, 1]);
  const extremes = groupAdvantages([-Number.MAX_VALUE, Number.MAX_VALUE]);
  assert.equal(extremes.mean, 0);
  assert.equal(extremes.std, Number.MAX_VALUE);
  assert.deepEqual(extremes.advantages, [-1, 1]);
});

test("rewards require a finite nonempty numeric array", () => {
  for (const rewards of [undefined, null, 1, "0,1", [], [NaN], [Infinity], [-Infinity], [0, "1"], [null], Array(2)]) {
    assert.throws(() => groupAdvantages(rewards));
  }
});

test("the default positive surrogate selects the upper clipping plateau", () => {
  assert.deepEqual(clippedSurrogate(), { ratio: 2, clamped: 1.2, objective: 1.2, clipped: true });
});

test("clipping depends on advantage sign rather than ratio bounds alone", () => {
  const cases = [
    { old: 0.2, current: 0.4, advantage: 2, ratio: 2, clamped: 1.2, objective: 2.4, clipped: true },
    { old: 0.2, current: 0.4, advantage: -2, ratio: 2, clamped: 1.2, objective: -4, clipped: false },
    { old: 0.5, current: 0.2, advantage: 2, ratio: 0.4, clamped: 0.8, objective: 0.8, clipped: false },
    { old: 0.5, current: 0.2, advantage: -2, ratio: 0.4, clamped: 0.8, objective: -1.6, clipped: true },
    { old: 0.5, current: 0.55, advantage: 1, ratio: 1.1, clamped: 1.1, objective: 1.1, clipped: false },
    { old: 0.5, current: 0.55, advantage: -1, ratio: 1.1, clamped: 1.1, objective: -1.1, clipped: false },
  ];
  for (const { old, current, advantage, ratio, clamped, objective, clipped } of cases) {
    const result = clippedSurrogate({ old, current, advantage });
    close(result.ratio, ratio);
    close(result.clamped, clamped);
    close(result.objective, objective);
    assert.equal(result.clipped, clipped);
  }
});

test("zero advantage and exact clipping boundaries do not activate clipping", () => {
  for (const current of [0.1, 0.4, 0.5, 0.6, 1]) {
    const result = clippedSurrogate({ old: 0.5, current, advantage: 0 });
    assert.equal(result.objective, 0);
    assert.equal(result.clipped, false);
  }
  for (const current of [0.4, 0.6]) {
    for (const advantage of [-1, 1]) {
      const result = clippedSurrogate({ old: 0.5, current, advantage });
      assert.equal(result.ratio, result.clamped);
      assert.equal(result.clipped, false);
    }
  }
  assert.equal(clippedSurrogate({ old: Number.MIN_VALUE, current: 1, advantage: 0 }).objective, 0);
});

test("valid probability endpoints and custom clip widths obey the minimum", () => {
  for (const old of [0.01, 0.2, 0.99, 1]) {
    for (const current of [0.01, 0.4, 0.99, 1]) {
      for (const advantage of [-1, 0, 1]) {
        for (const clip of [0.1, 0.2, 0.5]) {
          const result = clippedSurrogate({ old, current, advantage, clip });
          close(result.ratio, current / old);
          assert.ok(result.clamped >= 1 - clip && result.clamped <= 1 + clip);
          close(result.objective, Math.min(result.ratio * advantage, result.clamped * advantage));
          assert.equal(result.clipped, result.objective < result.ratio * advantage);
        }
      }
    }
  }
});

test("surrogates reject invalid probabilities, advantages and clip widths", () => {
  for (const value of [0, -0.1, 1.01, NaN, Infinity, -Infinity, null, "0.2"]) {
    assert.throws(() => clippedSurrogate({ old: value }));
    assert.throws(() => clippedSurrogate({ current: value }));
  }
  for (const clip of [0, 1, -0.1, 1.01, NaN, Infinity, null, "0.2"]) {
    assert.throws(() => clippedSurrogate({ clip }));
  }
  for (const advantage of [NaN, Infinity, -Infinity, null, "1"]) {
    assert.throws(() => clippedSurrogate({ advantage }));
  }
});

test("prefix states expose fixed distributions and both natural-log KL directions", () => {
  const result = prefixDistillation();
  assert.deepEqual(result.states.map(state => state.name), ["teacher", "student"]);
  assert.deepEqual(result.states.map(state => state.teacher), [[0.9, 0.1], [0.8, 0.2]]);
  assert.deepEqual(result.states.map(state => state.student), [[0.6, 0.4], [0.2, 0.8]]);
  close(result.states[0].forwardKL, 0.22628916118535888);
  close(result.states[0].reverseKL, 0.31123867958305756);
  close(result.states[1].forwardKL, 0.8317766166719345);
  close(result.states[1].reverseKL, 0.8317766166719345);
  assert.notEqual(result.states[0].forwardKL, result.states[0].reverseKL);
  for (const state of result.states) {
    close(sum(state.teacher), 1);
    close(sum(state.student), 1);
    assert.ok(state.forwardKL >= 0 && state.reverseKL >= 0);
  }
  close(result.teacherLoss, 0.22628916118535888);
  close(result.onPolicyLoss, 0.41534626700083294);
  close(result.recoveryCoverage, 0.2);
});

test("prefix occupancies and training weights sum to one across the slider range", () => {
  for (let percent = 0; percent <= 100; percent++) {
    const mix = percent / 100;
    const result = prefixDistillation(mix);
    assert.deepEqual(result.states.map(state => state.teacherWeight), [1, 0]);
    assert.deepEqual(result.states.map(state => state.occupancy), [mix, 1 - mix]);
    assert.deepEqual(result.states.map(state => state.studentWeight), [mix, 1 - mix]);
    close(sum(result.states.map(state => state.occupancy)), 1);
    close(sum(result.states.map(state => state.teacherWeight)), 1);
    close(sum(result.states.map(state => state.studentWeight)), 1);
    assert.ok(result.states.every(state => state.occupancy >= 0 && state.occupancy <= 1));
    assert.ok(result.recoveryCoverage >= 0 && result.recoveryCoverage <= 1);
    close(result.teacherLoss, result.states[0].forwardKL);
    close(result.onPolicyLoss, mix * result.states[0].reverseKL + (1 - mix) * result.states[1].reverseKL);
    assert.ok(result.onPolicyLoss >= result.states[0].reverseKL && result.onPolicyLoss <= result.states[1].reverseKL);
  }
});

test("occupancy endpoints include only the occupied state's reverse KL", () => {
  const teacherOnly = prefixDistillation(1);
  const recoveryOnly = prefixDistillation(0);
  assert.deepEqual(teacherOnly.states.map(state => state.occupancy), [1, 0]);
  assert.deepEqual(recoveryOnly.states.map(state => state.occupancy), [0, 1]);
  assert.equal(teacherOnly.onPolicyLoss, teacherOnly.states[0].reverseKL);
  assert.equal(recoveryOnly.onPolicyLoss, recoveryOnly.states[1].reverseKL);
  assert.equal(teacherOnly.recoveryCoverage, 0);
  assert.equal(recoveryOnly.recoveryCoverage, 1);
  assert.equal(teacherOnly.teacherLoss, recoveryOnly.teacherLoss);
});

test("prefix mix requires a finite value in the closed unit interval", () => {
  for (const mix of [-0.01, 1.01, NaN, Infinity, -Infinity, null, "0.8"]) {
    assert.throws(() => prefixDistillation(mix));
  }
});

function withElements(values, run) {
  const previous = Object.getOwnPropertyDescriptor(globalThis, "document");
  const elements = new Map(Object.entries(values).map(([name, value]) => {
    const listeners = new Map();
    return [`[${name}]`, {
      value: String(value), textContent: "", innerHTML: "",
      addEventListener(event, listener) {
        if (!listeners.has(event)) listeners.set(event, []);
        listeners.get(event).push(listener);
      },
      emit(event) { for (const listener of listeners.get(event) || []) listener(); },
    }];
  }));
  Object.defineProperty(globalThis, "document", { configurable: true, value: { querySelector: selector => elements.get(selector) || null } });
  try {
    run(name => elements.get(`[${name}]`));
  } finally {
    if (previous) Object.defineProperty(globalThis, "document", previous);
    else delete globalThis.document;
  }
}

function tableRows(html) {
  const tree = parseFragment(`<table><tbody>${html}</tbody></table>`);
  const body = tree.childNodes[0].childNodes.find(node => node.tagName === "tbody");
  return body.childNodes.filter(node => node.tagName === "tr").map(row =>
    row.childNodes.filter(node => node.tagName === "td").map(cell => cell.childNodes.map(node => node.value || "").join(""))
  );
}

test("initializers safely skip pages without their guard inputs", () => {
  withElements({}, () => {
    assert.doesNotThrow(() => initRlDynamics());
    assert.doesNotThrow(() => initOnPolicyDistillation());
  });
});

test("RL hooks render group outcomes and sign-sensitive clipping on input changes", () => {
  const values = { "data-rl-group": "mixed", "data-rld-old": 20, "data-rld-current": 40, "data-rld-sign": 1 };
  for (const name of ["mean", "std", "signal", "rows", "note", "ratio", "objective", "clip-note"]) values[`data-rld-${name}`] = "";
  withElements(values, get => {
    initRlDynamics();
    assert.equal(get("data-rld-mean").textContent, "0.500");
    assert.equal(get("data-rld-std").textContent, "0.500");
    assert.equal(get("data-rld-ratio").textContent, "2.000");
    assert.equal(get("data-rld-objective").textContent, "1.200");
    for (const mode of ["allpass", "allfail"]) {
      get("data-rl-group").value = mode;
      get("data-rl-group").emit("change");
      assert.equal(get("data-rld-signal").textContent, "无");
      assert.equal(get("data-rld-std").textContent, "0.000");
      assert.deepEqual(tableRows(get("data-rld-rows").innerHTML).map(row => row[2]), Array(4).fill("0.000"));
      assert.match(get("data-rld-note").textContent, /组内|本组/);
      assert.match(get("data-rld-note").textContent, /KL.*熵.*其他组/);
    }
    get("data-rl-group").value = "graded";
    get("data-rl-group").emit("input");
    assert.equal(get("data-rld-mean").textContent, "0.475");
    assert.equal(get("data-rld-signal").textContent, "有");
    get("data-rld-sign").value = "-1";
    get("data-rld-sign").emit("change");
    assert.equal(get("data-rld-objective").textContent, "-2.000");
    get("data-rld-old").value = "50";
    get("data-rld-old").emit("input");
    get("data-rld-current").value = "20";
    get("data-rld-current").emit("input");
    assert.equal(get("data-rld-ratio").textContent, "0.400");
    assert.equal(get("data-rld-objective").textContent, "-0.800");
    assert.match(get("data-rld-clip-note").textContent, /负.*低于 0.8/);
    assert.match(get("data-rld-clip-note").textContent, /全模型 KL.*单独检查/);
  });
});

test("distillation hooks display percentage weights, distributions and both KL columns", () => {
  const values = { "data-opd-mix": 80 };
  for (const name of ["mix-value", "recovery", "teacher-loss", "student-loss", "rows", "note"]) values[`data-opd-${name}`] = "";
  withElements(values, get => {
    initOnPolicyDistillation();
    assert.equal(get("data-opd-mix-value").textContent, "80%");
    assert.equal(get("data-opd-recovery").textContent, "20%");
    assert.equal(get("data-opd-teacher-loss").textContent, "0.226");
    assert.equal(get("data-opd-student-loss").textContent, "0.415");
    assert.deepEqual(tableRows(get("data-opd-rows").innerHTML), [
      ["常见前缀", "100%", "80%", "[0.90, 0.10]", "[0.60, 0.40]", "0.226", "0.311"],
      ["需要恢复的前缀", "0%", "20%", "[0.80, 0.20]", "[0.20, 0.80]", "0.832", "0.832"],
    ]);
    for (const percent of [0, 100]) {
      get("data-opd-mix").value = String(percent);
      get("data-opd-mix").emit("input");
      assert.equal(get("data-opd-mix-value").textContent, `${percent}%`);
      assert.equal(get("data-opd-recovery").textContent, `${100 - percent}%`);
      assert.equal(get("data-opd-student-loss").textContent, percent === 0 ? "0.832" : "0.311");
    }
    assert.match(get("data-opd-note").textContent, /访问权重.*恢复成功率需单独测试/);
    assert.match(get("data-opd-note").textContent, /教师轨迹目标保持不变/);
  });
});
