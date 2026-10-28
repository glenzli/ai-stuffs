import { $ } from "../core.js";

const INITIAL_STATE = Object.freeze({ stock: 10, reserved: 0, otherChanged: false });
const trajectories = {
  good: {
    description: "正确完成的预设轨迹", reserved: 2, otherChanged: false,
    actions: ["预设动作（仅模拟）：为商品 A 预留 2 件。"],
  },
  claim: {
    description: "只声称完成的预设轨迹", reserved: 0, otherChanged: false,
    actions: ["预设动作（仅模拟）：没有预留动作，库存和预留数保持初值。"],
  },
  overreserve: {
    description: "超出授权数量的预设轨迹", reserved: 3, otherChanged: false,
    actions: ["预设动作（仅模拟）：为商品 A 预留 3 件，超过授权的 2 件。"],
  },
  sideeffect: {
    description: "带有其他项目改动的预设轨迹", reserved: 2, otherChanged: true,
    actions: ["预设动作（仅模拟）：为商品 A 预留 2 件。", "预设动作（仅模拟）：另行改动其他项目，超出授权范围。"],
  },
};

export function environmentReadout(scenario = "good", grader = "state") {
  if (typeof scenario !== "string" || !Object.hasOwn(trajectories, scenario)) {
    throw new RangeError("scenario 必须为 good、claim、overreserve 或 sideeffect。");
  }
  if (!["reply", "state", "strict"].includes(grader)) {
    throw new RangeError("grader 必须为 reply、state 或 strict。");
  }
  const trajectory = trajectories[scenario];
  const state = {
    ...INITIAL_STATE,
    stock: INITIAL_STATE.stock - trajectory.reserved,
    reserved: trajectory.reserved,
    otherChanged: trajectory.otherChanged,
  };
  const claimed = true;
  const checks = {
    claimed,
    reservation: state.reserved === 2,
    stock: state.stock === 8,
    sideEffects: !state.otherChanged,
  };
  const passed = grader === "reply" ? checks.claimed
    : grader === "state" ? checks.reservation
      : checks.reservation && checks.stock && checks.sideEffects;
  const reward = passed ? 1 : 0;
  let reason;
  if (grader === "reply") {
    reason = `回复验证器只检查完成声明：本轨迹声称完成，奖励 ${reward}；终态未计入奖励。`;
  } else if (grader === "state") {
    reason = `数量验证器只检查预留恰为 2 件：观测为 ${state.reserved} 件，${passed ? "满足" : "不满足"}，奖励 ${reward}；库存和其他项目改动未计入奖励。`;
  } else {
    const failures = [
      [checks.reservation, `预留 ${state.reserved} 件（要求 2 件）`],
      [checks.stock, `库存 ${state.stock} 件（要求 8 件）`],
      [checks.sideEffects, "其他项目有改动（要求无改动）"],
    ].filter(([pass]) => !pass).map(([, observed]) => observed);
    reason = passed
      ? "严格验证器的预留数量、剩余库存、无副作用三项均满足，奖励 1。"
      : `严格验证器不满足：${failures.join("；")}，奖励 0。`;
  }
  const trace = [
    `人工初始快照：商品 A 库存 ${INITIAL_STATE.stock}，预留 ${INITIAL_STATE.reserved}，其他项目未改动。`,
    "任务授权：仅为商品 A 预留 2 件，其他项目不得改变。",
    ...trajectory.actions,
    `模拟终态：库存 ${state.stock}，预留 ${state.reserved}，其他项目${state.otherChanged ? "有改动" : "未改动"}。`,
    "预设最终回复（未发送）：声称已为商品 A 预留 2 件。",
  ];
  return {
    description: trajectory.description, initialState: INITIAL_STATE, trace, state,
    claimed, format: true, done: true, reward, checks, reason,
  };
}

export function initTrainingEnvironments() {
  const scenarioInput = $("[data-env-scenario]");
  if (!scenarioInput) return;
  const graderInput = $("[data-env-grader]");
  const setText = (selector, value) => {
    const output = $(selector);
    if (output) output.textContent = value;
  };
  const render = () => {
    const result = environmentReadout(scenarioInput.value, graderInput?.value ?? "state");
    setText("[data-env-reward]", String(result.reward));
    setText("[data-env-stock]", String(result.state.stock));
    setText("[data-env-reserved]", String(result.state.reserved));
    const trace = $("[data-env-trace]");
    if (trace) trace.innerHTML = result.trace.map(step => `<li>${step}</li>`).join("");
    const rows = $("[data-env-checks]");
    if (rows) rows.innerHTML = [
      ["claimed", "回复声称完成", result.claimed ? "声称完成" : "未声称完成"],
      ["reservation", "预留恰为 2 件", `${result.state.reserved} 件`],
      ["stock", "剩余库存恰为 8 件", `${result.state.stock} 件`],
      ["sideEffects", "其他项目无改动", result.state.otherChanged ? "有改动" : "未改动"],
    ].map(([check, label, observed]) =>
      `<tr><td>${label}</td><td>${observed}</td><td>${result.checks[check] ? "满足" : "不满足"}</td></tr>`
    ).join("");
    setText("[data-env-note]", `${result.description}。${result.reason} 每次读数均从冻结的人工初始快照开始；这是有限的固定轨迹，无自适应智能体，没有真实库存操作或交易。`);
  };
  for (const input of [scenarioInput, graderInput]) {
    for (const event of ["input", "change"]) input?.addEventListener(event, render);
  }
  render();
}
