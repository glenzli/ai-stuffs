import { $ } from "../core.js";

const dot = (a, b) => a.reduce((s, v, i) => s + v * b[i], 0);
const multiply = (x, w) => [x[0] * w[0] + x[1] * w[2], x[0] * w[1] + x[1] * w[3]];
const initial = { W1: [.3,-.2,.5,.4], W2: [.6,.1,-.3,.7] };
const samples = [{ x: [1,.5], target: [.2,-.1] }, { x: [-.3,.8], target: [-.4,.6] }];

export function fsdpReference(parameters = initial) {
  const local = samples.map(({ x, target }) => {
    const z = multiply(x, parameters.W1), h = z.map(Math.tanh), y = multiply(h, parameters.W2), dy = y.map((v, i) => v - target[i]);
    const dh = [dot(dy, parameters.W2.slice(0,2)), dot(dy, parameters.W2.slice(2))], dz = dh.map((v, i) => v * (1 - h[i] ** 2));
    return { x, h, y, loss: .5 * dot(dy, dy), gradients: { W1: x.flatMap(v => dz.map(g => v * g)), W2: h.flatMap(v => dy.map(g => v * g)) } };
  });
  const gradients = Object.fromEntries(Object.keys(initial).map(key => [key, local[0].gradients[key].map((v, i) => (v + local[1].gradients[key][i]) / 2)]));
  const updated = Object.fromEntries(Object.keys(initial).map(key => [key, parameters[key].map((v, i) => v - .1 * gradients[key][i])]));
  return { loss: (local[0].loss + local[1].loss) / 2, local, gradients, updated };
}

export function fsdpLifecycle({ reshard = true, prefetch = false } = {}) {
  if (typeof reshard !== "boolean" || typeof prefetch !== "boolean") throw new RangeError("FSDP 策略必须为布尔值。");
  const shards = [0,1].map(rank => Object.fromEntries(Object.entries(initial).map(([key, vector]) => [key, vector.slice(rank * 2, rank * 2 + 2)])));
  const full = [{},{}], localGrads = [{},{}], ownedGrads = [{},{}], momentum = [{ W1: [0,0], W2: [0,0] },{ W1: [0,0], W2: [0,0] }];
  const states = samples.map(sample => ({ x: [...sample.x], target: [...sample.target] }));
  const events = []; let gathers = 0, scatters = 0, peak = 0;
  const record = (name, note) => {
    const ranks = shards.map((parameters, rank) => {
      const fullCount = Object.values(full[rank]).flat().length, localCount = Object.values(localGrads[rank]).flat().length, ownedCount = Object.values(ownedGrads[rank]).flat().length;
      return { rank, parameters: structuredClone(parameters), momentum: structuredClone(momentum[rank]), full: structuredClone(full[rank]),
        localGradients: structuredClone(localGrads[rank]), gradients: structuredClone(ownedGrads[rank]),
        elements: 8 + fullCount + localCount + ownedCount, temporary: fullCount + localCount, ownedCount };
    });
    peak = Math.max(peak, ...ranks.map(r => r.elements));
    events.push({ name, note, ranks, gathers, scatters, peak });
  };
  const gather = (key, label = "AllGather") => {
    if (full[0][key]) return;
    for (let rank = 0; rank < 2; rank++) full[rank][key] = shards.flatMap(s => s[key]);
    gathers++; record(`${label} ${key}`, "每卡按 rank 顺序拼接两片，得到本层完整参数；本例另分配完整缓冲区。");
  };
  record("初始分片", "每卡持有 4 个参数元素和 4 个动量元素；两个 DP rank 使用不同样本。");
  gather("W1");
  if (prefetch) gather("W2", "预取");
  states.forEach((state, rank) => { state.h = multiply(state.x, full[rank].W1).map(Math.tanh); });
  record("前向 W1 + tanh", "每卡使用完整 W1 计算自己的样本，得到本地激活 H。");
  if (reshard) { full.forEach(s => delete s.W1); record("释放完整 W1", "本地参数片保留；临时完整参数不再存活。"); }
  gather("W2");
  states.forEach((state, rank) => { state.y = multiply(state.h, full[rank].W2); state.dy = state.y.map((v, i) => v - state.target[i]); });
  record("前向 W2 与损失", "每卡目标为 ½‖Y−target‖²；最终目标是两卡损失的平均。");
  if (reshard) { full.forEach(s => delete s.W2); record("释放完整 W2", "反向前需要再次获得完整参数。"); }
  gather("W2", "反向 AllGather");
  if (prefetch) gather("W1", "反向预取");
  for (let rank = 0; rank < 2; rank++) {
    const s = states[rank], w = full[rank].W2;
    s.dh = [dot(s.dy, w.slice(0,2)), dot(s.dy, w.slice(2))];
    localGrads[rank].W2 = s.h.flatMap(v => s.dy.map(g => v * g));
  }
  record("反向 W2", "完整本地 W2 梯度对应本卡样本，尚未进行 DP 平均。");
  const scatter = key => {
    const global = localGrads[0][key].map((v, i) => (v + localGrads[1][key][i]) / 2);
    for (let rank = 0; rank < 2; rank++) ownedGrads[rank][key] = global.slice(rank * 2, rank * 2 + 2);
    scatters++;
    record(`ReduceScatter ${key}`, "先求和并除以 DP=2，再保留本卡片段；此事件同时计入输入与输出梯度缓冲区。");
    for (let rank = 0; rank < 2; rank++) { delete localGrads[rank][key]; delete full[rank][key]; }
    record(`释放 ${key} 临时缓冲区`, "保留归约后的本地梯度片，释放完整参数和完整本地梯度。");
  };
  scatter("W2"); gather("W1", "反向 AllGather");
  for (let rank = 0; rank < 2; rank++) {
    const s = states[rank], dz = s.dh.map((v, i) => v * (1 - s.h[i] ** 2));
    localGrads[rank].W1 = s.x.flatMap(v => dz.map(g => v * g));
    s.dx = [dot(dz, full[rank].W1.slice(0,2)), dot(dz, full[rank].W1.slice(2))];
  }
  record("反向 W1", "激活导数和本地输入产生 W1 梯度；完整 W1 还用于计算输入梯度。");
  scatter("W1");
  const gradients = Object.fromEntries(Object.keys(initial).map(key => [key, ownedGrads.flatMap(s => s[key])]));
  for (let rank = 0; rank < 2; rank++) for (const key of Object.keys(initial)) {
    momentum[rank][key] = momentum[rank][key].map((m, i) => .9 * m + ownedGrads[rank][key][i]);
    shards[rank][key] = shards[rank][key].map((v, i) => v - .1 * momentum[rank][key][i]);
  }
  record("本地 optimizer step", "每卡只更新自己拥有的参数片；本例动量初值为零，学习率为 0.1。");
  ownedGrads.forEach(s => { delete s.W1; delete s.W2; });
  record("清空梯度", "回到参数片与动量片的持久状态，等待下一步。");
  const updated = Object.fromEntries(Object.keys(initial).map(key => [key, shards.flatMap(s => s[key])]));
  const reference = fsdpReference();
  const error = Math.max(...Object.keys(initial).flatMap(key => updated[key].map((v, i) => Math.abs(v - reference.updated[key][i]))));
  return { events, updated, gradients, reference, error, peak, gathers, scatters };
}

const vector = v => `[${v.map(x => x.toFixed(5)).join(", ")}]`;
export function initFsdpLifecycle() {
  const strategy = $("[data-fl-strategy]"); if (!strategy) return;
  const prefetch = $("[data-fl-prefetch]"), select = $("[data-fl-event]");
  let step = 0;
  const set = (key, value) => { $(`[data-fl-${key}]`).textContent = value; };
  const render = () => {
    const result = fsdpLifecycle({ reshard: strategy.value === "reshard", prefetch: prefetch.checked });
    step = Math.min(step, result.events.length - 1); const event = result.events[step];
    select.innerHTML = result.events.map((e, i) => `<option value="${i}"${i === step ? " selected" : ""}>${i + 1}. ${e.name}</option>`).join("");
    set("live", `${event.ranks[0].elements} 元素`); set("peak", `${event.peak} 元素`); set("collectives", `${event.gathers} / ${event.scatters}`);
    set("note", event.note); set("error", result.error.toExponential(2));
    $("[data-fl-ranks]").innerHTML = event.ranks.map(r => `<tr><td>${r.rank}</td><td>W1 ${vector(r.parameters.W1)}<br>W2 ${vector(r.parameters.W2)}</td><td>${Object.keys(r.full).join(", ") || "无"}</td><td>${Object.keys(r.localGradients).join(", ") || "无"}</td><td>${Object.entries(r.gradients).map(([k,v]) => `${k} ${vector(v)}`).join("<br>") || "无"}</td><td>${r.elements * 4} B</td></tr>`).join("");
    $("[data-fl-memory]").innerHTML = result.events.map((e, i) => `<li class="${i === step ? "is-current" : ""}"><span>${i + 1}</span><div><i style="width:${100 * e.ranks[0].elements / result.peak}%"></i></div><span>${e.ranks[0].elements}</span></li>`).join("");
    $("[data-fl-result]").innerHTML = ["W1", "W2"].map(key => `<tr><td>${key}</td><td>${vector(result.gradients[key])}</td><td>${vector(result.updated[key])}</td><td>${vector(result.reference.updated[key])}</td></tr>`).join("");
    $("[data-fl-prev]").disabled = step === 0; $("[data-fl-next]").disabled = step === result.events.length - 1;
  };
  for (const control of [strategy, prefetch]) control.addEventListener("change", () => { step = 0; render(); });
  select.addEventListener("change", () => { step = Number(select.value); render(); });
  $("[data-fl-next]").addEventListener("click", () => { step++; render(); });
  $("[data-fl-prev]").addEventListener("click", () => { step--; render(); });
  render();
}
