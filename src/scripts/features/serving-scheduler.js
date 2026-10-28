import { $ } from "../core.js";

export function servingTrace({ capacity = 6, budget = 8, chunk = 4, prefix = true, mode = "continuous" } = {}) {
  for (const [name, value, min, max] of [["capacity", capacity, 3, 12], ["budget", budget, 2, 16], ["chunk", chunk, 1, 8]]) {
    if (!Number.isInteger(value) || value < min || value > max) throw new RangeError(`${name} 超出教学范围。`);
  }
  if (typeof prefix !== "boolean" || !["continuous", "wave"].includes(mode)) throw new RangeError("调度选项无效。");
  const requests = [
    { id: "A", arrival: 0, prompt: 8, output: 4, common: true },
    { id: "B", arrival: 1, prompt: 8, output: 3, common: true },
    { id: "C", arrival: 2, prompt: 4, output: 2, common: false },
  ].map(r => ({ ...r, status: "waiting", admitted: null, first: null, finish: null, processedPrompt: 0, cached: 0, generated: 0, hit: 0, reservation: 0, blocks: [] }));
  const physical = Array.from({ length: capacity }, (_, id) => ({ id, owner: null, used: 0 }));
  if (prefix) physical[0] = { id: 0, owner: "共享前缀", used: 4 };
  let computed = 0, reused = 0, peak = prefix ? 1 : 0;
  const events = [];
  const allocate = request => {
    const privateTokens = request.cached - request.hit;
    while (request.blocks.length < Math.ceil(privateTokens / 4)) {
      const page = physical.find(p => p.owner === null);
      if (!page) throw new Error("教学容量预留不变量被破坏。");
      page.owner = request.id; request.blocks.push(page.id);
    }
    request.blocks.forEach((id, i) => { physical[id].used = Math.min(4, privateTokens - i * 4); });
    peak = Math.max(peak, physical.filter(p => p.owner !== null).length);
  };
  const snapshot = (tick, actions, work) => {
    events.push({ tick, actions, work, computed, reused, peak,
      reserved: requests.filter(r => r.status === "active").reduce((n, r) => n + r.reservation, 0) + Number(prefix),
      requests: requests.map(r => ({ ...r, blocks: [...r.blocks] })),
      blocks: physical.map(p => ({ ...p, references: p.id === 0 && prefix ? requests.filter(r => r.status === "active" && r.hit).map(r => r.id) : p.owner ? [p.owner] : [] })),
    });
  };
  snapshot(-1, ["初始：共享前缀块已预热（若启用）；其他物理块空闲。"], 0);
  for (let tick = 0; tick < 100 && requests.some(r => r.status !== "done"); tick++) {
    const actions = [];
    const active = () => requests.filter(r => r.status === "active");
    if (mode === "continuous" || active().length === 0) {
      for (const request of requests.filter(r => r.status === "waiting" && r.arrival <= tick)) {
        const hit = prefix && request.common ? 4 : 0;
        const reservation = Math.ceil((request.prompt + request.output - 1) / 4) - hit / 4;
        const reserved = active().reduce((n, r) => n + r.reservation, 0) + Number(prefix);
        if (active().length === 2 || reserved + reservation > capacity) break;
        Object.assign(request, { status: "active", admitted: tick, hit, reservation, cached: hit, processedPrompt: hit });
        reused += hit;
        actions.push(`${request.id} 入队执行，复用 ${hit} 个前缀位置，预留 ${reservation} 个私有块容量。`);
      }
    }
    let available = budget;
    const decode = active().filter(r => r.generated > 0);
    const prefill = active().filter(r => r.generated === 0);
    for (const request of decode) {
      if (!available) break;
      request.cached++; request.generated++; available--; computed++;
      allocate(request);
      actions.push(`${request.id} decode 输入 1 个位置，采样第 ${request.generated} 个输出。`);
    }
    for (const request of prefill) {
      if (!available) break;
      const n = Math.min(chunk, available, request.prompt - request.processedPrompt);
      request.processedPrompt += n; request.cached += n; computed += n; available -= n;
      allocate(request);
      actions.push(`${request.id} prefill 计算 ${n} 个位置。`);
      if (request.processedPrompt === request.prompt) { request.generated = 1; request.first = tick + 1; actions.push(`${request.id} prefill 完成，采样首 token。`); }
    }
    for (const request of active().filter(r => r.generated === r.output)) {
      request.status = "done"; request.finish = tick + 1;
      for (const id of request.blocks) { physical[id].owner = null; physical[id].used = 0; }
      request.blocks = [];
      actions.push(`${request.id} 完成，释放私有块；末个输出未再次输入。`);
    }
    snapshot(tick, actions.length ? actions : ["等待请求到达或容量释放。"], budget - available);
  }
  if (requests.some(r => r.status !== "done")) throw new Error("调度未能完成。");
  return { events, computed, reused, peak, requests, finish: Math.max(...requests.map(r => r.finish)), blockSize: 4, capacity };
}

export function initServingScheduler() {
  const capacity = $("[data-ss-capacity]"); if (!capacity) return;
  const budget = $("[data-ss-budget]"), chunk = $("[data-ss-chunk]"), prefix = $("[data-ss-prefix]"), mode = $("[data-ss-mode]");
  let trace, step = 0;
  const render = () => {
    const state = trace.events[step];
    $("[data-ss-step]").textContent = `${step}/${trace.events.length - 1}`;
    $("[data-ss-computed]").textContent = state.computed;
    $("[data-ss-reused]").textContent = state.reused;
    $("[data-ss-peak]").textContent = state.peak;
    $("[data-ss-reserved]").textContent = state.reserved;
    $("[data-ss-blocks]").innerHTML = state.blocks.map(p => `<div class="cache-page ${p.owner ? "is-used" : ""}"><span>块 ${p.id}</span><strong>${p.owner || "空闲"}</strong><small>${p.used}/4 · 引用 ${p.references.join(",") || "无"}</small></div>`).join("");
    $("[data-ss-rows]").innerHTML = state.requests.map(r => `<tr><th scope="row">${r.id}</th><td>${r.arrival}</td><td>${{waiting:"等待",active:r.generated ? "Decode" : "Prefill",done:"完成"}[r.status]}</td><td>${r.processedPrompt}/${r.prompt}</td><td>${r.generated}/${r.output}</td><td>${r.status === "active" ? r.cached : 0}</td><td>${r.first === null ? "—" : r.first - r.arrival}</td><td>${r.finish === null ? "—" : r.finish - r.arrival}</td></tr>`).join("");
    $("[data-ss-log]").innerHTML = state.actions.map(a => `<li>${a}</li>`).join("");
    $("[data-ss-next]").disabled = step === trace.events.length - 1;
    $("[data-ss-prev]").disabled = step === 0;
  };
  const reset = () => { trace = servingTrace({capacity:Number(capacity.value),budget:Number(budget.value),chunk:Number(chunk.value),prefix:prefix.checked,mode:mode.value});step=0;render(); };
  [capacity,budget,chunk,prefix,mode].forEach(x => x.addEventListener("change",reset));
  $("[data-ss-next]").addEventListener("click",()=>{step=Math.min(step+1,trace.events.length-1);render();});
  $("[data-ss-prev]").addEventListener("click",()=>{step=Math.max(0,step-1);render();});
  $("[data-ss-reset]").addEventListener("click",reset); reset();
}
