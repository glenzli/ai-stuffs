import { $ } from "../core.js";

const tokens = [[1,.5],[-.3,.8],[.7,-.2],[.1,-.6]];
const experts = [[.2,.4,-.1,.6],[-.3,.5,.7,.2],[.6,-.2,.1,.8],[.4,.3,-.5,.2]];
const patterns = {
  balanced: [[2,1,-1,-2],[-2,-1,2,1],[1,-2,2,-1],[-1,2,-2,1]],
  skewed: [[4,3,0,-1],[3,2,-1,-2],[5,4,1,0],[2,1,-2,-3]],
};
const softmax = values => { const m = Math.max(...values), ex = values.map(x => Math.exp(x - m)), sum = ex.reduce((s,x) => s+x,0); return ex.map(x=>x/sum); };
const applyExpert = (x, w) => [x[0] * w[0] + x[1] * w[2], x[0] * w[1] + x[1] * w[3]];
const upstream = [[.2,-.3],[.1,.4],[-.2,.5],[.3,.1]];

export function dispatchExample({ pattern = "balanced", capacity = 0, input = tokens, expertWeights = experts } = {}) {
  if (!Object.hasOwn(patterns, pattern) || !Number.isInteger(capacity) || capacity < 0 || capacity > 4
      || !Array.isArray(input) || input.length !== 4 || !input.every(x => Array.isArray(x) && x.length === 2 && x.every(Number.isFinite))
      || !Array.isArray(expertWeights) || expertWeights.length !== 4 || !expertWeights.every(x => Array.isArray(x) && x.length === 4 && x.every(Number.isFinite))) throw new RangeError("MoE 配置无效。");
  const counts = [0,0,0,0], routes = input.flatMap((x, token) => {
    const logits = patterns[pattern][token], chosen = [0,1,2,3].sort((a,b) => logits[b]-logits[a] || a-b).slice(0,2);
    const gates = softmax(chosen.map(e => logits[e]));
    return chosen.map((expert, branch) => {
      const order = counts[expert]++, kept = capacity === 0 || order < capacity;
      return { token, branch, expert, gate: gates[branch], source: Math.floor(token / 2), destination: Math.floor(expert / 2), kept, x: [...x] };
    });
  });
  const dispatched = routes.filter(r => r.kept).map(r => ({ ...r })).sort((a,b) => a.destination-b.destination || a.expert-b.expert || a.source-b.source || a.token-b.token);
  const output = input.map(() => [0,0]), dx = input.map(() => [0,0]), dw = experts.map(() => [0,0,0,0]);
  dispatched.forEach((route, permutation) => {
    const w = expertWeights[route.expert];
    route.permutation = permutation; route.output = applyExpert(route.x, w); route.weighted = route.output.map(v => route.gate * v);
    route.dy = upstream[route.token].map(v => route.gate * v);
    route.dx = [route.dy[0] * w[0] + route.dy[1] * w[1], route.dy[0] * w[2] + route.dy[1] * w[3]];
    for (let j = 0; j < 2; j++) { output[route.token][j] += route.weighted[j]; dx[route.token][j] += route.dx[j]; }
    route.x.forEach((x,i) => route.dy.forEach((g,j) => { dw[route.expert][i * 2 + j] += x * g; }));
  });
  const reference = input.map((x, token) => [0,1].map(j => routes.filter(r => r.token === token && r.kept).reduce((s,r) => s + r.gate * (x[0] * expertWeights[r.expert][j] + x[1] * expertWeights[r.expert][2+j]),0)));
  const sendCounts = [0,1].map(source => [0,1].map(destination => dispatched.filter(r => r.source === source && r.destination === destination).length));
  return { routes, dispatched, output, reference, dx, dw, sendCounts, counts, dropped: routes.length-dispatched.length,
    configuration: structuredClone({ input, expertWeights, routerLogits: patterns[pattern] }),
    received: [0,1,2,3].map(expert => dispatched.filter(r => r.expert === expert).length),
    loss: output.flatMap((row,i)=>row.map((v,j)=>v*upstream[i][j])).reduce((a,b)=>a+b,0),
    error: Math.max(...output.flatMap((row,i)=>row.map((v,j)=>Math.abs(v-reference[i][j])))) };
}

const vec = v => `[${v.map(x=>x.toFixed(5)).join(", ")}]`;
export function initMoeDispatch() {
  const pattern = $("[data-md-pattern]"); if (!pattern) return;
  const capacity = $("[data-md-capacity]");
  const set = (name, value) => { $(`[data-md-${name}]`).textContent = value; };
  const render = () => {
    const result = dispatchExample({ pattern: pattern.value, capacity: Number(capacity.value) });
    set("kept", `${result.dispatched.length} / 8`); set("load", result.received.join(" / ")); set("error", result.error.toExponential(2));
    $("[data-md-routes]").innerHTML = result.routes.map(r => `<tr><td>T${r.token}</td><td>R${r.source} → R${r.destination}</td><td>E${r.expert}</td><td>${r.gate.toFixed(5)}</td><td>${r.kept ? "保留" : "容量丢弃"}</td></tr>`).join("");
    $("[data-md-dispatch]").innerHTML = result.dispatched.map(r => `<tr><td>${r.permutation}</td><td>T${r.token} / E${r.expert}</td><td>${vec(r.x)}</td><td>${vec(r.output)}</td><td>${vec(r.weighted)}</td></tr>`).join("");
    $("[data-md-output]").innerHTML = result.output.map((v,i) => `<tr><td>T${i}</td><td>${vec(v)}</td><td>${vec(result.reference[i])}</td><td>${vec(result.dx[i])}</td></tr>`).join("");
    $("[data-md-send]").innerHTML = result.sendCounts.map((row,i)=>`<tr><td>R${i}</td><td>${row[0]}</td><td>${row[1]}</td></tr>`).join("");
    $("[data-md-config-input]").innerHTML = result.configuration.input.map((x,i)=>`<tr><td>T${i}</td><td>${vec(x)}</td><td>${vec(result.configuration.routerLogits[i])}</td></tr>`).join("");
    $("[data-md-config-weights]").innerHTML = result.configuration.expertWeights.map((w,i)=>`<tr><td>E${i}</td><td>${vec(w.slice(0,2))}<br>${vec(w.slice(2))}</td></tr>`).join("");
    $("[data-md-gradients]").innerHTML = result.dw.map((gradient,i)=>`<tr><td>E${i}</td><td>${vec(gradient.slice(0,2))}<br>${vec(gradient.slice(2))}</td></tr>`).join("");
    set("note", `专家收到的分支数为 ${result.received.join("、")}；丢弃 ${result.dropped} 个分支。本例按原 token 顺序保留容量内分支，丢弃后不重新归一化门控权重。合并时按原 token ID 求和，空专家没有分支梯度。`);
  };
  pattern.addEventListener("change", render); capacity.addEventListener("change", render); render();
}
