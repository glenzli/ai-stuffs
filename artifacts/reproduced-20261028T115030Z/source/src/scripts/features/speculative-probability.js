import { $ } from "../core.js";

function validateDistribution(values) {
  if (!Array.isArray(values) || !values.length || !values.every(v => Number.isFinite(v) && v >= 0) || Math.abs(values.reduce((a,b)=>a+b,0)-1) > 1e-12) throw new RangeError("分布应由非负有限概率组成，且和为 1。");
}
export function speculativeMass(p, q) {
  validateDistribution(p); validateDistribution(q);
  if (p.length !== q.length) throw new RangeError("目标与草稿词表必须相同。");
  const accepted = p.map((v,i)=>Math.min(v,q[i])), residual = p.map((v,i)=>Math.max(v-q[i],0));
  const rejection = residual.reduce((a,b)=>a+b,0), correction = rejection === 0 ? null : residual.map(v=>v/rejection);
  const output = accepted.map((v,i)=>v+(correction ? rejection*correction[i] : 0));
  return { accepted, acceptance: p.map((v,i)=>q[i]===0 ? null : Math.min(1,v/q[i])), rejection, correction, output,
    error: Math.max(...p.map((v,i)=>Math.abs(v-output[i]))) };
}
function sample(p, u) {
  let cumulative = 0, lastPositive = -1;
  for (let i = 0; i < p.length; i++) {
    if (p[i] > 0) lastPositive = i;
    cumulative += p[i];
    if (u < cumulative) return i;
  }
  return lastPositive;
}
export function speculativeToken(p, q, { draft = .4, accept = .5, repair = .2 } = {}) {
  if (![draft,accept,repair].every(v=>Number.isFinite(v)&&v>=0&&v<1))throw new RangeError("均匀数必须在 [0,1) 内。");
  const mass=speculativeMass(p,q), candidate=sample(q,draft), accepted=accept<mass.acceptance[candidate];
  return { candidate, accepted, token: accepted ? candidate : sample(mass.correction,repair) };
}

export function speculativeRound(p, q, { drafts = [.4,.8], accepts = [.5,.8], repair = .2, extra = .3 } = {}) {
  const mass = speculativeMass(p, q);
  if (!Array.isArray(drafts) || !drafts.length || !Array.isArray(accepts) || accepts.length !== drafts.length
      || ![...drafts,...accepts,repair,extra].every(v=>Number.isFinite(v)&&v>=0&&v<1)) throw new RangeError("每个草稿需一个 [0,1) 验收均匀数。");
  const proposed = drafts.map(u => sample(q,u)), emitted = [];
  for (let i = 0; i < proposed.length; i++) {
    if (accepts[i] < mass.acceptance[proposed[i]]) emitted.push(proposed[i]);
    else {
      emitted.push(sample(mass.correction,repair));
      return { proposed, emitted, acceptedCount:i, rejectedAt:i, discarded:proposed.slice(i+1), extraToken:false };
    }
  }
  emitted.push(sample(p,extra));
  return { proposed, emitted, acceptedCount:proposed.length, rejectedAt:null, discarded:[], extraToken:true };
}
const scenarios = {
  overlap: { p:[.5,.3,.2],q:[.2,.5,.3] }, equal:{p:[.5,.3,.2],q:[.5,.3,.2]},
  zeros:{p:[.6,0,.4],q:[0,.7,.3]}, disjoint:{p:[1,0,0],q:[0,.4,.6]},
};
export function initSpeculativeProbability() {
  const scenario=$("[data-sp-scenario]");if(!scenario)return;
  const draw=$("[data-sp-accept]"),second=$("[data-sp-second]"),set=(k,v)=>{$(`[data-sp-${k}]`).textContent=v;};
  const render=()=>{
    const {p,q}=scenarios[scenario.value], result=speculativeMass(p,q), trial=speculativeToken(p,q,{accept:Number(draw.value)});
    set("accept-value",draw.value);set("rejection",result.rejection.toFixed(5));set("error",result.error.toExponential(2));set("correction",result.correction?`[${result.correction.map(x=>x.toFixed(3)).join(", ")}]`:"不需要修正");
    $("[data-sp-rows]").innerHTML=p.map((v,i)=>`<tr><td>${"ABC"[i]}</td><td>${v.toFixed(3)}</td><td>${q[i].toFixed(3)}</td><td>${result.acceptance[i]===null?"不从 q 抽到":result.acceptance[i].toFixed(3)}</td><td>${result.accepted[i].toFixed(3)}</td><td>${(result.correction?result.rejection*result.correction[i]:0).toFixed(3)}</td><td>${result.output[i].toFixed(3)}</td></tr>`).join("");
    set("trial",`固定草稿抽样 u=0.4，抽到 ${"ABC"[trial.candidate]}；验收 u=${draw.value}，${trial.accepted?"接受":"拒绝并用修正分布采样（u=0.2）"}，最终输出 ${"ABC"[trial.token]}。`);
    const round = speculativeRound(p,q,{accepts:[Number(draw.value),Number(second.value)]}),names=tokens=>tokens.map(i=>"ABC"[i]).join(" ");
    set("second-value",second.value);
    set("round",`两个草稿为 ${names(round.proposed)}；${round.rejectedAt===null?"全部接受，再从 p 采样一个额外 token":`第 ${round.rejectedAt+1} 个草稿被拒绝，丢弃其后的 ${round.discarded.length} 个候选并补一个修正 token`}。本轮确认输出：${names(round.emitted)}。`);
    set("note",result.correction?"目标中草稿未覆盖的概率质量由拒绝后的修正分布补齐。表格计算整个分布，单次输出只用于观察采样步骤。":"p=q 时接受质量已等于目标分布，拒绝概率为 0；不计算 0/0 的修正分布。");
  };
  scenario.addEventListener("change",render);draw.addEventListener("input",render);second.addEventListener("input",render);render();
}
