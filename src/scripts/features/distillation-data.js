import { $ } from "../core.js";

export function softenedDistribution(logits, temperature) {
  const max = Math.max(...logits);
  const values = logits.map(x => Math.exp((x - max) / temperature));
  const sum = values.reduce((a,b)=>a+b,0);
  return values.map(x=>x/sum);
}
export function distillationReadout(temperature) {
  const teacher = softenedDistribution([4,2,1,0],temperature);
  const student = softenedDistribution([1.4,1.2,.7,.2],temperature);
  const kl = teacher.reduce((sum,p,i)=>sum+p*Math.log(p/student[i]),0);
  return { teacher, student, kl, scaled: temperature ** 2 * kl };
}
export function initDistillationData() {
  const input = $("[data-distill-temperature]");
  if (!input) return;
  const render = () => {
    const t = Number(input.value), result = distillationReadout(t);
    $("[data-distill-temperature-value]").textContent = t.toFixed(1);
    $("[data-distill-max]").textContent = `${(result.teacher[0]*100).toFixed(1)}%`;
    $("[data-distill-kl]").textContent = result.kl.toFixed(3);
    $("[data-distill-scaled]").textContent = result.scaled.toFixed(3);
    $("[data-distill-distribution]").innerHTML = ["teacher","student"].map((key)=>`<p class="micro-label">${key}</p><div class="horizontal-bars">${result[key].map((p,i)=>`<div class="bar-row"><span>${["A","B","C","D"][i]}</span><div class="bar-track"><i style="width:${100*p}%"></i></div><output>${(100*p).toFixed(1)}%</output></div>`).join("")}</div>`).join("");
  };
  input.addEventListener("input",render);render();
}
