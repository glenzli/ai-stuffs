import { $ } from "../core.js";

export const curationSamples = [
  { name:"数学讲解", source:"教育", score:.91, cluster:"A", holdout:false },
  { name:"转载的数学讲解", source:"教育", score:.90, cluster:"A", holdout:false },
  { name:"代码调试记录", source:"代码", score:.63, cluster:"B", holdout:false },
  { name:"少数语言社区问答", source:"多语种", score:.42, cluster:"C", holdout:false },
  { name:"已知评测题解析", source:"教育", score:.96, cluster:"D", holdout:true },
  { name:"导航与重复模板", source:"网页", score:.18, cluster:"E", holdout:false },
  { name:"技术故障长文", source:"技术", score:.72, cluster:"F", holdout:false },
  { name:"常见知识短答", source:"教育", score:.86, cluster:"G", holdout:false },
];
export function curateSamples(threshold, dedup, isolate) {
  const seen=new Set();
  return curationSamples.map(sample=>{
    let reason="保留";
    if(sample.score<threshold)reason="低于阈值";
    else if(isolate && sample.holdout)reason="评测隔离";
    else if(dedup && seen.has(sample.cluster))reason="重复簇";
    else seen.add(sample.cluster);
    return {...sample,reason,kept:reason==="保留"};
  });
}
export function initDataCuration() {
  const threshold=$("[data-curation-threshold]");if(!threshold)return;
  const dedup=$("[data-curation-dedup]"),holdout=$("[data-curation-holdout]");
  const render=()=>{
    const rows=curateSamples(Number(threshold.value)/100,dedup.checked,holdout.checked),kept=rows.filter(x=>x.kept);
    $("[data-curation-threshold-value]").textContent=(Number(threshold.value)/100).toFixed(2);
    $("[data-curation-kept]").textContent=`${kept.length}/8`;
    $("[data-curation-sources]").textContent=new Set(kept.map(x=>x.source)).size;
    $("[data-curation-leaks]").textContent=kept.filter(x=>x.holdout).length;
    $("[data-curation-rows]").innerHTML=rows.map(x=>`<tr><td>${x.name}<br><small>${x.source}</small></td><td>${x.score.toFixed(2)}</td><td>${x.cluster}${x.holdout?' · 评测重叠':''}</td><td style="color:var(${x.kept?'--accent':'--ink-faint'})">${x.reason}</td></tr>`).join("");
    $("[data-curation-note]").textContent=kept.length===0?"没有样本通过。可降低阈值，再检查保留样本的覆盖与质量。":kept.some(x=>x.holdout)?"质量分很高的文档仍可能污染评测。还需要按评测重叠标记执行隔离。":"查看保留来源是否变窄：这个人工评分器给少数语言打了低分，过滤阈值可能把偏差带入训练分布。";
  };
  [threshold,dedup,holdout].forEach(x=>x.addEventListener("input",render));render();
}
