import { experimentNavigation } from "../src/experiment-navigation.mjs";
import { pageLabel } from "../src/wiki.mjs";
import { renderExperimentSource } from "./experiment-sources.mjs";

const esc = value => String(value).replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;").replaceAll('"', "&quot;");

export function experimentStudies(experiments, navigation = experimentNavigation) {
  const ids = navigation.map(item => item.id);
  if (new Set(ids).size !== ids.length || JSON.stringify([...ids].sort()) !== JSON.stringify(experiments.map(item => item.id).sort())) throw new Error("Experiment navigation must cover every source group exactly once");
  return navigation.map(study => {
    const source = experiments.find(item => item.id === study.id);
    if (!source.pages.some(slug => study.href.split("#")[0] === `${slug}.html`)) throw new Error(`Experiment navigation points outside its owner: ${study.id}`);
    for (const field of ["question", "reference", "boundary", "requirements", "terms"]) if (!study[field]?.trim()) throw new Error(`Missing experiment navigation ${field}: ${study.id}`);
    return { ...study, source };
  });
}

export function renderExperimentNavigator(experiments) {
  const studies = experimentStudies(experiments);
  return `<details class="experiment-navigator" data-experiment-navigator-panel>
    <summary>按症状找实测 · ${studies.length} 组完整来源</summary>
    <p>先找与你的现象相近的问题，再看参考对象和结论边界。这里的“组”按来源登记；同一组可包含多个实验与页面。</p>
    <div class="experiment-navigator-controls" data-experiment-controls hidden>
      <label for="experiment-query">搜索症状或方法<input id="experiment-query" type="search" autocomplete="off" placeholder="例如：padding、DDP、检索 过滤" data-experiment-query></label>
      <button type="button" data-experiment-clear>清空</button>
      <p role="status" aria-live="polite" aria-atomic="true" data-experiment-count>显示 ${studies.length} / ${studies.length} 组</p>
    </div>
    <p class="experiment-navigator-empty" data-experiment-empty hidden>没有匹配的实测。试试更短的词，或清空搜索查看全部。</p>
    <div class="experiment-navigator-grid">${studies.map(study => `<article class="experiment-study" data-experiment-study="${study.id}" data-experiment-terms="${esc([study.id, study.question, study.reference, study.boundary, study.requirements, study.terms].join(" "))}">
      <h3><a href="${esc(study.href)}">${esc(study.question)}</a></h3>
      <dl><div><dt>对照什么</dt><dd>${esc(study.reference)}</dd></div><div><dt>能说明到哪</dt><dd>${esc(study.boundary)}</dd></div><div><dt>独立重跑条件</dt><dd>${esc(study.requirements)}</dd></div></dl>
      <div class="experiment-study-links"><a href="${esc(study.href)}">查看实测 →</a>${study.source.pages.map(slug => `<a href="${slug}.html">${esc(pageLabel(slug))}</a>`).join("")}</div>
      <details class="experiment-study-source"><summary>下载来源与独立重跑 · ${(study.source.archive.bytes / 1048576).toFixed(1)} MiB</summary>${renderExperimentSource(study.source)}</details>
    </article>`).join("\n")}</div>
    <p class="experiment-navigator-note">先阅读原始报告即可核验展示结果；重跑需要仓库、固定输入和对应依赖环境。CPU 时间只描述报告所记环境，独立重跑结果需另行核验与采纳。</p>
  </details>`;
}
