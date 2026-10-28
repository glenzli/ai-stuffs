import { learningPath, learningBranches, readingProfile } from "../src/learning.mjs";
import { wikiGroups } from "../src/wiki.mjs";

const esc = value => String(value).replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;").replaceAll('"', "&quot;");
const links = items => items.map(([href, label]) => `<a href="${esc(href)}">${esc(label)}</a>`).join("");

export function renderLearningPath(detailed = false) {
  return `<ol class="learning-path${detailed ? " learning-path-detailed" : ""}">${learningPath.map((step, index) => `<li${detailed ? ` id="learn-${step.id}"` : ""}>
    <span class="learning-path-number">${String(index + 1).padStart(2, "0")}</span>
    <div><h3><a href="${esc(step.href)}">${esc(step.title)}</a></h3><p>${esc(step.question)}</p>
    ${detailed ? `<p class="learning-path-focus"><strong>先读：</strong>${esc(step.focus)}</p><p><strong>前置：</strong>${esc(step.prerequisite)}</p><p class="learning-path-check"><strong>读完自检：</strong>${esc(step.checkpoint)}</p><div class="learning-path-further"><span>按需深入</span>${links(step.further)}</div>` : ""}
    </div>${detailed ? "" : `<a class="learning-path-open" href="${esc(step.href)}">进入<span class="sr-only">${esc(step.title)}</span> →</a>`}
  </li>`).join("")}</ol>`;
}

export function renderLearningBranches() {
  return `<div class="learning-branches">${learningBranches.map(branch => `<article><h3><a href="${esc(branch.href)}">${esc(branch.title)}</a></h3><p>${esc(branch.description)}</p><div class="learning-path-further">${links(branch.links)}</div></article>`).join("")}</div>`;
}

export function renderReadingContext(slug) {
  const group = wikiGroups.find(item => item.pages.some(([id]) => id === slug));
  if (!group) throw new Error(`Missing reading topic: ${slug}`);
  const profile = readingProfile(slug), index = learningPath.findIndex(step => step.href === `${slug}.html`);
  return `<div class="reading-context" aria-label="阅读定位" data-reading-topic="${esc(group.title)}"><span>${esc(group.title)}</span><span class="reading-tag">${esc(profile.level)}</span><span class="reading-tag">${esc(profile.format)}</span>${index < 0 ? "" : `<a href="deep-dive.html#learn-${learningPath[index].id}">学习主线 ${index + 1} / ${learningPath.length}</a>`}</div>`;
}

export function renderLearningContinuation(slug) {
  const index = learningPath.findIndex(step => step.href === `${slug}.html`);
  if (index < 0) return "";
  const step = learningPath[index], next = learningPath[index + 1], previous = learningPath[index - 1];
  return `<nav class="learning-continuation" aria-label="学习主线前后篇"><div><span>第 ${index + 1} / ${learningPath.length} 步 · ${esc(step.title)}</span><p>${esc(step.checkpoint)}</p></div><div class="learning-continuation-links">${previous ? `<a href="${esc(previous.href)}">← ${esc(previous.title)}</a>` : ""}<a href="deep-dive.html#learn-${step.id}">完整路线</a>${next ? `<a href="${esc(next.href)}">下一步：${esc(next.title)} →</a>` : `<a href="deep-dive.html#learning-branches">选择专题分支 →</a>`}</div></nav>`;
}

export function learningLinks() {
  return [...learningPath.flatMap(step => [[step.href, step.label], [`deep-dive.html#learn-${step.id}`, step.title], ...step.further]), ...learningBranches.flatMap(branch => [[branch.href, branch.title], ...branch.links])];
}
