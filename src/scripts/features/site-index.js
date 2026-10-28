import { $ } from "../core.js";
import { searchIndex } from "../data/search-index.js";

export function initSiteIndex() {
    const search = $("[data-site-search]");
    const results = $("[data-site-index-results]");
    if (!search || !results) return;


    const render = () => {
      const raw = search.value.trim().toLowerCase();
      const matches = raw
        ? searchIndex.filter((entry) =>
          `${entry.title} ${entry.summary} ${entry.terms}`.toLowerCase().includes(raw),
        ).sort((a, b) => b.priority - a.priority)
        : searchIndex.filter((entry) => entry.priority >= 8).sort((a, b) => b.priority - a.priority);
      $("[data-site-index-count]").textContent = raw
        ? `找到 ${matches.length} 个页面`
        : "常用入口";
      results.innerHTML = matches.length
        ? matches.map((entry) => `
          <a href="${entry.href}">
            <strong>${entry.title}</strong>
            <span>${entry.summary}</span>
          </a>
        `).join("")
        : `<p class="site-index-empty">没有找到。试搜 “GQA”“OOM”“RLVR”“tokenizer” 或清空搜索。</p>`;
    };
    search.addEventListener("input", render);
    render();
}
