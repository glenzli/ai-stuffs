import { $, $$ } from "../core.js";
import { searchIndex } from "../data/search-index.js";

export function initWikiShell() {
  const searchDialog = $("[data-search-dialog]");
  const libraryDialog = $("[data-library-dialog]");
  const search = $("[data-global-search]");
  const results = $("[data-global-search-results]");
  const renderSearch = () => {
    const terms = search.value.trim().toLocaleLowerCase().split(/\s+/).filter(Boolean);
    const matches = searchIndex.filter(entry => terms.every(term => `${entry.title} ${entry.summary} ${entry.terms}`.toLocaleLowerCase().includes(term)))
      .sort((a, b) => b.priority - a.priority);
    $("[data-global-search-count]").textContent = terms.length ? `${matches.length} 篇相关专题` : `${searchIndex.length} 篇专题`;
    results.replaceChildren(...matches.map(entry => {
      const link = document.createElement("a");
      link.href = entry.href;
      const title = document.createElement("strong");
      title.textContent = entry.title;
      const summary = document.createElement("span");
      summary.textContent = entry.summary;
      link.append(title, summary);
      return link;
    }));
    if (!matches.length) {
      const empty = document.createElement("p");
      empty.textContent = "没有匹配的专题。";
      results.append(empty);
    }
  };
  $("[data-open-search]").addEventListener("click", () => {
    renderSearch();
    searchDialog.showModal();
    search.focus();
  });
  $("[data-open-library]").addEventListener("click", () => libraryDialog.showModal());
  search.addEventListener("input", renderSearch);
  [searchDialog, libraryDialog].forEach(dialog => {
    $("[data-close-dialog]", dialog).addEventListener("click", () => dialog.close());
    dialog.addEventListener("keydown", event => {
      if (event.key === "Escape") {
        event.preventDefault();
        dialog.close();
      }
    });
    dialog.addEventListener("close", () => {
      $(dialog === searchDialog ? "[data-open-search]" : "[data-open-library]").focus({ preventScroll: true });
    });
    dialog.addEventListener("click", event => {
      const bounds = dialog.getBoundingClientRect();
      if (event.target === dialog && (event.clientX < bounds.left || event.clientX > bounds.right || event.clientY < bounds.top || event.clientY > bounds.bottom)) dialog.close();
    });
  });

  const library = $(".wiki-library");
  const selectedPage = $("[aria-current=page]", library);
  if (selectedPage && library.clientHeight) {
    const relativeTop = selectedPage.getBoundingClientRect().top - library.getBoundingClientRect().top;
    if (relativeTop > library.clientHeight - 60) library.scrollTop = relativeTop - library.clientHeight / 2;
  }

  const headings = $$("main h2[id]");
  const toc = $("[data-wiki-toc]");
  if (headings.length) {
    toc.hidden = false;
    $("[data-toc-count]").textContent = `${headings.length} 节`;
    const links = headings.map(heading => {
      const link = document.createElement("a");
      link.href = `#${heading.id}`;
      link.textContent = heading.textContent.replace(/\s+/g, " ").trim();
      return link;
    });
    $("[data-toc-links]").replaceChildren(...links);
    const observer = new IntersectionObserver(entries => {
      const current = entries.find(entry => entry.isIntersecting)?.target;
      if (current) links.forEach(link => link.toggleAttribute("data-current", link.hash === `#${current.id}`));
    }, { rootMargin: "-15% 0px -65% 0px" });
    headings.forEach(heading => observer.observe(heading));
  }

  const scrollRegions = new Set();
  // Some feature modules populate their table after the shared shell starts.
  // Observe the authored container even while it is still empty.
  $$(".math-display, main .compare-wrap").forEach(region => scrollRegions.add(region));
  $$("main table").forEach(table => {
    let container = table.parentElement;
    while (container && container.tagName !== "MAIN") {
      if (["auto", "scroll"].includes(getComputedStyle(container).overflowX)) {
        scrollRegions.add(container);
        break;
      }
      container = container.parentElement;
    }
  });
  const resizeObserver = new ResizeObserver(entries => {
    entries.forEach(({ target }) => {
      const overflowing = target.scrollWidth > target.clientWidth + 1;
      if (overflowing) {
        target.tabIndex = 0;
        target.setAttribute("role", "region");
        const heading = target.closest("section")?.querySelector("h2");
        target.setAttribute("aria-label", `${heading?.textContent.trim() || "内容"}：${target.classList.contains("math-display") ? "公式" : "表格"}`);
      } else {
        target.removeAttribute("tabindex");
        target.removeAttribute("role");
        target.removeAttribute("aria-label");
      }
    });
  });
  scrollRegions.forEach(region => resizeObserver.observe(region));
}
