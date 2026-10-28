export const $ = (selector, root = document) => root.querySelector(selector);
export const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));

export function initCore() {
  const root = document.documentElement;
  let savedTheme;
  try { savedTheme = localStorage.getItem("wiki-theme"); } catch { /* Storage can be disabled. */ }
  if (savedTheme === "dark" || savedTheme === "light") {
    root.dataset.theme = savedTheme;
  } else if (window.matchMedia("(prefers-color-scheme: dark)").matches) {
    root.dataset.theme = "dark";
  }

  const syncThemeLabel = () => {
    const label = $("[data-theme-label]");
    if (label) label.textContent = root.dataset.theme === "dark" ? "浅色" : "深色";
  };

  syncThemeLabel();
  const currentNavLink = $('.site-nav a[aria-current="page"]');
  currentNavLink?.scrollIntoView({ block: "nearest", inline: "center" });
  $("[data-theme-toggle]")?.addEventListener("click", () => {
    const next = root.dataset.theme === "dark" ? "light" : "dark";
    root.dataset.theme = next;
    try { localStorage.setItem("wiki-theme", next); } catch { /* The visible theme still changes. */ }
    syncThemeLabel();
  });
}
