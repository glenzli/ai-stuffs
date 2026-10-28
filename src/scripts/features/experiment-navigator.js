export function matchesExperiment(query, text) {
  const normalize = value => String(value).normalize("NFKC").toLocaleLowerCase("en-US");
  const haystack = normalize(text);
  return normalize(query).trim().split(/\s+/u).filter(Boolean).every(term => haystack.includes(term));
}

export function initExperimentNavigator() {
  const panel = document.querySelector("[data-experiment-navigator-panel]");
  if (!panel || panel.dataset.initialized) return;
  panel.dataset.initialized = "true";
  const input = panel.querySelector("[data-experiment-query]");
  const cards = [...panel.querySelectorAll("[data-experiment-study]")];
  const apply = () => {
    let count = 0;
    for (const card of cards) {
      card.hidden = !matchesExperiment(input.value, card.dataset.experimentTerms);
      if (!card.hidden) count++;
    }
    panel.querySelector("[data-experiment-count]").textContent = `显示 ${count} / ${cards.length} 组`;
    panel.querySelector("[data-experiment-empty]").hidden = count !== 0;
  };
  const restore = () => {
    const url = new URL(location.href);
    input.value = url.searchParams.get("verify") || "";
    if (input.value || url.hash === "#verified-experiments") panel.open = true;
    apply();
  };
  input.addEventListener("input", () => {
    apply();
    const url = new URL(location.href);
    if (input.value.trim()) url.searchParams.set("verify", input.value);
    else url.searchParams.delete("verify");
    history.replaceState(history.state, "", url);
  });
  panel.querySelector("[data-experiment-clear]").addEventListener("click", () => {
    input.value = "";
    input.dispatchEvent(new Event("input"));
    input.focus();
  });
  window.addEventListener("popstate", restore);
  window.addEventListener("hashchange", restore);
  panel.querySelector("[data-experiment-controls]").hidden = false;
  restore();
}
