// URL state and share controls have one owner; experiment renderers own their results.
export function resolveViewState(search, choices, defaults, prefix = "") {
  const params = new URLSearchParams(search), state = { ...defaults }, invalid = [];
  for (const [name, values] of Object.entries(choices)) {
    const raw = params.get(prefix + name);
    if (raw === null) continue;
    if (values.includes(raw)) state[name] = raw;
    else invalid.push(prefix + name);
  }
  return { state, invalid };
}
export function viewUrl(current, state, defaults, prefix = "") {
  const url = new URL(current);
  for (const [key, value] of Object.entries(state)) {
    if (String(value) === String(defaults[key])) url.searchParams.delete(prefix + key);
    else url.searchParams.set(prefix + key, value);
  }
  return url.href;
}
export function bindExperimentView(root, { attribute = "viewKey", prefix = "", onChange }) {
  const controls = [...root.querySelectorAll('select')];
  const defaults = Object.fromEntries(controls.map(control => [control.dataset[attribute], control.value]));
  const choices = Object.fromEntries(controls.map(control => [control.dataset[attribute], [...control.options].map(option => option.value)]));
  const toolbar = document.createElement('div'); toolbar.className = 'experiment-share';
  const share = document.createElement('button'); share.type = 'button'; share.textContent = '复制当前实验链接';
  const status = document.createElement('span'); status.setAttribute('role', 'status');
  const fallback = document.createElement('input'); fallback.type = 'text'; fallback.readOnly = true; fallback.hidden = true; fallback.setAttribute('aria-label', '当前实验链接，可手动复制');
  toolbar.append(share, status, fallback); root.insertBefore(toolbar, root.children[1] || null);
  const read = () => {
    const value = resolveViewState(location.search, choices, defaults, prefix);
    for (const control of controls) control.value = value.state[control.dataset[attribute]];
    fallback.hidden = true;
    status.textContent = value.invalid.length ? '链接含不支持的选项，已使用默认值。' : '';
    onChange(value.state); return value.state;
  };
  let state = read();
  for (const control of controls) control.addEventListener('change', () => {
    state = Object.fromEntries(controls.map(item => [item.dataset[attribute], item.value]));
    history.replaceState(null, '', viewUrl(location.href, state, defaults, prefix));
    fallback.hidden = true; status.textContent = ''; onChange(state);
  });
  const pop = () => { state = read(); }; window.addEventListener('popstate', pop);
  share.addEventListener('click', async () => {
    const url = viewUrl(location.href, state, defaults, prefix);
    fallback.value = url; fallback.hidden = false;
    try { if (!navigator.clipboard?.writeText) throw new Error('Clipboard unavailable'); await navigator.clipboard.writeText(url); status.textContent = '链接已复制，也可从输入框复制。'; }
    catch { fallback.focus(); fallback.select(); status.textContent = '可从输入框复制链接。'; }
  });
  return () => window.removeEventListener('popstate', pop);
}
