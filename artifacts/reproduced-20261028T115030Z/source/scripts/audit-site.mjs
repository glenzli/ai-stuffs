import { pages as pageConfig, site } from "../src/pages.mjs";

const port = process.env.CDP_PORT || "9222";
const baseUrl = process.env.BASE_URL || "http://127.0.0.1:4173";
const pages = pageConfig.map((page) => `/${page.file}`);
const width = Number(process.env.VIEWPORT_WIDTH || 390);
const height = Number(process.env.VIEWPORT_HEIGHT || 844);
const theme = process.env.AUDIT_THEME || "light";
const expandMath = process.env.EXPAND_MATH === "1";
const timeoutMs = 8000;
const issues = [];
const pageLinks = new Map();
let target, socket, activePage = "initialization";

async function request(url, options = {}) {
  const response = await fetch(url, { ...options, signal: AbortSignal.timeout(timeoutMs) });
  if (!response.ok) throw new Error(`HTTP ${response.status}: ${url}`);
  return response;
}

try {
if (!(width > 0 && height > 0)) throw new Error("Invalid viewport dimensions");
target = await (await request(`http://127.0.0.1:${port}/json/new`, { method: "PUT" })).json();
socket = new WebSocket(target.webSocketDebuggerUrl);
await new Promise((resolve, reject) => {
  const timer = setTimeout(() => reject(new Error("CDP connection timeout")), timeoutMs);
  socket.addEventListener("open", () => { clearTimeout(timer); resolve(); }, { once: true });
  socket.addEventListener("error", () => { clearTimeout(timer); reject(new Error("CDP connection failed")); }, { once: true });
});

let seq = 0;
const pending = new Map();
const runtimeErrors = [];
socket.addEventListener("message", (event) => {
  const message = JSON.parse(event.data);
  if (message.method === "Runtime.exceptionThrown") {
    runtimeErrors.push(`${activePage}: ${message.params.exceptionDetails.exception?.description || message.params.exceptionDetails.text}`);
  }
  if (message.method === "Runtime.consoleAPICalled" && message.params.type === "error") issues.push(`${activePage}: console.error ${message.params.args.map(arg => arg.value ?? arg.description).join(" ")}`);
  if (message.method === "Network.responseReceived" && message.params.response.status >= 400) issues.push(`${activePage}: HTTP ${message.params.response.status} ${message.params.response.url}`);
  if (message.method === "Network.loadingFailed" && !message.params.canceled) issues.push(`${activePage}: resource failure ${message.params.errorText}`);
  if (!message.id || !pending.has(message.id)) return;
  const { resolve, reject, timer } = pending.get(message.id);
  clearTimeout(timer);
  pending.delete(message.id);
  if (message.error) reject(new Error(message.error.message));
  else resolve(message.result);
});

const send = (method, params = {}) =>
  new Promise((resolve, reject) => {
    const id = ++seq;
    const timer = setTimeout(() => { pending.delete(id); reject(new Error(`CDP timeout: ${method} (${activePage})`)); }, timeoutMs);
    pending.set(id, { resolve, reject, timer });
    socket.send(JSON.stringify({ id, method, params }));
  });

await send("Page.enable");
await send("Runtime.enable");
await send("Network.enable");
await send("Network.setCacheDisabled", { cacheDisabled: true });
await send("Emulation.setDeviceMetricsOverride", {
  width,
  height,
  deviceScaleFactor: 1,
  mobile: false,
});

const navSignature = JSON.stringify(site.navigation.map(item => item.href));

async function waitReady(path) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    const result = await send("Runtime.evaluate", {
      expression: `location.pathname === ${JSON.stringify(path)} && document.readyState === 'complete' && document.documentElement.dataset.wikiReady === 'true'`, returnByValue: true,
    });
    if (result.result?.value) {
      await send("Runtime.evaluate", { expression: "document.fonts.ready.then(() => true)", awaitPromise: true });
      return;
    }
    await new Promise(resolve => setTimeout(resolve, 60));
  }
  throw new Error(`Page did not finish initialization: ${path}`);
}

for (const path of pages) {
  activePage = path;
  const navigation = await send("Page.navigate", { url: `${baseUrl}${path}` });
  if (navigation.errorText) throw new Error(navigation.errorText);
  await waitReady(path);
  await send("Runtime.evaluate", { expression: `document.documentElement.dataset.theme = ${JSON.stringify(theme)}` });
  if (expandMath) {
    await send("Runtime.evaluate", { expression: `document.querySelectorAll('.math-derivation').forEach(el => { el.open = true; }); document.body.getBoundingClientRect(); document.fonts.ready.then(() => true)`, awaitPromise: true });
  }
  const result = await send("Runtime.evaluate", {
    expression: `(() => {
      const accessibleName = (el) =>
        (el.getAttribute("aria-label") || el.textContent || el.getAttribute("title") || "").trim();
      const controls = [...document.querySelectorAll("input:not([type=hidden]), select, textarea")];
      const unlabeledControls = controls.filter((control) =>
        !control.labels?.length && !control.closest("label") && !control.getAttribute("aria-label")
      ).map((control) => ({
        tag: control.tagName.toLowerCase(),
        type: control.getAttribute("type") || "",
        name: control.getAttribute("name") || "",
      }));
      const duplicateIds = [...document.querySelectorAll("[id]")]
        .map((el) => el.id)
        .filter((id, index, ids) => ids.indexOf(id) !== index);
      const unnamedButtons = [...document.querySelectorAll("button")].filter((button) =>
        !accessibleName(button)
      ).map((button) => button.outerHTML.slice(0, 120));
      const namelessSvg = [...document.querySelectorAll("svg[role=img]")].filter((svg) =>
        !svg.getAttribute("aria-label") && !svg.querySelector("title")
      ).map((svg) => svg.outerHTML.slice(0, 120));
      const navLinks = [...document.querySelectorAll(".site-nav a")].map((link) =>
        new URL(link.href).pathname.replace(/^\\//, "") || "index.html"
      );
      const currentNav = document.querySelectorAll(".site-nav [aria-current=page]").length;
      return {
        ids: [...document.querySelectorAll('[id]')].map(el => el.id),
        localLinks: [...document.querySelectorAll('a[href]')]
          .map(el => new URL(el.href))
          .filter(url => url.origin === location.origin)
          .map(url => ({ path: url.pathname === '/' ? '/index.html' : url.pathname, hash: decodeURIComponent(url.hash.slice(1)) })),
        sourceCount: document.querySelectorAll('.page-evidence a[href^="https:"]').length,
        libraryCount: document.querySelectorAll('.wiki-library nav a').length,
        libraryCurrent: document.querySelectorAll('.wiki-library [aria-current="page"]').length,
        tocCount: document.querySelectorAll('[data-toc-links] a').length,
        headingCount: document.querySelectorAll('main h2[id]').length,
        brokenImages: [...document.images].filter(img => img.complete && !img.naturalWidth).map(img => img.src),
        mainOverflow: document.querySelector('main').scrollWidth > document.querySelector('main').clientWidth + 2 || document.querySelector('main').getBoundingClientRect().right > innerWidth + 1,
        mathCount: document.querySelectorAll('[data-math]').length,
        mathMLCount: document.querySelectorAll('[data-math] math').length,
        mathErrors: document.querySelectorAll('.katex-error').length,
        mathFontsLoaded: [...document.fonts].filter(font => font.family.startsWith('KaTeX') && font.status === 'loaded').length,
        title: document.title,
        lang: document.documentElement.lang,
        h1Count: document.querySelectorAll("h1").length,
        duplicateIds,
        unlabeledControls,
        unnamedButtons,
        namelessSvg,
        navLinks,
        currentNav,
        docWidth: document.documentElement.scrollWidth,
        viewportWidth: window.innerWidth,
      };
    })()`,
    returnByValue: true,
  });
  if (result.exceptionDetails) throw new Error(result.exceptionDetails.exception?.description || result.exceptionDetails.text);
  const data = result.result.value;
  pageLinks.set(path, data);
  const prefix = `${path}: `;
  if (!data.sourceCount) issues.push(prefix + "missing primary sources");
  if (data.libraryCount !== pageConfig.length || data.libraryCurrent !== 1) issues.push(prefix + "library coverage/current page mismatch");
  if (data.tocCount !== data.headingCount) issues.push(prefix + "TOC headings mismatch");
  if (data.lang !== "zh-CN") issues.push(prefix + `lang=${data.lang}`);
  if (data.h1Count !== 1) issues.push(prefix + `h1 count=${data.h1Count}`);
  if (data.duplicateIds.length) issues.push(prefix + `duplicate ids=${data.duplicateIds.join(",")}`);
  if (data.unlabeledControls.length) issues.push(prefix + `unlabeled controls=${JSON.stringify(data.unlabeledControls)}`);
  if (data.unnamedButtons.length) issues.push(prefix + `unnamed buttons=${data.unnamedButtons.length}`);
  if (data.namelessSvg.length) issues.push(prefix + `nameless SVGs=${data.namelessSvg.length}`);
  if (data.brokenImages.length) issues.push(prefix + `broken images=${JSON.stringify(data.brokenImages)}`);
  if (data.mathErrors || data.mathCount !== data.mathMLCount || (data.mathCount && !data.mathFontsLoaded)) issues.push(prefix + `math render/font mismatch: ${data.mathCount}/${data.mathMLCount}, errors=${data.mathErrors}, loaded fonts=${data.mathFontsLoaded}`);
  if (data.currentNav !== 1) issues.push(prefix + `current nav count=${data.currentNav}`);
  if (JSON.stringify(data.navLinks) !== navSignature) issues.push(prefix + "nav signature mismatch");
  if (data.docWidth > data.viewportWidth || data.mainOverflow) issues.push(prefix + `content overflow at ${width}px`);
}

for (const [path, data] of pageLinks) {
  for (const link of data.localLinks) {
    const destination = pageLinks.get(link.path);
    if (!destination) issues.push(`${path}: missing linked page ${link.path}`);
    else if (link.hash && !destination.ids.includes(link.hash)) {
      issues.push(`${path}: missing anchor ${link.path}#${link.hash}`);
    }
  }
}

if (runtimeErrors.length) issues.push(`runtime exceptions=${JSON.stringify(runtimeErrors)}`);
} catch (error) {
  issues.push(`${activePage}: ${error.message}`);
} finally {
  socket?.close();
  if (target?.id) {
    try { await request(`http://127.0.0.1:${port}/json/close/${target.id}`); }
    catch (error) { issues.push(`Could not close audit tab: ${error.message}`); }
  }
}
console.log(JSON.stringify({ checkedPages: pageLinks.size, expectedPages: pages.length, width, theme, expandMath, issues: [...new Set(issues)] }, null, 2));
if (issues.length || pageLinks.size !== pages.length) process.exitCode = 1;
