import { writeFile } from "node:fs/promises";

const port = process.env.CDP_PORT || "9222";
const width = Number(process.env.VIEWPORT_WIDTH || 390);
const height = Number(process.env.VIEWPORT_HEIGHT || 844);
const captureDir = process.env.CAPTURE_DIR || "";
const scrollSelector = process.env.SCROLL_SELECTOR || "";
const captureSuffix = process.env.CAPTURE_SUFFIX ? `-${process.env.CAPTURE_SUFFIX}` : "";
const clipSelector = process.env.CLIP_SELECTOR || "";
const urls = process.argv.slice(2);

if (!urls.length) {
  throw new Error("Pass at least one page URL.");
}

const newTarget = await fetch(`http://127.0.0.1:${port}/json/new`, { method: "PUT" });
const target = await newTarget.json();
const socket = new WebSocket(target.webSocketDebuggerUrl);
await new Promise((resolve, reject) => {
  socket.addEventListener("open", resolve, { once: true });
  socket.addEventListener("error", reject, { once: true });
});

let seq = 0;
const pending = new Map();
socket.addEventListener("message", (event) => {
  const message = JSON.parse(event.data);
  if (!message.id || !pending.has(message.id)) return;
  const { resolve, reject } = pending.get(message.id);
  pending.delete(message.id);
  if (message.error) reject(new Error(message.error.message));
  else resolve(message.result);
});

const send = (method, params = {}) =>
  new Promise((resolve, reject) => {
    const id = ++seq;
    pending.set(id, { resolve, reject });
    socket.send(JSON.stringify({ id, method, params }));
  });

await send("Page.enable");
await send("Runtime.enable");
await send("Emulation.setDeviceMetricsOverride", {
  width,
  height,
  deviceScaleFactor: 1,
  mobile: false,
});

for (const url of urls) {
  await send("Page.navigate", { url });
  const deadline = Date.now() + 10000;
  let ready = false;
  while (Date.now() < deadline) {
    const state = await send("Runtime.evaluate", {
      expression: "document.readyState === 'complete' && document.documentElement.dataset.wikiReady === 'true'",
      returnByValue: true,
    });
    if (state.result.value) { ready = true; break; }
    await new Promise(resolve => setTimeout(resolve, 80));
  }
  if (!ready) throw new Error(`Page initialization timed out: ${url}`);
  await send("Runtime.evaluate", { expression: "document.fonts.ready", awaitPromise: true });
  if (scrollSelector) {
    await send("Runtime.evaluate", {
      expression: `document.querySelector(${JSON.stringify(scrollSelector)})?.scrollIntoView({ block: "start" })`,
    });
    await new Promise((resolve) => setTimeout(resolve, 120));
  }
  const result = await send("Runtime.evaluate", {
    expression: `(() => {
      const vw = window.innerWidth;
      const offenders = [...document.querySelectorAll("body *")]
        .map((el) => {
          const rect = el.getBoundingClientRect();
          return {
            tag: el.tagName.toLowerCase(),
            className: typeof el.className === "string" ? el.className : "",
            parentClassName: typeof el.parentElement?.className === "string" ? el.parentElement.className : "",
            text: (el.textContent || "").trim().slice(0, 60),
            left: Math.round(rect.left),
            right: Math.round(rect.right),
            width: Math.round(rect.width),
            scrollWidth: el.scrollWidth,
          };
        })
        .filter((item) => item.right > vw + 1 || item.left < -1 || item.width > vw + 1)
        .slice(0, 30);
      return {
        title: document.title,
        viewport: vw,
        documentScrollWidth: document.documentElement.scrollWidth,
        bodyScrollWidth: document.body.scrollWidth,
        offenders,
      };
    })()`,
    returnByValue: true,
  });
  console.log(JSON.stringify(result.result.value, null, 2));
  if (captureDir) {
    const filename = new URL(url).pathname === "/" ? "home" : new URL(url).pathname.replace(/[\/.]/g, "-").replace(/^-|-$/g, "");
    let clip;
    if (clipSelector) {
      const bounds = await send("Runtime.evaluate", {
        expression: `(() => {
          const rect = document.querySelector(${JSON.stringify(clipSelector)}).getBoundingClientRect();
          return {
            x: rect.left + window.scrollX,
            y: rect.top + window.scrollY,
            width: rect.width,
            height: rect.height,
          };
        })()`,
        returnByValue: true,
      });
      clip = { ...bounds.result.value, scale: 1 };
    }
    const shot = await send("Page.captureScreenshot", {
      format: "png",
      captureBeyondViewport: Boolean(clip),
      ...(clip ? { clip } : {}),
    });
    await writeFile(`${captureDir}/${filename}-${width}${captureSuffix}.png`, Buffer.from(shot.data, "base64"));
  }
}

socket.close();
await fetch(`http://127.0.0.1:${port}/json/close/${target.id}`);
