import assert from "node:assert/strict";
import test from "node:test";
import { initCore } from "../src/scripts/core.js";

function mount(t, storage, prefersDark = false) {
  const root = { dataset: {} }, label = {}, listeners = new Map();
  const button = { addEventListener: (name, fn) => listeners.set(name, fn) };
  const values = {
    document: { documentElement: root, querySelector: selector =>
      selector === "[data-theme-label]" ? label : selector === "[data-theme-toggle]" ? button : null },
    window: { matchMedia: () => ({ matches: prefersDark }) },
    localStorage: storage,
  };
  for (const [name, value] of Object.entries(values)) {
    const previous = Object.getOwnPropertyDescriptor(globalThis, name);
    Object.defineProperty(globalThis, name, { configurable: true, value });
    t.after(() => previous ? Object.defineProperty(globalThis, name, previous) : delete globalThis[name]);
  }
  return { root, label, toggle: () => listeners.get("click")() };
}

test("blocked storage leaves page initialization and theme controls usable", t => {
  const view = mount(t, null, true);
  Object.defineProperty(globalThis, "localStorage", { configurable: true, get() { throw new Error("Storage denied"); } });
  assert.doesNotThrow(initCore);
  assert.equal(view.root.dataset.theme, "dark");
  assert.doesNotThrow(view.toggle);
  assert.equal(view.root.dataset.theme, "light");
  assert.equal(view.label.textContent, "深色");
});

test("a failed preference write still changes the visible theme and label", t => {
  const view = mount(t, { getItem: () => "light", setItem() { throw new Error("Quota exceeded"); } });
  initCore();
  assert.doesNotThrow(view.toggle);
  assert.equal(view.root.dataset.theme, "dark");
  assert.equal(view.label.textContent, "浅色");
});

test("saved preferences override the OS and subsequent changes persist", t => {
  const writes = [];
  const view = mount(t, { getItem: () => "dark", setItem: (...args) => writes.push(args) });
  initCore();
  assert.equal(view.root.dataset.theme, "dark");
  view.toggle();
  assert.deepEqual(writes, [["wiki-theme", "light"]]);
  assert.equal(view.root.dataset.theme, "light");
});
