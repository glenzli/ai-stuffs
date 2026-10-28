import { $ } from "../core.js";

const layouts = {
  base: { shape: [3, 4], strides: [4, 1], offset: 0 },
  transpose: { shape: [4, 3], strides: [1, 4], offset: 0 },
  slice: { shape: [3, 2], strides: [4, 2], offset: 0 },
  offset: { shape: [2, 4], strides: [4, 1], offset: 4 },
};

export function layoutReadout({ mode = "transpose", materialize = false, write = false } = {}) {
  if (typeof mode !== "string" || !Object.hasOwn(layouts, mode)) {
    throw new RangeError("布局须为 base、transpose、slice 或 offset。");
  }
  if (typeof materialize !== "boolean" || typeof write !== "boolean") {
    throw new TypeError("复制与写入选项须为布尔值。");
  }
  const shape = layouts[mode].shape.slice();
  let strides = layouts[mode].strides.slice(), offset = layouts[mode].offset;
  const baseStorage = Array.from({ length: 12 }, (_, index) => index);
  let tensorStorage = baseStorage;
  const addressGrid = () => Array.from({ length: shape[0] }, (_, i) =>
    Array.from({ length: shape[1] }, (_, j) => offset + i * strides[0] + j * strides[1])
  );
  let addresses = addressGrid();
  if (materialize) {
    tensorStorage = addresses.flat().map(address => baseStorage[address]);
    strides = [shape[1], 1];
    offset = 0;
    addresses = addressGrid();
  }
  if (write) tensorStorage[addresses[0][0]] = 99;
  const values = addresses.map(row => row.map(address => tensorStorage[address]));
  const flatAddresses = addresses.flat(), step = flatAddresses[1] - flatAddresses[0];
  // A strided flat view can exist even when the tensor is not contiguous.
  const flatViewPossible = flatAddresses.every((address, index) =>
    index === 0 || address - flatAddresses[index - 1] === step
  );
  return {
    shape, strides, offset, addresses, values, baseStorage, tensorStorage,
    sharesStorage: tensorStorage === baseStorage,
    contiguous: strides[1] === 1 && strides[0] === shape[1],
    flatViewPossible, flatStride: flatViewPossible ? step : null,
    flatValues: values.flat(),
  };
}

export function initTensorLayout() {
  const mode = $("[data-tl-mode]");
  if (!mode) return;
  const materialize = $("[data-tl-materialize]"), write = $("[data-tl-write]");
  const set = (name, value) => {
    const node = $(`[data-tl-${name}]`);
    if (node) node.textContent = String(value);
  };
  const render = () => {
    const result = layoutReadout({ mode: mode.value, materialize: materialize?.checked ?? false, write: write?.checked ?? false });
    set("shape", `[${result.shape.join(", ")}]`);
    set("strides", `[${result.strides.join(", ")}]`);
    set("offset", result.offset);
    set("shared", result.sharesStorage ? "共享" : "独立副本");
    set("view", result.flatViewPossible ? `可行（步幅 ${result.flatStride}）` : "不可行（需复制）");

    const addressed = new Set(result.sharesStorage ? result.addresses.flat() : []);
    const storage = $("[data-tl-storage]");
    if (storage) storage.innerHTML = result.baseStorage.map((value, index) => {
      const state = addressed.has(index) ? "当前张量引用此地址" : "当前张量未引用此地址";
      return `<span class="storage-cell${addressed.has(index) ? " is-addressed" : ""}" title="原始存储地址 ${index}，值 ${value}；${state}"><small>地址 ${index}</small><strong>值 ${value}</strong></span>`;
    }).join("");

    const rows = $("[data-tl-rows]");
    if (rows) rows.innerHTML = result.addresses.flatMap((row, i) => row.map((address, j) =>
      `<tr><td>[${i}, ${j}]</td><td>${address - result.offset}</td><td>${address}</td><td>${result.values[i][j]}</td></tr>`
    )).join("");
    const matrix = $("[data-tl-matrix]");
    if (matrix) {
      matrix.style.gridTemplateColumns = `repeat(${result.shape[1]}, minmax(0, 1fr))`;
      matrix.innerHTML = result.values.flatMap((row, i) => row.map((value, j) =>
        `<span title="逻辑坐标 [${i}, ${j}]，存储地址 ${result.addresses[i][j]}">${value}</span>`
      )).join("");
    }

    const flatten = result.flatViewPossible
      ? `逻辑遍历的相邻地址差恒定，一维视图步幅 ${result.flatStride}。`
      : "逻辑行边界的地址差改变，展平为一维排列需复制。";
    const sharing = result.sharesStorage ? "当前张量共享原始存储。" : "已按逻辑行优先顺序复制到独立存储，原始存储保持不变。";
    const mutation = !write?.checked ? "" : result.sharesStorage
      ? `逻辑 [0, 0] 写入 99，原始存储地址 ${result.addresses[0][0]} 同步改变。`
      : "逻辑 [0, 0] 写入 99，仅独立副本改变。";
    set("note", `${result.contiguous ? "连续" : "非连续"}。${flatten}${sharing}${mutation}`);
  };
  for (const control of [mode, materialize, write]) {
    for (const event of ["input", "change"]) control?.addEventListener(event, render);
  }
  render();
}
