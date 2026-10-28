import { $ } from "../core.js";

const formats = { fp16: [5, 10], bf16: [8, 7], fp32: [8, 23] };
function formatSpec(format) {
  if (!Object.hasOwn(formats, format)) throw new RangeError("格式必须为 fp16、bf16 或 fp32。");
  const [exponentBits, fractionBits] = formats[format];
  const maxExponent = 2 ** (exponentBits - 1) - 1, minExponent = 1 - maxExponent;
  return { fractionBits, maxExponent, minExponent,
    minSubnormal: 2 ** (minExponent - fractionBits),
    maxFinite: (2 - 2 ** -fractionBits) * 2 ** maxExponent };
}
function binaryExponent(value) {
  let exponent = Math.floor(Math.log2(value));
  // log2 may round up immediately below a power of two.
  if (2 ** exponent > value) exponent--;
  return exponent;
}
function nearestEven(value) {
  const lower = Math.floor(value), fraction = value - lower;
  return fraction > .5 || (fraction === .5 && lower % 2 === 1) ? lower + 1 : lower;
}

export function roundFormat(value, format = "fp16") {
  const spec = formatSpec(format);
  if (typeof value !== "number") throw new TypeError("数值必须为 number。");
  if (!Number.isFinite(value) || value === 0) return value;
  const sign = value < 0 ? -1 : 1, magnitude = Math.abs(value);
  const exponent = binaryExponent(magnitude);
  if (exponent > spec.maxExponent) return sign * Infinity;
  const step = 2 ** (Math.max(spec.minExponent, exponent) - spec.fractionBits);
  const rounded = nearestEven(magnitude / step) * step;
  return sign * (rounded > spec.maxFinite ? Infinity : rounded);
}

export function formatReadout(value, format = "fp16") {
  if (!Number.isFinite(value)) throw new RangeError("请输入有限数值。");
  const spec = formatSpec(format), rounded = roundFormat(value, format);
  const exponent = value === 0 ? spec.minExponent : Math.max(spec.minExponent, binaryExponent(Math.abs(value)));
  return { value, rounded, error: rounded - value, ulp: 2 ** (exponent - spec.fractionBits), ...spec };
}

export function sumTrace(values, format = "fp16") {
  formatSpec(format);
  if (!Array.isArray(values) || !values.every(Number.isFinite)) throw new RangeError("求和输入必须为有限数值数组。");
  let result = 0;
  const rows = values.map((input, index) => {
    const rounded = roundFormat(input, format), before = result;
    result = roundFormat(before + rounded, format);
    return { index, input, rounded, before, after: result };
  });
  return { rows, result };
}

export function stabilityExample(offset = 1000) {
  if (!Number.isFinite(offset) || Math.abs(offset) > 1000) throw new RangeError("偏移须在 −1000 到 1000 之间。");
  const logits = [offset, offset - 1, offset - 2], exp = logits.map(Math.exp), total = exp.reduce((s, x) => s + x);
  const stableExp = logits.map(x => Math.exp(x - offset)), stableSum = stableExp.reduce((s, x) => s + x);
  return { logits, naive: exp.map(x => x / total), stable: stableExp.map(x => x / stableSum), lse: offset + Math.log(stableSum) };
}

const number = x => Number.isNaN(x) ? "NaN" : x === Infinity ? "+∞" : x === -Infinity ? "−∞" : Object.is(x, -0) ? "−0" : Number(x.toPrecision(9)).toString();
export function initNumericalComputing() {
  const format = $("[data-nc-format]"); if (!format) return;
  const input = $("[data-nc-value]"), order = $("[data-nc-order]"), offset = $("[data-nc-offset]");
  const set = (name, value) => { const node = $(`[data-nc-${name}]`); if (node) node.textContent = value; };
  const render = () => {
    const values = order.value === "smallfirst" ? [10000, -10000, 1] : [10000, 1, -10000];
    const sum = sumTrace(values, format.value);
    set("sum", number(sum.result)); set("exact", "1");
    $("[data-nc-rows]").innerHTML = sum.rows.map(row => `<tr><td>${row.index + 1}</td><td>${number(row.input)}</td><td>${number(row.rounded)}</td><td>${number(row.before)}</td><td>${number(row.after)}</td></tr>`).join("");
    try {
      const value = input.value.trim() ? Number(input.value) : NaN;
      const result = formatReadout(value, format.value);
      set("rounded", number(result.rounded)); set("error", number(result.error)); set("ulp", number(result.ulp));
      set("note", `最小正 subnormal 为 ${number(result.minSubnormal)}，最大有限值为 ${number(result.maxFinite)}。${Number.isFinite(result.rounded) ? "每次加法都在所选格式下舍入，旧的舍入误差不会由后续相减自动恢复。" : "此输入舍入后上溢到无穷；误差不再是有限值。"}`);
    } catch (error) {
      if (!(error instanceof RangeError)) throw error;
      ["rounded", "error", "ulp"].forEach(name => set(name, "输入无效")); set("note", error.message);
    }
    const stable = stabilityExample(Number(offset.value));
    set("offset-value", offset.value);
    set("naive", `[${stable.naive.map(number).join(", ")}]${stable.naive.some(x => !Number.isFinite(x)) ? "（直接指数出现上溢或下溢）" : ""}`);
    set("stable", `[${stable.stable.map(x => x.toFixed(6)).join(", ")}]`); set("lse", number(stable.lse));
  };
  [format, input, order, offset].forEach(control => { control.addEventListener("input", render); control.addEventListener("change", render); });
  render();
}
