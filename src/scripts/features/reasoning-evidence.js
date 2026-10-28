import { $ } from "../core.js";

const distributions = {
  base: [[0.2, 0.1, 0.7], [0.04, 0.01, 0.95]],
  efficient: [[0.6, 0.1, 0.3], [0.009, 0.001, 0.99]],
  process: [[0.29, 0.01, 0.7], [0.049, 0.001, 0.95]],
};

export function pathEvidence(mode = "base", k = 16) {
  if (typeof mode !== "string" || !Object.hasOwn(distributions, mode)) {
    throw new RangeError("证据模式无效。");
  }
  if (!Number.isInteger(k) || k < 1 || k > 256) {
    throw new RangeError("采样次数须为 1 到 256 的整数。");
  }
  const rows = distributions[mode].map(([pSound, pLucky, pWrong], index) => {
    const answer = pSound + pLucky;
    return { task: index === 0 ? "A" : "B", pSound, pLucky, pWrong, answer, passk: 1 - (1 - answer) ** k, process: pSound };
  });
  const pass1 = rows.reduce((sum, row) => sum + row.answer, 0) / rows.length;
  const passk = rows.reduce((sum, row) => sum + row.passk, 0) / rows.length;
  const sound = rows.reduce((sum, row) => sum + row.process, 0) / rows.length;
  return { rows, pass1, passk, sound, soundGivenCorrect: sound / pass1 };
}

const modeNotes = {
  base: "正确答案包含可靠推理与侥幸答对，答案分数不能替代过程证据。",
  efficient: "单次均值上升，任务乙的正确概率下降，较大采样次数的平均覆盖更低。",
  process: "各题答案正确率不变，可靠推理占比提高。",
};

export function initReasoningEvidence() {
  const modeInput = $("[data-re-mode]");
  if (!modeInput) return;
  const kInput = $("[data-re-k]");
  const percent = value => `${(value * 100).toFixed(2)}%`;
  const setText = (selector, value) => {
    const output = $(selector);
    if (output) output.textContent = value;
  };
  const render = () => {
    const k = kInput ? Number(kInput.value) : 16;
    const result = pathEvidence(modeInput.value, k);
    setText("[data-re-k-value]", String(k));
    setText("[data-re-pass1]", percent(result.pass1));
    setText("[data-re-passk]", percent(result.passk));
    setText("[data-re-sound]", percent(result.sound));
    setText("[data-re-conditional]", percent(result.soundGivenCorrect));
    const rows = $("[data-re-rows]");
    if (rows) rows.innerHTML = result.rows.map(row =>
      `<tr><th scope="row">${row.task === "A" ? "任务甲" : "任务乙"}</th><td>${percent(row.pSound)}</td><td>${percent(row.pLucky)}</td><td>${percent(row.pWrong)}</td><td>${percent(row.passk)}</td></tr>`
    ).join("");
    setText("[data-re-note]", `固定教学分布；两题等权、各题独立同分布采样。${modeNotes[modeInput.value]}三类路径均保留正概率。`);
  };
  modeInput.addEventListener("change", render);
  kInput?.addEventListener("input", render);
  render();
}
