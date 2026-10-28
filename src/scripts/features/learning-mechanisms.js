import { $ } from "../core.js";

export function inductionTrace({ example = "B", previous = true, matching = true, copy = true } = {}) {
  if (example !== "B" && example !== "D") throw new RangeError("example 必须为 B 或 D。");
  for (const [name, enabled] of [["previous", previous], ["matching", matching], ["copy", copy]]) {
    if (typeof enabled !== "boolean") throw new TypeError(`${name} 必须为布尔值。`);
  }

  const tokens = ["A", example, "C", "A"];
  const last = tokens.length - 1;
  const previousTokens = tokens.map((token, index) => previous && index > 0 ? tokens[index - 1] : null);
  const found = matching ? previousTokens.findIndex((key, index) => index < last && key === tokens[last]) : -1;
  const matchIndex = found < 0 ? null : found;
  const copied = copy && matchIndex !== null ? tokens[matchIndex] : null;
  const logits = { B: 0, D: 0 };
  if (copied !== null) logits[copied] += 2;
  const massB = Math.exp(logits.B);
  const massD = Math.exp(logits.D);
  const total = massB + massD;
  const probabilities = { B: massB / total, D: massD / total };
  const answer = probabilities.B === probabilities.D ? "不区分" : probabilities.B > probabilities.D ? "B" : "D";
  const stages = [
    {
      id: "previous", label: "前项写入", enabled: previous, active: previous,
      description: previous
        ? `前项写入得到 [无, A, ${example}, C]。`
        : "前项写入已关闭，各位置均未写入前项信息。",
    },
    {
      id: "matching", label: "历史匹配", enabled: matching, active: matchIndex !== null,
      description: !matching
        ? "历史匹配已关闭。"
        : matchIndex === null
          ? "当前查询 A 没有找到前项信息为 A 的历史位置。"
          : `当前查询 A 匹配历史位置 ${matchIndex} 的前项 A，该位置的值为 ${tokens[matchIndex]}；仅搜索位置小于 ${last} 的键。`,
    },
    {
      id: "copy", label: "复制到输出分数", enabled: copy, active: copied !== null,
      description: !copy
        ? "复制已关闭，B/D 的 logits 均保持 0。"
        : copied === null
          ? "没有可复制的匹配值，B/D 的 logits 均保持 0。"
          : `复制位置 ${matchIndex} 的 ${copied}，给该候选的 logit 加 2。`,
    },
  ];
  return { tokens, previousTokens, matchIndex, copied, probabilities, answer, stages };
}

export function initLearningMechanisms() {
  const exampleInput = $("[data-icl-example]");
  if (!exampleInput) return;
  const previousInput = $("[data-icl-previous]");
  const matchingInput = $("[data-icl-matching]");
  const copyInput = $("[data-icl-copy]");
  const setText = (selector, value) => {
    const output = $(selector);
    if (output) output.textContent = value;
  };
  const render = () => {
    const result = inductionTrace({
      example: exampleInput.value,
      previous: previousInput?.checked ?? true,
      matching: matchingInput?.checked ?? true,
      copy: copyInput?.checked ?? true,
    });
    const tape = $("[data-icl-tokens]");
    if (tape) tape.innerHTML = result.tokens.map((token, index) => {
      let role = "历史 token";
      let className = "";
      if (index === result.tokens.length - 1) {
        role = "当前查询";
        className = "pending";
      } else if (index === result.matchIndex) {
        role = result.copied === null ? "历史匹配位置" : "历史匹配位置与复制值";
        className = "accepted";
      } else if (result.stages[0].active && index === 0) {
        role = "前项写入来源";
        className = "corrected";
      }
      return `<span class="${className}" title="位置 ${index}：${role}">${token}</span>`;
    }).join("");
    const rows = $("[data-icl-rows]");
    if (rows) rows.innerHTML = result.tokens.map((token, index) =>
      `<tr><td>${index}</td><td>${token}</td><td>${result.previousTokens[index] ?? "未写入"}</td></tr>`
    ).join("");
    setText("[data-icl-copy-result]", result.copied ?? "未复制");
    setText("[data-icl-prob-b]", `${(result.probabilities.B * 100).toFixed(1)}%`);
    setText("[data-icl-prob-d]", `${(result.probabilities.D * 100).toFixed(1)}%`);
    setText("[data-icl-note]", `${result.stages.map(stage => stage.description).join(" ")} 判定：${result.answer}。这是固定规则的因果示意，未加载或训练 Transformer；消融只验证本例手工构造的写入、匹配、复制依赖。`);
  };
  for (const input of [exampleInput, previousInput, matchingInput, copyInput]) {
    for (const event of ["input", "change"]) input?.addEventListener(event, render);
  }
  render();
}
