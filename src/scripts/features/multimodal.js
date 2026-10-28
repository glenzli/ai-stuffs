import { $ } from "../core.js";

export function visualTokenBudget(resolution, patch, merge, frames) {
  const patches = Math.ceil(resolution / patch) ** 2;
  return { patches, tokens: Math.ceil(patches / merge) * frames };
}

export function initMultimodal() {
  const resolution = $("[data-mm-resolution]");
  if (!resolution) return;
  const patch = $("[data-mm-patch]"), merge = $("[data-mm-merge]"), frames = $("[data-mm-frames]");
  const canvas = $("[data-mm-canvas]"), ctx = canvas.getContext("2d");
  const render = () => {
    const r = Number(resolution.value), p = Number(patch.value), m = Number(merge.value), f = Number(frames.value);
    const { patches, tokens } = visualTokenBudget(r, p, m, f);
    $("[data-mm-patches]").textContent = patches.toLocaleString();
    $("[data-mm-tokens]").textContent = tokens.toLocaleString();
    $("[data-mm-total]").textContent = (tokens + 512).toLocaleString();
    $("[data-mm-frames-value]").textContent = f;
    $("[data-mm-note]").textContent = `${r}×${r} 输入，每帧 ${patches} 个 patch，合并后 ${Math.ceil(patches / m)} 个视觉位置。语言序列的长度变化和视觉塔本身的计算量要分开记账。`;
    ctx.fillStyle = "#edf1f5"; ctx.fillRect(0,0,448,448);
    ctx.fillStyle = "#ffffff"; ctx.fillRect(58,34,332,380);
    ctx.fillStyle = "#182b36"; ctx.font = "bold 27px sans-serif"; ctx.fillText("SAMPLE RECEIPT",78,84);
    ctx.font = "14px monospace"; ctx.fillText("2026-10-26   /   NO. 018",78,111);
    ctx.strokeStyle = "#dce2e6"; ctx.beginPath(); ctx.moveTo(78,134);ctx.lineTo(370,134);ctx.stroke();
    ["NOTEBOOK        2   36.00", "PENCIL          4   12.00", "FOLDER          1    8.00"].forEach((line,i)=>ctx.fillText(line,78,172+i*37));
    ctx.fillStyle = "#b44939";ctx.font = "bold 24px monospace";ctx.fillText("TOTAL       56.00",78,308);
    ctx.fillStyle = "#4d5963";ctx.font = "12px sans-serif";ctx.fillText("A synthetic input, not model output",78,346);
    for(let i=0;i<52;i++){ctx.fillStyle="#20272d";ctx.fillRect(78+i*5.4,366,(i%3)+1,24);}
    ctx.strokeStyle = "rgba(38,110,174,.24)";ctx.lineWidth=0.75;
    const grid = Math.ceil(r/p), cell = 448/grid;
    ctx.beginPath();for(let i=1;i<grid;i++){ctx.moveTo(i*cell,0);ctx.lineTo(i*cell,448);ctx.moveTo(0,i*cell);ctx.lineTo(448,i*cell);}ctx.stroke();
  };
  [resolution,patch,merge,frames].forEach(input=>input.addEventListener("input",render));
  render();
}
