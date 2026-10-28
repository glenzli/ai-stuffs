import { $ } from "../core.js";

export function quantizeSymmetric(values, bits) {
  const qmax = 2 ** (bits-1)-1;
  const max = Math.max(...values.map(Math.abs));
  const scale = max === 0 ? 1 : max/qmax;
  const rows = values.map(value=> {
    const integer = Math.max(-qmax,Math.min(qmax,Math.round(value/scale)));
    return { value, integer, restored: integer*scale, error: integer*scale-value };
  });
  return { rows, scale, mse: rows.reduce((sum,row)=>sum+row.error**2,0)/rows.length };
}
export function initEfficientInference() {
  const bits = $("[data-quant-bits]"), outlier = $("[data-quant-outlier]");
  if (!bits) return;
  const renderQuant = () => {
    const b=Number(bits.value), v=Number(outlier.value), result=quantizeSymmetric([-.92,-.43,.08,.37,.84,v],b);
    $("[data-quant-outlier-value]").textContent = v.toFixed(1);
    $("[data-quant-scale]").textContent = result.scale.toFixed(3);
    $("[data-quant-mse]").textContent = result.mse.toFixed(4);
    $("[data-quant-ratio]").textContent = `${16/b}×`;
    $("[data-quant-table]").innerHTML=result.rows.map(row=>`<tr><td>${row.value.toFixed(2)}</td><td>${row.integer}</td><td>${row.restored.toFixed(3)}</td><td>${row.error.toFixed(3)}</td></tr>`).join("");
  };
  [bits,outlier].forEach(input=>input.addEventListener("input",renderQuant));renderQuant();
  const reject = $("[data-spec-reject]");let step=0;
  const renderSpec=()=>{
    const rejected=Number(reject.value), labels=["先","检查","资料","再答"];
    const tokens=labels.map((label,i)=>`<span class="${step===0?'':step===1?'pending':i<rejected?'accepted':'rejected'}">${step===0?'?':label}</span>`);
    if(step===3)tokens.push('<span class="corrected">修正/额外</span>');
    $("[data-spec-tape]").innerHTML=tokens.join("");
    const states=[
      ["共同前缀已缓存","draft 与 target 都从同一已确认前缀出发。"],
      ["draft 提出 4 个候选","这四个位置来自便宜模型的连续采样，尚未被 target 接受。"],
      ["target 一次前向验证",rejected===4?"四个草稿均被接受；已有 logits 可继续给出额外 token。":`前 ${rejected} 个接受，第 ${rejected+1} 个拒绝，后续草稿全部作废。它们对应的 KV 也不能继续当作已确认前缀。`],
      ["确认输出并修正状态",rejected===4?"发出 4 个已接受 token 和 1 个额外 token；对齐两模型的下一轮输入与缓存。":`发出 ${rejected} 个已接受 token 与 1 个修正 token；回滚无效草稿缓存，下一轮从正确前缀重新提出。`],
    ];
    $("[data-spec-title]").textContent=states[step][0];$("[data-spec-copy]").textContent=states[step][1];$("[data-spec-next]").disabled=step===3;
  };
  $("[data-spec-next]").addEventListener("click",()=>{step=Math.min(3,step+1);renderSpec();});
  $("[data-spec-reset]").addEventListener("click",()=>{step=0;renderSpec();});
  reject.addEventListener("change",()=>{step=0;renderSpec();});renderSpec();
}
