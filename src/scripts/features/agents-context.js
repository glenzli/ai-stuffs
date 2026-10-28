import { $, $$ } from "../core.js";

export function initAgentsContext() {
  if (!$("[data-agent-title]")) return;
  const states=[
    ["目标进入上下文","用户要查库存；模型尚未观察到数据库当前值。权重中的历史知识不能代替实时查询。"],
    ["模型生成动作","assistant 输出工具名 inventory.lookup 与参数 sku=A17。此时只是动作提议，不能报告查询成功。"],
    ["运行时返回观测","校验后执行查询，tool 返回 available=12、查询时间和状态 success。这是本次查询返回的环境观测。"],
    ["基于观测继续生成","模型看到工具结果，再回答“当前可用库存 12”。本轮增加了上下文与缓存，未执行训练反传，也没有修改模型权重。"],
  ];
  let step=0;
  const render=()=>{
    $("[data-agent-title]").textContent=states[step][0];$("[data-agent-copy]").textContent=states[step][1];
    $$("[data-agent-stages] > div").forEach((node,i)=>{node.style.background=i===step?"var(--accent-soft)":"transparent";});
    $("[data-agent-next]").disabled=step===3;
  };
  $("[data-agent-next]").addEventListener("click",()=>{step=Math.min(3,step+1);render();});
  $("[data-agent-reset]").addEventListener("click",()=>{step=0;render();});render();
}
