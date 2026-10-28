import { $ } from "../core.js";

export const engineeringQuestions = [
  { id:"delivery",title:"数据实际被读了多少次？",question:"10 条记录、3 个 DP rank。Sampler 补齐，Loader 每卡 batch=2，不丢尾。单轮实际读取多少次？",options:["10 次","12 次","9 次"],correct:1,why:"Sampler 先补齐到 12 个索引，每卡 4 条、2 个 batch。原始 10 条都覆盖，其中 2 次读取重复。",link:"finetuning-recipe.html#sample-delivery" },
  { id:"supervision",title:"一次更新有多少监督位置？",question:"使用上一题数据；每条输入3 token、回答2 token，共5 token，完整保留。仅回答计 loss，两个 microbatch 累积后更新一次。这次全局更新有多少监督 token？",options:["24","30","60"],correct:0,why:"3 rank × 每卡2条 × 累积2步 × 每条2个回答标签 = 24。输入上下文仍参与计算，监督分母只统计标签。",link:"parallel-correctness.html#global-objective" },
  { id:"shape",title:"前向张量的形状是什么？",question:"每卡 microbatch=2，序列长度5，hidden=4。查表后 X 的形状是什么？",options:["[3,2,5,4]","[2,5,4]","[2,2,4]"],correct:1,why:"单卡逻辑输入为 [batch,sequence,hidden]=[2,5,4]。DP rank 数不作为本地张量的额外维度，监督 mask 不缩短输入序列。",link:"backpropagation.html#vjp-rules" },
  { id:"vocab",title:"目标类别位于哪个分片？",question:"6 类词表按顺序分到两个 TP rank：0..2 与 3..5，目标 ID=4。哪个 rank 提供目标 logit？",options:["两个 rank 都各提供一份","rank 1","rank 0"],correct:1,why:"目标4由 rank1持有。所有分片参与全局最大值与指数和，非目标分片的目标值贡献为0；最终梯度仍覆盖全部类别。",link:"distributed-execution.html#vocab-parallel" },
  { id:"cache",title:"采样数量与缓存长度怎样对应？",question:"标准自回归输入8 token，已完成 prefill 并总共采样4个输出，最后一个输出尚未作为输入。应有多少位置的 KV？",options:["12","8","11"],correct:2,why:"缓存只覆盖已输入位置：8个 prompt + 前3个输出 = 11。最后采样的第4个输出尚未计算它自己的 K/V。",link:"inference-engine.html#serving-schedule" },
  { id:"layout",title:"非连续视图能否直接展平？",question:"一个视图 shape=[3,2]、stride=[4,2]、offset=0，逻辑地址依次为0,2,4,6,8,10。展平为长度6是否必须复制？",options:["必须，因为不是连续张量","不必，可以使用 stride=2 的一维视图","可以，但只能把值重新排序"],correct:1,why:"逻辑访问地址是等差序列，可以用 [6]、stride=[2] 表示，无需改变顺序或复制。非连续不是 view 失败的充分条件。",link:"tensor-layout.html#view-copy" },
  { id:"checkpoint",title:"重算需要保持什么？",question:"激活 checkpoint 区域含 dropout。希望反向沿原前向路径求导，除了输入和权重，还应处理什么？",options:["只恢复 loss 数值","保持对应随机状态，并检查重执行副作用","把学习率减半"],correct:1,why:"重算若生成不同 mask，就可能对应另一条路径。需要按框架规则保留 RNG，并检查函数中的状态更新或副作用。",link:"activation-checkpointing.html#recompute-correctness" },
];

export function gradeEngineeringAnswer(id, choice) {
  const item=engineeringQuestions.find(q=>q.id===id);
  if(!item||!Number.isInteger(choice)||choice<0||choice>=item.options.length) throw new RangeError("题目或选项无效。");
  return {correct:choice===item.correct,why:item.why,link:item.link};
}

export function initEngineeringCheck(){
  const root=$("[data-check-board]");if(!root)return;
  let step=0,choice=null,revealed=false;
  const answers=new Map();
  const render=()=>{
    const q=engineeringQuestions[step];
    $("[data-check-progress]").textContent=`${step+1}/${engineeringQuestions.length}`;
    $("[data-check-title]").textContent=q.title;$("[data-check-question]").textContent=q.question;
    $("[data-check-options]").innerHTML=q.options.map((label,index)=>`<label><input type="radio" name="engineering-answer" value="${index}" ${choice===index?"checked":""} ${revealed?"disabled":""}><span>${label}</span></label>`).join("");
    $("[data-check-explanation]").hidden=!revealed;
    if(revealed){const result=gradeEngineeringAnswer(q.id,choice);$("[data-check-explanation]").innerHTML=`<strong>${result.correct?"这项判断正确。":"再核对计算口径。"}</strong><p>${result.why}</p><a href="${result.link}">回到对应机制</a>`;}
    $("[data-check-reveal]").disabled=choice===null||revealed;
    $("[data-check-next]").disabled=!revealed||step===engineeringQuestions.length-1;
    $("[data-check-score]").textContent=`已核对 ${answers.size} 项，其中 ${[...answers.values()].filter(Boolean).length} 项首次选择正确`;
    $("[data-check-options]").querySelectorAll("input").forEach(input=>input.addEventListener("change",()=>{choice=Number(input.value);$("[data-check-reveal]").disabled=false;}));
  };
  $("[data-check-reveal]").addEventListener("click",()=>{if(choice===null)return;answers.set(step,gradeEngineeringAnswer(engineeringQuestions[step].id,choice).correct);revealed=true;render();});
  $("[data-check-next]").addEventListener("click",()=>{if(!revealed)return;step=Math.min(step+1,engineeringQuestions.length-1);choice=null;revealed=false;render();});
  $("[data-check-reset]").addEventListener("click",()=>{step=0;choice=null;revealed=false;answers.clear();render();});render();
}
