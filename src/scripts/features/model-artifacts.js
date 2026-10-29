import { $, $$ } from "../core.js";

export function initModelArtifacts() {
    const buttons = $$("[data-artifact]");
    if (!buttons.length) return;
    const artifacts = {
      config: {
        kicker: "MODEL CONFIG",
        title: "定义权重形状与前向规则",
        copy: "config 与模型实现共同定义结构和计算规则。宽度、词表等不匹配可能在加载时暴露；RoPE 频率、Norm epsilon、注意力窗口等即使不改变权重形状，也可能改变输出。",
        deploy: "必须",
        resume: "结构侧必须；还需数据/优化器状态",
        risk: "权重可加载也不证明配置兼容；核对配置与实现版本，再用固定输入比对 logits。",
        code: `{
  "hidden_size": 4096,
  "num_hidden_layers": 32,
  "num_attention_heads": 32,
  "num_key_value_heads": 8
}`,
      },
      tokenizer: {
        kicker: "TOKENIZER ARTIFACT",
        title: "没有 tokenizer，权重无法接收同一种输入",
        copy: "词表、merges、normalization、special tokens、EOS、chat template 与 byte fallback 必须一起版本化。文本相同但 tokenizer 不同，模型看到的是另一串 ID。",
        deploy: "必须",
        resume: "必须",
        risk: "推理服务换 template 或 special token，生成边界与训练分布错位。",
        code: `tokens = tokenizer.apply_chat_template(
  messages,
  tokenize=True,
  add_generation_prompt=True,
)`,
      },
      safetensors: {
        kicker: "WEIGHT SHARDS",
        title: "一张或多张 tensor 文件承载参数",
        copy: "safetensors 以张量字典保存权重，避免任意 pickle 反序列化风险；大模型常分成多个 shard，并配合 index 定位每个张量。",
        deploy: "必须",
        resume: "必须，但还需恢复训练状态",
        risk: "shard 缺失/损坏、dtype 不一致或加载时静默替换成随机初始化。",
        code: `model-00001-of-00004.safetensors
model-00002-of-00004.safetensors
...`,
      },
      index: {
        kicker: "SHARD INDEX",
        title: "记录哪个 tensor 存在哪一片",
        copy: "index 让 loader 不用猜测 shard 内容。检查所有层、embedding、head 和 norm 张量是否齐全，以及是否有未预期前缀。",
        deploy: "必须",
        resume: "必须",
        risk: "index 与 shard 版本不一致；某些层漏加载但框架只给 warning。",
        code: `{
  "weight_map": {
    "model.layers.0.self_attn.q_proj.weight":
      "model-00001-of-00004.safetensors"
  }
}`,
      },
      generation: {
        kicker: "GENERATION CONFIG",
        title: "采样与停止策略决定生成行为",
        copy: "temperature、top_p、max_new_tokens、EOS、停止字符串和 repetition penalty 会显著改变生成结果；评估时必须与部署协议一起版本化。",
        deploy: "通常必须",
        resume: "训练恢复不依赖，评估需要一致",
        risk: "离线评估与线上采样参数不同，导致同一模型表现差异巨大。",
        code: `{
  "temperature": 0.7,
  "top_p": 0.9,
  "max_new_tokens": 512
}`,
      },
      optimizer: {
        kicker: "OPTIMIZER STATE",
        title: "优化器状态是继续训练的一部分",
        copy: "Adam 的 m/v、master weights、参数分片与分布式 optimizer shards 都必须恢复。只加载模型权重会重新开始优化器轨迹。",
        deploy: "通常不需要",
        resume: "必须",
        risk: "只恢复参数，继续训练出现 loss 跳变；shard 映射错误导致状态对错参数。",
        code: `optimizer.state_dict():
  exp_avg, exp_avg_sq,
  step, param_groups,
  sharded_states`,
      },
      trainer: {
        kicker: "TRAINER STATE",
        title: "step、scheduler 与数据游标共同定义位置",
        copy: "训练 step、学习率 scheduler、gradient accumulation 边界、RNG、data sampler cursor 与已消耗 token 数需要一起恢复。",
        deploy: "不需要",
        resume: "必须",
        risk: "恢复后重复/跳过数据，学习率与 warmup 位置错位，loss 曲线不可比较。",
        code: `{
  "global_step": 18420,
  "lr": 2.7e-5,
  "epoch": 0.42,
  "tokens_seen": 918273000
}`,
      },
      manifest: {
        kicker: "RUN MANIFEST",
        title: "把分散的版本信息绑在一起",
        copy: "记录代码 commit、数据 shard hash、tokenizer hash、config、环境、并行/精度、评估集与输出摘要。它是“这个模型从哪来”的总索引。",
        deploy: "强烈建议",
        resume: "必须",
        risk: "只有模型文件名和 seed，无法重建数据混合、环境与评估版本。",
        code: `{
  "code_commit": "...",
  "data_manifest": "...",
  "tokenizer_hash": "...",
  "parallelism": "dp8-zero3",
  "eval_suite": "..."
}`,
      },
    };
    const render = (key) => {
      const data = artifacts[key];
      buttons.forEach((button) => {
        button.setAttribute("aria-pressed", String(button.dataset.artifact === key));
      });
      $("[data-artifact-kicker]").textContent = data.kicker;
      $("[data-artifact-title]").textContent = data.title;
      $("[data-artifact-copy]").textContent = data.copy;
      $("[data-artifact-deploy]").textContent = data.deploy;
      $("[data-artifact-resume]").textContent = data.resume;
      $("[data-artifact-risk]").textContent = data.risk;
      $("[data-artifact-code]").textContent = data.code;
    };
    buttons.forEach((button) => {
      button.addEventListener("click", () => render(button.dataset.artifact));
    });
    render("config");
}
