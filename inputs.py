"""Pipeline 统一输入声明。

整个 pipeline 的全部外部输入集中在配置文件的 `inputs:` 段声明,
本模块负责把它展开成各 Step 的参数 (Step 段里显式写的值优先级更高,
inputs 只补默认值; 命令行 --set 最后应用, 优先级最高)。

inputs 段结构 (configs/*.yaml 里有完整注释示例):

```yaml
inputs:
  # 基座模型 (训练 + 评估共用): 可换任意 HF 模型或本地路径
  base_model:
    hf_id: Qwen/Qwen3.5-9B                 # HF repo id (下载兜底)
    local_dir: /dockerdata/models/Qwen3.5-9B   # 本地落地位置 (训练/评估读这里)
    cfs_src: /cfs/.../models/Qwen3.5-9B    # CFS 备份 (优先于 HF 下载)

  # 扩散模型三件套 (prepare_patterns 的 pattern -> 隐藏文字图):
  # 可换成其他 SD1.5/SDXL 底模 + 对应 ControlNet
  diffusion:
    base: SG161222/Realistic_Vision_V5.1_noVAE
    controlnet: monster-labs/control_v1p_sd15_qrcode_monster
    vae: stabilityai/sd-vae-ft-mse

  # pattern 合成输入: 文字池 + 字体 + 已有 pattern (优先复用)
  pattern_text_pool: synthesis/inputs/patterns      # 含 txt_from_segments/ txt_from_api/
  pattern_fonts: synthesis/inputs/fonts             # NotoSansSC Bold/Black
  patterns_dir: synthesis/inputs/patterns/patterns  # 已有 pattern 目录

  # 载体场景 prompt 池 (合成时画面场景描述)
  prompt_pool: synthesis/inputs/prompts/out_domain_carrier_prompts.jsonl

  # 开源数据 (step2 按各源 weight × 合成数 抽样混入训练集)
  # 支持三种格式适配器:
  #   registry: all.jsonl 式 {"image_path","ocr_gt"(多区域\n分隔),...}
  #   sharegpt: {"image","conversations":[human/gpt]} 检测数据 (DocVQA/POIE/FUNSD)
  #   chat:     ms-swift messages 多轮问答 (OCR-VQA), GT=assistant answer
  open_data:
    sources:
      - {path: data/raw/all.jsonl, format: registry, weight: 4.0}
      - {path: /cfs/.../intern_139549_train_data4_detection.json,
         format: sharegpt, weight: 0.5}
      - {path: data/sft/v3/ocr_vqa/train_5k_chat.jsonl, format: chat, weight: 0.25}
    # 单源简写 (等价于 sources 只有一条 registry):
    # registry: data/raw/all.jsonl
    # ratio: 4.0

  # 评估 GT (记录血缘; benchmark 脚本内部也读这些路径)
  eval_gt:
    - data/gt/level23_all.jsonl
    - data/gt/level23_hidden.jsonl
```

open_data registry 格式 (jsonl 每行一条):
  {"image_path": "/abs/x.jpg",   # 图片绝对路径
   "level1": "1",                # 1=隐藏文字任务图 0=普通
   "ocr_gt": "文字A\\n文字B",     # 图中文字 GT, 多区域 \\n 分隔, 空串=无文字图
   "level3": "0",                # 1=高风险
   "source_split": "na"}         # na/in_domain_holdout/out_domain/ipa_outdomain
                                  # (holdout/out_domain 被用作评估集, 不入训练)

映射关系 (inputs -> step 参数):
  base_model        -> models(检查清单) + training.env.MODEL(SFT) + evaluation.model
  diffusion.*       -> models(检查清单) + synthesis 的模型替换参数
  pattern_text_pool -> patterns.text_pool_dir
  pattern_fonts     -> patterns.font_dir
  patterns_dir      -> patterns.pattern_dir + synthesis.pattern (单步缺省)
  prompt_pool       -> synthesis.prompt_file
  open_data.sources -> processing.open_sources (多源) 或
  open_data.registry/ratio -> processing.open_data / open_ratio (单源)
  eval_gt           -> evaluation.gt_files (血缘记录用)
"""
from __future__ import annotations


def expand_inputs(cfg: dict) -> dict:
    """把 cfg['inputs'] 展开进各 step 段 (只补缺省, 不覆盖 step 段显式值)。"""
    inputs = cfg.get("inputs")
    if not inputs:
        return cfg

    cfg = dict(cfg)

    def fill(section: str, key: str, value):
        if value is None:
            return
        sec = cfg.setdefault(section, {})
        sec.setdefault(key, value)

    # ---- base_model: 模型检查清单 + SFT 训练 + 评估 ----
    bm = inputs.get("base_model") or {}
    if bm:
        model_entry = {"name": "base_model"}
        model_entry.update({k: v for k, v in bm.items() if v})
        models_sec = cfg.setdefault("models", {})
        models_list = models_sec.setdefault("models", [])
        if not any(m.get("name") == "base_model" for m in models_list):
            models_list.insert(0, model_entry)
        if bm.get("local_dir"):
            fill("training", "env", {})
            cfg["training"].setdefault("env", {}).setdefault("MODEL", bm["local_dir"])
            fill("evaluation", "model", bm["local_dir"])

    # ---- diffusion 三件套: 模型检查 + 合成模型替换 ----
    diff = inputs.get("diffusion") or {}
    if diff:
        models_sec = cfg.setdefault("models", {})
        models_list = models_sec.setdefault("models", [])
        for key, name in (("base", "diffusion_base"),
                          ("controlnet", "diffusion_controlnet"),
                          ("vae", "diffusion_vae")):
            hf_id = diff.get(key)
            if hf_id and not any(m.get("name") == name for m in models_list):
                entry = {"name": name, "hf_id": hf_id}
                if diff.get(f"{key}_local_dir"):
                    entry["local_dir"] = diff[f"{key}_local_dir"]
                models_list.append(entry)
        fill("synthesis", "diffusion_base", diff.get("base"))
        fill("synthesis", "diffusion_controlnet", diff.get("controlnet"))
        fill("synthesis", "diffusion_vae", diff.get("vae"))

    # ---- pattern / prompt / 开源数据 / 评估 GT ----
    fill("patterns", "text_pool_dir", inputs.get("pattern_text_pool"))
    fill("patterns", "font_dir", inputs.get("pattern_fonts"))
    fill("patterns", "pattern_dir", inputs.get("patterns_dir"))
    fill("synthesis", "prompt_file", inputs.get("prompt_pool"))
    # 单步运行 synthesis (无 prepare_patterns 上游注入) 时的 pattern 缺省值;
    # 完整 pipeline 中 prepare_patterns 产物会覆盖它
    fill("synthesis", "pattern", inputs.get("patterns_dir"))

    od = inputs.get("open_data") or {}
    if od.get("sources"):
        # 多源混合: [{path, format, weight}]
        fill("processing", "open_sources", od["sources"])
    else:
        fill("processing", "open_data", od.get("registry"))
        fill("processing", "open_ratio", od.get("ratio"))

    fill("evaluation", "gt_files", inputs.get("eval_gt"))

    return cfg
