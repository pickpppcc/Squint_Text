"""数据合成 Step — 包装 synthesis/augmentation/method1_txt2img_illusion CLI。

real 模式: subprocess 调用
    python -m synthesis.augmentation.method1_txt2img_illusion.run \
        --pattern <pattern> --prompt-file <prompts> --output-dir <out> --n N --seed S
    产出图片 + manifest.json (每行一条血缘记录) + data.txt, 由现有代码写出。

mock 模式: 直接写 N 张 1x1 占位 PNG + 同 schema 的 manifest.json / data.txt,
    不加载扩散模型、不占 GPU, 用于低成本流程验证。

输出 (返回值, 作为 ZenML Artifact 传给下游 Step):
    { images_dir, manifest_path, data_txt_path, n_images, run_id, git_commit, mock }
"""
from __future__ import annotations

import base64
import json
import logging

from zenml import step

from pipelines.config import PROJECT_ROOT, git_commit, resolve_path, run_cmd

# 1x1 透明 PNG, mock 模式当占位图片, 无需 Pillow 依赖
_TINY_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGNgYGBgAAAABQAB"
    "h6FO1AAAAABJRU5ErkJggg=="
)


@step(enable_cache=False)  # 有副作用 (写文件/占 GPU), 关闭缓存
def synthesize_data(mock: bool, params: dict, run_id: str,
                    patterns_output: dict = None) -> dict:
    """生成合成数据, 返回产物路径与元信息 dict。

    patterns_output: 可选, prepare_patterns 的产物; 若提供且未在 params.pattern
    显式覆盖, 则使用其中的 pattern_dir (保证 pattern 与本次 pattern 生成联动)。
    """
    log = logging.getLogger(__name__)
    log.info("synthesize_data: mock=%s run_id=%s params=%s", mock, run_id, params)

    # prepare_patterns 产物优先于配置里的 pattern (可用 synthesis.use_patterns_step
    # = false 关闭联动, 完全走配置)
    if patterns_output and params.get("use_patterns_step", True):
        params = dict(params)
        params["pattern"] = patterns_output["pattern_dir"]
        log.info("使用 prepare_patterns 产物: %s (%s 张)",
                 params["pattern"], patterns_output.get("n_patterns"))

    out_dir = resolve_path(params["output_dir"]) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    n = int(params.get("n", 8))
    seed = int(params.get("seed", 42))

    if mock:
        log.info("[MOCK] 生成 %d 张占位 PNG (seed=%s)", n, seed)
        entries = []
        for i in range(n):
            img_path = out_dir / f"{i:06d}.png"
            img_path.write_bytes(_TINY_PNG)
            entries.append({
                "gid": i,
                "output_path": str(img_path),
                "synthesis_family": "method1_txt2img_illusion",
                "synthesis_version": "v1",
                "ocr_text": f"mock_hidden_text_{i}",
                "level1": 1, "level3": 1,
                "source_tag": "zenml_mock",
                "control_image": "mock_pattern.png",
                "prompt": "mock prompt for pipeline validation",
                "negative_prompt": "",
                "controlnet_conditioning_scale": 1.5,
                "guidance_scale": 7.5,
                "num_inference_steps": 20,
                "seed": seed + i,
                "sampler": "Euler",
            })
        manifest_path = out_dir / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            for e in entries:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        data_txt = out_dir / "data.txt"
        with open(data_txt, "w", encoding="utf-8") as f:
            for e in entries:
                # 与 synthesis/manifest.py 一致的 legacy 管道分隔格式
                f.write("|".join([
                    e["output_path"], str(e["level1"]), e["ocr_text"],
                    str(e["level3"]), e["source_tag"], "-1", "-1",
                ]) + "\n")
    else:
        # ---- real: 调用 generate_method1_method2 --methods method1 ----
        # 与历史数据 synthesis/output/method1_2_cn2.0/ 的生成方式保持一致:
        #   同一生成器 (illusion_diffusion_gen), 同默认参数 (512->1024, 15/20 步),
        #   cn_scale=2.0 (历史批量配置, 目录名 cn2.0 即指此); 仅 method1, 不含 method2。
        # 该脚本支持 --num-shards/--shard-id 分片, 单进程即 --num-shards 1。
        pattern = params["pattern"]
        if "," not in str(pattern):
            p = resolve_path(pattern)
            if not p.exists():
                raise FileNotFoundError(f"pattern 不存在: {pattern}")
        # 指定了 diffusion 模型时走 vendored wrapper (运行时替换原脚本的
        # BASE_MODEL/CONTROLNET_MODEL/VAE_MODEL 常量, 不改原代码);
        # 否则直调原模块, 行为与历史完全一致
        has_model_override = any(params.get(k) for k in (
            "diffusion_base", "diffusion_controlnet", "diffusion_vae"))
        module = ("pipelines.vendor.run_method1" if has_model_override
                  else "synthesis.augmentation.generate_method1_method2")
        cmd = [
            params.get("python", "python"), "-m", module,
        ]
        if has_model_override:
            for k, flag in (("diffusion_base", "--base-model"),
                            ("diffusion_controlnet", "--controlnet-model"),
                            ("diffusion_vae", "--vae-model")):
                if params.get(k):
                    cmd += [flag, str(params[k])]
        cmd += [
            "--pattern", str(pattern),
            "--output-root", str(out_dir),
            "--methods", "method1",
            "--n", str(n),
            "--seed", str(seed),
            "--device", params.get("device", "cuda:0"),
            "--cn-scale", str(params.get("cn_scale", 2.0)),
        ]
        if params.get("prompt_file"):
            cmd += ["--prompt-file", str(resolve_path(params["prompt_file"]))]
        elif params.get("prompt"):
            cmd += ["--prompt", str(params["prompt"])]
        else:
            raise ValueError("real 模式需要 prompt 或 prompt_file")
        if params.get("num_shards", 1) > 1:
            cmd += ["--num-shards", str(params["num_shards"])]

        rc = run_cmd(cmd, cwd=PROJECT_ROOT)
        if rc != 0:
            raise RuntimeError(f"数据合成失败, exit code {rc} (见上方日志)")

        # 产物: <out_dir>/method1/{NNNNNN.png, manifest_shard*.jsonl}
        images_dir = out_dir / "method1"
        if not images_dir.is_dir():
            raise FileNotFoundError(f"合成完成但未找到输出目录: {images_dir}")
        # 合并 shard manifest 为单一 manifest.json (与 generate_method1_method2
        # 的 manifest 行 schema 一致: file/seed/prompt/pattern/source/ocr/...)
        shards = sorted(images_dir.glob("manifest_shard*.jsonl"))
        manifest_path = images_dir / "manifest.json"
        n = 0
        with open(manifest_path, "w", encoding="utf-8") as out_f:
            for shard in shards:
                with open(shard, encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            out_f.write(line)
                            n += 1
        if n == 0:
            raise RuntimeError(f"合成完成但 manifest 为空: {manifest_path}")
        out_dir = images_dir
        data_txt = images_dir / "data.txt"  # generate_method1_method2 不产 data.txt, 仅占位

    result = {
        "images_dir": str(out_dir),
        "manifest_path": str(manifest_path),
        "data_txt_path": str(data_txt),
        "n_images": n,
        "seed": seed,
        "run_id": run_id,
        "git_commit": git_commit(),
        "mock": mock,
    }
    # example: 第一条 manifest 记录 (截断长字段), Dashboard artifact 页可直接查看
    try:
        with open(manifest_path, encoding="utf-8") as f:
            first = json.loads(f.readline())
        result["example"] = {
            "file": first.get("file") or first.get("output_path"),
            "ocr_text": first.get("ocr") or first.get("ocr_text"),
            "prompt": (first.get("prompt") or "")[:200],
            "seed": first.get("seed"),
        }
    except Exception:
        pass
    log.info("synthesize_data 完成: %s", result)
    return result
