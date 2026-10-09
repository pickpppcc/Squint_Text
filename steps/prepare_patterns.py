"""prepare_patterns Step: 准备合成用的隐藏文字 pattern 图。

策略 (只增不覆盖):
  1. 检查配置的 pattern_dir: 已有 >= min_patterns 张 png 且目录内含
     pattern_ocr.jsonl -> 直接复用 (本机现状: 10001 张, 历史 cn2.0 同源)
  2. 不足/不存在 -> 调 pipelines/vendor/make_patterns.py 生成到独立输出目录
     (默认 zenml_runs/patterns/<run_id>/), 不触碰已有 pattern 目录

pattern 是 synthesize_data 的输入, 产物: pattern_dir + pattern_ocr.jsonl
(内含每张图的 OCR 文字, 即合成数据的 GT 来源)。
"""
import json
import logging
import os

from zenml import step

from pipelines.config import git_commit, resolve_path, run_cmd

log = logging.getLogger(__name__)

VENDOR_SCRIPT = "pipelines/vendor/make_patterns.py"


def _count_patterns(pattern_dir) -> int:
    if not os.path.isdir(str(pattern_dir)):
        return 0
    return sum(1 for f in os.listdir(str(pattern_dir)) if f.endswith(".png"))


@step(enable_cache=False)
def prepare_patterns(mock: bool, params: dict, run_id: str) -> dict:
    """确保 pattern 图 + OCR 标注就绪, 返回供 synthesize_data 使用的路径。

    params (yaml patterns 段):
      pattern_dir:    已有 pattern 目录 (优先复用)
      min_patterns:   复用所需的最少图片数 (默认 = synthesis.n)
      generate_n:     需要生成时的图片数量
      seed:           生成随机种子
      text_pool_dir:  文字池目录 (txt_from_segments/ + txt_from_api/)
      font_dir:       字体目录 (可空, 自动搜索)
      output_dir:     生成时的输出根目录 (默认 zenml_runs/patterns)
      python:         运行解释器 (需 PIL/numpy/scipy)
    """
    existing = resolve_path(params.get("pattern_dir", "synthesis/inputs/patterns/patterns"))
    min_patterns = int(params.get("min_patterns") or 0)
    ocr_in_dir = existing / "pattern_ocr.jsonl"

    n_existing = _count_patterns(existing)
    if n_existing >= max(min_patterns, 1) and ocr_in_dir.exists():
        log.info("prepare_patterns: 复用已有 pattern 目录 %s (%d 张, 含 pattern_ocr.jsonl)",
                 existing, n_existing)
        result = {
            "pattern_dir": str(existing),
            "pattern_ocr": str(ocr_in_dir),
            "n_patterns": n_existing,
            "reused": True,
            "run_id": run_id,
            "git_commit": git_commit(),
            "mock": mock,
            "example": {"note": "reused existing patterns, no new generation"},
        }
        return result

    # ---- 需要生成 ----
    generate_n = int(params.get("generate_n") or max(min_patterns, 100))
    out_root = resolve_path(params.get("output_dir", "zenml_runs/patterns"))
    out_dir = out_root / run_id / "patterns"

    if mock:
        # mock: 不渲染真实字体图, 只产出占位 png + 同 schema 的 pattern_ocr.jsonl
        from PIL import Image
        out_dir.mkdir(parents=True, exist_ok=True)
        records = []
        for i in range(1, generate_n + 1):
            text = f"MOCK文字{i:03d}"
            fp = out_dir / f"pattern_{i:06d}.png"
            Image.new("RGB", (512, 512), "black").save(fp)
            records.append({"file": fp.name, "path": str(fp.resolve()),
                            "ocr": text, "n_lines": 1})
        ocr_path = out_dir / "pattern_ocr.jsonl"
        with open(ocr_path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        result = {
            "pattern_dir": str(out_dir),
            "pattern_ocr": str(ocr_path),
            "n_patterns": generate_n,
            "reused": False,
            "run_id": run_id,
            "git_commit": git_commit(),
            "mock": True,
            "example": records[0],
        }
        log.info("prepare_patterns [MOCK] 完成: %d 张占位 pattern -> %s", generate_n, out_dir)
        return result

    # ---- real: 调 vendored make_patterns.py ----
    script = resolve_path(VENDOR_SCRIPT)
    cmd = [
        params.get("python", "/opt/conda/envs/torch-base/bin/python"),
        str(script),
        "--n", str(generate_n),
        "--seed", str(params.get("seed", 42)),
        "--out-dir", str(out_dir),
    ]
    if params.get("text_pool_dir"):
        cmd += ["--text-pool-dir", str(resolve_path(params["text_pool_dir"]))]
    if params.get("font_dir"):
        cmd += ["--font-dir", str(resolve_path(params["font_dir"]))]
    rc = run_cmd(cmd, cwd=str(resolve_path(".")))
    if rc != 0:
        raise RuntimeError(f"pattern 生成失败, exit code {rc} (见上方日志)")

    ocr_path = out_dir / "pattern_ocr.jsonl"
    if not ocr_path.exists():
        raise FileNotFoundError(f"生成完成但未找到 {ocr_path}")

    example = None
    with open(ocr_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                example = json.loads(line)
                break

    result = {
        "pattern_dir": str(out_dir),
        "pattern_ocr": str(ocr_path),
        "n_patterns": generate_n,
        "reused": False,
        "run_id": run_id,
        "git_commit": git_commit(),
        "mock": False,
        "example": example,
    }
    log.info("prepare_patterns 完成: %d 张 pattern -> %s", generate_n, out_dir)
    return result
