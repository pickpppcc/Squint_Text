#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""[ZenML Pipeline 拷贝版 — 拷贝自 synthesis/inputs/patterns/make_patterns.py]

pattern 生成逻辑 (clean_text/decide_lines/split_text/fit_font/make_one) 逐字未改。
仅参数化 (默认行为与原脚本一致):
  --text-pool-dir   文字池目录 (默认 synthesis/inputs/patterns, 读其下
                    txt_from_segments/from_text_only.txt, txt_from_api/*.txt)
  --out-dir         pattern 图片输出目录 (默认 <text-pool-dir>/patterns)
  --ocr-out         pattern_ocr.jsonl 路径 (默认 <out-dir>/pattern_ocr.jsonl,
                    与下游 _load_pattern_ocr_map 约定的"图案目录内"位置一致)
  --font-dir        字体目录 (默认依次找 /usr/share/fonts/google-noto-cjk ->
                    synthesis/inputs/fonts; 支持 .ttc/.ttf/.otf)

注意: 默认输出会覆盖 --out-dir 下同名 png 与 --ocr-out (与原脚本行为一致)。
pipeline 的 prepare_patterns Step 只在 pattern 不足时才调用本脚本,
且默认输出到独立目录, 不会触碰已有 synthesis/inputs/patterns/patterns。
"""
import argparse
import glob
import json
import os
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_POOL_DIR = PROJECT_ROOT / "synthesis/inputs/patterns"

IMG = 512
MARGIN = 48  # 四周留白 (调大让字更收敛)
DILATION = 0  # 不做 dilation, 防止笔画黏成一团; Bold 字体本身够粗

# 字体目录搜索顺序: 系统字体 -> 仓库内自带字体
FONT_DIR_CANDIDATES = [
    "/usr/share/fonts/google-noto-cjk",
    str(PROJECT_ROOT / "synthesis/inputs/fonts"),
]
# 优先粗体 (更粗), 支持 ttc/otf/ttf
FONT_NAME_PATTERNS = [
    "NotoSansCJK-Bold.ttc", "NotoSansCJK-Black.ttc",
    "NotoSansSC-Bold.otf", "NotoSansSC-Black.otf",
]


def resolve_font_dir():
    """找到第一个含可用字体的目录; 返回 (font_dir, [font_paths])."""
    for d in FONT_DIR_CANDIDATES:
        if not os.path.isdir(d):
            continue
        fonts = []
        for name in FONT_NAME_PATTERNS:
            p = os.path.join(d, name)
            if os.path.exists(p):
                fonts.append(p)
        if fonts:
            return d, fonts
        # 兜底: 目录里任意 CJK/SC 字体
        any_fonts = sorted(
            glob.glob(os.path.join(d, "*.ttc"))
            + glob.glob(os.path.join(d, "*.otf"))
            + glob.glob(os.path.join(d, "*.ttf"))
        )
        if any_fonts:
            return d, any_fonts
    raise SystemExit(
        "未找到 CJK 字体。请安装 fonts-noto-cjk 或把 NotoSansSC-Bold.otf "
        "放到 synthesis/inputs/fonts/ (见 pipelines/README 说明)")

FONT_DIR, FONT_PATHS = None, []  # main() 里初始化


def clean_text(s):
    """去空格/制表/换行 + 去所有标点符号 (只保留 CJK 汉字与 ASCII 字母数字)"""
    return ''.join(c for c in s if ('一' <= c <= '鿿') or (c.isascii() and c.isalnum()))


def load_pool(pool_dir):
    """加载三个来源 -> clean_text -> 全局去重 -> 返回 list."""
    sources = [
        ("segment", os.path.join(pool_dir, "txt_from_segments", "from_text_only.txt")),
        ("api_malicious", os.path.join(pool_dir, "txt_from_api", "api_texts.txt")),
        ("api_benign", os.path.join(pool_dir, "txt_from_api", "api_text_benign.txt")),
    ]
    seen = set()
    pool = []
    per_source = {}
    for name, fp in sources:
        n = 0
        if not os.path.exists(fp):
            print(f"  [skip] {name}: {fp} 不存在")
            continue
        with open(fp, encoding="utf-8") as f:
            for ln in f:
                t = clean_text(ln)
                if not t or len(t) > 16:
                    continue
                if t in seen:
                    continue
                seen.add(t)
                pool.append(t)
                n += 1
        per_source[name] = n
        print(f"  {name}: +{n} 条 (去重后) <- {fp}")
    print(f"全局去重后合计: {len(pool)} 条")
    return pool


def decide_lines(L, rng):
    """按字数决定拆几行: 短文 1 行(偶拆 2), 长文 2-3 行不挤一行"""
    if L <= 3:
        return 1
    if L == 4:
        return rng.choice([1, 1, 1, 2])
    if L <= 6:
        return rng.choice([1, 2, 2])
    if L <= 8:
        return rng.choice([2, 2, 3])
    return rng.choice([2, 3, 3])


def split_text(text, n):
    """把 text 尽量均匀拆成 n 段(按字符数)"""
    if n <= 1:
        return [text]
    L = len(text)
    base, extra = L // n, L % n
    out, idx = [], 0
    for i in range(n):
        sz = base + (1 if i < extra else 0)
        out.append(text[idx:idx + sz])
        idx += sz
    return out


def fit_font(draw, text, max_w, max_h, rng, floor=40, start=85):
    """随机选一个粗体字体, 取最大字号使 ink bbox <= max_w x max_h."""
    fpath = rng.choice(FONT_PATHS)
    for size in range(start, floor - 1, -2):
        try:
            f = ImageFont.truetype(fpath, size)
        except Exception:
            continue
        bb = draw.textbbox((0, 0), text, font=f)
        w, h = bb[2] - bb[0], bb[3] - bb[1]
        if w <= max_w and h <= max_h:
            return f, w, h, size, fpath, bb
    f = ImageFont.truetype(fpath, floor)
    bb = draw.textbbox((0, 0), text, font=f)
    return f, bb[2] - bb[0], bb[3] - bb[1], floor, fpath, bb


def make_one(lines, out_path, rng):
    lines = [clean_text(t) for t in lines if clean_text(t)]
    if not lines:
        return
    n = len(lines)
    img = Image.new("RGB", (IMG, IMG), "black")
    draw = ImageDraw.Draw(img)
    uw = IMG - 2 * MARGIN
    gap = 18 if n >= 2 else 0
    block_target = int(IMG * (0.50 if n >= 2 else 0.40))
    line_h_budget = (block_target - gap * (n - 1)) // n

    fitted = []
    for text in lines:
        font, w, h, size, fname, ink_bb = fit_font(draw, text, uw, line_h_budget, rng)
        fitted.append((font, w, h, ink_bb))

    y0 = (IMG - block_target) // 2
    for i, (text, (font, w, h, ink_bb)) in enumerate(zip(lines, fitted)):
        slot_top = y0 + i * (line_h_budget + gap)
        y_ink = slot_top + (line_h_budget - h) // 2
        y = y_ink - ink_bb[1]
        x = (IMG - w) // 2 - ink_bb[0]
        draw.text((x, y), text, fill="white", font=font)

    arr = np.array(img.convert("L"))
    if DILATION > 0:
        arr = ndimage.grey_dilation(arr, size=(DILATION, DILATION))
    out = np.where(arr > 128, 255, 0).astype(np.uint8)
    Image.fromarray(out).save(out_path)


def main():
    global FONT_DIR, FONT_PATHS
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--text-pool-dir", default=str(DEFAULT_POOL_DIR))
    ap.add_argument("--out-dir", default=None,
                    help="默认 <text-pool-dir>/patterns")
    ap.add_argument("--ocr-out", default=None,
                    help="默认 <out-dir>/pattern_ocr.jsonl (下游 loader 约定位置)")
    ap.add_argument("--font-dir", default=None,
                    help="默认自动搜索系统字体与 synthesis/inputs/fonts")
    args = ap.parse_args()

    if args.font_dir:
        FONT_DIR_CANDIDATES.insert(0, args.font_dir)
    FONT_DIR, FONT_PATHS = resolve_font_dir()
    print(f"字体目录: {FONT_DIR} ({len(FONT_PATHS)} 个字体: "
          f"{[os.path.basename(p) for p in FONT_PATHS]})")

    out_dir = args.out_dir or os.path.join(args.text_pool_dir, "patterns")
    ocr_out = args.ocr_out or os.path.join(out_dir, "pattern_ocr.jsonl")
    os.makedirs(out_dir, exist_ok=True)

    rng = random.Random(args.seed)
    pool = load_pool(args.text_pool_dir)
    if not pool:
        raise SystemExit("文字池为空, 请先跑 collect_segments.py / gen_texts_api.py")
    rng.shuffle(pool)

    records = []
    for i in range(1, args.n + 1):
        text = pool[(i - 1) % len(pool)]
        n_lines = decide_lines(len(text), rng)
        lines = split_text(text, n_lines)
        out_path = os.path.join(out_dir, f"pattern_{i:06d}.png")
        make_one(lines, out_path, rng)
        records.append({
            "file": os.path.basename(out_path),
            "path": os.path.abspath(out_path),
            "ocr": text,
            "n_lines": n_lines,
        })
        if i % 25 == 0 or i == args.n:
            print(f"  {i}/{args.n}")

    with open(ocr_out, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n完成: {args.n} 张 -> {out_dir}")
    print(f"ocr -> {ocr_out}")


if __name__ == "__main__":
    main()
