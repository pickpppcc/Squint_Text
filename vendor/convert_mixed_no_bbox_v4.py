#!/usr/bin/env python3
"""v4 数据转换: 基于 v3, teacher privilege 改为 3 张图 (原图 + 1/8 缩略图 + 40x40 缩略图),
并保留 GT 文字 hint (方案 B)。

相对 v3 改动:
  1. teacher_images: [原图, ds8, ds40]  (新增第 3 张 40x40 缩略图)
  2. teacher_prompt: 3 个 <image> + 新 PREFIX (三张图说明) + 任务指令 + 任务专属 FMT + GT 末尾
  3. student / images / reward_model / extra_info 全部不动
  4. GT 文字 hint 保留 (末尾单独成句, "该图的正确输出为{gt}。")

40x40 缓存规则 (与 ds8_cache 平行):
  - 目录: data/sft/v3/ds40_cache/
  - 命名: <orig_path 不含开头/>.40x40.png
  - 生成: Image.open(orig).convert("RGB").thumbnail((40, 40), Image.LANCZOS)
          (保宽高比, 长边缩到 40)
  - 已存在则跳过

输入:
  train: parquet_v5_6k_gt_opsd_aigc68w_mixed_no_bbox_v3/train.parquet  (5905 条)
  val:   parquet_v5_6k_gt_opsd_aigc68w_mixed_no_bbox_v3/val.parquet    (1045 条)
输出:
  parquet_v5_6k_gt_opsd_aigc68w_mixed_no_bbox_v4/{train,val}.{jsonl,parquet}

==============================================================================
[ZenML Pipeline 拷贝版说明]

来源: data/opsd/convert_mixed_no_bbox_v4.py (逐行拷贝, 供 pipeline 自包含运行,
      不依赖 data/ 目录下的代码)

改动 (仅两处, 转换逻辑 ds40_cache_path / make_ds40 / to_jsonable / convert_row /
     convert_file 逐字未动):
  1. ROOT/DATA_DIR/SRC_DIR/OUT_DIR/DS40_CACHE_ROOT 从模块级常量改为 main() 内
     变量, 值仍默认原路径 (行为不变), 可通过 --src-dir/--out-dir/--ds40-cache 覆盖;
  2. main() 增加 argparse, 并支持 --limit N (取前 N 条做小规模验证, 默认全量)。

安全: pipeline 调用时通过 --out-dir 指向 zenml_runs/<run_id>/ 下的独立目录,
      不会覆盖 data/opsd/ 下已有的 v4 产物。
==============================================================================
"""

import argparse
import json
import os
from pathlib import Path

import pandas as pd
from PIL import Image

# 默认路径与原脚本完全一致
DEFAULT_ROOT = Path("/cfs/cfs-izx10hg3/cinniepeng/AIGC_CONTROLNET")
DEFAULT_SRC_DIR = DEFAULT_ROOT / "data/opsd/parquet_v5_6k_gt_opsd_aigc68w_mixed_no_bbox_v3"
DEFAULT_OUT_DIR = DEFAULT_ROOT / "data/opsd/parquet_v5_6k_gt_opsd_aigc68w_mixed_no_bbox_v4"
DEFAULT_DS40_CACHE_ROOT = DEFAULT_ROOT / "data/sft/v3/ds40_cache"

# 运行时可被 main() --ds40-cache 覆盖 (ds40_cache_path 引用)
DS40_CACHE_ROOT = DEFAULT_DS40_CACHE_ROOT

PREFIX = ("<image>\n<image>\n<image>\n"
          "第一张图为原图,第二张图为同一张图的 1/8 下采样视图,"
          "第三张图为同一张图的 40×40 缩放视图。")


def ds40_cache_path(orig_path: str) -> str:
    """原图路径 -> ds40 缓存路径 (与 ds8_cache 规则平行)。"""
    return str(DS40_CACHE_ROOT / orig_path.lstrip("/")) + ".40x40.png"


def make_ds40(orig_path: str, cache_path: str) -> None:
    """长边=40 保宽高比缩略图 (thumbnail 自动按长边缩)。"""
    if os.path.exists(cache_path):
        return
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with Image.open(orig_path) as im:
        im = im.convert("RGB")
        # thumbnail 保宽高比, 长边缩到 40 (<=40); LANCZOS 与 ds8 一致
        im.thumbnail((40, 40), Image.LANCZOS)
        im.save(cache_path, "PNG")


def to_jsonable(v):
    import numpy as np
    if isinstance(v, np.ndarray):
        return [to_jsonable(x) for x in v.tolist()]
    if isinstance(v, dict):
        return {k: to_jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [to_jsonable(x) for x in v]
    return v


def convert_row(row):
    """teacher_images 加第 3 张 ds40; teacher_prompt 改 3 图 PREFIX (保留 GT hint)。
    其余字段不动。"""
    row = dict(row)
    orig_path = row["teacher_images"][0]["path"]
    ds40_path = ds40_cache_path(orig_path)
    make_ds40(orig_path, ds40_path)

    # teacher_images: 追加 ds40 (保持 dict 结构与 ds8 一致)
    ti = list(row["teacher_images"])
    ti.append({"path": ds40_path})
    row["teacher_images"] = ti

    # teacher_prompt: 从 v3 的 2 图 PREFIX 换成 3 图 PREFIX, 其余 (任务指令 + FMT + GT 末尾) 不动
    tp_content = row["teacher_prompt"][0]["content"]
    v3_prefix = ("<image>\n<image>\n"
                 "第一张图为原图,第二张图为同一张图的 1/8 下采样视图。")
    # v3 teacher_prompt 结构: "{v3_prefix}\n{instr}\n{fmt}\n该图的正确输出为{gt}。"
    assert tp_content.startswith(v3_prefix), \
        f"v3 teacher_prompt 前缀不匹配: {tp_content[:80]!r}"
    rest = tp_content[len(v3_prefix):]  # 以 "\n{instr}..." 开头
    new_tp_content = PREFIX + rest
    row["teacher_prompt"] = [{"role": "user", "content": new_tp_content}]
    return row


def convert_file(in_path, out_jsonl, out_parquet, limit=None):
    df = pd.read_parquet(in_path)
    print(f"  {in_path.name}: {len(df)} rows")
    if limit is not None:
        df = df.head(limit)
        print(f"  --limit {limit}: 取前 {len(df)} rows")
    rows = [dict(zip(df.columns, df.iloc[i])) for i in range(len(df))]
    converted = [to_jsonable(convert_row(r)) for r in rows]
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for r in converted:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    pd.DataFrame(converted, columns=list(df.columns)).to_parquet(out_parquet, index=False)
    print(f"    -> {out_jsonl.name} + {out_parquet.name}")
    return converted


def main():
    parser = argparse.ArgumentParser(description="v4 数据转换 (ZenML pipeline 拷贝版)")
    parser.add_argument("--src-dir", type=Path, default=DEFAULT_SRC_DIR,
                        help=f"v3 输入目录 (默认: {DEFAULT_SRC_DIR})")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                        help=f"v4 输出目录 (默认: {DEFAULT_OUT_DIR})")
    parser.add_argument("--ds40-cache", type=Path, default=DEFAULT_DS40_CACHE_ROOT,
                        help=f"ds40 缓存目录 (默认: {DEFAULT_DS40_CACHE_ROOT})")
    parser.add_argument("--limit", type=int, default=None,
                        help="只转换前 N 条 (小规模验证用; 默认全量)")
    args = parser.parse_args()

    # 原模块级常量 -> 本次运行的值 (全局函数 ds40_cache_path 引用)
    global DS40_CACHE_ROOT
    DS40_CACHE_ROOT = args.ds40_cache

    src_dir, out_dir = args.src_dir, args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print("v4 数据转换: 3 图 privilege (原图 + 1/8 + 40x40) + GT hint (方案 B)")
    print(f"输入目录: {src_dir}")
    print(f"输出目录: {out_dir}")
    print(f"ds40 缓存: {DS40_CACHE_ROOT}")
    print("=" * 70)

    print("\n[1/2] 转换 train (v3, 5905 条):")
    train = convert_file(src_dir / "train.parquet",
                         out_dir / "train.jsonl", out_dir / "train.parquet",
                         limit=args.limit)

    print("\n[2/2] 转换 val (v3, 1045 条):")
    val = convert_file(src_dir / "val.parquet",
                       out_dir / "val.jsonl", out_dir / "val.parquet",
                       limit=args.limit)

    # 校验
    print("\n=== 校验 ===")
    bad_nimg = sum(1 for r in train + val if len(r["teacher_images"]) != 3)
    bad_nplaceholder = sum(1 for r in train + val
                           if r["teacher_prompt"][0]["content"].count("<image>") != 3)
    bad_gt = sum(1 for r in train + val
                 if "该图的正确输出为" not in r["teacher_prompt"][0]["content"])
    print(f"  teacher_images != 3: {bad_nimg} (应为 0)")
    print(f"  teacher_prompt <image> 数 != 3: {bad_nplaceholder} (应为 0)")
    print(f"  teacher_prompt 缺 GT 末尾句: {bad_gt} (应为 0)")
    if bad_nimg or bad_nplaceholder or bad_gt:
        raise SystemExit("校验失败! 转换结果不符合 v4 schema")

    # 抽样
    print("\n=== 抽样验证 ===")
    seen = set()
    for r in train:
        pv = r["extra_info"]["prompt_version"]
        if pv in seen:
            continue
        seen.add(pv)
        print(f"\n--- {pv}  index={r['extra_info']['index']}  GT={r['reward_model']['ground_truth']} ---")
        print("TEACHER images:")
        for i, t in enumerate(r["teacher_images"]):
            print(f"  [{i}] {t['path']}")
        print("TEACHER prompt:")
        print("  " + r["teacher_prompt"][0]["content"].replace("\n", "\n  "))
        print("STUDENT prompt:")
        print("  " + r["prompt"][0]["content"].replace("\n", "\n  "))

    print(f"\n完成! v4 数据在 {out_dir}/")


if __name__ == "__main__":
    main()
