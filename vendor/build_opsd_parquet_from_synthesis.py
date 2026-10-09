#!/usr/bin/env python3
"""从 ZenML 合成产物直接构建 v4 格式 OPSD 训练 parquet (打通 合成->训练 连续性)。

输入:
  1. synthesize_data 产物目录 (<images_dir>/manifest.json, 每行含
     file/seed/prompt/pattern/source/ocr/... — 由 generate_method1_method2 的
     manifest_shard*.jsonl 合并而来, ocr 即图中隐藏文字 GT)
  2. 开源数据注册表 (--open-data, 默认 data/raw/all.jsonl)。开源数据无法由
     合成链路产生, 但作为训练数据的一部分在 Step2 合并进同一批 v4 记录。

开源数据格式 (data/raw/all.jsonl, 每行一条):
    {
      "image_path": "/cfs/.../xxx.jpg",        # 图片绝对路径
      "level1": "1",                           # 是否隐藏文字任务图: "1"/"0"
      "ocr_gt":  "文字A\n文字B",               # 图中文字 GT, 多区域 \n 分隔; "" = 无文字
      "level3": "0",                           # 是否高风险: "1"/"0"
      "source_split": "na"                     # na / in_domain_holdout / out_domain / ...
    }
  抽样规则: 有 ocr_gt 的优先 (teach OCR); 自动剔除 benchmark GT 图片
  (data/gt/level1.jsonl + level23.jsonl 中的 image_path, 防止训练/评测泄漏);
  自动跳过图片文件不存在的记录。采样数 = --open-ratio × 合成样本数
  (默认 4.0, 对齐历史 v3 约 4~5:1 的真实:合成比例)。
输出: <out_dir>/{train,val}.parquet + {train,val}.jsonl
      与 data/opsd/parquet_v5_6k_gt_opsd_aigc68w_mixed_no_bbox_v4 完全同 schema
      (columns: data_source/prompt/images/teacher_prompt/teacher_images/ability/
       reward_model/extra_info), 可直接给 run_opsd_gt_mixed_no_bbox_v4b.sh 训练。
      另产 SFT 格式 jsonl (sft_{train,val}.jsonl, ms-swift messages 格式, 对齐
      data/sft/v4/build_v4.py 的 ds8 双图变体), 供 SFT 训练使用。

prompt/teacher 模板逐字沿用 v3/v4 转换脚本 (fix2 prompt + v4 3图 teacher):
  - OCR_FMT / INSTR: data/opsd/convert_mixed_no_bbox_v3.py
  - PREFIX (3图):    pipelines/vendor/convert_mixed_no_bbox_v4.py
teacher_images = [原图, 1/8 缩略图(ds8), 40x40 缩略图(ds40)], 缩略图缓存规则与
convert_v5_6k_gt_opsd_replace_method2_aigc68w.py / convert_mixed_no_bbox_v4.py 一致
(ds8_cache / ds40_cache, 已存在则跳过, 只增不覆盖)。

用法:
  python pipelines/vendor/build_opsd_parquet_from_synthesis.py \
      --images-dir zenml_runs/synthesis/<run_id>/method1 \
      --out-dir    zenml_runs/processing/<run_id>/v4
"""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ---- 模板常量 (逐字来自 data/opsd/convert_mixed_no_bbox_v3.py 与 v4) ----
INSTR = "请检测图中所有文字区域，并识别其中的文字。"
OCR_FMT = ('每个文字区域用 {"text_content": "文字"} 表示，写在一个列表里。\n'
           '单条示例：[{"text_content": "文字"}]\n'
           '若图中无文字则输出 []。仅输出 JSON 列表，禁止使用代码块或输出解释性文字。')
PREFIX_V4 = ("<image>\n<image>\n<image>\n"
             "第一张图为原图,第二张图为同一张图的 1/8 下采样视图,"
             "第三张图为同一张图的 40×40 缩放视图。")
SFT_INSTR_DS8 = "<image>\n<image>\n识别出图中所有的文字。"   # build_v4.py ds8 变体

DS8_CACHE_ROOT = PROJECT_ROOT / "data/sft/v3/ds8_cache"
DS40_CACHE_ROOT = PROJECT_ROOT / "data/sft/v3/ds40_cache"

# benchmark GT 图片 (训练时必须排除, 防泄漏; 对齐 sft/split.py exclude_benchmark_gt)
BENCH_GT_FILES = [PROJECT_ROOT / "data/gt/level1.jsonl",
                  PROJECT_ROOT / "data/gt/level23.jsonl"]


def _gt_to_regions(ocr_gt: str) -> list[str]:
    """ocr_gt 纯文本 (多区域 \n 分隔) -> 区域文本列表。"""
    return [t for t in (ocr_gt or "").split("\n") if t.strip()]


def load_bench_excludes() -> set:
    """benchmark GT 图片路径集合 (这些图在评测集里, 训练必须排除)。"""
    excludes = set()
    for p in BENCH_GT_FILES:
        if not p.exists():
            continue
        with open(p, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    o = json.loads(line)
                    if o.get("image_path"):
                        excludes.add(o["image_path"])
    return excludes


def load_open_records(open_data: Path, excludes: set, n: int, seed: int,
                      index_base: int) -> list:
    """从开源注册表抽 n 条 (有 ocr_gt 优先), 返回 (image_path, ocr_gt, index)。

    注意: 不逐条 os.path.exists (CFS 网络文件系统 stat 14 万条会挂死);
    先按条件过滤+抽样, 只对小倍数候选做存在性检查。"""
    return _sample_records(_iter_registry(open_data), excludes, n, seed, index_base)


# ===== 开源数据源格式适配器: 统一产出 (image_path, ocr_text) =====
# ocr_text 多区域以 \n 分隔; 空串 = 无文字图

def _iter_registry(path: Path):
    """all.jsonl 注册表格式: {"image_path", "ocr_gt", ...}"""
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            o = json.loads(line)
            if o.get("image_path"):
                yield o["image_path"], (o.get("ocr_gt") or "")


def _iter_sharegpt(path: Path):
    """sharegpt 检测格式 (DocVQA/POIE/FUNSD/TextCaps detection,
    如 intern_139549_train_data4_detection.json):
    {"image": path, "conversations": [{from: human, value: 问},
                                      {from: gpt, value: 答案</s>}]}
    同一图可有多条 (不同颜色框), 按图聚合为一条多区域记录。"""
    by_image: dict[str, list] = {}
    order = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            img = d.get("image", "")
            if not img:
                continue
            # 路径修复 (沿用 integrate_ocr_detection.py 的映射)
            img = img.replace("/dockerdata/data4_clear/",
                              "/cfs/cfs-18a3pglt/xinghaozhao/data4_clear/")
            if img not in by_image:
                by_image[img] = []
                order.append(img)
            for c in d.get("conversations", []):
                if c.get("from") == "gpt":
                    t = (c.get("value") or "").replace("</s>", "").strip()
                    if t:
                        by_image[img].append(t)
    for img in order:
        yield img, "\n".join(by_image[img])


def _iter_chat(path: Path):
    """ms-swift messages 格式 (如 OCR-VQA train_5k_chat.jsonl):
    {"messages": [...], "images": [path, ...]}
    assistant 答案是 json {"answer": "...", ["bbox": ...]} 或纯文本;
    聚合所有 assistant 答案为一条多区域记录。"""
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            images = d.get("images") or []
            if not images:
                continue
            texts = []
            for m in d.get("messages", []):
                if m.get("role") != "assistant":
                    continue
                content = (m.get("content") or "").strip()
                try:
                    content = str(json.loads(content).get("answer", "")).strip()
                except (json.JSONDecodeError, AttributeError):
                    pass
                if content:
                    texts.append(content)
            yield images[0], "\n".join(texts)


_OPEN_FORMATS = {"registry": _iter_registry, "sharegpt": _iter_sharegpt,
                 "chat": _iter_chat}


def _sample_records(records_iter, excludes: set, n: int, seed: int,
                    index_base: int) -> list:
    """从 (image_path, ocr_text) 迭代器抽 n 条, 有文字的优先。"""
    pool_with_ocr, pool_empty = [], []
    for image_path, ocr_text in records_iter:
        if not image_path or image_path in excludes:
            continue
        (pool_with_ocr if ocr_text.strip() else pool_empty).append(
            (image_path, ocr_text))
    rng = random.Random(seed)
    chosen = []
    for pool in (pool_with_ocr, pool_empty):
        if len(chosen) >= n:
            break
        rng.shuffle(pool)
        for item in pool[: (n - len(chosen)) * 2]:
            if os.path.exists(item[0]):
                chosen.append(item)
            if len(chosen) >= n:
                break
    return [(p, t, index_base + i) for i, (p, t) in enumerate(chosen)]


def load_open_sources(specs: list, excludes: set, n_synth: int, seed: int,
                      index_base: int) -> list:
    """按 sources 配置分别从各开源源抽样并合并。

    specs: [{"path": ..., "format": registry|sharegpt|chat, "weight": float}]
    每源采样数 = round(weight × n_synth)。返回合并列表 (image_path, ocr, index)。
    """
    merged = []
    for i, spec in enumerate(specs):
        path = Path(spec["path"])
        fmt = spec.get("format", "registry")
        weight = float(spec.get("weight", 1.0))
        if fmt not in _OPEN_FORMATS:
            raise ValueError(f"未知开源数据格式: {fmt} (支持 {list(_OPEN_FORMATS)})")
        if not path.exists():
            print(f"  [skip] 开源源不存在: {path}")
            continue
        n = int(round(weight * n_synth))
        if n <= 0:
            continue
        recs = _sample_records(_OPEN_FORMATS[fmt](path), excludes, n,
                               seed + i, index_base + len(merged))
        print(f"  开源源 [{fmt}] {path.name}: 采 {len(recs)}/{n} 条")
        merged.extend(recs)
    return merged


def ds8_cache_path(orig_path: str) -> str:
    """与原 convert 脚本一致: ds8_cache/<orig 不含开头/ >.ds8.png"""
    return str(DS8_CACHE_ROOT / orig_path.lstrip("/")) + ".ds8.png"


def ds40_cache_path(orig_path: str) -> str:
    return str(DS40_CACHE_ROOT / orig_path.lstrip("/")) + ".40x40.png"


def make_thumb(orig_path: str, cache_path: str, size: int) -> None:
    """已存在则跳过 (幂等, 不覆盖)。size=8 表示 1/8 缩放; size=40 表示长边40。"""
    if os.path.exists(cache_path):
        return
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with Image.open(orig_path) as im:
        im = im.convert("RGB")
        if size == 8:
            w, h = im.size
            im.resize((max(1, w // 8), max(1, h // 8)), Image.LANCZOS).save(cache_path, "PNG")
        else:
            im.thumbnail((40, 40), Image.LANCZOS)
            im.save(cache_path, "PNG")


def build_record(image_path: str, ocr_text: str, index: int) -> dict:
    """一条样本 -> v4 格式 verl 记录 (ocr_only_v1 任务, 无 bbox)。
    ocr_text 多区域以 \n 分隔 (开源数据格式), 拆成多个 text_content 区域。"""
    regions = [{"text_content": t} for t in _gt_to_regions(ocr_text)]
    gt_json = json.dumps(regions, ensure_ascii=False)
    ds8 = ds8_cache_path(image_path)
    ds40 = ds40_cache_path(image_path)
    make_thumb(image_path, ds8, 8)
    make_thumb(image_path, ds40, 40)
    return {
        "data_source": "opsd_ocr_bbox",
        "prompt": [{"role": "user", "content": f"<image>\n{INSTR}\n{OCR_FMT}"}],
        "images": [{"path": image_path}],
        "teacher_prompt": [{"role": "user", "content":
                            f"{PREFIX_V4}\n{INSTR}\n{OCR_FMT}\n"
                            f"该图的正确输出为{gt_json}。"}],
        "teacher_images": [{"path": image_path}, {"path": ds8}, {"path": ds40}],
        "ability": "visual_question_answering",
        "reward_model": {"ground_truth": gt_json, "style": "none"},
        "extra_info": {"answer": gt_json, "index": index,
                       "original_gt": gt_json, "prompt_version": "ocr_only_v1"},
    }


def build_sft_record(image_path: str, ocr_text: str, index: int) -> dict:
    """同一样本的 SFT (ms-swift) 格式, 对齐 build_v4.py ds8 双图变体。"""
    regions = [{"text_content": t} for t in _gt_to_regions(ocr_text)]
    gt_json = json.dumps(regions, ensure_ascii=False)
    ds8 = ds8_cache_path(image_path)
    return {
        "messages": [
            {"role": "user", "content": SFT_INSTR_DS8},
            {"role": "assistant", "content": gt_json},
        ],
        "images": [image_path, ds8],
        "prompt_version": "ocr_text_v4",
        "id": f"synth_{index}",
    }


def main() -> None:
    p = argparse.ArgumentParser(description="合成 manifest -> v4 OPSD parquet + SFT jsonl")
    p.add_argument("--images-dir", required=True,
                   help="合成产物目录 (含 manifest.json 与 NNNNNN.png)")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--val-ratio", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--index-base", type=int, default=500000,
                   help="extra_info.index 起始值 (避开历史 0..24927 与 50000+)")
    p.add_argument("--open-data", default=str(PROJECT_ROOT / "data/raw/all.jsonl"),
                   help="开源数据注册表 jsonl (registry 格式, 见模块 docstring); "
                        "传空字符串禁用。多源混合请用 --open-sources")
    p.add_argument("--open-ratio", type=float, default=4.0,
                   help="--open-data 的采样倍数: 样本数 = ratio × 合成样本数")
    p.add_argument("--open-sources", default=None,
                   help="多源混合: JSON 列表 [{\"path\":..., \"format\": registry|"
                        "sharegpt|chat, \"weight\": float}], 每源采样 weight × 合成数。"
                        "指定后覆盖 --open-data/--open-ratio")
    args = p.parse_args()

    images_dir = Path(args.images_dir)
    manifest_path = images_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest 不存在: {manifest_path}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 读 manifest, 跳过无 OCR 文本的样本 (无 GT 无法训练)
    records, skipped_no_ocr, skipped_missing = [], 0, 0
    with open(manifest_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            e = json.loads(line)
            ocr = (e.get("ocr") or "").strip()
            if not ocr:
                skipped_no_ocr += 1
                continue
            img = images_dir / e["file"]
            if not img.exists():
                skipped_missing += 1
                continue
            records.append((str(img), ocr, args.index_base + i))

    if not records:
        raise RuntimeError("manifest 中没有可用样本 (全部无 OCR 或缺图)")
    print(f"manifest 合成样本: {len(records)} 可用 "
          f"(跳过: 无OCR={skipped_no_ocr}, 缺图={skipped_missing})")

    # ---- 合并开源数据 (合成链路无法产生, 但训练需要真实数据混合) ----
    if args.open_sources:
        sources = json.loads(args.open_sources)
    elif args.open_data and args.open_ratio > 0:
        sources = [{"path": args.open_data, "format": "registry",
                    "weight": args.open_ratio}]
    else:
        sources = []
    if sources:
        excludes = load_bench_excludes()
        open_recs = load_open_sources(sources, excludes, len(records),
                                      args.seed, args.index_base + len(records))
        print(f"开源样本: 合计 {len(open_recs)} 条 "
              f"(已剔除 benchmark GT {len(excludes)} 张)")
        records += open_recs

    rng = random.Random(args.seed)
    rng.shuffle(records)
    n_val = max(1, int(len(records) * args.val_ratio))
    val_recs, train_recs = records[:n_val], records[n_val:]
    if not train_recs:  # 样本太少时 val 全部退回 train, 保证训练集非空
        train_recs, val_recs = records, records[:1]

    cols = ["data_source", "prompt", "images", "teacher_prompt", "teacher_images",
            "ability", "reward_model", "extra_info"]
    for split, recs in (("train", train_recs), ("val", val_recs)):
        rows = [build_record(img, ocr, idx) for img, ocr, idx in recs]
        sft_rows = [build_sft_record(img, ocr, idx) for img, ocr, idx in recs]
        # parquet (OPSD/verl) + jsonl (同内容, 便于人读) + SFT jsonl
        pd.DataFrame(rows, columns=cols).to_parquet(
            out_dir / f"{split}.parquet", index=False)
        with open(out_dir / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        with open(out_dir / f"sft_{split}.jsonl", "w", encoding="utf-8") as f:
            for r in sft_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  {split}: {len(rows)} 条 -> {split}.parquet + {split}.jsonl + sft_{split}.jsonl")

    print(f"完成: {out_dir} (train={len(train_recs)}, val={len(val_recs)})")


if __name__ == "__main__":
    main()
