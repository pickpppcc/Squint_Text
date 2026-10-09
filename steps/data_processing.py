"""数据处理与过滤 Step — 包装 data/opsd/convert_*.py 等转换脚本。

real 模式: subprocess 运行配置指定的转换脚本 (如
    data/opsd/convert_mixed_no_bbox_v3.py: jsonl -> OPSD train/val parquet,
    或 data/sft/v4/build_v4.py: 合成数据 -> SFT jsonl)。
    转换脚本自带输入/输出路径 (在其文件头注明), 本 Step 负责执行与产物校验。

mock 模式: 读取上游 manifest.json, 切分成小份 train/val jsonl
    (schema 对齐 ms-swift messages 格式), 不依赖 pyarrow。

输出:
    { train_file, val_file, output_dir, n_train, n_val, run_id, git_commit, mock }
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from zenml import step

from pipelines.config import git_commit, resolve_path, run_cmd


@step(enable_cache=False)  # 副作用步骤, 关闭缓存
def process_data(synthesis_output: dict, mock: bool, params: dict, run_id: str) -> dict:
    """处理/过滤合成数据, 返回训练数据路径与统计信息。"""
    log = logging.getLogger(__name__)
    log.info("process_data: mock=%s run_id=%s", mock, run_id)

    out_dir = resolve_path(params["output_dir"]) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    train_file = out_dir / "train.jsonl"
    val_file = out_dir / "val.jsonl"

    if mock:
        manifest_path = synthesis_output["manifest_path"]
        log.info("[MOCK] 从 manifest 生成 train/val jsonl: %s", manifest_path)
        records = []
        with open(manifest_path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if not line.strip():
                    continue
                e = json.loads(line)
                # 对齐 ms-swift 的 messages 格式 (参考 data/sft/v4 数据)
                records.append({
                    "id": f"{run_id}_{i}",
                    "images": [e["output_path"]],
                    "messages": [
                        {"role": "user", "content": "<image>\n识别出图中所有的文字。"},
                        {"role": "assistant",
                         "content": json.dumps(
                             [{"text_content": e.get("ocr_text", "")}],
                             ensure_ascii=False)},
                    ],
                })
        n_val = max(1, len(records) // 5)
        val_records, train_records = records[:n_val], records[n_val:]
        for path, recs in ((val_file, val_records), (train_file, train_records)):
            with open(path, "w", encoding="utf-8") as f:
                for r in recs:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        log.info("[MOCK] train=%d val=%d -> %s", len(train_records), len(val_records), out_dir)
        sft_train_file, sft_val_file = train_file, val_file  # mock jsonl 即 ms-swift 格式
    else:
        # ---- real ----
        # mode=from_synthesis (默认, 链路连续): 用本次合成的 manifest 直接构建
        #   v4 格式 OPSD parquet + SFT jsonl (vendored 转换器, 模板与 v3/v4 逐字一致),
        #   不依赖任何历史中间产物。
        # mode=convert_existing: 跑 vendored v4 转换器 (v3 parquet -> v4), 用于
        #   复现历史数据集。
        mode = params.get("mode", "from_synthesis")
        if mode == "from_synthesis":
            script = resolve_path(
                "pipelines/vendor/build_opsd_parquet_from_synthesis.py")
            images_dir = synthesis_output.get("images_dir")
            if not images_dir or not Path(images_dir).is_dir():
                raise FileNotFoundError(
                    f"from_synthesis 模式需要上游合成产物目录, 收到 images_dir={images_dir!r}")
            v4_dir = out_dir / "v4"
            cmd = [
                params.get("python", "/opt/conda/envs/torch-base/bin/python"),
                str(script),
                "--images-dir", str(images_dir),
                "--out-dir", str(v4_dir),
                "--val-ratio", str(params.get("val_ratio", 0.05)),
                "--seed", str(params.get("seed", 42)),
            ]
            if params.get("open_sources"):
                # 多源混合: [{path, format, weight}] (优先于 open_data/open_ratio)
                cmd += ["--open-sources",
                        json.dumps(params["open_sources"], ensure_ascii=False)]
            elif params.get("open_data") is not None:
                cmd += ["--open-data", str(params["open_data"])]
                if params.get("open_ratio") is not None:
                    cmd += ["--open-ratio", str(params["open_ratio"])]
            rc = run_cmd(cmd)
            if rc != 0:
                raise RuntimeError(f"合成数据转换失败, exit code {rc} (见上方日志)")
            train_file = v4_dir / "train.parquet"
            val_file = v4_dir / "val.parquet"
            sft_train_file = v4_dir / "sft_train.jsonl"
            sft_val_file = v4_dir / "sft_val.jsonl"
        elif mode == "convert_existing":
            # 运行 vendored v4 转换脚本 (拷贝自 data/opsd/convert_mixed_no_bbox_v4.py,
            # 转换逻辑未改; 输出到 run_id 隔离目录, 不触碰 data/opsd/ 已有产物)
            script = resolve_path(params.get(
                "convert_script", "pipelines/vendor/convert_mixed_no_bbox_v4.py"))
            if not script.exists():
                raise FileNotFoundError(f"转换脚本不存在: {script}")
            src_dir = resolve_path(params.get(
                "src_dir", "data/opsd/parquet_v5_6k_gt_opsd_aigc68w_mixed_no_bbox_v3"))
            if not (src_dir / "train.parquet").exists():
                raise FileNotFoundError(f"v3 输入数据不存在: {src_dir}/train.parquet")
            v4_dir = out_dir / "v4"
            cmd = [
                params.get("python", "/opt/conda/envs/torch-base/bin/python"),
                str(script),
                "--src-dir", str(src_dir),
                "--out-dir", str(v4_dir),
            ]
            if params.get("limit"):
                cmd += ["--limit", str(int(params["limit"]))]
            cmd += list(params.get("extra_args", []))
            rc = run_cmd(cmd)
            if rc != 0:
                raise RuntimeError(f"数据转换失败, exit code {rc} (见上方日志)")
            train_file = v4_dir / "train.parquet"
            val_file = v4_dir / "val.parquet"
            sft_train_file = sft_val_file = None
        else:
            raise ValueError(f"未知 processing mode: {mode} (支持 from_synthesis | convert_existing)")
        for f in (train_file, val_file):
            if not f.exists():
                raise FileNotFoundError(f"转换脚本运行后未找到预期产物: {f}")

    def _count(p) -> int:
        p = Path(str(p))
        # parquet 是二进制, 数同名 .jsonl 姊妹文件的行数 (转换器两者同产)
        if p.suffix == ".parquet":
            p = p.with_suffix(".jsonl")
        try:
            with open(p, encoding="utf-8") as fh:
                return sum(1 for line in fh if line.strip())
        except (OSError, UnicodeDecodeError):
            return -1

    result = {
        "train_file": str(train_file),
        "val_file": str(val_file),
        "sft_train_file": str(sft_train_file) if sft_train_file else None,
        "sft_val_file": str(sft_val_file) if sft_val_file else None,
        "output_dir": str(out_dir),
        "n_train": _count(train_file),
        "n_val": _count(val_file),
        "n_images_input": synthesis_output.get("n_images"),
        "run_id": run_id,
        "git_commit": git_commit(),
        "mock": mock,
    }
    # example: train.jsonl 第一条记录摘要 (student prompt + GT + 图片数),
    # Dashboard artifact 页可直接查看 (parquet 的姊妹 jsonl 同内容)
    try:
        jsonl_sibling = Path(str(train_file)).with_suffix(".jsonl")
        if jsonl_sibling.exists():
            with open(jsonl_sibling, encoding="utf-8") as f:
                rec = json.loads(f.readline())
            # 兼容 verl parquet 格式与 ms-swift messages 格式
            if "prompt" in rec:
                example = {
                    "student_prompt": rec["prompt"][0]["content"][:300],
                    "teacher_prompt_tail": rec["teacher_prompt"][0]["content"][-120:],
                    "images": [i["path"] for i in rec["images"]],
                    "teacher_images": [i["path"] for i in rec["teacher_images"]],
                    "ground_truth": rec["reward_model"]["ground_truth"][:200],
                    "prompt_version": rec["extra_info"].get("prompt_version"),
                }
            else:
                example = {
                    "user": rec["messages"][0]["content"][:200],
                    "assistant": rec["messages"][1]["content"][:200],
                    "images": rec.get("images"),
                }
            result["example"] = example
    except Exception:
        pass
    log.info("process_data 完成: %s", result)
    return result
