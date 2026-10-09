"""运行元数据 Step — 汇总一次 Run 的完整实验血缘 (P1 实验管理)。

把上游四个 Step 的产物元信息 (路径 / 条数 / 指标 / git commit / seed / 配置快照)
写成一份 run_metadata.json, 同时作为 ZenML Artifact 在 Dashboard 可见,
实现: 一次实验的配置、输入数据、Checkpoint、评估结果可追溯。

另外维护一份集中式「文件位置清单」(artifact_locations):
  - 基座/teacher 模型权重路径
  - 各 Stage 的关键文件/目录 (pattern、合成图、训练数据、checkpoint、评估 summary)
  - 训练/推理日志位置
并写入 ZenML Artifact Metadata, 在 Dashboard 的 run/step 页面直接可见,
无需打开 json 查找。

输出:
    { run_id, created_at, git_commit, seed, config, steps: {...},
      artifact_locations: {...}, metadata_path }
"""
from __future__ import annotations

import json
import logging
import os
import time

from zenml import step
from zenml.client import Client
from zenml.logger import get_logger

from pipelines.config import git_commit, resolve_path

log = get_logger(__name__)


def _loc(path: str | None, note: str = "") -> dict:
    """构造一条位置记录: {path, exists, note}。path 为空则标记 missing。"""
    if not path:
        return {"path": None, "exists": False, "note": note or "未提供"}
    p = resolve_path(path)
    return {"path": str(p), "exists": p.exists(), "note": note}


@step(enable_cache=False)  # 汇总的是本次运行的实时状态, 必须每次执行
def log_run_metadata(
    synthesis: dict,
    data: dict,
    model: dict,
    evaluation: dict,
    params: dict,
    run_id: str,
) -> dict:
    """汇总并落盘一次运行的完整元数据 + 文件位置清单。"""
    log.info("log_run_metadata: run_id=%s", run_id)
    cfg = params.get("config_snapshot", {})

    # ---------- 集中式文件位置清单 ----------
    tr_cfg = cfg.get("training", {})
    ev_cfg = cfg.get("evaluation", {})
    sy_cfg = cfg.get("synthesis", {})

    locations = {
        "models": {
            # 基座/teacher 模型 (训练与评估共用)
            "base_model": _loc(tr_cfg.get("model") or ev_cfg.get("model"), "基座模型权重"),
        },
        "inputs": {
            "pattern_dir": _loc(sy_cfg.get("pattern"), "合成用 control pattern 目录"),
            "prompt_file": _loc(sy_cfg.get("prompt_file"), "合成 carrier prompt 池"),
            "train_data_src": _loc(
                (cfg.get("processing", {}) or {}).get("src_dir"),
                "数据转换输入 (v3 parquet 目录)"),
        },
        "outputs": {
            "synthesized_images": _loc(synthesis.get("images_dir"), "合成图片目录"),
            "synthesis_manifest": _loc(synthesis.get("manifest_path"), "合成血缘 manifest"),
            "processed_train": _loc(data.get("train_file"), "处理后训练数据"),
            "processed_val": _loc(data.get("val_file"), "处理后验证数据"),
            "checkpoint": _loc(model.get("checkpoint_dir"), "训练产出 checkpoint"),
            "eval_summaries": {
                k: _loc(v, f"{k} 评估 summary")
                for k, v in (evaluation.get("summaries") or {}).items()
            },
        },
        "training": {
            "script": _loc(tr_cfg.get("script"), "训练入口脚本"),
            "experiment_name": model.get("experiment_name"),
        },
        "evaluation": {
            "script": _loc(ev_cfg.get("script"), "评估入口脚本"),
            "eval_tag": evaluation.get("tag"),
            "predictions_root": _loc(
                ev_cfg.get("out_root", "data/predictions/hf"), "预测输出根目录"),
        },
        "logs": {
            "metadata_dir": str(resolve_path(
                params.get("output_dir", "zenml_runs")) / run_id),
        },
    }

    meta = {
        "run_id": run_id,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "git_commit": git_commit(),
        "seed": params.get("seed"),
        "mock": synthesis.get("mock"),
        "config": cfg,
        "artifact_locations": locations,
        "steps": {
            "synthesis": synthesis,
            "processing": data,
            "training": model,
            "evaluation": evaluation,
        },
    }

    out_dir = resolve_path(params.get("output_dir", "zenml_runs")) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = out_dir / "run_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # ---------- 写入 ZenML Artifact Metadata: Dashboard 直接可见 ----------
    try:
        from zenml import get_step_context
        from zenml.client import Client
        from zenml.models import RunMetadataResource

        def _flatten(node, prefix=""):
            """位置清单 -> 扁平 {label: value} 表, 适合 Dashboard metadata 表格。"""
            flat = {}
            if isinstance(node, dict):
                if "path" in node and "exists" in node:  # _loc 记录
                    flat[prefix or "path"] = node["path"] or "—"
                    flat[(prefix + " [存在?]" if prefix else "exists")] = (
                        "是" if node["exists"] else "否")
                else:
                    for k, v in node.items():
                        flat.update(_flatten(v, f"{prefix}.{k}" if prefix else k))
            else:
                flat[prefix] = str(node)
            return flat

        loc_flat = _flatten(locations, "loc")
        client = Client()
        ctx = get_step_context()
        run_id_uuid = ctx.pipeline_run.id
        client.create_run_metadata(
            metadata={**loc_flat, "run_id": run_id,
                      "metadata_json": str(metadata_path)},
            resources=[
                RunMetadataResource(id=run_id_uuid, type="pipeline_run"),
                # 同时挂到本 step, step 详情页也能看到
                RunMetadataResource(id=ctx.step_run.id, type="step_run"),
            ],
        )
        log.info("artifact_locations 已写入 ZenML run metadata (%d 项)", len(loc_flat))
    except Exception as e:  # 元数据写入失败不影响 pipeline 结果
        log.warning("写入 ZenML run metadata 失败 (不影响产物): %s", e)

    meta["metadata_path"] = str(metadata_path)
    log.info("run_metadata 已写入: %s", metadata_path)
    return meta
