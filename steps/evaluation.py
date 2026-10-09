"""模型评估 Step — 包装 inference/experiments/run_benchmarks_hf.sh。

real 模式: subprocess 运行
    bash inference/experiments/run_benchmarks_hf.sh \
        --model <model> [--adapter <checkpoint>] --tag <tag> --benchmarks <benchmarks> \
        [--gpus ...] [--dry-run] [--scan-only]
    预测与评分由现有脚本完成, summary 输出在 data/predictions/hf/<Bench>/<tag>/。
    配置默认 dry_run=true / scan_only 可选, 先低成本验证评估链路。

mock 模式: 直接写与真实脚本同结构的 <tag>_summary.json (含假指标)。

输出:
    { summaries: {bench: path}, metrics: {bench: score...}, tag, run_id, git_commit, mock }
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from zenml import step

from pipelines.config import git_commit, resolve_path, run_cmd

# benchmark -> (OUT_ROOT 子目录, 主指标提取函数名); 与 run_benchmarks_hf.sh setup_bench 对齐
_BENCH_DIRS = {
    "ai2d": "AI2D",
    "docvqa": "DocVQA",
    "ocrbench": "OCRBench",
    "ours": "ourBenchmark/single",
}


def _headline_metric(bench: str, summary: dict):
    """从 summary.json 提取主指标 (与 run_benchmarks_hf.sh 汇总表一致)。"""
    if bench == "ai2d":
        return {"accuracy": summary["accuracy"]}
    if bench == "docvqa":
        return {"anls": summary["anls"]}
    if bench == "ocrbench":
        return {"final_score": summary["final_score"]}
    if bench == "ours":
        ov = summary["overall"]
        return {
            "anls_level2_all": ov["level2_all_text"]["anls"],
            "anls_level2_hidden": ov["level2_hidden_only"]["anls"],
        }
    return summary  # 未知 benchmark: 原样返回


@step(enable_cache=False)  # 副作用步骤 (推理/写预测文件), 关闭缓存
def evaluate_model(train_info: dict, mock: bool, params: dict, run_id: str) -> dict:
    """评估模型, 返回 summary 路径与主指标。"""
    log = logging.getLogger(__name__)
    log.info("evaluate_model: mock=%s run_id=%s params=%s", mock, run_id, params)

    tag = params.get("tag") or (run_id if run_id.startswith("zenml") else f"zenml_{run_id}")
    benchmarks = [b.strip() for b in str(params.get("benchmarks", "ours")).split(",") if b.strip()]
    out_root = resolve_path(params.get("out_root", "data/predictions/hf"))

    if mock:
        log.info("[MOCK] 写假 summary (tag=%s, benchmarks=%s)", tag, benchmarks)
        summaries, metrics = {}, {}
        for bench in benchmarks:
            bench_dir = out_root / _BENCH_DIRS.get(bench, bench) / tag
            bench_dir.mkdir(parents=True, exist_ok=True)
            if bench == "ai2d":
                summary = {"accuracy": 0.75, "n_ok": 3088}
            elif bench == "docvqa":
                summary = {"anls": 0.82, "n_scored": 5349}
            elif bench == "ocrbench":
                summary = {"final_score": 480, "final_score_max": 1000, "n_ok": 1000}
            elif bench == "ours":
                summary = {"overall": {
                    "n": 2063, "n_ok": 2063,
                    "level2_all_text": {"anls": 0.71},
                    "level2_hidden_only": {"anls": 0.68},
                }}
            else:
                summary = {"mock": True}
            summary["_mock"] = True
            summary["_run_id"] = run_id
            summary["_checkpoint_dir"] = train_info.get("checkpoint_dir")
            summary["_git_commit"] = train_info.get("git_commit")
            spath = bench_dir / f"{tag}_summary.json"
            spath.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
            summaries[bench] = str(spath)
            metrics[bench] = _headline_metric(bench, summary) if bench in _BENCH_DIRS else summary
    else:
        # ---- real: 调用现有 benchmark 脚本 ----
        script = resolve_path(params["script"])
        adapter = train_info.get("checkpoint_dir", "")
        cmd = [
            "/bin/bash", str(script),
            "--model", params["model"],
            "--tag", tag,
            "--benchmarks", ",".join(benchmarks),
            "--gpus", str(params.get("gpus", "0,1,2,3")),
            "--batch-size", str(params.get("batch_size", 4)),
        ]
        if adapter and not params.get("skip_adapter"):
            cmd += ["--adapter", adapter]
        if params.get("scales"):
            cmd += ["--scales", str(params["scales"])]
        if params.get("dry_run"):
            cmd += ["--dry-run"]
        if params.get("scan_only"):
            cmd += ["--scan-only"]
        env = {k: str(v) for k, v in params.get("env", {}).items()}
        rc = run_cmd(cmd, env=env)
        if rc != 0:
            raise RuntimeError(f"评估脚本失败, exit code {rc} (见上方日志)")

        summaries, metrics = {}, {}
        if params.get("dry_run") or params.get("scan_only"):
            # dry-run/scan-only 不产出 summary, 直接返回 (链路验证用)
            log.info("dry_run/scan_only 模式: 跳过 summary 收集, 评估命令已验证可执行")
        else:
            for bench in benchmarks:
                spath = out_root / _BENCH_DIRS.get(bench, bench) / tag / f"{tag}_summary.json"
                if not spath.exists():
                    raise FileNotFoundError(
                        f"评估完成但未找到 summary: {spath} "
                        f"(dry-run 模式不产生预测, 请关闭 dry_run 后重试)")
                summaries[bench] = str(spath)
                with open(spath, encoding="utf-8") as f:
                    metrics[bench] = _headline_metric(bench, json.load(f))

    result = {
        "summaries": summaries,
        "metrics": metrics,
        "tag": tag,
        "checkpoint_dir": train_info.get("checkpoint_dir"),
        "train_file": train_info.get("train_file"),
        "run_id": run_id,
        "git_commit": git_commit(),
        "mock": mock,
    }
    # example: 第一条预测记录 (real 模式) 或 summary 内容摘要 (mock),
    # Dashboard artifact 页可直接查看
    try:
        if mock:
            first_bench = next(iter(summaries))
            with open(summaries[first_bench], encoding="utf-8") as f:
                result["example"] = {"benchmark": first_bench,
                                     "summary": json.load(f)}
        else:
            # 预测文件: <out_root>/<Bench>/<tag>/<tag>_predictions.jsonl
            first_bench = next(iter(summaries))
            pred = (out_root / _BENCH_DIRS.get(first_bench, first_bench) / tag
                    / f"{tag}_predictions.jsonl")
            if pred.exists():
                with open(pred, encoding="utf-8") as f:
                    rec = json.loads(f.readline())
                result["example"] = {
                    "benchmark": first_bench,
                    "idx": rec.get("idx"),
                    "predict": (rec.get("predict") or rec.get("pred_ocr_text")
                                or rec.get("raw") or "")[:300],
                    "gt": str(rec.get("gt") or rec.get("answer") or "")[:200],
                }
    except Exception:
        pass
    log.info("evaluate_model 完成: %s", json.dumps(result, ensure_ascii=False))
    return result
