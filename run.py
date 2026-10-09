#!/usr/bin/env python3
"""ZenML Pipeline 统一执行入口。

用法:
    # 完整 Pipeline — Mock 端到端验证 (无 GPU, 秒级)
    ~/venvs/zenml/bin/python pipelines/run.py --config pipelines/configs/mock.yaml

    # 完整 Pipeline — 真实运行 (GPU, 先阅读 real_*.yaml 内注释)
    ~/venvs/zenml/bin/python pipelines/run.py --config pipelines/configs/real_sft.yaml

    # 单独执行某个 Step (复用已有数据/Checkpoint, 不跑完整 pipeline)
    ~/venvs/zenml/bin/python pipelines/run.py --config pipelines/configs/mock.yaml \
        --step synthesis
    ~/venvs/zenml/bin/python pipelines/run.py --config pipelines/configs/real_sft.yaml \
        --step training --set data_info.train_file=data/opsd/.../train.parquet
    ~/venvs/zenml/bin/python pipelines/run.py --config pipelines/configs/real_sft.yaml \
        --step evaluation --set train_info.checkpoint_dir=/path/to/existing_ckpt

    # 覆盖个别参数
    ~/venvs/zenml/bin/python pipelines/run.py --config pipelines/configs/mock.yaml \
        --set synthesis.n=16

    Dashboard:
    ~/venvs/zenml/bin/zenml up   # 默认 http://127.0.0.1:8237

每次运行自动生成唯一 run_id, 所有产物写入 <output_dir>/<run_id>/, 不覆盖已有数据。
"""
from __future__ import annotations

import argparse
import importlib
import json
import logging
import sys
import time
import uuid
from pathlib import Path

# 保证从任意 cwd 运行都能 import 项目内包 (pipelines.*, synthesis.* 等)
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from pipelines.config import git_commit, load_config  # noqa: E402
from pipelines.inputs import expand_inputs  # noqa: E402
from pipelines.training_pipeline import aigc_training_pipeline  # noqa: E402

# 单 Step 直调表: step 名 -> (模块, 函数, 配置 section, 上游 artifact 参数名)
_STEP_FUNCS = {
    "prepare": ("pipelines.steps.prepare_models", "prepare_models", "models", None),
    "patterns": ("pipelines.steps.prepare_patterns", "prepare_patterns", "patterns", None),
    "synthesis": ("pipelines.steps.data_synthesis", "synthesize_data", "synthesis", None),
    "processing": ("pipelines.steps.data_processing", "process_data", "processing", "synthesis_output"),
    "training": ("pipelines.steps.training", "train_model", "training", "data_info"),
    "evaluation": ("pipelines.steps.evaluation", "evaluate_model", "evaluation", "train_info"),
}


def _apply_set(cfg: dict, sets: list[str]) -> dict:
    """--set key=value 覆盖嵌套配置 (点号分隔 key, value 尝试 JSON 解析)。"""
    for s in sets:
        key, _, val = s.partition("=")
        if not _:
            raise SystemExit(f"--set 格式应为 key=value: {s}")
        try:
            val = json.loads(val)
        except json.JSONDecodeError:
            pass
        node = cfg
        keys = key.split(".")
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        node[keys[-1]] = val
    return cfg


def _default_upstream(step: str) -> dict:
    """单独运行某 Step 时, 补齐其上游 artifact dict 的保守占位值
    (实际值应通过 --set upstream.xxx / --set data_info.xxx 注入)。"""
    if step == "processing":
        return {"manifest_path": "", "n_images": 0, "run_id": "", "git_commit": ""}
    if step == "training":
        return {"train_file": "", "val_file": "", "n_train": None,
                "output_dir": "", "run_id": "", "git_commit": ""}
    if step == "evaluation":
        return {"checkpoint_dir": "", "experiment_name": "", "train_file": "",
                "val_file": "", "run_id": "", "git_commit": ""}
    return {}


def main() -> None:
    parser = argparse.ArgumentParser(description="AIGC_CONTROLNET ZenML Pipeline 入口")
    parser.add_argument("--config", required=True,
                        help="YAML 配置文件 (相对项目根目录)")
    parser.add_argument("--mock", choices=["true", "false"], default=None,
                        help="强制指定 mock/real 模式 (覆盖配置文件)")
    parser.add_argument("--step", choices=list(_STEP_FUNCS),
                        help="只运行单个 Step (不跑完整 pipeline); 上游输入用 --set 注入")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="覆盖配置项, 点号分隔嵌套 key (可重复使用, 如 --set synthesis.n=16)")
    parser.add_argument("--run-id", default=None,
                        help="自定义 run ID (默认自动生成 zenml-<时间戳>-<随机>)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    cfg = load_config(args.config)
    cfg = _apply_set(cfg, args.set)
    # 统一输入声明 inputs: 展开为各 step 参数 (step 段显式值优先, --set 最高;
    # inputs 内被 --set 修改的值也能传播到 step)
    cfg = expand_inputs(cfg)
    mock = str(args.mock if args.mock is not None
               else cfg.get("run", {}).get("mock", True)).lower() in ("1", "true", "yes")
    run_id = args.run_id or f"zenml-{time.strftime('%Y%m%d_%H%M%S')}-{uuid.uuid4().hex[:6]}"
    seed = cfg.get("run", {}).get("seed", 42)
    cfg.setdefault("synthesis", {}).setdefault("seed", seed)  # 全局 seed 注入

    logging.info("run_id=%s mock=%s seed=%s git=%s config=%s",
                 run_id, mock, seed, git_commit(), args.config)

    if args.step:
        # ---- 单 Step 独立运行: 直接调用 step 函数 (ZenML 支持在 pipeline 外执行) ----
        mod_name, fn_name, cfg_key, upstream_arg = _STEP_FUNCS[args.step]
        fn = getattr(importlib.import_module(mod_name), fn_name)
        params = cfg.get(cfg_key, {})
        upstream = cfg.get("upstream", None) or _default_upstream(args.step)
        logging.info("单独运行 Step '%s'", args.step)
        kwargs = {upstream_arg: upstream} if upstream_arg else {}
        out = fn(mock=mock, params=params, run_id=run_id, **kwargs)
        logging.info("Step '%s' 输出:\n%s", args.step,
                     json.dumps(out, ensure_ascii=False, indent=2, default=str))
        return

    # ---- 完整 Pipeline ----
    meta_params = {
        "seed": seed,
        "output_dir": cfg.get("outputs", {}).get("metadata", "zenml_runs"),
        "config_snapshot": cfg,
    }
    pipeline_instance = aigc_training_pipeline
    try:  # 给 run 打 tag, Dashboard 里可按 run_id / mock|real 过滤
        pipeline_instance = pipeline_instance.with_options(
            tags=[run_id, "mock" if mock else "real"]
        )
    except Exception:
        pass  # with_options 不支持 tags 时直接裸跑

    pipeline_instance(
        mock=mock,
        run_id=run_id,
        models_params=cfg.get("models", {}),
        patterns_params=cfg.get("patterns", {}),
        synthesis_params=cfg.get("synthesis", {}),
        processing_params=cfg.get("processing", {}),
        training_params=cfg.get("training", {}),
        evaluation_params=cfg.get("evaluation", {}),
        meta_params=meta_params,
    )


if __name__ == "__main__":
    main()
