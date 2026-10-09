"""模型训练 Step — 包装现有训练脚本 (OPSD / SFT), 保留 Accelerate/DeepSpeed/verl 机制。

real 模式 (mode=opsd):
    subprocess 运行 bash other/Vision-OPD/scripts/run_opsd_gt_mixed_no_bbox_v4b.sh [extra_args...]
    脚本内部走 conda env vision-opd + verl.trainer.main_ppo;
    extra_args 会透传给 verl (脚本已支持 "$@"), 如 trainer.total_epochs=1。
    checkpoint 输出到 other/Vision-OPD/checkpoints/<EXPERIMENT_NAME>/ (由脚本管理)。

real 模式 (mode=sft):
    subprocess 运行 bash sft/experiments/run_sft_v4.sh,
    通过环境变量配置 (MODE/TAG/GPUS/NPROC/BS/ACCUM), 脚本内部用 ~/venvs/qwen3vl + ms-swift。
    TAG 默认加 zenml_<run_id> 前缀, 保证不覆盖已有实验目录。

mock 模式: 仅创建带标记文件的假 checkpoint 目录, 不启动 GPU 训练。

输出:
    { checkpoint_dir, adapter_dir, experiment_name, train_file, val_file,
      run_id, git_commit, mock }
"""
from __future__ import annotations

import json
import logging
import time

from zenml import step

from pipelines.config import git_commit, resolve_path, run_cmd


@step(enable_cache=False)  # 训练是重副作用步骤, 必须关闭缓存
def train_model(data_info: dict, mock: bool, params: dict, run_id: str) -> dict:
    """执行训练, 返回 checkpoint 路径与元信息。"""
    log = logging.getLogger(__name__)
    log.info("train_model: mock=%s run_id=%s params=%s", mock, run_id, params)
    commit = git_commit()

    if mock:
        ckpt_dir = resolve_path(params.get("mock_output_dir", "zenml_runs")) / "checkpoints" / run_id
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        log.info("[MOCK] 创建假 checkpoint: %s", ckpt_dir)
        (ckpt_dir / "adapter_config.json").write_text(json.dumps({
            "mock": True, "run_id": run_id, "git_commit": commit,
            "base_model": params.get("model", ""),
            "train_file": data_info["train_file"],
            "val_file": data_info["val_file"],
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, ensure_ascii=False, indent=2))
        (ckpt_dir / "README.md").write_text(
            f"# Mock checkpoint (zenml run {run_id})\n\n"
            f"由 ZenML pipeline mock 模式生成, 非真实模型权重。\n"
            f"- git_commit: {commit}\n- train_file: {data_info['train_file']}\n")
        result = {
            "checkpoint_dir": str(ckpt_dir),
            "experiment_name": f"mock_{run_id}",
            "train_file": data_info["train_file"],
            "val_file": data_info["val_file"],
            "n_train": data_info.get("n_train"),
            "run_id": run_id,
            "git_commit": commit,
            "mock": True,
            "example": {"checkpoint_files": sorted(p.name for p in ckpt_dir.iterdir())},
        }
        log.info("train_model [MOCK] 完成: %s", result)
        return result

    # ---- real ----
    mode = params.get("mode", "opsd")
    if mode == "opsd":
        script = resolve_path(params["script"])
        env = {k: str(v) for k, v in params.get("env", {}).items()}
        extra_args = list(params.get("extra_args", []))
        # 可选: 用 pipeline 上一步产出的 v4 数据替换脚本默认数据
        # (verl hydra 覆盖: data.train_files / data.val_files; 脚本 "$@" 透传已支持)
        if params.get("use_pipeline_data") and data_info.get("train_file"):
            train_f, val_f = data_info["train_file"], data_info["val_file"]
            extra_args += [
                f'data.train_files="[\"{train_f}\"]"',
                f'data.val_files="[\"{val_f}\"]"',
            ]
            log.info("use_pipeline_data=True: 训练数据替换为 pipeline 产物: "
                     "train=%s val=%s", train_f, val_f)
        rc = run_cmd(["/bin/bash", str(script)] + extra_args, env=env)
        if rc != 0:
            raise RuntimeError(f"OPSD 训练失败, exit code {rc} (见上方日志)")
        ckpt_dir = resolve_path(params["checkpoint_dir"])  # 脚本的 EXPERIMENT_NAME 目录
        if not ckpt_dir.exists():
            raise FileNotFoundError(
                f"训练完成但未找到 checkpoint_dir={ckpt_dir}; "
                f"请核对配置 checkpoint_dir 与脚本内 EXPERIMENT_NAME 是否一致")
        experiment_name = params.get("experiment_name", ckpt_dir.name)
    elif mode == "sft":
        script = resolve_path(params["script"])
        tag = params.get("tag") or f"zenml_{run_id}"
        env = {k: str(v) for k, v in params.get("env", {}).items()}
        env["TAG"] = tag
        # 可选: 用 pipeline 上一步产出的 SFT jsonl 训练 (vendored 脚本支持
        # TRAIN_DATA/VAL_DATA 环境变量覆盖; 原版脚本数据路径写死不支持)
        if params.get("use_pipeline_data") and data_info.get("sft_train_file"):
            env["TRAIN_DATA"] = str(data_info["sft_train_file"])
            env["VAL_DATA"] = str(data_info["sft_val_file"])
            log.info("use_pipeline_data=True: SFT 数据替换为 pipeline 产物: "
                     "train=%s val=%s", env["TRAIN_DATA"], env["VAL_DATA"])
        rc = run_cmd(["/bin/bash", str(script)] + list(params.get("extra_args", [])),
                     env=env)
        if rc != 0:
            raise RuntimeError(f"SFT 训练失败, exit code {rc} (见上方日志)")
        date = time.strftime("%Y%m%d")
        ckpt_dir = resolve_path(
            params.get("checkpoint_dir_fmt", "/cfs/cfs-izx10hg3/cinniepeng/models")
        ) / f"qwen3_5_9b_lora_{tag}_{date}"
        if not ckpt_dir.exists():
            raise FileNotFoundError(f"SFT 完成但未找到输出目录: {ckpt_dir}")
        experiment_name = f"qwen3_5_9b_lora_{tag}"
    else:
        raise ValueError(f"未知训练 mode: {mode} (支持 opsd | sft)")

    result = {
        "checkpoint_dir": str(ckpt_dir),
        "experiment_name": experiment_name,
        "train_file": data_info["train_file"],
        "val_file": data_info["val_file"],
        "n_train": data_info.get("n_train"),
        "run_id": run_id,
        "git_commit": commit,
        "mock": False,
    }
    # example: checkpoint 目录内容清单 (Dashboard artifact 页可核对权重文件是否落盘)
    try:
        ckpt_path = ckpt_dir
        # sft 的 swift 会在 ckpt_dir 下再建 v0-* 版本子目录, 找最深一层
        for sub in sorted(ckpt_path.glob("v0-*")):
            ckpt_path = sub
        files = sorted(p.name for p in ckpt_path.iterdir())
        # 若含 checkpoint-N 子目录, 列出其中一个的内容
        ckpts = [f for f in files if f.startswith("checkpoint")]
        if ckpts:
            inner = sorted(p.name for p in (ckpt_path / ckpts[-1]).iterdir())[:10]
            files = files + [f"{ckpts[-1]}/{f}" for f in inner]
        result["example"] = {"checkpoint_dir": str(ckpt_path),
                             "checkpoint_files": files[:20]}
    except Exception:
        pass
    log.info("train_model 完成: %s", result)
    return result
