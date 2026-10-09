"""模型准备 Step — 检查并补齐 pipeline 需要的模型权重 (首个 Step)。

按配置中的模型清单逐个检查:
  1. 本地路径已就绪 -> 跳过
  2. 本地缺失但 CFS 备份存在 -> rsync 拷贝 (参考 dev/env.sh 的模型拷贝逻辑,
     CFS/NFS 慢, 拷到本地 NVMe 加速训练读取)
  3. 都没有 -> 从 HuggingFace 下载 (huggingface_hub.snapshot_download)

mock 模式: 只做检查与打印, 不做拷贝/下载。

输出:
    { models: {name: {local_dir, status, source}}, run_id, git_commit, mock }
"""
from __future__ import annotations

import logging
from pathlib import Path

from zenml import step

from pipelines.config import git_commit, run_cmd

# HF cache 中的模型 (不写 local_dir 时, snapshot_download 走默认缓存)
_HF_CACHE = Path.home() / ".cache/huggingface/hub"


def _cache_present(hf_id: str) -> bool:
    """HF 默认缓存中是否已有该模型 (目录名 models--org--name)。"""
    marker = _HF_CACHE / ("models--" + hf_id.replace("/", "--")) / "snapshots"
    return marker.is_dir() and any(marker.iterdir())


@step(enable_cache=False)  # 涉及外部文件系统副作用, 关闭缓存
def prepare_models(mock: bool, params: dict, run_id: str) -> dict:
    """检查/下载模型权重, 返回各模型状态。"""
    log = logging.getLogger(__name__)
    log.info("prepare_models: mock=%s run_id=%s", mock, run_id)

    specs = params.get("models", [])
    if not specs:
        raise ValueError("prepare_models 需要配置 models 清单 (见 configs 内 models: 段)")

    results = {}
    for spec in specs:
        name = spec["name"]
        local_dir = spec.get("local_dir")          # 如 /dockerdata/models/Qwen3.5-9B
        cfs_src = spec.get("cfs_src")              # 如 /cfs/.../models/Qwen3.5-9B
        hf_id = spec.get("hf_id")                  # 如 Qwen/Qwen3.5-9B
        entry = {"local_dir": local_dir, "hf_id": hf_id}

        # ---- 1. 已就绪 ----
        if local_dir and Path(local_dir).is_dir() and any(Path(local_dir).iterdir()):
            log.info("[%s] 已就绪: %s (跳过)", name, local_dir)
            results[name] = {**entry, "status": "ready", "source": "local"}
            continue
        if not local_dir and hf_id and _cache_present(hf_id):
            log.info("[%s] HF 缓存已存在: %s (跳过)", name, hf_id)
            results[name] = {**entry, "status": "ready", "source": "hf_cache"}
            continue

        if mock:
            log.info("[MOCK] [%s] 缺失 (local=%s), 仅记录不下载", name, local_dir or hf_id)
            results[name] = {**entry, "status": "missing", "source": None}
            continue

        # ---- 2. 从 CFS 拷贝 (env.sh 模式: NFS 慢, 拷本地 NVMe) ----
        if cfs_src and Path(cfs_src).is_dir() and local_dir:
            log.info("[%s] 从 CFS 拷贝: %s -> %s", name, cfs_src, local_dir)
            Path(local_dir).parent.mkdir(parents=True, exist_ok=True)
            rc = run_cmd([
                "rsync", "-a", "-W", "--info=progress2",
                f"{cfs_src.rstrip('/')}/", f"{local_dir}/",
            ])
            if rc != 0:
                raise RuntimeError(f"[{name}] CFS 拷贝失败, exit code {rc}")
            results[name] = {**entry, "status": "copied", "source": cfs_src}
            continue

        # ---- 3. 从 HuggingFace 下载 ----
        if hf_id:
            target = local_dir or None   # None -> HF 默认缓存
            log.info("[%s] 从 HuggingFace 下载: %s -> %s", name, hf_id,
                     target or "(HF 默认缓存)")
            cmd = ["hf", "download", hf_id]
            if target:
                cmd += ["--local-dir", target]
            rc = run_cmd(cmd)
            if rc != 0:
                raise RuntimeError(
                    f"[{name}] HF 下载失败 (exit {rc}); "
                    f"请检查网络或 HF_TOKEN (私有模型需认证)")
            results[name] = {**entry, "status": "downloaded",
                             "source": f"huggingface:{hf_id}"}
            continue

        raise FileNotFoundError(
            f"[{name}] 模型缺失且无法获取: local_dir={local_dir} 不存在, "
            f"cfs_src={cfs_src} 不存在, 未配置 hf_id")

    result = {"models": results, "run_id": run_id,
              "git_commit": git_commit(), "mock": mock}
    # example: 各模型目录的关键文件清单 (前 8 个), 核对权重/配置是否齐
    example = {}
    for name, entry in results.items():
        d = entry.get("local_dir")
        if d and Path(d).is_dir():
            example[name] = sorted(p.name for p in Path(d).iterdir())[:8]
    if example:
        result["example"] = example
    log.info("prepare_models 完成: %s", {k: v["status"] for k, v in results.items()})
    return result
