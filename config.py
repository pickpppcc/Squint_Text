"""Pipeline 统一配置管理: YAML 加载、路径解析、git commit 记录。

所有 Step 的可调参数集中在 configs/zenml/*.yaml, 避免硬编码。
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import yaml

# 项目根目录 (pipelines/ 的上一级)
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path) -> dict:
    """加载 YAML 配置文件。相对路径以项目根目录为基准。"""
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(p: str | Path) -> Path:
    """相对路径 -> 项目根目录下的绝对路径; 绝对路径原样返回。"""
    p = Path(p)
    return p if p.is_absolute() else PROJECT_ROOT / p


def git_commit() -> str:
    """当前 git commit SHA (用于实验溯源; 失败返回 'unknown')。"""
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT, capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def run_cmd(cmd: list[str], *, env: dict | None = None, cwd: Path | None = None) -> int:
    """执行外部命令, stdout/stderr 实时打到 logging (会被 ZenML 记录到 Dashboard)。

    返回 returncode, 不抛异常 — 由调用方决定如何处理非零退出。
    """
    import logging
    import os

    full_env = os.environ.copy()
    if env:
        full_env.update({k: str(v) for k, v in env.items()})

    # 展开 ~ / 环境变量 (subprocess 不做 shell 展开)
    cmd = [os.path.expanduser(os.path.expandvars(c)) for c in cmd]

    logging.info("[cmd] %s (cwd=%s)", " ".join(cmd), cwd or PROJECT_ROOT)
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd or PROJECT_ROOT),
        env=full_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    for line in proc.stdout:
        logging.info("[out] %s", line.rstrip())
    proc.wait()
    if proc.returncode != 0:
        logging.error("[cmd] FAILED with exit code %s", proc.returncode)
    return proc.returncode
