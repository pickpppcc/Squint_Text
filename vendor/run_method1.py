#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""[ZenML Pipeline 包装] 可换扩散模型的 method1 合成入口。

原脚本 synthesis/augmentation/generate_method1_method2.py 的模型 ID 写死在
illusion_diffusion_gen.py 模块常量 (BASE_MODEL / CONTROLNET_MODEL / VAE_MODEL),
load_models() 运行时读取。本 wrapper 在 import 后、调用 main() 前替换这三个常量,
实现不改原代码的模型替换。

用法:
  python -m pipelines.vendor.run_method1 \
      [--base-model HF_ID] [--controlnet-model HF_ID] [--vae-model HF_ID] \
      [generate_method1_method2 原参数...]

不带 --base-model 等参数时与原命令完全等价。
"""
import sys
from pathlib import Path

# 保证项目根在 sys.path (python -m 运行时已在, 直跑脚本时补上)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main():
    argv = sys.argv[1:]
    overrides = {}
    rest = []
    i = 0
    while i < len(argv):
        if argv[i] in ("--base-model", "--controlnet-model", "--vae-model"):
            overrides[argv[i]] = argv[i + 1]
            i += 2
        else:
            rest.append(argv[i])
            i += 1

    from synthesis.augmentation import illusion_diffusion_gen as gen_mod

    if "--base-model" in overrides:
        gen_mod.BASE_MODEL = overrides["--base-model"]
    if "--controlnet-model" in overrides:
        gen_mod.CONTROLNET_MODEL = overrides["--controlnet-model"]
    if "--vae-model" in overrides:
        gen_mod.VAE_MODEL = overrides["--vae-model"]
    if overrides:
        print(f"[run_method1] 模型替换: base={gen_mod.BASE_MODEL} "
              f"controlnet={gen_mod.CONTROLNET_MODEL} vae={gen_mod.VAE_MODEL}")

    from synthesis.augmentation import generate_method1_method2

    sys.argv = [sys.argv[0]] + rest
    generate_method1_method2.main()


if __name__ == "__main__":
    main()
