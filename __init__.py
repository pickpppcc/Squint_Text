"""ZenML pipelines for AIGC_CONTROLNET.

数据合成 → 数据处理 → 模型训练 → 模型评估 的自动化 Pipeline。

设计原则:
  - 只包装现有 CLI/脚本 (subprocess 调用), 不修改原有科研代码
  - Step 之间传递路径 + 元信息的小 dict (ZenML Artifact), 大文件不拷贝
  - 所有输出目录带 run_id, 只新增不覆盖已有 Checkpoint / 数据 / 实验结果
  - 每个 Step 支持 mock 模式, 用于低成本端到端验证

结构:
  run.py               统一执行入口
  training_pipeline.py @pipeline DAG 定义
  steps/               各 ZenML Step (包装现有科研代码)
  configs/             pipeline 运行配置 (mock / real_sft / real_opsd)
  config.py            配置加载 / 路径解析 / git commit / subprocess 工具
"""
