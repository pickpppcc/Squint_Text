"""端到端训练 Pipeline DAG。

    prepare_models ─→ prepare_patterns ─→ synthesize_data ─→ process_data
    (模型检查/下载)    (pattern 图生成)     (method1 合成)      (转换/过滤)
    ─→ train_model ─→ evaluate_model ─→ log_run_metadata
       (OPSD/SFT)      (benchmarks)      (血缘汇总)

Step 之间只传递轻量元数据 dict (路径/条数/指标), 大文件 (图片/parquet/checkpoint)
留在 CFS 原地, 由路径引用 — 避免 ZenML artifact 拷贝大文件。
"""
from __future__ import annotations

from zenml import pipeline

from pipelines.steps import (
    evaluate_model,
    log_run_metadata,
    prepare_models,
    prepare_patterns,
    process_data,
    synthesize_data,
    train_model,
)


@pipeline
def aigc_training_pipeline(
    mock: bool,
    run_id: str,
    models_params: dict,
    patterns_params: dict,
    synthesis_params: dict,
    processing_params: dict,
    training_params: dict,
    evaluation_params: dict,
    meta_params: dict,
):
    """AIGC_CONTROLNET 端到端 pipeline。

    Args:
        mock: 全局 mock 开关, true 时所有 Step 走低成本路径 (无 GPU, 秒级)
        run_id: 本次运行唯一标识, 所有产物写入 <output_dir>/<run_id>/
        *_params: 各 Step 的参数 dict, 来自 pipelines/configs/*.yaml
    """
    models = prepare_models(
        mock=mock, params=models_params, run_id=run_id
    )
    patterns = prepare_patterns(
        mock=mock, params=patterns_params, run_id=run_id,
        after="prepare_models",
    )
    synthesis = synthesize_data(
        mock=mock, params=synthesis_params, run_id=run_id,
        patterns_output=patterns,
    )
    data = process_data(
        synthesis_output=synthesis, mock=mock, params=processing_params, run_id=run_id
    )
    model = train_model(
        data_info=data, mock=mock, params=training_params, run_id=run_id
    )
    evaluation = evaluate_model(
        train_info=model, mock=mock, params=evaluation_params, run_id=run_id
    )
    log_run_metadata(
        synthesis=synthesis,
        data=data,
        model=model,
        evaluation=evaluation,
        params=meta_params,
        run_id=run_id,
    )
