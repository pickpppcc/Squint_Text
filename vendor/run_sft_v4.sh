#!/usr/bin/env bash
# =============================================================================
# SFT 训练脚本 (v4) — canonical text-only OCR, 两个模式:
#   MODE=orig: 原图单图输入  (all_v4_{train,val}.jsonl)
#   MODE=ds8:  原图 + 1/8 下采样图双图输入 (all_v4_{train,val}_ds8.jsonl)
#
# 数据: data/sft/v4/ (由 data/sft/v4/build_v4.py 生成)
#   label = original_gt 去掉 bbox_2d 的 canonical JSON: [{"text_content":".."}, ...] / []
#   user  = "<image>[\n<image>]\n识别出图中所有的文字。"
# 超参与 run_sft_v3_ds8.sh 一致 (LoRA r16/a32, 3 epoch, lr 1e-4, freeze_vit+aligner);
# ds8 模式另设 MIN_PIXELS=784 避免 1/8 小图被 processor 上采样。
# 系统 prompt 沿用 v3 通用 prompt (覆盖文字提取/定位/问答)。
#
# 用法:
#   MODE=orig bash sft/experiments/run_sft_v4.sh
#   MODE=ds8  bash sft/experiments/run_sft_v4.sh
#   MODE=ds8 TAG=v4_ds8_1 NPROC=4 GPUS=0,1,2,3 bash sft/experiments/run_sft_v4.sh
# =============================================================================
# [ZenML Pipeline 拷贝版 — 拷贝自 sft/experiments/run_sft_v4.sh, 仅两处改动:
#  1. TRAIN_DATA / VAL_DATA 支持环境变量覆盖 (默认仍走 MODE 分支的原路径),
#     以便 pipeline 把本次处理 Step 产出的 SFT jsonl 直接喂给训练;
#  2. swift 命令末尾透传 "$@" (如 --max_steps 2 --save_strategy steps, 小规模冒烟用)。
#  其余逐字未改。]
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/../.."
VENV=~/venvs/qwen3vl
PY="$VENV/bin/python"

# --- 可调参数 ---
MODE="${MODE:-orig}"              # orig | ds8
TAG="${TAG:-v4_${MODE}_$(date +%Y%m%d)}"
NPROC="${NPROC:-4}"
GPUS="${GPUS:-0,1,2,3}"
BS="${BS:-2}"                      # per_device_train_batch_size (原 A100-40G 调的 2, H20-97G 可调大)
ACCUM="${ACCUM:-8}"               # gradient_accumulation_steps
MODEL="${MODEL:-/dockerdata/models/Qwen3.5-9B}"

if [ -n "${TRAIN_DATA:-}" ] && [ -n "${VAL_DATA:-}" ]; then
  :  # pipeline 注入: 直接用外部传入的数据路径
else
  case "$MODE" in
    orig)
      TRAIN_DATA=./data/sft/v4/all_v4_train.jsonl
      VAL_DATA=./data/sft/v4/all_v4_val.jsonl
      ;;
    ds8)
      TRAIN_DATA=./data/sft/v4/all_v4_train_ds8.jsonl
      VAL_DATA=./data/sft/v4/all_v4_val_ds8.jsonl
      ;;
    *) echo "Unknown MODE: $MODE (expected: orig | ds8)"; exit 1 ;;
  esac
fi

OUT_TAG="qwen3_5_9b_lora_${TAG}"

# --- Env ---
export LOG_LEVEL=INFO
export NPROC_PER_NODE="$NPROC"
export CUDA_VISIBLE_DEVICES="$GPUS"
export MASTER_PORT="${MASTER_PORT:-29500}"
export TOKENIZERS_PARALLELISM=false
export LD_LIBRARY_PATH="/usr/lib64:/usr/local/cuda-12.6/lib64:/usr/local/cuda-12.6/extras/CUPTI/lib64/:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/lib:/usr/local/lib"
if [ "$MODE" = "ds8" ]; then
  export MIN_PIXELS=784   # 28x28, 避免 1/8 小图被上采样偏离真实尺度
fi

# --- 输出目录 (CFS) ---
DATE=$(date +%Y%m%d)
CKPT_DIR=/cfs/cfs-izx10hg3/cinniepeng/models/${OUT_TAG}_${DATE}
OUTPUT_DIR=./output/${OUT_TAG}_${DATE}
mkdir -p "$CKPT_DIR"
ln -sfn "$CKPT_DIR" "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

# --- 校验数据 ---
for f in "$TRAIN_DATA" "$VAL_DATA"; do
  if [ ! -f "$f" ]; then
    echo "ERROR: $f not found. Run: python data/sft/v4/build_v4.py"; exit 1
  fi
done

echo "===== SFT v4 训练 (canonical text-only OCR) ====="
echo "  Mode:      $MODE"
echo "  Tag:       $TAG"
echo "  Model:     $MODEL"
echo "  Train:     $TRAIN_DATA ($(wc -l < "$TRAIN_DATA") records)"
echo "  Val:       $VAL_DATA ($(wc -l < "$VAL_DATA") records)"
echo "  GPUs:      $GPUS (nproc=$NPROC)"
echo "  BS/ACCUM:  $BS / $ACCUM  (effective batch = $((NPROC * BS * ACCUM)))"
echo "  Output:    $CKPT_DIR"
echo "  MIN_PIXELS: ${MIN_PIXELS:-<unset>}"
echo "  Think:     OFF (直接答案 SFT)"
echo ""

# --- 系统 prompt (沿用 v3 通用 prompt; 每条数据 user 已带任务指令) ---
SYSTEM_PROMPT="你是一个视觉理解助手。根据用户指令,对图片进行文字提取、文字位置定位(bounding box,坐标0-1000)或图文问答。"

# --- 启动 ms-swift SFT ---
echo ">>> 启动 swift SFT..."
"$PY" -m swift.cli.main sft \
    --model "$MODEL" \
    --model_type qwen3_5 \
    --template qwen3_5 \
    --system "$SYSTEM_PROMPT" \
    --tuner_type lora \
    --dataset "$TRAIN_DATA" \
    --val_dataset "$VAL_DATA" \
    --loss_scale default \
    --freeze_vit true \
    --freeze_aligner true \
    --lora_rank 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --target_modules all-linear \
    --torch_dtype bfloat16 \
    --attn_impl sdpa \
    --max_length 2048 \
    --max_pixels 1048576 \
    --num_train_epochs 3 \
    --per_device_train_batch_size "$BS" \
    --per_device_eval_batch_size "$BS" \
    --gradient_accumulation_steps "$ACCUM" \
    --gradient_checkpointing true \
    --learning_rate 1e-4 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.03 \
    --max_grad_norm 1.0 \
    --eval_strategy epoch \
    --save_strategy epoch \
    --save_total_limit 3 \
    --logging_steps 10 \
    --dataloader_num_workers 16 \
    --dataset_num_proc 16 \
    --output_dir "$OUTPUT_DIR" \
    --report_to tensorboard \
    --logging_first_step true \
    --log_level info \
    --seed 42 \
    "$@" \
    2>&1 | tee -a "$OUTPUT_DIR/train.log"

echo "===== 训练完成: $CKPT_DIR ====="
