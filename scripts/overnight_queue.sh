#!/usr/bin/env bash
# 通宵实验队列（M3.5）：D2 → A（offset 头）→ B → C。
#
# 特性：
#   - 每个实验有 done 标记，脚本被中断后重新运行会跳过已完成的实验；
#   - 训练统一用 --resume auto：被中断的实验会从 latest.pt 续训；
#   - 日志与结果都在 outputs/overnight/ 与 outputs/eval/。
#
# 用法（容器内）：
#   bash scripts/overnight_queue.sh
set -uo pipefail
cd /Docker/git/piano2 || exit 1

STD_STEPS=71160          # 20 轮
V2=/Docker/piano2_data/processed/maestro_processed_v2
V3A=/Docker/piano2_data/processed/maestro_processed_v3a
V3B=/Docker/piano2_data/processed/maestro_processed_v3b
QUEUE_DIR=outputs/overnight
mkdir -p "$QUEUE_DIR" outputs/eval

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$QUEUE_DIR/queue.log"; }

run_train() {
  local name="$1" data_root="$2" steps="$3" eval_offset="$4"; shift 4
  local done_marker="$QUEUE_DIR/$name.done"
  local model="outputs/M1/$name/checkpoints/best.pt"
  local eval_flag=""
  if [ "$eval_offset" = "1" ]; then eval_flag="--offset-end"; fi
  if [ -f "$done_marker" ]; then log "skip $name（已完成）"; return 0; fi

  log "start $name：steps=$steps data=$data_root 额外参数=$*"
  PIANO2_DATA_ROOT="$data_root" python -m piano2.train \
    --max-steps "$steps" --batch 16 --val-every 500 --save-every 200 \
    --val-segments 120 --num-workers 4 --run-name "$name" --resume auto "$@" \
    >> "$QUEUE_DIR/$name.log" 2>&1
  local rc=$?
  log "train $name 结束 rc=$rc，开始评测"

  python scripts/evaluate.py --model "$model" --out "outputs/eval/$name.json" \
    $eval_flag >> "$QUEUE_DIR/$name.log" 2>&1 || true
  python scripts/evaluate_holdout.py --model "$model" \
    --out "outputs/eval/${name}_holdout.json" $eval_flag >> "$QUEUE_DIR/$name.log" 2>&1 || true

  { echo "finished $(date '+%F %T')"; echo "rc=$rc"; } > "$done_marker"
  log "done $name"
}

log "===== 通宵队列开始 ====="

if [ ! -f "$V3A/manifest.json" ]; then
  log "构建数据 v3a（frame_key）"
  python scripts/build_labels_v3a.py --workers 8 >> "$QUEUE_DIR/build_v3a.log" 2>&1
fi
if [ ! -f "$V3B/manifest.json" ]; then
  log "构建数据 v3b（+offset 通道）"
  python scripts/build_labels_v3b.py --workers 8 >> "$QUEUE_DIR/build_v3b.log" 2>&1
fi

# 1) D2：frame 改为按键时长（不含踏板）
run_train d2_frame_key "$V3A" "$STD_STEPS" 0
# 2) A：offset 头（v3b 数据 + offset 损失 + 用 offset 定结束）
run_train a_offset "$V3B" "$STD_STEPS" 1 --offset-loss 1.0
# 3) B：1 帧 onset
run_train b_onset1 "$V2" "$STD_STEPS" 0 --label-channel onset1
# 4) C：数据增强（频率遮蔽 + 时间遮蔽）
run_train c_aug "$V2" "$STD_STEPS" 0 --aug-freq-mask 2 --aug-time-mask 2

log "===== 通宵队列结束 ====="
