#!/usr/bin/env bash
# 时值专项：offset 标签窗口/损失权重对照。
#
# 用法（容器内）：
#   bash scripts/duration_queue.sh
set -uo pipefail
cd /Docker/git/piano2 || exit 1

QUEUE_DIR=outputs/duration
mkdir -p "$QUEUE_DIR" outputs/eval
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$QUEUE_DIR/queue.log"; }

run() {
  local name="$1"
  local data="$2"
  local loss="$3"
  local done_marker="$QUEUE_DIR/$name.done"
  local model="outputs/M1/$name/checkpoints/best.pt"
  if [ -f "$done_marker" ]; then log "skip $name"; return 0; fi
  log "start $name data=$data offset_loss=$loss"
  PIANO2_DATA_ROOT="$data" python -m piano2.train \
    --max-steps 71160 --batch 16 --val-every 500 --save-every 200 \
    --val-segments 120 --num-workers 4 --run-name "$name" --resume auto \
    --label-channel onset1 --offset-loss "$loss" --reg-loss 0 \
    >> "$QUEUE_DIR/$name.log" 2>&1
  local rc=$?
  log "train $name rc=$rc"
  if [ $rc -ne 0 ]; then echo "train_rc=$rc" > "$QUEUE_DIR/HOLD_REASON.txt"; exit 1; fi
  python scripts/evaluate_holdout.py --model "$model" --offset-end \
    --out "outputs/eval/${name}_holdout.json" >> "$QUEUE_DIR/$name.log" 2>&1 || true
  { echo "finished $(date '+%F %T')"; } > "$done_marker"
  log "done $name"
}

log "===== 时值专项队列开始 ====="
# 1) 1 帧 offset + 权重 3.0（用户直觉的对照组）
run duration_1f_w3 /Docker/piano2_data/processed/maestro_processed_v3c 3.0
# 2) 2 帧 offset + 权重 3.0（窗口宽度对照）
run duration_2f_w3 /Docker/piano2_data/processed/maestro_processed_v3b 3.0
{ echo "finished $(date '+%F %T')"; echo "all_done"; } > "$QUEUE_DIR/DURATION_DONE.txt"
log "===== 时值专项队列结束 ====="
