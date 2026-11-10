#!/usr/bin/env bash
# 回归对照队列（下午）：去掉 offset 分类头竞争，加强回归权重。
#
# 用法（容器内）：
#   bash scripts/reg2_queue.sh
set -uo pipefail
cd /Docker/git/piano2 || exit 1

V4=/Docker/piano2_data/processed/maestro_processed_v4
QUEUE_DIR=outputs/reg2
mkdir -p "$QUEUE_DIR" outputs/eval
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$QUEUE_DIR/queue.log"; }

run() {
  local name="$1"; shift
  local done_marker="$QUEUE_DIR/$name.done"
  local model="outputs/M1/$name/checkpoints/best.pt"
  if [ -f "$done_marker" ]; then log "skip $name"; return 0; fi
  log "start $name: $*"
  PIANO2_DATA_ROOT="$V4" python -m piano2.train \
    --max-steps 71160 --batch 16 --val-every 500 --save-every 200 \
    --val-segments 120 --num-workers 4 --run-name "$name" --resume auto "$@" \
    >> "$QUEUE_DIR/$name.log" 2>&1
  local rc=$?
  log "train $name rc=$rc"
  if [ $rc -ne 0 ]; then echo "train_rc=$rc" > "$QUEUE_DIR/HOLD_REASON.txt"; exit 1; fi
  python scripts/evaluate_reg.py --model "$model" --out "outputs/eval/$name.json" \
    >> "$QUEUE_DIR/$name.log" 2>&1 || true
  { echo "finished $(date '+%F %T')"; } > "$done_marker"
  log "done $name"
}

log "===== 回归对照队列开始 ====="
# 1) 去掉 offset 分类头损失，回归权重 1.0
run reg_pure --label-channel onset1 --offset-loss 0.0 --reg-loss 1.0
# 2) 去掉 offset 分类头损失，回归权重 2.0
run reg_pure2 --label-channel onset1 --offset-loss 0.0 --reg-loss 2.0
{ echo "finished $(date '+%F %T')"; echo "all_done"; } > "$QUEUE_DIR/REG2_DONE.txt"
log "===== 回归对照队列结束 ====="
