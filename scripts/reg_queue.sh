#!/usr/bin/env bash
# 回归原型训练队列（今天第一组）：data_v4 + 回归头，训练 20 轮。
#
# 用法（容器内）：
#   bash scripts/reg_queue.sh
set -uo pipefail
cd /Docker/git/piano2 || exit 1

V4=/Docker/maestro_processed_v4
QUEUE_DIR=outputs/reg
mkdir -p "$QUEUE_DIR" outputs/eval
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$QUEUE_DIR/queue.log"; }

log "===== 回归原型训练开始 ====="
PIANO2_DATA_ROOT="$V4" python -m piano2.train \
  --max-steps 71160 --batch 16 --val-every 500 --save-every 200 \
  --val-segments 120 --num-workers 4 --run-name reg_base --resume auto \
  --label-channel onset1 --offset-loss 1.0 --reg-loss 0.5 \
  >> "$QUEUE_DIR/reg_base.log" 2>&1
rc=$?
log "训练结束 rc=$rc"
if [ $rc -ne 0 ]; then
  echo "train_rc=$rc" > "$QUEUE_DIR/HOLD_REASON.txt"
  exit 1
fi

python scripts/evaluate_reg.py --model outputs/M1/reg_base/checkpoints/best.pt \
  --out outputs/eval/reg_base.json >> "$QUEUE_DIR/reg_base.log" 2>&1 || true

{ echo "finished $(date '+%F %T')"; echo "rc=$rc"; } > "$QUEUE_DIR/reg_base.done"
{ echo "finished $(date '+%F %T')"; echo "all_done"; } > "$QUEUE_DIR/REG_DONE.txt"
log "===== 回归原型训练结束 ====="
