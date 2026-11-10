#!/usr/bin/env bash
# 混合模型：offset 分类 + 回归头，训练后做亚帧 onset 评测。
set -uo pipefail
cd /Docker/git/piano2 || exit 1

V4=/Docker/piano2_data/processed/maestro_processed_v4
QUEUE=outputs/hybrid
mkdir -p "$QUEUE" outputs/eval
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$QUEUE/queue.log"; }

name=hybrid_reg2
model=outputs/M1/$name/checkpoints/best.pt
if [ -f "$QUEUE/$name.done" ]; then log "skip $name"; exit 0; fi
log "start $name"
PIANO2_DATA_ROOT="$V4" python -m piano2.train \
  --max-steps 71160 --batch 16 --val-every 500 --save-every 200 \
  --val-segments 120 --num-workers 4 --run-name "$name" --resume auto \
  --label-channel onset1 --offset-loss 2.0 --reg-loss 0.5 \
  >> "$QUEUE/$name.log" 2>&1
rc=$?
log "train rc=$rc"
if [ $rc -ne 0 ]; then echo "train_rc=$rc" > "$QUEUE/HOLD_REASON.txt"; exit 1; fi
python scripts/evaluate_hybrid.py --model "$model" \
  --out "outputs/eval/${name}_hybrid.json" >> "$QUEUE/$name.log" 2>&1 || true
python scripts/evaluate_holdout.py --model "$model" --offset-end \
  --out "outputs/eval/${name}_holdout.json" >> "$QUEUE/$name.log" 2>&1 || true
{ echo "finished $(date '+%F %T')"; } > "$QUEUE/$name.done"
echo "finished $(date '+%F %T')" > "$QUEUE/HYBRID_DONE.txt"
log "done $name"
