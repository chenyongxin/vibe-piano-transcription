#!/usr/bin/env bash
# 夜间队列（2026-11-07 夜）：offset 损失权重确认与扫描，4 组。
# 跑完自动评测并写 NIGHT2_DONE.txt，不自动进入下一阶段。
#
# 用法（容器内）：
#   bash scripts/night2_queue.sh
set -uo pipefail
cd /Docker/git/piano2 || exit 1

STEPS=71160
V3B=/Docker/maestro_processed_v3b
QUEUE_DIR=outputs/night2
mkdir -p "$QUEUE_DIR" outputs/eval

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$QUEUE_DIR/queue.log"; }

run_train() {
  local name="$1" data_root="$2" eval_offset="$3"; shift 3
  local done_marker="$QUEUE_DIR/$name.done"
  local model="outputs/M1/$name/checkpoints/best.pt"
  local eval_flag=""
  if [ "$eval_offset" = "1" ]; then eval_flag="--offset-end"; fi
  if [ -f "$done_marker" ]; then log "skip $name（已完成）"; return 0; fi

  log "start $name：data=$data_root 额外参数=$*"
  PIANO2_DATA_ROOT="$data_root" python -m piano2.train \
    --max-steps "$STEPS" --batch 16 --val-every 500 --save-every 200 \
    --val-segments 120 --num-workers 4 --run-name "$name" --resume auto "$@" \
    >> "$QUEUE_DIR/$name.log" 2>&1
  local rc=$?
  log "train $name 结束 rc=$rc，开始评测"
  if [ $rc -ne 0 ]; then
    echo "train_rc=$rc" > "$QUEUE_DIR/HOLD_REASON.txt"
    log "ERROR: $name 训练失败，队列停止"
    exit 1
  fi

  python scripts/evaluate.py --model "$model" --out "outputs/eval/$name.json" \
    $eval_flag >> "$QUEUE_DIR/$name.log" 2>&1 || true
  python scripts/evaluate_holdout.py --model "$model" \
    --out "outputs/eval/${name}_holdout.json" $eval_flag >> "$QUEUE_DIR/$name.log" 2>&1 || true

  if [ ! -f "outputs/eval/$name.json" ] || [ ! -f "outputs/eval/${name}_holdout.json" ]; then
    echo "eval_missing" > "$QUEUE_DIR/HOLD_REASON.txt"
    log "ERROR: $name 评测缺失，队列停止"
    exit 1
  fi
  { echo "finished $(date '+%F %T')"; echo "rc=$rc"; } > "$done_marker"
  log "done $name"
}

log "===== 夜间队列开始（offset 损失权重）====="

# 1) loss 2.0 复现（seed 1）
run_train combo_loss2_seed1 "$V3B" 1 --label-channel onset1 --offset-loss 2.0 --seed 1
# 2) loss 2.0 复现（seed 2）
run_train combo_loss2_seed2 "$V3B" 1 --label-channel onset1 --offset-loss 2.0 --seed 2
# 3) loss 3.0
run_train combo_loss3 "$V3B" 1 --label-channel onset1 --offset-loss 3.0
# 4) loss 4.0
run_train combo_loss4 "$V3B" 1 --label-channel onset1 --offset-loss 4.0

{ echo "finished $(date '+%F %T')"; echo "all_four_done"; } > "$QUEUE_DIR/NIGHT2_DONE.txt"
log "===== 夜间队列结束（等待分析）====="
