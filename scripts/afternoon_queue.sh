#!/usr/bin/env bash
# 下午队列（2026-11-07）：3 组单变量实验，约 5 小时，跑完停下等人分析。
#
# 用法（容器内）：
#   bash scripts/afternoon_queue.sh
set -uo pipefail
cd /Docker/git/piano2 || exit 1

STEPS=71160
V3B=/Docker/maestro_processed_v3b
V3D=/Docker/maestro_processed_v3d
QUEUE_DIR=outputs/afternoon
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

log "===== 下午队列开始 ====="

if [ ! -f "$V3D/manifest.json" ]; then
  log "构建数据 v3d（offset 3 帧居中）"
  python scripts/build_labels_v3d.py --workers 8 >> "$QUEUE_DIR/build_v3d.log" 2>&1
fi

# 1) combo_seed2：combo_base 的第二个复现种子（方差）
run_train combo_seed2 "$V3B" 1 --label-channel onset1 --offset-loss 1.0 --seed 2
# 2) combo_offset_loss2：offset 损失权重加倍，验证 offset 监督是否偏弱
run_train combo_offset_loss2 "$V3B" 1 --label-channel onset1 --offset-loss 2.0
# 3) combo_offset3：offset 标签改 3 帧居中（v3d），验证窗口中心偏差
run_train combo_offset3 "$V3D" 1 --label-channel onset1 --offset-loss 1.0

{ echo "finished $(date '+%F %T')"; echo "all_three_done"; } > "$QUEUE_DIR/AFTERNOON_DONE.txt"
log "===== 下午队列结束（等待分析）====="
