#!/usr/bin/env bash
# Round 2 队列（2026-11-06 周五夜）：combo 系列 6 组，跑到周六中午。
# 跑完自动评测并写 ROUND2_DONE.txt，不自动进入 Round 3（等人一起分析）。
#
# 用法（容器内）：
#   bash scripts/round2_queue.sh
set -uo pipefail
cd /Docker/git/piano2 || exit 1

STD_STEPS=71160
LONG_STEPS=142320
V3A=/Docker/piano2_data/processed/maestro_processed_v3a
V3B=/Docker/piano2_data/processed/maestro_processed_v3b
V3C=/Docker/piano2_data/processed/maestro_processed_v3c
QUEUE_DIR=outputs/overnight2
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
  if [ $rc -ne 0 ]; then
    log "ERROR: $name 训练失败，队列停止"
    echo "train_rc=$rc" > "$QUEUE_DIR/HOLD_REASON.txt"
    exit 1
  fi

  python scripts/evaluate.py --model "$model" --out "outputs/eval/$name.json" \
    $eval_flag >> "$QUEUE_DIR/$name.log" 2>&1 || true
  python scripts/evaluate_holdout.py --model "$model" \
    --out "outputs/eval/${name}_holdout.json" $eval_flag >> "$QUEUE_DIR/$name.log" 2>&1 || true

  if [ ! -f "outputs/eval/$name.json" ] || [ ! -f "outputs/eval/${name}_holdout.json" ]; then
    log "ERROR: $name 评测缺失，队列停止"
    echo "eval_missing" > "$QUEUE_DIR/HOLD_REASON.txt"
    exit 1
  fi

  { echo "finished $(date '+%F %T')"; echo "rc=$rc"; } > "$done_marker"
  log "done $name"
}

log "===== Round 2 开始 ====="

if [ ! -f "$V3C/manifest.json" ]; then
  log "构建数据 v3c（offset 1 帧）"
  python scripts/build_labels_v3c.py --workers 8 >> "$QUEUE_DIR/build_v3c.log" 2>&1
fi

# 1) combo_base：v3b + 1 帧 onset + offset 损失
run_train combo_base "$V3B" "$STD_STEPS" 1 --label-channel onset1 --offset-loss 1.0
# 2) combo_offset1：v3c（offset 1 帧）+ 1 帧 onset + offset 损失
run_train combo_offset1 "$V3C" "$STD_STEPS" 1 --label-channel onset1 --offset-loss 1.0
# 3) combo_aug：combo_base + 频率/时间遮蔽
run_train combo_aug "$V3B" "$STD_STEPS" 1 --label-channel onset1 --offset-loss 1.0 \
  --aug-freq-mask 2 --aug-time-mask 2
# 4) combo_v3a_onset1：v3a + 1 帧 onset，不加 offset 头损失（对照组）
run_train combo_v3a_onset1 "$V3A" "$STD_STEPS" 0 --label-channel onset1
# 5) combo_seed1：combo_base 换随机种子（方差检查）
run_train combo_seed1 "$V3B" "$STD_STEPS" 1 --label-channel onset1 --offset-loss 1.0 --seed 1
# 6) combo_base_40ep：combo_base 跑 40 轮（训练量曲线）
run_train combo_base_40ep "$V3B" "$LONG_STEPS" 1 --label-channel onset1 --offset-loss 1.0

{ echo "finished $(date '+%F %T')"; echo "all_six_done"; } > "$QUEUE_DIR/ROUND2_DONE.txt"
log "===== Round 2 结束（等待分析）====="
