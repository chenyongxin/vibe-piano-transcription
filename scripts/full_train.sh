#!/usr/bin/env bash
# piano2 全量训练（71,160 步）启动/续训脚本。
#
# 用法（容器内）：
#   bash scripts/full_train.sh            # 全新开始
#   RESUME=1 bash scripts/full_train.sh   # 从 latest.pt 续训（关机后第二天用）
#
# 环境变量：
#   RUN_NAME    实验目录名，默认 m3_full_4060
#   MAX_STEPS   总步数，默认 71160（20 轮 × 3558 步/轮）
#   RESUME      1 = 续训，其他 = 新开
set -euo pipefail

cd /Docker/git/piano2

RUN="${RUN_NAME:-m3_full_4060}"
MAX_STEPS="${MAX_STEPS:-71160}"
LOG="outputs/M1/${RUN}_train.log"
mkdir -p outputs/M1

EXTRA=""
if [ "${RESUME:-0}" = "1" ]; then
  EXTRA="--resume auto"
  echo "[resume] 将从 outputs/M1/${RUN}/checkpoints/latest.pt 继续" | tee -a "$LOG"
else
  echo "[fresh] 全新训练 ${MAX_STEPS} 步 → outputs/M1/${RUN}" | tee -a "$LOG"
fi

python -m piano2.train \
  --max-steps "$MAX_STEPS" \
  --batch 16 \
  --val-every 500 \
  --save-every 200 \
  --val-segments 120 \
  --num-workers 4 \
  --run-name "$RUN" \
  $EXTRA 2>&1 | tee -a "$LOG"
