"""piano2 的运行配置：所有路径都可以用环境变量覆盖。

默认值与开发机（容器里挂载 /Docker）一致；换机器时设环境变量即可，不用改代码：
    PIANO2_SRC_ROOT       旧版压缩 npz 的根目录
    PIANO2_DATA_ROOT      数据 v2（memmap npy）根目录
    PIANO2_MAESTRO_ROOT   MAESTRO 原始数据根目录
    PIANO2_OUTPUT_ROOT    输出根目录（日志、检查点、评测结果）
    PIANO2_CKPT           默认模型检查点
"""

from __future__ import annotations

import os
from pathlib import Path

# ── 音频与特征参数 ──
SR = 16000
HOP = 512
FPS = SR / HOP          # 31.25 帧/秒，1 帧 = 32 ms
N_BINS = 88 * 3
BINS_PER_OCTAVE = 36
FMIN = 27.5             # A0 的频率（Hz）
SEG_FRAMES = 313        # 10 秒一段
FRAME_MS = 1000.0 / FPS

# ── 路径（均可用环境变量覆盖） ──
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = Path(os.environ.get("PIANO2_SRC_ROOT", "/Docker/piano2_data/processed/maestro_processed"))
DATA_ROOT = Path(os.environ.get("PIANO2_DATA_ROOT", "/Docker/piano2_data/processed/maestro_processed_v2"))
MAESTRO_ROOT = Path(os.environ.get("PIANO2_MAESTRO_ROOT", "/Docker/piano2_data/raw/maestro-v3.0.0"))
OUTPUT_ROOT = Path(os.environ.get("PIANO2_OUTPUT_ROOT", str(REPO_ROOT / "outputs")))
CKPT = Path(os.environ.get("PIANO2_CKPT", str(REPO_ROOT / "models" / "best.pt")))

# ── 解码默认参数（M0.5 步骤 3 选定） ──
DECODE_DEFAULTS = {
    "method": "peak_frame",
    "harmonic_mode": "conditional",
    "harmonic_strength": 0.5,
    "harmonic_window_frames": 5,
    "harmonic_frame_max": 0.6,
    "frame_gate": 0.15,
    "onset_threshold": 0.6,
    "onset_prominence": 0.005,
    "peak_distance": 5,
    "frame_off_threshold": 0.18,
    "release_frames": 3,
    # offset 结束时间：只在起点后 20 帧内找第一个高置信度 offset 峰。
    # 旧的 160 帧窗口会把后面的释放峰误配给当前音，note F1 明显偏低。
    "offset_threshold": 0.7,
    "offset_max_frames": 20,
}

# ── 固定评测集（MAESTRO test；与 M0/M0.5 报告一致） ──
EVAL_TRACKS = {
    "haydn": "2004/MIDI-Unprocessed_XP_19_R1_2004_01-02_ORIG_MID--AUDIO_19_R1_2004_02_Track02_wav",
    "mozart": "2008/MIDI-Unprocessed_08_R1_2008_01-05_ORIG_MID--AUDIO_08_R1_2008_wav--2",
    "beethoven": "2014/MIDI-UNPROCESSED_14-15_R1_2014_MID--AUDIO_15_R1_2014_wav--2",
}
