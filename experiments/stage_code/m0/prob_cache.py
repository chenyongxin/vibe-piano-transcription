"""M0.5：把模型概率缓存到磁盘，调参时只做解码，不再重复推理。

缓存内容：每首曲目的 onset / frame / velocity 三个概率矩阵 + 音频时长。
形状约定：(时间帧 T, 88)。T 由 CQT 帧数决定，SR/HOP = 16000/512。

用法（容器内）：
    cd /Docker/git/piano_practice/m0 && \
    PYTHONPATH=/Docker/git/piano_practice/m0:/Docker/git/piano_practice \
    python prob_cache.py
"""

from __future__ import annotations

import os
import sys

import librosa
import numpy as np
import torch

ROOT = "/Docker/git/piano_practice"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "m0"))

from transcribe_readable import compute_cqt, infer_sliding_window, load_model  # noqa: E402

MAESTRO = "/Docker/maestro-v3.0.0"
CKPT = os.path.join(ROOT, "trained_data", "checkpoints", "model_best.pth")
CACHE_DIR = os.path.join(ROOT, "outputs", "M0", "cache")

# M0 里用密度 10/50/90 分位选出来的三首 GT 曲目，外加用户自己的 nezha2
TRACKS = {
    "haydn": os.path.join(MAESTRO, "2004/MIDI-Unprocessed_XP_19_R1_2004_01-02_ORIG_MID--AUDIO_19_R1_2004_02_Track02_wav.wav"),
    "mozart": os.path.join(MAESTRO, "2008/MIDI-Unprocessed_08_R1_2008_01-05_ORIG_MID--AUDIO_08_R1_2008_wav--2.wav"),
    "beethoven": os.path.join(MAESTRO, "2014/MIDI-UNPROCESSED_14-15_R1_2014_MID--AUDIO_15_R1_2014_wav--2.wav"),
    "nezha2": os.path.join(ROOT, "data", "nezha2.wav"),
}

GT_MIDI = {
    "haydn": os.path.join(MAESTRO, "2004/MIDI-Unprocessed_XP_19_R1_2004_01-02_ORIG_MID--AUDIO_19_R1_2004_02_Track02_wav.midi"),
    "mozart": os.path.join(MAESTRO, "2008/MIDI-Unprocessed_08_R1_2008_01-05_ORIG_MID--AUDIO_08_R1_2008_wav--2.midi"),
    "beethoven": os.path.join(MAESTRO, "2014/MIDI-UNPROCESSED_14-15_R1_2014_MID--AUDIO_15_R1_2014_wav--2.midi"),
}


def cache_path(name):
    """返回某个曲目的缓存文件路径。

    参数:
        name (str): 曲目短名（haydn / mozart / beethoven / nezha2）。

    返回:
        str: 形如 outputs/M0/cache/nezha2.npz 的路径。
    """
    return os.path.join(CACHE_DIR, f"{name}.npz")


def compute_and_cache(name, wav_path, device="cuda"):
    """对一首曲目做 CQT + 模型推理，并把概率矩阵写盘。

    参数:
        name (str): 曲目短名。
        wav_path (str): 音频路径。
        device (str): "cuda" 或 "cpu"。

    返回:
        dict: {"onset": (T,88), "frame": (T,88), "velocity": (T,88), "duration_s": float}
    """
    model = load_model(CKPT, device)
    cqt = compute_cqt(wav_path)
    x = torch.tensor(cqt, dtype=torch.float32).unsqueeze(0).to(device)
    onset, frame, vel = infer_sliding_window(model, x)
    duration = float(librosa.get_duration(path=wav_path))
    payload = {
        "onset": onset.astype(np.float32),
        "frame": frame.astype(np.float32),
        "velocity": vel.astype(np.float32),
        "duration_s": np.float32(duration),
    }
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez_compressed(cache_path(name), **payload)
    return payload


def load_cache(name):
    """读缓存；不存在时抛 FileNotFoundError，提示先跑 compute_and_cache。

    参数:
        name (str): 曲目短名。

    返回:
        dict: 同 compute_and_cache 的返回值。
    """
    path = cache_path(name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"没有缓存 {path}，请先运行 prob_cache.py")
    data = np.load(path)
    return {
        "onset": data["onset"],
        "frame": data["frame"],
        "velocity": data["velocity"],
        "duration_s": float(data["duration_s"]),
    }


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device = {device}")
    for name, wav in TRACKS.items():
        out = cache_path(name)
        if os.path.exists(out):
            print(f"[skip] {name} 已有缓存: {out}")
            continue
        print(f"[run ] {name}: {wav}")
        payload = compute_and_cache(name, wav, device)
        print(f"       T={payload['onset'].shape[0]}, duration={payload['duration_s']:.1f}s -> {out}")
