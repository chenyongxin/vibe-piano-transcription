"""数据 v5：v4 的精确事件时间回归目标。

v4 的回归目标用 `int(round(note.start × FPS))` 作为中心，三角峰落在最近整数帧，
没有编码亚帧相位。v5 保留 v4 的 7 个分类通道，重新计算 onset_reg / offset_reg：

    g(frame) = 1 - |frame/FPS - event_time| / (J × 32 ms)
    target = round(g × 255)

事件时间用 MIDI 的精确秒数，不取整。

用法（容器内）：
    cd /Docker/git/piano2 && python scripts/build_labels_v5.py --workers 8
"""

from __future__ import annotations

import concurrent.futures
import json
import os
import shutil
import time

import numpy as np
import pandas as pd
import pretty_midi

FPS = 31.25
FRAME_S = 1.0 / FPS
J = 5
V4 = "/Docker/piano2_data/processed/maestro_processed_v4"
V5 = "/Docker/piano2_data/processed/maestro_processed_v5"
MAESTRO = "/Docker/piano2_data/raw/maestro-v3.0.0"


def link_or_copy(src, dst):
    """同卷硬链接复用大文件，失败则复制。"""
    if os.path.exists(dst):
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copyfile(src, dst)


def exact_regression_targets(lab, n_frames, midi_path):
    """用精确事件时间写 onset_reg / offset_reg。

    参数:
        lab (numpy.ndarray): (T, 9, 88) uint8，前 7 通道已从 v4 复制。
        n_frames (int): 帧数。
        midi_path (str): MAESTRO MIDI 路径。

    返回:
        numpy.ndarray: 填好第 7/8 通道的 lab。
    """
    pm = pretty_midi.PrettyMIDI(midi_path)
    for inst in pm.instruments:
        for note in inst.notes:
            pitch = int(note.pitch) - 21
            if not 0 <= pitch < 88:
                continue
            for event_time, channel in ((float(note.start), 7), (float(note.end), 8)):
                center = event_time * FPS
                lo = max(0, int(np.floor(center)) - J - 1)
                hi = min(n_frames - 1, int(np.ceil(center)) + J + 1)
                for frame in range(lo, hi + 1):
                    distance_frames = abs(frame - center)
                    g = 1.0 - distance_frames / J
                    if g <= 0:
                        continue
                    target = int(round(g * 255))
                    if target > lab[frame, channel, pitch]:
                        lab[frame, channel, pitch] = target
    return lab


def process_one(args):
    """处理一首曲目：复制 v4 分类通道，重算精确回归通道，链接 CQT/pedal。"""
    split, idx, stem, midi_path = args
    base = f"{idx:04d}_{stem}"
    v4_lab = os.path.join(V4, split, base + ".lab.npy")
    v5_lab = os.path.join(V5, split, base + ".lab.npy")
    if os.path.exists(v5_lab):
        return {"status": "skipped"}
    if not os.path.exists(v4_lab):
        return {"status": "missing_v4"}
    lab4 = np.load(v4_lab, mmap_mode="r")
    n_frames = int(lab4.shape[0])
    lab = np.zeros((n_frames, 9, 88), dtype=np.uint8)
    lab[:, :7, :] = np.asarray(lab4[:, :7, :])
    exact_regression_targets(lab, n_frames, midi_path)
    np.save(v5_lab, lab)
    link_or_copy(os.path.join(V4, split, base + ".cqt.npy"),
                 os.path.join(V5, split, base + ".cqt.npy"))
    link_or_copy(os.path.join(V4, split, base + ".pedal.npy"),
                 os.path.join(V5, split, base + ".pedal.npy"))
    return {"status": "ok", "n_frames": n_frames}


def main():
    import argparse

    ap = argparse.ArgumentParser(description="构建数据 v5（精确时间回归目标）")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    df = pd.read_csv(os.path.join(MAESTRO, "maestro-v3.0.0.csv"))
    tasks = []
    for idx, row in df.iterrows():
        split = str(row["split"])
        os.makedirs(os.path.join(V5, split), exist_ok=True)
        stem = os.path.basename(str(row["audio_filename"]))[:-4]
        tasks.append((split, idx, stem, os.path.join(MAESTRO, str(row["midi_filename"]))))

    t0 = time.time()
    counts = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, res in enumerate(pool.map(process_one, tasks, chunksize=8), 1):
            counts[res["status"]] = counts.get(res["status"], 0) + 1
            if i % 300 == 0:
                print(f"  {i}/{len(tasks)}  {counts}", flush=True)
    print(f"v5 标签完成，用时 {time.time() - t0:.0f}s，状态 {counts}")

    for split in ("train", "validation", "test"):
        src = os.path.join(V4, split, "segments.json")
        dst = os.path.join(V5, split, "segments.json")
        if not os.path.exists(dst):
            shutil.copyfile(src, dst)

    # 抽样统计，确认相位信息不再取整
    d = os.path.join(V5, "train")
    names = sorted(f for f in os.listdir(d) if f.endswith(".lab.npy"))
    rates = {"onset_reg": [], "offset_reg": []}
    for name in names[:: max(1, len(names) // 6)][:6]:
        lab = np.load(os.path.join(d, name), mmap_mode="r")
        rates["onset_reg"].append(float(np.asarray(lab[:, 7, :]).mean()) / 255.0)
        rates["offset_reg"].append(float(np.asarray(lab[:, 8, :]).mean()) / 255.0)
    manifest = {
        "version": "data_v5",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "note": "v4 分类通道 + 用精确事件时间计算的 onset_reg/offset_reg 三角目标（J=5）",
        "channels": ["frame_key", "onset1", "onset2", "onset3", "velocity",
                     "pedal", "offset", "onset_reg", "offset_reg"],
        "J": J,
        "exact_event_time": True,
        "mean_onset_reg": float(np.mean(rates["onset_reg"])),
        "mean_offset_reg": float(np.mean(rates["offset_reg"])),
    }
    with open(os.path.join(V5, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)
    print("写入 manifest:", os.path.join(V5, "manifest.json"))
    print("回归目标均值: onset %.3f / offset %.3f"
          % (manifest["mean_onset_reg"], manifest["mean_offset_reg"]))


if __name__ == "__main__":
    main()
