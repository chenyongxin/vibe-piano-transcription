"""数据 v4：在 v3b（7 通道）基础上增加两个回归目标通道。

新增通道：
    onset_reg  (索引 7)：每帧到最近 onset 的时间差的三角目标，g = 1 - |d|/J
    offset_reg (索引 8)：每帧到最近 offset 的时间差的三角目标

J = 5 帧，值域 0–1，用 uint8 存 0–255（dataset 读入时除以 255）。
CQT / pedal 继续硬链接 v3a，segments.json 复制 v3b。

用法（容器内）：
    cd /Docker/git/piano2 && python scripts/build_labels_v4.py --workers 8
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
J = 5
V3B = "/Docker/piano2_data/processed/maestro_processed_v3b"
V3A = "/Docker/piano2_data/processed/maestro_processed_v3a"
V4 = "/Docker/piano2_data/processed/maestro_processed_v4"
MAESTRO = "/Docker/piano2_data/raw/maestro-v3.0.0"


def link_or_copy(src, dst):
    """同卷硬链接复用大文件，失败则退化为复制。"""
    if os.path.exists(dst):
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copyfile(src, dst)


def add_regression_targets(lab, n_frames, midi_path):
    """在 lab 的第 7/8 通道写入 onset/offset 回归目标。

    参数:
        lab (numpy.ndarray): (T, 9, 88) uint8，前 7 通道已填好。
        n_frames (int): 帧数。
        midi_path (str): 对齐 MIDI 路径。

    返回:
        numpy.ndarray: 填好第 7/8 通道的 lab。
    """
    pm = pretty_midi.PrettyMIDI(midi_path)
    for inst in pm.instruments:
        for note in inst.notes:
            pitch = int(note.pitch) - 21
            if not 0 <= pitch < 88:
                continue
            for event_frame, channel in (
                (int(round(note.start * FPS)), 7),
                (int(round(note.end * FPS)), 8),
            ):
                for d in range(-J, J + 1):
                    frame = event_frame + d
                    if 0 <= frame < n_frames:
                        target = int(round((1.0 - abs(d) / J) * 255))
                        lab[frame, channel, pitch] = max(lab[frame, channel, pitch], target)
    return lab


def process_one(args):
    """处理一首曲目：硬链接 CQT/pedal，复制 v3b 标签并补回归通道。"""
    split, idx, stem, midi_path = args
    base = f"{idx:04d}_{stem}"
    b_lab = os.path.join(V3B, split, base + ".lab.npy")
    v4_lab = os.path.join(V4, split, base + ".lab.npy")
    if os.path.exists(v4_lab):
        return {"status": "skipped"}
    if not os.path.exists(b_lab):
        return {"status": "missing_v3b"}
    lab_b = np.load(b_lab, mmap_mode="r")
    n_frames = int(lab_b.shape[0])
    lab = np.zeros((n_frames, 9, 88), dtype=np.uint8)
    lab[:, :7, :] = np.asarray(lab_b[:, :7, :])
    lab = add_regression_targets(lab, n_frames, midi_path)
    np.save(v4_lab, lab)
    link_or_copy(os.path.join(V3A, split, base + ".cqt.npy"),
                 os.path.join(V4, split, base + ".cqt.npy"))
    link_or_copy(os.path.join(V3A, split, base + ".pedal.npy"),
                 os.path.join(V4, split, base + ".pedal.npy"))
    return {"status": "ok", "n_frames": n_frames}


def main():
    import argparse

    ap = argparse.ArgumentParser(description="构建数据 v4（回归目标通道）")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    df = pd.read_csv(os.path.join(MAESTRO, "maestro-v3.0.0.csv"))
    tasks = []
    for idx, row in df.iterrows():
        split = str(row["split"])
        os.makedirs(os.path.join(V4, split), exist_ok=True)
        stem = os.path.basename(str(row["audio_filename"]))[:-4]
        tasks.append((split, idx, stem, os.path.join(MAESTRO, str(row["midi_filename"]))))

    t0 = time.time()
    counts = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, res in enumerate(pool.map(process_one, tasks, chunksize=8), 1):
            counts[res["status"]] = counts.get(res["status"], 0) + 1
            if i % 300 == 0:
                print(f"  {i}/{len(tasks)}  {counts}", flush=True)
    print(f"回归标签完成，用时 {time.time() - t0:.0f}s，状态 {counts}")

    for split in ("train", "validation", "test"):
        src = os.path.join(V3B, split, "segments.json")
        dst = os.path.join(V4, split, "segments.json")
        if not os.path.exists(dst):
            shutil.copyfile(src, dst)

    d = os.path.join(V4, "train")
    names = sorted(f for f in os.listdir(d) if f.endswith(".lab.npy"))
    rates = {"onset_reg": [], "offset_reg": []}
    for name in names[:: max(1, len(names) // 6)][:6]:
        lab = np.load(os.path.join(d, name), mmap_mode="r")
        rates["onset_reg"].append(float(np.asarray(lab[:, 7, :]).mean()) / 255.0)
        rates["offset_reg"].append(float(np.asarray(lab[:, 8, :]).mean()) / 255.0)
    manifest = {
        "version": "data_v4",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "note": "v3b + onset_reg/offset_reg 三角回归目标（J=5，uint8 0–255）",
        "channels": ["frame_key", "onset1", "onset2", "onset3", "velocity",
                     "pedal", "offset", "onset_reg", "offset_reg"],
        "J": J,
        "mean_onset_reg": float(np.mean(rates["onset_reg"])),
        "mean_offset_reg": float(np.mean(rates["offset_reg"])),
    }
    with open(os.path.join(V4, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)
    print("写入 manifest:", os.path.join(V4, "manifest.json"))
    print("回归目标均值: onset %.3f / offset %.3f"
          % (manifest["mean_onset_reg"], manifest["mean_offset_reg"]))


if __name__ == "__main__":
    main()
