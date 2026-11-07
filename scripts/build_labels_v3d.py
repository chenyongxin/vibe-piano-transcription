"""数据 v3d：offset 通道改成 3 帧窗口 [end-1, end+2)，正样本中心正好在 note-off 那一帧。

背景：v3b 的 offset 是 2 帧窗口 [end, end+2)，中心 +16ms（偏晚）；v3c 的 1 帧窗口又太稀疏
（留出集 note F1 0.251 < v3b 的 0.372）。v3d 用「以 end 为中心的 3 帧」，既保留足够正样本，
又消除窗口中心的系统偏差——这是从 onset「窗口中心偏差」直接推出来的对照实验。

目录结构（/Docker/maestro_processed_v3d）：与 v3b 相同，只是 offset 通道为 3 帧。

用法（容器内）：
    cd /Docker/git/piano2 && python scripts/build_labels_v3d.py --workers 8
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
V3A = "/Docker/maestro_processed_v3a"
V3D = "/Docker/maestro_processed_v3d"
MAESTRO = "/Docker/maestro-v3.0.0"


def link_or_copy(src, dst):
    """同卷硬链接复用大文件，失败则退化为复制。"""
    if os.path.exists(dst):
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copyfile(src, dst)


def offset_channel(pm, n_frames):
    """生成 offset 通道：以 note-off 帧为中心的 3 帧窗口 [end-1, end+2)。

    参数:
        pm (pretty_midi.PrettyMIDI): 对齐 MIDI。
        n_frames (int): 帧数。

    返回:
        numpy.ndarray: (n_frames, 88) uint8。
    """
    offset = np.zeros((n_frames, 88), dtype=np.uint8)
    for inst in pm.instruments:
        for note in inst.notes:
            pitch = int(note.pitch) - 21
            if not (0 <= pitch < 88):
                continue
            end = int(round(note.end * FPS))
            for f in (end - 1, end, end + 1):
                if 0 <= f < n_frames:
                    offset[f, pitch] = 1
    return offset


def process_one(args):
    """处理一首曲目：硬链接 CQT/pedal，重建 7 通道 lab（offset 3 帧居中）。"""
    split, idx, stem, midi_path = args
    base = f"{idx:04d}_{stem}"
    a_lab_path = os.path.join(V3A, split, base + ".lab.npy")
    d_lab_path = os.path.join(V3D, split, base + ".lab.npy")
    if os.path.exists(d_lab_path):
        return {"status": "skipped"}
    if not os.path.exists(a_lab_path):
        return {"status": "missing_v3a"}
    lab_a = np.load(a_lab_path, mmap_mode="r")
    n_frames = int(lab_a.shape[0])
    lab = np.zeros((n_frames, 7, 88), dtype=np.uint8)
    lab[:, :6, :] = np.asarray(lab_a[:, :6, :])
    lab[:, 6, :] = offset_channel(pretty_midi.PrettyMIDI(midi_path), n_frames)
    np.save(d_lab_path, lab)
    link_or_copy(os.path.join(V3A, split, base + ".cqt.npy"),
                 os.path.join(V3D, split, base + ".cqt.npy"))
    link_or_copy(os.path.join(V3A, split, base + ".pedal.npy"),
                 os.path.join(V3D, split, base + ".pedal.npy"))
    return {"status": "ok", "n_frames": n_frames}


def main():
    import argparse

    ap = argparse.ArgumentParser(description="构建数据 v3d（offset 3 帧居中）")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    df = pd.read_csv(os.path.join(MAESTRO, "maestro-v3.0.0.csv"))
    tasks = []
    for idx, row in df.iterrows():
        split = str(row["split"])
        os.makedirs(os.path.join(V3D, split), exist_ok=True)
        stem = os.path.basename(str(row["audio_filename"]))[:-4]
        tasks.append((split, idx, stem, os.path.join(MAESTRO, str(row["midi_filename"]))))

    t0 = time.time()
    counts = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, res in enumerate(pool.map(process_one, tasks, chunksize=8), 1):
            counts[res["status"]] = counts.get(res["status"], 0) + 1
            if i % 300 == 0:
                print(f"  {i}/{len(tasks)}  {counts}", flush=True)
    print(f"offset(3 帧) 标签完成，用时 {time.time() - t0:.0f}s，状态 {counts}")

    for split in ("train", "validation", "test"):
        src = os.path.join(V3A, split, "segments.json")
        dst = os.path.join(V3D, split, "segments.json")
        if not os.path.exists(dst):
            shutil.copyfile(src, dst)

    d = os.path.join(V3D, "train")
    names = sorted(f for f in os.listdir(d) if f.endswith(".lab.npy"))
    rates = []
    for name in names[:: max(1, len(names) // 6)][:6]:
        lab = np.load(os.path.join(d, name), mmap_mode="r")
        rates.append(float(np.asarray(lab[:, 6, :]).mean()))
    manifest = {
        "version": "data_v3d",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "note": "offset 通道为 [end-1, end+2) 三帧，中心在 note-off；CQT/pedal 硬链接",
        "channels": ["frame_key", "onset1", "onset2", "onset3", "velocity", "pedal", "offset"],
        "offset_positive_rate": float(np.mean(rates)),
    }
    with open(os.path.join(V3D, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)
    print("写入 manifest:", os.path.join(V3D, "manifest.json"))
    print("抽样 offset 正样本率: %.4f（v3b 2 帧约 0.0069，v3c 1 帧约 0.0034）"
          % manifest["offset_positive_rate"])


if __name__ == "__main__":
    main()
