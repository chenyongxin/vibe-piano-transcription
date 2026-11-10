"""数据 v3c：在 v3a 基础上增加 offset 通道，但窗口改为 1 帧（只标 note-off 那一帧）。

背景：v3b 的 offset 标签是 2 帧窗口 [end, end+2)，正样本中心比真实键释放晚半帧（16 ms）。
这与 onset 的 2 帧标签是同一个问题；v3c 把 offset 改为 1 帧，验证能否进一步改善时值。

目录结构（/Docker/piano2_data/processed/maestro_processed_v3c）：
    {split}/{name}.cqt.npy   ← 硬链接 v3a（最终指向 v2 同一份数据）
    {split}/{name}.lab.npy   ← 7 通道：v3a 的 6 通道 + offset（1 帧）
    {split}/{name}.pedal.npy ← 硬链接 v3a
    {split}/segments.json    ← 复制 v3a
    manifest.json

用法（容器内）：
    cd /Docker/git/piano2 && python scripts/build_labels_v3c.py --workers 8
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
V3A = "/Docker/piano2_data/processed/maestro_processed_v3a"
V3C = "/Docker/piano2_data/processed/maestro_processed_v3c"
MAESTRO = "/Docker/piano2_data/raw/maestro-v3.0.0"


def link_or_copy(src, dst):
    """同卷硬链接复用大文件，失败则退化为复制。"""
    if os.path.exists(dst):
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copyfile(src, dst)


def offset_channel(pm, n_frames):
    """生成 offset 通道：只在 note-off 那一帧标 1（1 帧窗口）。

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
            if 0 <= end < n_frames:
                offset[end, pitch] = 1
    return offset


def process_one(args):
    """处理一首曲目：硬链接 CQT/pedal，重建 7 通道 lab（offset 1 帧）。"""
    split, idx, stem, midi_path = args
    base = f"{idx:04d}_{stem}"
    a_lab_path = os.path.join(V3A, split, base + ".lab.npy")
    c_lab_path = os.path.join(V3C, split, base + ".lab.npy")
    if os.path.exists(c_lab_path):
        return {"status": "skipped"}
    if not os.path.exists(a_lab_path):
        return {"status": "missing_v3a"}
    lab_a = np.load(a_lab_path, mmap_mode="r")
    n_frames = int(lab_a.shape[0])
    lab = np.zeros((n_frames, 7, 88), dtype=np.uint8)
    lab[:, :6, :] = np.asarray(lab_a[:, :6, :])
    lab[:, 6, :] = offset_channel(pretty_midi.PrettyMIDI(midi_path), n_frames)
    np.save(c_lab_path, lab)
    link_or_copy(os.path.join(V3A, split, base + ".cqt.npy"),
                 os.path.join(V3C, split, base + ".cqt.npy"))
    link_or_copy(os.path.join(V3A, split, base + ".pedal.npy"),
                 os.path.join(V3C, split, base + ".pedal.npy"))
    return {"status": "ok", "n_frames": n_frames}


def main():
    import argparse

    ap = argparse.ArgumentParser(description="构建数据 v3c（offset 1 帧）")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    df = pd.read_csv(os.path.join(MAESTRO, "maestro-v3.0.0.csv"))
    tasks = []
    for idx, row in df.iterrows():
        split = str(row["split"])
        os.makedirs(os.path.join(V3C, split), exist_ok=True)
        stem = os.path.basename(str(row["audio_filename"]))[:-4]
        tasks.append((split, idx, stem, os.path.join(MAESTRO, str(row["midi_filename"]))))

    t0 = time.time()
    counts = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, res in enumerate(pool.map(process_one, tasks, chunksize=8), 1):
            counts[res["status"]] = counts.get(res["status"], 0) + 1
            if i % 300 == 0:
                print(f"  {i}/{len(tasks)}  {counts}", flush=True)
    print(f"offset(1 帧) 标签完成，用时 {time.time() - t0:.0f}s，状态 {counts}")

    for split in ("train", "validation", "test"):
        src = os.path.join(V3A, split, "segments.json")
        dst = os.path.join(V3C, split, "segments.json")
        if not os.path.exists(dst):
            shutil.copyfile(src, dst)

    d = os.path.join(V3C, "train")
    names = sorted(f for f in os.listdir(d) if f.endswith(".lab.npy"))
    rates = []
    for name in names[:: max(1, len(names) // 6)][:6]:
        lab = np.load(os.path.join(d, name), mmap_mode="r")
        rates.append(float(np.asarray(lab[:, 6, :]).mean()))
    manifest = {
        "version": "data_v3c",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "note": "在 v3a 基础上增加 offset 通道（只标 note-off 那一帧）；CQT/pedal 硬链接",
        "channels": ["frame_key", "onset1", "onset2", "onset3", "velocity", "pedal", "offset"],
        "offset_positive_rate": float(np.mean(rates)),
    }
    with open(os.path.join(V3C, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)
    print("写入 manifest:", os.path.join(V3C, "manifest.json"))
    print("抽样 offset 正样本率: %.4f（v3b 的 2 帧窗口约 0.0069）"
          % manifest["offset_positive_rate"])


if __name__ == "__main__":
    main()
