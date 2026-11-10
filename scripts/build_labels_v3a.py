"""数据 v3a：frame 改为「按键时长」（不含踏板），其余与 v2 完全一致。

目的（M3.5 实验 D2）：验证 note F1 偏低是否主要因为「训练 frame 含踏板、评测用键释放」
口径不一致。CQT 与 pedal 直接从 v2 硬链接复用，只重建标签（约 2 分钟）。

目录结构（/Docker/piano2_data/processed/maestro_processed_v3a）：
    {split}/{name}.cqt.npy   ← 硬链接 v2
    {split}/{name}.lab.npy   ← 重建：通道 0 改为 frame_key（不含踏板），通道 5 复制 v2 的 pedal
    {split}/{name}.pedal.npy ← 硬链接 v2
    {split}/segments.json    ← 复制 v2
    manifest.json

用法（容器内）：
    cd /Docker/git/piano2 && python scripts/build_labels_v3a.py --workers 8
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
V2 = "/Docker/piano2_data/processed/maestro_processed_v2"
V3A = "/Docker/piano2_data/processed/maestro_processed_v3a"
MAESTRO = "/Docker/piano2_data/raw/maestro-v3.0.0"


def link_or_copy(src, dst):
    """同卷用硬链接复用大文件，失败则退化为复制。"""
    if os.path.exists(dst):
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copyfile(src, dst)


def build_labels(pm, n_frames):
    """生成 v3a 标签：frame_key（键按住）+ 与 v2 相同的 onset/velocity。

    参数:
        pm (pretty_midi.PrettyMIDI): 对齐 MIDI。
        n_frames (int): 目标帧数。

    返回:
        numpy.ndarray: (n_frames, 6, 88) uint8；通道 5 由调用方从 v2 复制。
    """
    lab = np.zeros((n_frames, 6, 88), dtype=np.uint8)
    for inst in pm.instruments:
        for note in inst.notes:
            pitch = int(note.pitch) - 21
            if not (0 <= pitch < 88):
                continue
            start = int(round(note.start * FPS))
            end = int(round(note.end * FPS))
            if start >= n_frames:
                continue
            end = max(end, start + 1)
            # 通道 0：frame_key —— 按键按住期间为 1，不含踏板延音
            lab[max(0, start):min(n_frames, end), 0, pitch] = 1
            # 通道 1/2/3：三种 onset 变体（与 v2 相同）
            if start >= 0:
                lab[start, 1, pitch] = 1
            velocity = int(np.clip(note.velocity, 1, 127))
            for f in (start, start + 1):
                if 0 <= f < n_frames:
                    lab[f, 2, pitch] = 1
                    lab[f, 4, pitch] = velocity
            for f in (start - 1, start, start + 1):
                if 0 <= f < n_frames:
                    lab[f, 3, pitch] = 1
    return lab


def process_one(args):
    """处理一首曲目：硬链接 CQT/pedal，重建 lab。"""
    split, idx, stem, midi_path = args
    base = f"{idx:04d}_{stem}"
    v2_lab_path = os.path.join(V2, split, base + ".lab.npy")
    v3a_lab_path = os.path.join(V3A, split, base + ".lab.npy")
    if os.path.exists(v3a_lab_path):
        return {"status": "skipped"}
    if not os.path.exists(v2_lab_path):
        return {"status": "missing_v2"}

    v2_lab = np.load(v2_lab_path, mmap_mode="r")
    n_frames = int(v2_lab.shape[0])
    lab = build_labels(pretty_midi.PrettyMIDI(midi_path), n_frames)
    lab[:, 5, :] = np.asarray(v2_lab[:, 5, :])  # pedal 状态通道沿用 v2
    np.save(v3a_lab_path, lab)

    link_or_copy(os.path.join(V2, split, base + ".cqt.npy"),
                 os.path.join(V3A, split, base + ".cqt.npy"))
    link_or_copy(os.path.join(V2, split, base + ".pedal.npy"),
                 os.path.join(V3A, split, base + ".pedal.npy"))
    return {"status": "ok", "n_frames": n_frames}


def main():
    import argparse

    ap = argparse.ArgumentParser(description="构建数据 v3a（frame_key）")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    df = pd.read_csv(os.path.join(MAESTRO, "maestro-v3.0.0.csv"))
    tasks = []
    for idx, row in df.iterrows():
        split = str(row["split"])
        os.makedirs(os.path.join(V3A, split), exist_ok=True)
        stem = os.path.basename(str(row["audio_filename"]))[:-4]
        tasks.append((split, idx, stem, os.path.join(MAESTRO, str(row["midi_filename"]))))

    t0 = time.time()
    counts = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, res in enumerate(pool.map(process_one, tasks, chunksize=8), 1):
            counts[res["status"]] = counts.get(res["status"], 0) + 1
            if i % 300 == 0:
                print(f"  {i}/{len(tasks)}  {counts}", flush=True)
    print(f"标签完成，用时 {time.time() - t0:.0f}s，状态 {counts}")

    for split in ("train", "validation", "test"):
        src = os.path.join(V2, split, "segments.json")
        dst = os.path.join(V3A, split, "segments.json")
        if not os.path.exists(dst):
            shutil.copyfile(src, dst)

    sample = []
    d = os.path.join(V3A, "train")
    names = sorted(f for f in os.listdir(d) if f.endswith(".lab.npy"))
    for name in names[::max(1, len(names) // 8)][:8]:
        lab = np.load(os.path.join(d, name), mmap_mode="r")
        sample.append(lab.shape[0])
        rates = {c: float(np.asarray(lab[:, i, :]).mean()) for i, c in
                 enumerate(("frame_key", "onset1", "onset2", "onset3", "velocity", "pedal"))}
    manifest = {
        "version": "data_v3a",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "note": "frame 通道改为按键时长（不含踏板）；CQT/pedal 硬链接自 v2",
        "channels": ["frame_key", "onset1", "onset2", "onset3", "velocity", "pedal"],
        "segments_source": "v2（复制 segments.json）",
        "sample_frames": sample,
        "sampled_label_rates": rates,
    }
    with open(os.path.join(V3A, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)
    print("写入 manifest:", os.path.join(V3A, "manifest.json"))
    print("抽样 frame_key 正样本率: %.4f（v2 含踏板 frame 约 0.06）" % rates["frame_key"])


if __name__ == "__main__":
    main()
