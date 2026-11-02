"""M1：构建数据 v2 —— memmap 友好的 CQT + 标签 v2 + 段索引 + manifest。

为什么要重建数据（对照 M0 审计结果）：
    1. onset 标签从 3 帧（[start-1, start+2)）改成 2 帧（[start, start+2)）；
       同时保留 1 帧、3 帧两个变体，供 M1 做对照实验。
    2. frame 标签保持「含踏板延音」（方案 A），并额外生成踏板标签，供新增的踏板头学习。
    3. 把 CQT 从压缩 npz 改成 .npy（这样训练时可以用 memmap 随机切片，
       不用每步解压整首歌——旧管线 9 倍的 I/O 浪费就来自这里）。
    4. 预生成 10 秒段索引；验证集固定段；生成 manifest（清单/指纹）。

输出目录结构（容器内 /Docker/maestro_processed_v2）：
    {split}/{idx:04d}_{stem}.cqt.npy      (T, 264) float32
    {split}/{idx:04d}_{stem}.lab.npy      (T, 6, 88) uint8
        通道顺序：frame, onset1, onset2, onset3, velocity(0-127), pedal(0/1 广播到 88 键)
    {split}/{idx:04d}_{stem}.pedal.npy    (T, 2) uint8  [pedal, pedal_on]
    {split}/segments.json                 段索引：[{file, start, n}]
    manifest.json                         配置、统计、指纹

用法（容器内）：
    cd /Docker/git/piano_practice/m1 && \
    python build_data_v2.py --workers 8
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import time

import numpy as np
import pandas as pd
import pretty_midi

SR = 16000
HOP = 512
FPS = SR / HOP  # 31.25 帧/秒
SEG_FRAMES = 313  # 10 秒
CHANNELS = ("frame", "onset1", "onset2", "onset3", "velocity", "pedal")

MAESTRO = "/Docker/maestro-v3.0.0"
SRC = "/Docker/maestro_processed"
DST = "/Docker/maestro_processed_v2"
OUT = "/Docker/git/piano_practice/outputs/M1"


def sha256_text(text):
    """对字符串做 sha256（用于给配置和脚本留指纹）。"""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path, limit=None):
    """对文件做 sha256；limit 给定时只读前 limit 字节（用于快速指纹）。"""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        remaining = limit
        while True:
            n = 1 << 20 if remaining is None else min(1 << 20, remaining)
            if n <= 0:
                break
            data = fh.read(n)
            if not data:
                break
            h.update(data)
            if remaining is not None:
                remaining -= len(data)
    return h.hexdigest()


def pad_or_trim(arr, n_frames):
    """把数组第一维对齐到 n_frames：不足补零，超出截断。"""
    n = arr.shape[0]
    if n == n_frames:
        return arr
    if n > n_frames:
        return arr[:n_frames]
    pad = np.zeros((n_frames - n,) + arr.shape[1:], dtype=arr.dtype)
    return np.concatenate([arr, pad], axis=0)


def pedal_arrays(pm, n_frames):
    """从 MIDI 的 CC64 生成逐帧踏板状态与踩下时刻。

    参数:
        pm (pretty_midi.PrettyMIDI): 已解析的 MIDI。
        n_frames (int): 目标帧数。

    返回:
        tuple[numpy.ndarray, numpy.ndarray]: (pedal, pedal_on)，均为 (n_frames,) uint8。
    """
    events = []
    for inst in pm.instruments:
        for cc in inst.control_changes:
            if cc.number == 64:
                events.append((float(cc.time), 1 if cc.value >= 64 else 0))
    events.sort()
    pedal = np.zeros(n_frames, dtype=np.uint8)
    pedal_on = np.zeros(n_frames, dtype=np.uint8)
    if not events:
        return pedal, pedal_on
    frames = np.clip(np.round(np.array([e[0] for e in events]) * FPS).astype(int), 0, n_frames - 1)
    values = np.array([e[1] for e in events], dtype=np.uint8)
    for i, (f, v) in enumerate(zip(frames, values)):
        end = int(frames[i + 1]) if i + 1 < len(frames) else n_frames
        pedal[f:end] = v
        if v == 1 and (i == 0 or values[i - 1] == 0):
            pedal_on[f] = 1
    return pedal, pedal_on


def build_labels(pm, n_frames):
    """由 MIDI 生成标签 v2 的 6 个通道。

    参数:
        pm (pretty_midi.PrettyMIDI): 已解析的 MIDI。
        n_frames (int): 目标帧数。

    返回:
        numpy.ndarray: (n_frames, 6, 88) uint8，通道顺序见模块文档。
    """
    lab = np.zeros((n_frames, len(CHANNELS), 88), dtype=np.uint8)
    # frame：含踏板的逐帧发声（pretty_midi 默认应用延音踏板）
    frame_roll = pm.get_piano_roll(fs=FPS)[21:109, :].T
    frame_roll = (frame_roll > 0).astype(np.uint8)
    lab[:, 0, :] = pad_or_trim(frame_roll, n_frames)

    for inst in pm.instruments:
        for note in inst.notes:
            pitch = int(note.pitch) - 21
            if not (0 <= pitch < 88):
                continue
            start = int(round(note.start * FPS))
            if start < 0:
                continue
            velocity = int(np.clip(note.velocity, 1, 127))
            if start < n_frames:
                lab[start, 1, pitch] = 1  # 1 帧变体
            for f in (start, start + 1):
                if 0 <= f < n_frames:
                    lab[f, 2, pitch] = 1  # 2 帧变体（默认）
                    lab[f, 4, pitch] = velocity  # 力度与 onset2 窗口对齐（被 mask 选中）
            for f in (start - 1, start, start + 1):
                if 0 <= f < n_frames:
                    lab[f, 3, pitch] = 1  # 旧 3 帧变体（对照用）
    return lab


def process_one(args):
    """处理一首曲目：读源 npz、写 CQT memmap、生成标签与踏板。"""
    split, idx, audio_stem, midi_path = args
    src_npz = os.path.join(SRC, split, f"{idx:04d}_{audio_stem}.npz")
    if not os.path.exists(src_npz):
        return {"split": split, "idx": idx, "status": "missing_src"}
    base = os.path.join(DST, split, f"{idx:04d}_{audio_stem}")
    cqt_path = base + ".cqt.npy"
    lab_path = base + ".lab.npy"
    pedal_path = base + ".pedal.npy"
    if os.path.exists(cqt_path) and os.path.exists(lab_path) and os.path.exists(pedal_path):
        return {"split": split, "idx": idx, "status": "skipped"}

    data = np.load(src_npz)
    cqt = data["cqt"].astype(np.float32)
    n_frames = cqt.shape[0]
    pm = pretty_midi.PrettyMIDI(midi_path)
    lab = build_labels(pm, n_frames)
    pedal, pedal_on = pedal_arrays(pm, n_frames)
    lab[:, 5, :] = pedal[:, None]  # 踏板状态广播到 88 键，方便和 frame 头对齐

    np.save(cqt_path, cqt)
    np.save(lab_path, lab)
    np.save(pedal_path, np.stack([pedal, pedal_on], axis=1))
    return {"split": split, "idx": idx, "status": "ok", "n_frames": int(n_frames)}


def main():
    ap = argparse.ArgumentParser(description="构建数据 v2（memmap CQT + 标签 v2 + 段索引）")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 首（调试用）")
    args = ap.parse_args()

    os.makedirs(DST, exist_ok=True)
    os.makedirs(OUT, exist_ok=True)
    df = pd.read_csv(os.path.join(MAESTRO, "maestro-v3.0.0.csv"))
    tasks = []
    for idx, row in df.iterrows():
        if args.limit and len(tasks) >= args.limit:
            break
        split = str(row["split"])
        stem = os.path.basename(str(row["audio_filename"]))[:-4]
        midi_path = os.path.join(MAESTRO, str(row["midi_filename"]))
        os.makedirs(os.path.join(DST, split), exist_ok=True)
        tasks.append((split, idx, stem, midi_path))

    t0 = time.time()
    counts = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(process_one, t) for t in tasks]
        for i, fut in enumerate(concurrent.futures.as_completed(futures), 1):
            res = fut.result()
            key = (res["split"], res["status"])
            counts[key] = counts.get(key, 0) + 1
            if i % 100 == 0 or i == len(tasks):
                print(f"  {i}/{len(tasks)} 完成，用时 {time.time() - t0:.0f}s，状态 {counts}", flush=True)

    # 段索引（10 秒一段，不重叠）
    segments_meta = {}
    for split in ("train", "validation", "test"):
        segs = []
        files = sorted(f for f in os.listdir(os.path.join(DST, split)) if f.endswith(".cqt.npy"))
        for fname in files:
            n = int(np.load(os.path.join(DST, split, fname), mmap_mode="r").shape[0])
            n_seg = max(n // SEG_FRAMES, 1)
            for k in range(n_seg):
                start = k * SEG_FRAMES
                segs.append({"file": fname, "start": start,
                             "n": int(min(SEG_FRAMES, n - start))})
        with open(os.path.join(DST, split, "segments.json"), "w", encoding="utf-8") as fh:
            json.dump(segs, fh)
        segments_meta[split] = {"n_files": len(files), "n_segments": len(segs)}

    # 标签统计（抽样 10 个文件）
    sample_files = []
    for split in ("train", "validation"):
        d = os.path.join(DST, split)
        names = sorted(f for f in os.listdir(d) if f.endswith(".lab.npy"))
        sample_files.extend(os.path.join(d, names[i]) for i in range(0, len(names), max(1, len(names) // 5))[:5])
    rates = {name: [] for name in CHANNELS}
    for path in sample_files:
        lab = np.load(path, mmap_mode="r")
        for ci, name in enumerate(CHANNELS):
            rates[name].append(float(np.asarray(lab[:, ci, :]).mean()))
    stats = {name: float(np.mean(v)) for name, v in rates.items() if v}

    manifest = {
        "version": "data_v2",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "sr": SR, "hop": HOP, "fps": FPS, "seg_frames": SEG_FRAMES,
            "seg_seconds": SEG_FRAMES / FPS,
            "channels": list(CHANNELS),
            "onset_default": "onset2 ([start, start+2))",
            "frame_contains_pedal": True,
            "input_norm": "(x + 80) / 80，把 dB 范围 [-80,0] 映射到 [0,1]",
        },
        "splits": segments_meta,
        "label_positive_rates": stats,
        "build": {
            "script_sha256": sha256_file(os.path.abspath(__file__)),
            "source_root": SRC, "dest_root": DST,
            "counts": {f"{k[0]}_{k[1]}": v for k, v in counts.items()},
        },
        "fingerprint": sha256_text(json.dumps(segments_meta, sort_keys=True)),
    }
    path = os.path.join(OUT, "data_manifest.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)
    print("段索引与 manifest 写入完成:", path)
    print("标签正样本率:", stats)
    print("总用时:", round(time.time() - t0, 1), "s")


if __name__ == "__main__":
    main()
