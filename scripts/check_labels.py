"""M1：检查数据 v2 的 onset 标签宽度是否已修正为 2 帧。

检查内容：
    1. onset2 通道的连续段长度：应几乎全是 2 帧；
    2. onset3（旧 3 帧变体）应比 onset2 多出「提前一帧」的部分；
    3. 抽样若干文件，统计 frame / pedal 正样本率是否与 manifest 一致。

用法（容器内）：
    cd /Docker/git/piano_practice/m1 && python check_labels.py
"""

from __future__ import annotations

import json
import os

import numpy as np

from piano2.constants import DATA_ROOT, OUTPUT_ROOT  # noqa: E402

MANIFEST = str(OUTPUT_ROOT / "M1" / "data_manifest.json")


def run_lengths(matrix):
    """统计 (T,88) 01 矩阵里每个连续 1 段的长度。"""
    widths = []
    for k in range(matrix.shape[1]):
        col = matrix[:, k]
        if not col.any():
            continue
        idx = np.flatnonzero(np.diff(np.concatenate(([0], col, [0]))) != 0)
        widths.extend(int(b - a) for a, b in zip(idx[::2], idx[1::2]))
    return np.array(widths)


def main():
    manifest = json.load(open(MANIFEST, encoding="utf-8"))
    split_dir = os.path.join(DATA_ROOT, "train")
    names = sorted(f for f in os.listdir(split_dir) if f.endswith(".lab.npy"))[:12]

    widths2, widths3 = [], []
    rates = {"frame": [], "pedal": []}
    pre_roll_extra = 0
    total_onset2 = 0
    for fname in names:
        lab = np.load(os.path.join(split_dir, fname))  # (T, 6, 88)
        widths2.append(run_lengths(lab[:, 2, :]))
        widths3.append(run_lengths(lab[:, 3, :]))
        rates["frame"].append(float(lab[:, 0, :].mean()))
        rates["pedal"].append(float(lab[:, 5, :].mean()))
        # 统计「onset3 有、onset2 没有」的帧数：应约等于 onset2 正样本数的一半（提前一帧）
        extra = int(((lab[:, 3, :] > 0) & (lab[:, 2, :] == 0)).sum())
        total_onset2 += int((lab[:, 2, :] > 0).sum())
        pre_roll_extra += extra

    w2 = np.concatenate(widths2)
    w3 = np.concatenate(widths3)
    result = {
        "sampled_files": len(names),
        "onset2_width_mean": float(w2.mean()),
        "onset2_width_mode_fraction_2": float((w2 == 2).mean()),
        "onset3_width_mean": float(w3.mean()),
        "onset3_width_mode_fraction_3": float((w3 == 3).mean()),
        "onset3_only_frames": pre_roll_extra,
        "onset2_positive_frames": total_onset2,
        "onset3_extra_ratio": pre_roll_extra / max(total_onset2, 1),
        "frame_positive_rate": float(np.mean(rates["frame"])),
        "pedal_positive_rate": float(np.mean(rates["pedal"])),
        "manifest_label_rates": manifest["label_positive_rates"],
    }
    out = "/Docker/git/piano_practice/outputs/M1/label_check.json"
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("写入:", out)


if __name__ == "__main__":
    main()
