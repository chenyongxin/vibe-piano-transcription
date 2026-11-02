"""M1：在数据 v2 的固定验证段上，测旧模型（3 头、旧标签训练）的 onset F1 作为基线。

旧模型的问题：训练时输入是原始 dB 值（未归一化），所以要用原始 dB 喂它；
输出只有 3 个头；架构里 MaxPool(1,2) 与琴键不对齐，但这里只做推理对比。

用法（容器内）：
    cd /Docker/git/piano_practice/m1 && \
    PYTHONPATH=/Docker/git/piano_practice/m0:/Docker/git/piano_practice/m1 \
    python eval_old_model.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = "/Docker/git/piano_practice"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "m0"))
sys.path.insert(0, os.path.join(ROOT, "m1"))

from dataset_v2 import SegmentDataset, collate  # noqa: E402
from train_v2 import match_f1, gt_onsets, decode_onsets  # noqa: E402
from transcribe_readable import load_model  # noqa: E402

CKPT = os.path.join(ROOT, "trained_data", "checkpoints", "model_best.pth")
OUT = os.path.join(ROOT, "outputs", "M1", "baseline_old_model.json")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = SegmentDataset("validation", label_channel="onset2", limit=120)
    loader = DataLoader(ds, batch_size=8, shuffle=False, collate_fn=collate)
    model = load_model(CKPT, device)
    model.eval()

    tp = fp = fn = 0
    with torch.no_grad():
        for batch in loader:
            # 旧模型训练时吃的是原始 dB 值：这里把归一化后的 [0,1] 还原回 [-80,0]
            x = batch["x"].to(device) * 80.0 - 80.0
            with torch.autocast(device_type="cuda", enabled=(device == "cuda")):
                onset_logits, _, _ = model(x)
            prob = torch.sigmoid(onset_logits.float()).cpu().numpy()
            gt = batch["onset"].numpy()
            for i in range(prob.shape[0]):
                n = int(batch["n_frames"][i])
                pred = decode_onsets(prob[i][:n])
                ref = gt_onsets(gt[i][:n])
                _, _, _, a, b, c = match_f1(pred, ref)
                tp += a
                fp += b
                fn += c
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    result = {"model": CKPT, "val_segments": 120, "P": precision, "R": recall,
              "F1": f1, "tp": tp, "fp": fp, "fn": fn}
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("写入:", OUT)


if __name__ == "__main__":
    main()
