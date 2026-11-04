"""M1：训练脚本 v2 —— 数据 v2 + 五头模型 + 按验证集 F1 选模型。

相对旧训练脚本（`train_full.py`）的改动：
    1. 每轮按「段」采样（10 秒段索引），不再每轮每首只截一次 65 秒；
    2. 验证集固定段，val F1 可跨步比较；
    3. 五个输出头分别记录 loss；按验证集 onset F1 选模型，而不是 val loss；
    4. AdamW + warmup + cosine，梯度裁剪，AMP；
    5. 检查点保存 optimizer/step/配置，可续训；
    6. 支持 --max-steps 与 --dry-run，用于本地小规模试跑（pilot run）。

用法（容器内）：
    cd /Docker/git/piano_practice/m1 && \
    PYTHONPATH=/Docker/git/piano_practice/m1 python train_v2.py --max-steps 1000
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from piano2.dataset import SEG_FRAMES, SegmentDataset, collate
from piano2.model import PianoModelV2

from piano2.constants import FPS, OUTPUT_ROOT

OUT_ROOT = str(OUTPUT_ROOT / "M1")


def set_seed(seed):
    """固定所有随机源，并告诉 DataLoader worker 该用哪个种子。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def worker_init(worker_id):
    """让每个 DataLoader worker 的 numpy 随机性也可复现。"""
    seed = torch.initial_seed() % 2 ** 32
    np.random.seed(seed)
    random.seed(seed)


def lr_at(step, base_lr, warmup, total):
    """线性预热 + cosine 衰减。"""
    if step < warmup:
        return base_lr * (step + 1) / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    return base_lr * (0.03 + 0.97 * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0))))


def decode_onsets(prob, threshold=0.5, tol_frames=1):
    """把连续概率解码成 (pitch, frame) 事件；这里用最简单的峰值检测。

    参数:
        prob (numpy.ndarray): (T, 88) onset 概率。
        threshold (float): 峰值高度阈值。
        tol_frames (int): 未使用，保留参数语义（在匹配时用）。

    返回:
        list[tuple[int, int]]: (pitch, frame)。
    """
    from scipy.signal import find_peaks

    events = []
    for k in range(88):
        peaks, _ = find_peaks(prob[:, k], height=threshold, prominence=0.01, distance=5)
        events.extend((k, int(p)) for p in peaks)
    return events


def gt_onsets(onset_label):
    """从 onset 标签矩阵提取 GT 事件：连续 1 的起点记为一个起音。"""
    events = []
    diff = np.diff(np.concatenate([np.zeros((1, 88)), onset_label, np.zeros((1, 88))]), axis=0)
    for k in range(88):
        starts = np.flatnonzero(diff[:, k] > 0.5)
        events.extend((k, int(s)) for s in starts)
    return events


def match_f1(pred, ref, tol_frames=1):
    """按音高 + 时间容差做贪心匹配，返回 P/R/F1。

    参数:
        pred (list[tuple[int,int]]): 预测 (pitch, frame)。
        ref (list[tuple[int,int]]): GT (pitch, frame)。
        tol_frames (int): 允许的帧差；1 帧 = 32 ms，对应 50 ms 容差的整数等价。

    返回:
        tuple[float, float, float, int, int, int]: P, R, F1, TP, FP, FN。
    """
    used = set()
    by_pitch = {}
    for i, (p, f) in enumerate(ref):
        by_pitch.setdefault(p, []).append((f, i))
    tp = 0
    for p, f in pred:
        candidates = by_pitch.get(p, [])
        best = None
        for gf, gi in candidates:
            if gi in used:
                continue
            if abs(gf - f) <= tol_frames:
                if best is None or abs(gf - f) < best[1]:
                    best = (gi, abs(gf - f))
        if best is not None:
            used.add(best[0])
            tp += 1
    fp = len(pred) - tp
    fn = len(ref) - len(pred) + fp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1, tp, fp, fn


@torch.no_grad()
def evaluate(model, loader, device, threshold=0.5):
    """在固定验证段上计算 onset F1（micro）。"""
    model.eval()
    tp = fp = fn = 0
    loss_sum = 0.0
    for batch in loader:
        x = batch["x"].to(device)
        with torch.autocast(device_type="cuda", enabled=(device == "cuda")):
            onset_logits, frame_logits, _, _, _ = model(x)
        prob = torch.sigmoid(onset_logits.float()).cpu().numpy()
        gt = batch["onset"].numpy()
        for i in range(prob.shape[0]):
            pred = decode_onsets(prob[i], threshold=threshold)
            ref = gt_onsets(gt[i])
            _, _, _, a, b, c = match_f1(pred, ref)
            tp += a
            fp += b
            fn += c
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"P": precision, "R": recall, "F1": f1, "tp": tp, "fp": fp, "fn": fn}


def main():
    ap = argparse.ArgumentParser(description="M1 训练脚本 v2")
    ap.add_argument("--max-steps", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--val-every", type=int, default=200)
    ap.add_argument("--save-every", type=int, default=500,
                    help="每多少步保存一次 latest.pt（断点续训用）")
    ap.add_argument("--val-segments", type=int, default=120)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--label-channel", type=str, default="onset2")
    ap.add_argument("--run-name", type=str, default="pilot")
    ap.add_argument("--dry-run", action="store_true", help="只跑几十步检查链路")
    ap.add_argument("--resume", type=str, default="",
                    help="检查点路径；写 auto 表示自动读 run 目录下的 latest.pt")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.dry_run:
        args.max_steps = min(args.max_steps, 40)
        args.val_every = 20

    run_dir = os.path.join(OUT_ROOT, args.run_name)
    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_ds = SegmentDataset("train", label_channel=args.label_channel, augment=True)
    val_ds = SegmentDataset("validation", label_channel=args.label_channel,
                            augment=False, limit=args.val_segments)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate,
                              pin_memory=True, drop_last=True,
                              persistent_workers=(args.num_workers > 0),
                              worker_init_fn=worker_init)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            num_workers=max(args.num_workers // 2, 0),
                            collate_fn=collate, pin_memory=True)
    print(f"train 段数: {len(train_ds)}；验证段数: {len(val_ds)}；设备: {device}")

    model = PianoModelV2().to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"参数量: {n_params:,}")

    pos_weight = {"onset": 20.0, "frame": 5.0, "pedal": 1.0, "pedal_on": 10.0}
    bce_onset = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight["onset"]).to(device))
    bce_frame = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight["frame"]).to(device))
    bce_pedal = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight["pedal"]).to(device))
    bce_pedal_on = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight["pedal_on"]).to(device))
    mse = nn.MSELoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler(device="cuda", enabled=(device == "cuda"))

    # 续训时用追加模式，避免把之前的训练日志覆盖掉
    log_path = os.path.join(run_dir, "train_log.csv")
    resume_path = args.resume
    if resume_path == "auto":
        resume_path = os.path.join(run_dir, "checkpoints", "latest.pt")
    resuming = bool(resume_path) and os.path.exists(resume_path)
    if resuming:
        log_fh = open(log_path, "a", newline="", encoding="utf-8")
        writer = csv.writer(log_fh)
        if os.path.getsize(log_path) == 0:
            writer.writerow(["step", "loss", "loss_onset", "loss_frame", "loss_vel",
                             "loss_pedal", "loss_pedal_on", "lr", "elapsed_s"])
    else:
        log_fh = open(log_path, "w", newline="", encoding="utf-8")
        writer = csv.writer(log_fh)
        writer.writerow(["step", "loss", "loss_onset", "loss_frame", "loss_vel",
                         "loss_pedal", "loss_pedal_on", "lr", "elapsed_s"])

    best_f1 = -1.0
    t0 = time.time()
    step = 0
    if resuming:
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        step = int(ckpt["step"])
        best_f1 = float(ckpt.get("best_f1", -1.0))
        old_max = ckpt.get("config", {}).get("max_steps")
        if old_max and int(old_max) != args.max_steps:
            print(f"[!] 注意：检查点记录的 max_steps={old_max}，本次传的是 {args.max_steps}；"
                  f"学习率调度会按新值计算，建议保持一致")
        print(f"从 {resume_path} 续训：step={step}，best_f1={best_f1:.4f}，"
              f"剩余 {args.max_steps - step} 步（约 {args.max_steps - step} × 0.08 秒）")
    if step >= args.max_steps:
        print("检查点已达到目标步数，无需继续训练")
        log_fh.close()
        return

    model.train()
    while step < args.max_steps:
        for batch in train_loader:
            if step >= args.max_steps:
                break
            lr = lr_at(step, args.lr, args.warmup, args.max_steps)
            for group in optimizer.param_groups:
                group["lr"] = lr
            x = batch["x"].to(device)
            yo = batch["onset"].to(device)
            yf = batch["frame"].to(device)
            yv = batch["velocity"].to(device)
            yp = batch["pedal"].to(device)
            ypo = batch["pedal_on"].to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=(device == "cuda")):
                po, pf, pv, pp, ppo = model(x)
                loss_o = bce_onset(po, yo)
                loss_f = bce_frame(pf, yf)
                mask = yo > 0.5
                loss_v = mse(torch.sigmoid(pv)[mask], yv[mask]) if mask.any() else torch.zeros((), device=device)
                loss_p = bce_pedal(pp, yp)
                loss_po = bce_pedal_on(ppo, ypo)
                total = loss_o + loss_f + loss_v + 0.5 * loss_p + 0.5 * loss_po
            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            step += 1
            writer.writerow([step, float(total.detach()), float(loss_o.detach()),
                             float(loss_f.detach()), float(loss_v.detach()),
                             float(loss_p.detach()), float(loss_po.detach()), lr,
                             round(time.time() - t0, 2)])
            if step % 50 == 0:
                log_fh.flush()
                print(f"  step {step}/{args.max_steps} loss={float(total):.4f} "
                      f"onset={float(loss_o):.4f} frame={float(loss_f):.4f} "
                      f"lr={lr:.2e} 用时 {time.time() - t0:.0f}s", flush=True)

            # 定期保存断点（与验证解耦），保证意外关机最多丢 save_every 步
            if args.save_every > 0 and step % args.save_every == 0:
                ckpt = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "step": step, "best_f1": best_f1, "config": vars(args), "val": {}}
                torch.save(ckpt, os.path.join(run_dir, "checkpoints", "latest.pt"))
                with open(os.path.join(run_dir, "progress.json"), "w", encoding="utf-8") as fh:
                    json.dump({"step": step, "max_steps": args.max_steps,
                               "best_f1": best_f1, "lr": lr,
                               "updated": time.strftime("%Y-%m-%d %H:%M:%S")},
                              fh, ensure_ascii=False)

            if step % args.val_every == 0:
                val = evaluate(model, val_loader, device)
                print(f"  [val] step {step}: P={val['P']:.3f} R={val['R']:.3f} F1={val['F1']:.3f}")
                model.train()
                ckpt = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "step": step, "best_f1": max(best_f1, val["F1"]),
                        "config": vars(args), "val": val}
                torch.save(ckpt, os.path.join(run_dir, "checkpoints", "latest.pt"))
                with open(os.path.join(run_dir, "progress.json"), "w", encoding="utf-8") as fh:
                    json.dump({"step": step, "max_steps": args.max_steps,
                               "best_f1": max(best_f1, val["F1"]), "val_f1": val["F1"],
                               "lr": lr, "updated": time.strftime("%Y-%m-%d %H:%M:%S")},
                              fh, ensure_ascii=False)
                if val["F1"] > best_f1:
                    best_f1 = val["F1"]
                    torch.save({**ckpt, "best_f1": best_f1},
                               os.path.join(run_dir, "checkpoints", "best.pt"))
                    print(f"  ⭐ 新最佳 F1={best_f1:.4f}")

    log_fh.close()
    # 最终一次验证
    val = evaluate(model, val_loader, device)
    summary = {"steps": args.max_steps, "best_val_f1": max(best_f1, val["F1"]),
               "final_val": val, "config": vars(args),
               "elapsed_s": round(time.time() - t0, 1)}
    with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    print("完成:", json.dumps(summary, ensure_ascii=False))

    # 画训练曲线（简单两张图）
    try:
        import matplotlib
        matplotlib.use("Agg")
        import sys
        from piano2 import plot_style
        plot_style.use_cjk_font()
        import matplotlib.pyplot as plt
        rows = list(csv.DictReader(open(log_path, encoding="utf-8")))
        steps = [int(r["step"]) for r in rows]
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
        for key, label in (("loss_onset", "onset loss"), ("loss_frame", "frame loss"),
                           ("loss_vel", "velocity loss"), ("loss_pedal", "pedal loss")):
            axes[0].plot(steps, [float(r[key]) for r in rows], label=label, linewidth=1.2)
        axes[0].set_xlabel("步数"); axes[0].set_ylabel("损失"); axes[0].legend(fontsize=8)
        axes[0].set_title("各头损失"); axes[0].grid(alpha=0.3)
        axes[1].plot(steps, [float(r["lr"]) for r in rows], color="#8e44ad")
        axes[1].set_xlabel("步数"); axes[1].set_ylabel("学习率")
        axes[1].set_title("学习率（warmup + cosine）"); axes[1].grid(alpha=0.3)
        fig.tight_layout()
        plot_style.save(fig, os.path.join(run_dir, "train_curves.png"))
        plt.close(fig)
        print("曲线写入:", os.path.join(run_dir, "train_curves.png"))
    except Exception as exc:
        print("画图失败（不影响训练）:", exc)


if __name__ == "__main__":
    main()
