"""M0.6 训练前门禁（pre-flight gate）：G1–G8 自动检查。

目的：在花钱上云训练之前，把「数据、标签、评测、解码、资源、可复现性」逐项检查一遍，
产出一张 PASS / FAIL 表。任何一项 FAIL，都意味着上云前必须先处理。

检查项：
    G1 数据完整性与标签统计
    G2 训练/验证/测试划分是否有泄漏
    G3 训练特征与推理特征是否一致
    G4 评测链路 oracle 自检
    G5 解码器是否已达天花板（证明模型不是被解码拖累）
    G6 基线指标登记（训练后对比用）
    G7 资源与预算：显存、每步耗时、训练总量外推
    G8 可复现性：重复解码一致、随机种子受控

输出：
    outputs/M0/preflight_report.json
    outputs/M0/preflight_summary.md
    outputs/M0/registry/EXP-000_baseline.json（G6）

用法（容器内）：
    cd /Docker/git/piano_practice/m0 && \
    PYTHONPATH=/Docker/git/piano_practice/m0:/Docker/git/piano_practice \
    python preflight.py
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import sys
import time
from collections import Counter

import numpy as np
import pandas as pd
import torch

ROOT = "/Docker/git/piano_practice"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "m0"))

import mir_eval.transcription as mtran  # noqa: E402

import decode_v2  # noqa: E402
import notation_v1  # noqa: E402
import prob_cache  # noqa: E402
from transcribe_readable import compute_cqt  # noqa: E402

MAESTRO = "/Docker/piano2_data/raw/maestro-v3.0.0"
PROCESSED = "/Docker/piano2_data/processed/maestro_processed"
OUT = os.path.join(ROOT, "outputs", "M0")
CKPT = os.path.join(ROOT, "trained_data", "checkpoints", "model_best.pth")
FPS = 16000 / 512
DEV = ("haydn", "mozart", "beethoven", "holdout2")
CEILING = {"haydn": 0.950, "mozart": 0.949, "beethoven": 0.797, "holdout2": 0.825}
EXPECTED_COUNTS = {"train": 962, "validation": 137, "test": 177}
RESULTS: list[dict] = []


def add(check_id, name, status, detail, metrics=None):
    """记录一条检查结果。

    参数:
        check_id (str): 检查项编号，如 "G1"。
        name (str): 检查项名称。
        status (str): "PASS" / "FAIL" / "WARN"。
        detail (str): 人话解释。
        metrics (dict | None): 支撑数字。
    """
    RESULTS.append({"id": check_id, "name": name, "status": status,
                    "detail": detail, "metrics": metrics or {}})
    print(f"[{status}] {check_id} {name}: {detail}")


def sha256_file(path, chunk=1 << 20):
    """计算文件 sha256（分块读，避免一次读进内存）。"""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            data = fh.read(chunk)
            if not data:
                break
            h.update(data)
    return h.hexdigest()


# ─────────────────────────── G1 数据完整性 ───────────────────────────


def check_g1():
    """检查 npz 数量、形状、数值健康度与标签统计。"""
    counts = {}
    for split in ("train", "validation", "test"):
        n = len([f for f in os.listdir(os.path.join(PROCESSED, split)) if f.endswith(".npz")])
        counts[split] = n
    count_ok = all(counts[k] == v for k, v in EXPECTED_COUNTS.items())

    files = []
    for split in ("train", "validation", "test"):
        d = os.path.join(PROCESSED, split)
        names = sorted(f for f in os.listdir(d) if f.endswith(".npz"))
        step = max(1, len(names) // 6)
        files.extend((split, os.path.join(d, names[i])) for i in range(0, len(names), step))

    onset_pos, frame_pos, widths, cqt_range = [], [], [], []
    shape_ok = dtype_ok = finite_ok = True
    for split, path in files:
        data = np.load(path)
        x, yf, yo, yv = data["cqt"], data["frame"], data["onset"], data["velocity"]
        if x.shape[1] != 264 or yf.shape[1] != 88 or x.shape[0] != yf.shape[0]:
            shape_ok = False
        if x.dtype != np.float32 or yf.dtype != np.float32:
            dtype_ok = False
        if not (np.isfinite(x).all() and np.isfinite(yf).all() and np.isfinite(yo).all()):
            finite_ok = False
        onset_pos.append(float(yo.mean()))
        frame_pos.append(float(yf.mean()))
        cqt_range.append((float(x.min()), float(x.max())))
        # onset 正样本块的宽度（帧）：每键沿时间找连续 1 的段
        for k in range(88):
            col = yo[:, k]
            if col.max() == 0:
                continue
            idx = np.flatnonzero(np.diff(np.concatenate(([0], col, [0]))) != 0)
            for a, b in zip(idx[::2], idx[1::2]):
                widths.append(int(b - a))

    width_mean = float(np.mean(widths)) if widths else 0.0
    width_mode3 = float(np.mean([w == 3 for w in widths])) if widths else 0.0
    label_ok = width_mean < 2.5  # 标准应为 2 帧；当前实现是 3 帧
    detail = (f"npz 数量 {counts}；抽样 {len(files)} 个文件；"
              f"onset 正样本率 {np.mean(onset_pos):.3f}、frame {np.mean(frame_pos):.3f}；"
              f"onset 目标宽度均值 {width_mean:.2f} 帧（标准 2 帧，长度 3 帧占比 {width_mode3:.1%}）")
    status = "PASS" if (count_ok and shape_ok and dtype_ok and finite_ok and label_ok) else "FAIL"
    reasons = []
    if not count_ok:
        reasons.append("数量不符")
    if not shape_ok:
        reasons.append("形状不一致")
    if not dtype_ok:
        reasons.append("dtype 不是 float32")
    if not finite_ok:
        reasons.append("存在 NaN/inf")
    if not label_ok:
        reasons.append("onset 标签宽 3 帧（应为 2 帧）")
    if reasons:
        detail += "；问题：" + "、".join(reasons)
    metrics = {
        "counts": counts, "n_sampled": len(files),
        "onset_pos_rate": float(np.mean(onset_pos)), "frame_pos_rate": float(np.mean(frame_pos)),
        "onset_width_mean_frames": width_mean,
        "onset_width3_fraction": width_mode3,
        "cqt_min": float(min(r[0] for r in cqt_range)),
        "cqt_max": float(max(r[1] for r in cqt_range)),
        "known_issue_frame_contains_pedal": True,
        "known_issue_short_file_pad_zero": True,
    }
    add("G1", "数据完整性与标签统计", status, detail, metrics)


# ─────────────────────────── G2 数据泄漏 ───────────────────────────


def check_g2():
    """检查文件级与曲目级划分泄漏。"""
    df = pd.read_csv(os.path.join(MAESTRO, "maestro-v3.0.0.csv"))
    df["work"] = df["canonical_composer"].astype(str) + " | " + df["canonical_title"].astype(str)
    dup_audio = int(df["audio_filename"].duplicated().sum())
    work_splits = df.groupby("work")["split"].nunique()
    cross_work = int((work_splits > 1).sum())
    frac = cross_work / len(work_splits)
    detail = (f"文件级重复 {dup_audio} 条；跨 split 的同一作品 {cross_work} 个"
              f"（占作品数 {frac:.1%}，MAESTRO 划分固有现象）")
    status = "PASS" if dup_audio == 0 else "FAIL"
    if status == "PASS" and frac > 0.005:
        status = "WARN"
        detail += "；需在报告里注明，看 test 分数时留意"
    add("G2", "划分无泄漏", status, detail,
        {"duplicate_audio": dup_audio, "cross_split_works": cross_work, "work_fraction": frac})


# ─────────────────────────── G3 特征一致性 ───────────────────────────


def check_g3():
    """对比训练 npz 里的 CQT 与推理端重算的 CQT。"""
    df = pd.read_csv(os.path.join(MAESTRO, "maestro-v3.0.0.csv"))
    # 选两首较短的 test 曲目，省时间
    te = df[df.split == "test"].sort_values("duration").head(2)
    rows = []
    for idx, row in te.iterrows():
        stem = os.path.basename(str(row["audio_filename"]))[:-4]
        npz_path = os.path.join(PROCESSED, "test", f"{idx:04d}_{stem}.npz")
        if not os.path.exists(npz_path):
            continue
        wav = os.path.join(MAESTRO, row["audio_filename"])
        stored = np.load(npz_path)["cqt"]
        recomputed = compute_cqt(wav)
        n = min(len(stored), len(recomputed))
        a = stored[:n].ravel()[::50]
        b = recomputed[:n].ravel()[::50]
        corr = float(np.corrcoef(a, b)[0, 1]) if len(a) > 10 else float("nan")
        rows.append({
            "track": os.path.basename(wav),
            "stored_frames": int(len(stored)), "recomputed_frames": int(len(recomputed)),
            "corr": corr,
            "mean_abs_diff": float(np.mean(np.abs(stored[:n] - recomputed[:n]))),
        })
    if not rows:
        add("G3", "训练/推理特征一致性", "SKIP", "没有找到可对比的 npz")
        return
    min_corr = min(r["corr"] for r in rows)
    frames_match = all(abs(r["stored_frames"] - r["recomputed_frames"]) <= 2 for r in rows)
    status = "PASS" if (min_corr > 0.999 and frames_match) else "FAIL"
    detail = "；".join(
        f"{r['track'][:28]} 帧数 {r['stored_frames']}/{r['recomputed_frames']}，相关 {r['corr']:.6f}"
        for r in rows)
    add("G3", "训练/推理特征一致性", status, detail,
        {"rows": rows, "min_corr": min_corr, "frames_match": frames_match})


# ─────────────────────────── G4 评测 oracle ───────────────────────────


def _gt(name):
    midi = prob_cache.GT_MIDI.get(name)
    if midi is None:
        step3 = json.load(open(os.path.join(OUT, "m05", "step3b_final.json"), encoding="utf-8"))
        midi = os.path.join(MAESTRO, step3["holdout2"]["midi"])
    ev = notation_v1.events_from_midi(midi)
    return (np.array([[s, e] for (_p, s, e, _v) in ev], float),
            np.array([p for (p, _s, _e, _v) in ev], int))


def _onset_f1(ref_iv, ref_p, est_iv, est_p):
    if len(est_iv) == 0:
        return 0.0, 0.0, 0.0
    p, r, f, _ = mtran.precision_recall_f1_overlap(
        ref_iv, ref_p, est_iv, est_p, onset_tolerance=0.05,
        pitch_tolerance=50.0, offset_ratio=None)
    return float(p), float(r), float(f)


def check_g4():
    """把 GT 当预测喂给评分函数，必须得满分；再验证对扰动的敏感度。"""
    ref_iv, ref_p = _gt("haydn")
    rng = np.random.default_rng(0)
    dur = ref_iv[:, 1] - ref_iv[:, 0]
    cases = {
        "exact": (ref_iv, ref_p),
        "jitter32ms": (ref_iv + 0.032, ref_p),
        "jitter64ms": (ref_iv + 0.064, ref_p),
        "shuffle_pitch": (ref_iv, ref_p[rng.permutation(len(ref_p))]),
    }
    scores = {name: _onset_f1(ref_iv, ref_p, iv, pp)[2] for name, (iv, pp) in cases.items()}
    ok = (scores["exact"] > 0.999 and scores["jitter32ms"] > 0.999
          and scores["jitter64ms"] < 0.2 and scores["shuffle_pitch"] < 0.35)
    add("G4", "评测链路 oracle 自检", "PASS" if ok else "FAIL",
        f"GT 当预测 F1={scores['exact']:.3f}；+32ms={scores['jitter32ms']:.3f}；"
        f"+64ms={scores['jitter64ms']:.3f}；打乱音高={scores['shuffle_pitch']:.3f}", scores)


# ─────────────────────────── G5 解码天花板 ───────────────────────────


def _final_cfg():
    step3 = json.load(open(os.path.join(OUT, "m05", "step3b_final.json"), encoding="utf-8"))
    return decode_v2.DecodeConfig(**step3["final_config"])


def _decode_track(name, cfg):
    c = prob_cache.load_cache(name)
    return decode_v2.decode(c["onset"], c["frame"], c["velocity"], cfg)


def check_g5():
    """当前解码器在缓存概率上应达到天花板的 90% 以上。"""
    cfg = _final_cfg()
    rows = {}
    ok = True
    for name in DEV:
        try:
            ev = _decode_track(name, cfg)
        except FileNotFoundError:
            continue
        ref_iv, ref_p = _gt(name)
        est = np.array([[e.start, e.end] for e in ev], float)
        pitches = np.array([e.pitch for e in ev], int)
        p, r, f = _onset_f1(ref_iv, ref_p, est, pitches)
        rows[name] = {"P": p, "R": r, "F1": f, "ceiling": CEILING[name]}
        ok = ok and f >= 0.9 * CEILING[name]
    add("G5", "解码器达到天花板", "PASS" if ok else "FAIL",
        "；".join(f"{k}: F1={v['F1']:.3f}（天花板 {v['ceiling']:.3f}）" for k, v in rows.items()), rows)


# ─────────────────────────── G6 基线登记 ───────────────────────────


def check_g6():
    """把当前权重 + 解码配置 + 固定测试集指标登记下来，供训练后对比。"""
    cfg = _final_cfg()
    model_sha = sha256_file(CKPT)
    rows = {}
    for name in DEV:
        try:
            ev = _decode_track(name, cfg)
        except FileNotFoundError:
            continue
        ref_iv, ref_p = _gt(name)
        est = np.array([[e.start, e.end] for e in ev], float)
        pitches = np.array([e.pitch for e in ev], int)
        p, r, f = _onset_f1(ref_iv, ref_p, est, pitches)
        p2, r2, f2, _ = mtran.precision_recall_f1_overlap(
            ref_iv, ref_p, est, pitches, onset_tolerance=0.05,
            pitch_tolerance=50.0, offset_ratio=0.2, offset_min_tolerance=0.05)
        rows[name] = {"onset": {"P": p, "R": r, "F1": f},
                      "note_with_offset": {"P": float(p2), "R": float(r2), "F1": float(f2)},
                      "n_pred": int(len(ev))}
    os.makedirs(os.path.join(OUT, "registry"), exist_ok=True)
    payload = {
        "experiment_id": "EXP-000_baseline",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "kind": "baseline before retraining",
        "checkpoint": {"path": CKPT, "sha256": model_sha,
                       "size_bytes": os.path.getsize(CKPT)},
        "decode_config": {k: getattr(cfg, k) for k in (
            "method", "harmonic_mode", "harmonic_strength", "harmonic_window_frames",
            "harmonic_frame_max", "frame_gate", "onset_threshold",
            "onset_prominence", "peak_distance", "frame_off_threshold", "release_frames")},
        "test_set": list(rows.keys()),
        "metrics": rows,
        "notes": "onset 容差 50 ms、pitch 50 cents；note_with_offset 用 offset_ratio=0.2",
    }
    path = os.path.join(OUT, "registry", "EXP-000_baseline.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    add("G6", "基线指标登记", "PASS" if rows else "FAIL",
        f"写入 {path}；模型 sha256={model_sha[:12]}…；曲目 {list(rows.keys())}",
        {"registry_path": path, "tracks": list(rows.keys())})


# ─────────────────────────── G7 资源与预算 ───────────────────────────


def check_g7():
    """用合成数据在 4060 上实测训练一步的耗时与显存，并外推预算。"""
    from model_industrial import TransformerPianoTranscriber

    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = []
    for batch, seq in ((8, 313), (16, 313), (8, 625), (16, 625)):
        model = TransformerPianoTranscriber().to(device)
        model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scaler = torch.amp.GradScaler(device="cuda")
        bce = torch.nn.BCEWithLogitsLoss()
        mse = torch.nn.MSELoss()
        x = torch.randn(batch, seq, 264, device=device)
        yo = (torch.rand(batch, seq, 88, device=device) > 0.95).float()
        yf = (torch.rand(batch, seq, 88, device=device) > 0.90).float()
        yv = torch.rand(batch, seq, 88, device=device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        times = []
        try:
            for step in range(3):
                t0 = time.time()
                opt.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda"):
                    po, pf, pv = model(x)
                    loss = bce(po, yo) + bce(pf, yf) + mse(torch.sigmoid(pv), yv)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                torch.cuda.synchronize()
                if step > 0:
                    times.append(time.time() - t0)
            peak = torch.cuda.max_memory_allocated() / 1024 ** 3
            results.append({"batch": batch, "seq_frames": seq, "step_s": float(np.mean(times)),
                            "peak_vram_gb": float(peak)})
        except RuntimeError as exc:
            results.append({"batch": batch, "seq_frames": seq, "error": str(exc)[:120]})
        finally:
            del model
            torch.cuda.empty_cache()

    feasible = [r for r in results if "step_s" in r and r["peak_vram_gb"] < 7.5]
    if not feasible:
        add("G7", "资源与预算", "FAIL", "所有配置都装不进 4060 的 8 GB 显存", {"results": results})
        return

    best = max(feasible, key=lambda r: r["batch"])
    # 训练量估算：159.24 小时训练音频，10 秒一段 → 片段数；batch 后得到每轮步数
    total_audio_h = 159.24
    seg_seconds = 10.0
    n_segments = total_audio_h * 3600.0 / seg_seconds
    steps_per_epoch = n_segments / best["batch"]
    epochs = 20
    total_steps = steps_per_epoch * epochs
    hours_4060 = total_steps * best["step_s"] / 3600.0
    hours_4090 = hours_4060 / 2.5  # 4090 相对 4060 的保守加速比假设
    status = "PASS" if hours_4090 <= 12.0 else "WARN"
    add("G7", "资源与预算", status,
        f"可行配置 batch={best['batch']}、seq={best['seq_frames']} 帧，"
        f"每步 {best['step_s']:.3f}s、峰值显存 {best['peak_vram_gb']:.2f} GB；"
        f"按 {total_audio_h} 小时音频、{seg_seconds:g} 秒片段、{epochs} 轮估算 "
        f"{total_steps:,.0f} 步 ≈ 4060 上 {hours_4060:.1f} 小时 / 4090 上约 {hours_4090:.1f} 小时"
        f"（4090 按 2.5 倍加速估算，未实测）",
        {"results": results, "feasible_best": best,
         "estimate": {"n_segments": n_segments, "steps_per_epoch": steps_per_epoch,
                      "epochs": epochs, "total_steps": total_steps,
                      "hours_4060": hours_4060, "hours_4090_assumed": hours_4090}})


# ─────────────────────────── G8 可复现性 ───────────────────────────


def check_g8():
    """重复解码是否一致；DataLoader 随机性是否受控。"""
    cfg = _final_cfg()
    ev1 = _decode_track("haydn", cfg)
    ev2 = _decode_track("haydn", cfg)
    same = (len(ev1) == len(ev2)
            and all(abs(a.start - b.start) < 1e-9 and a.pitch == b.pitch
                    for a, b in zip(ev1, ev2)))

    # DataLoader 顺序可复现性（shuffle + 固定 generator）
    from torch.utils.data import DataLoader, Dataset

    class Dummy(Dataset):
        def __len__(self):
            return 50

        def __getitem__(self, idx):
            return torch.tensor(idx)

    def order(seed):
        g = torch.Generator()
        g.manual_seed(seed)
        dl = DataLoader(Dummy(), batch_size=5, shuffle=True, generator=g)
        return [int(v) for batch in dl for v in batch]

    order_ok = order(123) == order(123)
    status = "PASS" if (same and order_ok) else "FAIL"
    add("G8", "可复现性", status,
        f"重复解码一致={same}；DataLoader 顺序固定种子可复现={order_ok}；"
        f"注意：多进程 worker 内的 numpy 随机性仍需在训练脚本里显式设种子（当前未验证）",
        {"decode_identical": bool(same), "dataloader_order_reproducible": bool(order_ok)})


# ─────────────────────────── 汇总输出 ───────────────────────────


def write_summary():
    """把检查结果写成 JSON 与 Markdown 摘要。"""
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "preflight_report.json"), "w", encoding="utf-8") as fh:
        json.dump({"date": time.strftime("%Y-%m-%d %H:%M:%S"),
                   "checks": RESULTS}, fh, ensure_ascii=False, indent=2)
    lines = ["# 训练前门禁检查结果（自动生成）", "",
             f"日期：{time.strftime('%Y-%m-%d %H:%M:%S')}", "",
             "| 检查项 | 状态 | 说明 |", "|---|---|---|"]
    for r in RESULTS:
        lines.append(f"| {r['id']} {r['name']} | {r['status']} | {r['detail']} |")
    lines += ["", "## 数字明细", "", "```json",
              json.dumps({r["id"]: r["metrics"] for r in RESULTS}, ensure_ascii=False, indent=2),
              "```", ""]
    path = os.path.join(OUT, "preflight_summary.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print("写入:", path)
    print("写入:", os.path.join(OUT, "preflight_report.json"))


def main():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    check_g1()
    check_g2()
    check_g3()
    check_g4()
    check_g5()
    check_g6()
    check_g7()
    check_g8()
    write_summary()
    n_fail = sum(1 for r in RESULTS if r["status"] == "FAIL")
    n_warn = sum(1 for r in RESULTS if r["status"] == "WARN")
    print(f"\n门禁结果：{len(RESULTS) - n_fail - n_warn} PASS / {n_warn} WARN / {n_fail} FAIL")
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
