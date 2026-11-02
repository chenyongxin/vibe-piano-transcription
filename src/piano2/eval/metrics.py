"""piano2.eval.metrics：note-level 评测指标与 oracle 自检。

口径（与 M0 报告一致）：
    onset-only F1：音高 + 起点（容差 50 ms），不看时值；
    note-with-offset F1：起点、音高、时值都要匹配（offset 容差 20%、最小 50 ms）。
"""

from __future__ import annotations

import numpy as np

import mir_eval.transcription as mtran


def to_arrays(events):
    """把 NoteEvent 列表转成 mir_eval 需要的 (intervals, pitches)。

    参数:
        events (list): 含 pitch/start/end 的对象列表。

    返回:
        tuple[numpy.ndarray, numpy.ndarray]: (N,2) 区间与 (N,) 音高。
    """
    if not events:
        return np.zeros((0, 2)), np.zeros(0, dtype=int)
    return (np.array([[e.start, e.end] for e in events], dtype=float),
            np.array([e.pitch for e in events], dtype=int))


def note_metrics(ref_iv, ref_p, est_iv, est_p):
    """同时算 onset-only 与 note-with-offset 两种口径。

    参数:
        ref_iv (numpy.ndarray): GT 区间。
        ref_p (numpy.ndarray): GT 音高。
        est_iv (numpy.ndarray): 预测区间。
        est_p (numpy.ndarray): 预测音高。

    返回:
        dict: onset_* 与 note_* 的 P/R/F1，以及样本数。
    """
    if len(est_iv) == 0 or len(ref_iv) == 0:
        return {"onset_P": 0.0, "onset_R": 0.0, "onset_F1": 0.0,
                "note_P": 0.0, "note_R": 0.0, "note_F1": 0.0,
                "n_ref": int(len(ref_p)), "n_est": int(len(est_p))}
    p1, r1, f1, _ = mtran.precision_recall_f1_overlap(
        ref_iv, ref_p, est_iv, est_p,
        onset_tolerance=0.05, pitch_tolerance=50.0, offset_ratio=None)
    p2, r2, f2, _ = mtran.precision_recall_f1_overlap(
        ref_iv, ref_p, est_iv, est_p,
        onset_tolerance=0.05, pitch_tolerance=50.0,
        offset_ratio=0.2, offset_min_tolerance=0.05)
    return {"onset_P": float(p1), "onset_R": float(r1), "onset_F1": float(f1),
            "note_P": float(p2), "note_R": float(r2), "note_F1": float(f2),
            "n_ref": int(len(ref_p)), "n_est": int(len(est_p))}


def oracle_check(gt_events):
    """评测链路自检：GT 当预测必须 1.0，并检查对抖动/打乱的敏感度。

    参数:
        gt_events (list): GT 音符事件。

    返回:
        dict: 各用例的 onset-only F1。
    """
    ref_iv, ref_p = to_arrays(gt_events)
    rng = np.random.default_rng(0)
    cases = {
        "exact": (ref_iv, ref_p),
        "jitter32ms": (ref_iv + 0.032, ref_p),
        "jitter64ms": (ref_iv + 0.064, ref_p),
        "shuffle_pitch": (ref_iv, ref_p[rng.permutation(len(ref_p))]),
    }
    scores = {}
    for name, (iv, pp) in cases.items():
        scores[name] = note_metrics(ref_iv, ref_p, iv, pp)["onset_F1"]
    return scores
