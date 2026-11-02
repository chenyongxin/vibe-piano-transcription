"""piano2.inference：音频 → CQT → 模型概率。

输入：wav 路径（任意采样率，内部重采样到 16 kHz）。
输出：三个 (T, 88) 概率矩阵（onset / frame / velocity）。
"""

from __future__ import annotations

import librosa
import numpy as np
import torch

from piano2.constants import BINS_PER_OCTAVE, FMIN, HOP, N_BINS, SR
from piano2.model import PianoModelV2


def compute_cqt(audio_path):
    """计算与训练一致的 log-CQT。

    参数:
        audio_path (str): 音频路径（任意采样率）。

    返回:
        numpy.ndarray: (T, 264) float32，dB 域，范围约 [-80, 0]。
    """
    y, _ = librosa.load(audio_path, sr=SR, mono=True)
    cqt_complex = librosa.cqt(
        y, sr=SR, hop_length=HOP, fmin=FMIN,
        n_bins=N_BINS, bins_per_octave=BINS_PER_OCTAVE,
    )
    return librosa.amplitude_to_db(np.abs(cqt_complex), ref=np.max).T.astype(np.float32)


def normalize_input(cqt):
    """把 dB 值 [-80, 0] 映射到 [0, 1]（与数据 v2 的训练输入一致）。"""
    return (np.clip(cqt, -80.0, 0.0) + 80.0) / 80.0


def load_model(checkpoint, device="cuda"):
    """加载五头模型权重。

    参数:
        checkpoint (str): 检查点路径；支持纯 state_dict（旧格式）或 train.py 保存的字典。
        device (str): "cuda" 或 "cpu"。

    返回:
        piano2.model.PianoModelV2: eval 模式的模型。

    异常:
        FileNotFoundError: 检查点不存在。
    """
    import os

    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"找不到模型检查点: {checkpoint}")
    model = PianoModelV2().to(device)
    payload = torch.load(checkpoint, map_location=device)
    state = payload.get("model", payload) if isinstance(payload, dict) else payload
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def infer_sliding_window(model, cqt, device="cuda", window=2048, stride=1024):
    """滑窗推理并做重叠平均。

    参数:
        model (PianoModelV2): 模型。
        cqt (numpy.ndarray): (T, 264) log-CQT（未归一化，函数内部会归一化）。
        device (str): 设备。
        window (int): 窗口帧数。
        stride (int): 步长帧数。

    返回:
        tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]: onset/frame/velocity 概率，形状 (T, 88)。
    """
    x = torch.tensor(normalize_input(cqt), dtype=torch.float32).unsqueeze(0).to(device)
    total = x.shape[1]
    out_o = torch.zeros((1, total, 88))
    out_f = torch.zeros((1, total, 88))
    out_v = torch.zeros((1, total, 88))
    count = torch.zeros((1, total, 1))
    for start in range(0, total, stride):
        end = min(start + window, total)
        chunk = x[:, start:end, :]
        if chunk.shape[1] == 0:
            continue
        o, f, v, _p, _po = model(chunk)  # 模型 v2 有 5 个头，这里只用前 3 个
        n = end - start
        out_o[:, start:end, :] += torch.sigmoid(o[:, :n, :]).cpu()
        out_f[:, start:end, :] += torch.sigmoid(f[:, :n, :]).cpu()
        out_v[:, start:end, :] += torch.sigmoid(v[:, :n, :]).cpu()
        count[:, start:end, :] += 1.0
    count[count == 0] = 1.0
    return ((out_o / count).squeeze(0).numpy(),
            (out_f / count).squeeze(0).numpy(),
            (out_v / count).squeeze(0).numpy())
