"""piano2.inference：音频 → CQT → 模型概率。

输入：wav 路径（任意采样率，内部重采样到 16 kHz）。
输出：三个 (T, 88) 概率矩阵（onset / frame / velocity）。
"""

from __future__ import annotations

import os

import librosa
import numpy as np
import torch

from piano2.constants import BINS_PER_OCTAVE, FMIN, HOP, N_BINS, SR
from piano2.model import PianoModelV2


def pick_device(preferred=None):
    """选择推理设备：环境变量 PIANO2_DEVICE 优先，否则按 CUDA → MPS → CPU 自动检测。

    参数:
        preferred (str | None): 显式指定 "auto" / "cuda" / "mps" / "cpu"；
            None 时读环境变量 PIANO2_DEVICE（默认 "auto"）。

    返回:
        str: 实际可用的设备名（"cuda" / "mps" / "cpu"）。

    异常:
        ValueError: 设备名不在允许集合内。
        RuntimeError: 显式请求了不可用的设备。
    """
    request = (preferred or os.environ.get("PIANO2_DEVICE", "auto")).lower()
    if request not in ("auto", "cuda", "mps", "cpu"):
        raise ValueError(f"未知设备: {request}（可选 auto/cuda/mps/cpu）")
    cuda_ok = torch.cuda.is_available()
    mps_ok = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    if request == "cuda":
        if not cuda_ok:
            raise RuntimeError("请求 cuda，但当前环境不可用")
        return "cuda"
    if request == "mps":
        if not mps_ok:
            raise RuntimeError("请求 mps，但当前环境不可用（容器内通常不可用）")
        return "mps"
    if request == "cpu":
        return "cpu"
    if cuda_ok:
        return "cuda"
    if mps_ok:
        return "mps"
    return "cpu"


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


def load_model(checkpoint, device="auto"):
    """加载五头模型权重。

    参数:
        checkpoint (str): 检查点路径；支持纯 state_dict（旧格式）或 train.py 保存的字典。
        device (str): "auto" / "cuda" / "cpu" / "mps"；auto 时自动检测（见 pick_device）。

    返回:
        piano2.model.PianoModelV2: eval 模式的模型。

    异常:
        FileNotFoundError: 检查点不存在。
    """
    import os

    device = pick_device(device)
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"找不到模型检查点: {checkpoint}")
    model = PianoModelV2().to(device)
    payload = torch.load(checkpoint, map_location=device)
    state = payload.get("model", payload) if isinstance(payload, dict) else payload
    try:
        model.load_state_dict(state)
    except RuntimeError:
        # 兼容旧 5 头检查点：offset 头随机初始化，不影响已有推理路径
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"[!] 检查点缺少 {len(missing)} 个参数（可能是旧 5 头模型），offset 头将未训练")
    model.eval()
    model.has_offset = "fc_offset.weight" in state
    model.has_regression = "fc_onset_reg.weight" in state
    return model


@torch.no_grad()
def infer_sliding_window(model, cqt, device="auto", window=2048, stride=1024,
                         with_offset=False, with_regression=False):
    """滑窗推理并做重叠平均。

    参数:
        model (PianoModelV2): 模型。
        cqt (numpy.ndarray): (T, 264) log-CQT（未归一化，函数内部会归一化）。
        device (str): 设备，默认 "auto"（自动检测）。
        window (int): 窗口帧数。
        stride (int): 步长帧数。
        with_offset (bool): 是否同时返回 offset 头概率（6 头以上模型才有效）。
        with_regression (bool): 是否返回 onset_reg / offset_reg 回归输出（8 头模型才有效）。

    返回:
        tuple: 默认 (onset, frame, velocity)；
            加 with_offset 时追加 offset；
            加 with_regression 时再追加 onset_reg、offset_reg。
    """
    device = pick_device(device)
    x = torch.tensor(normalize_input(cqt), dtype=torch.float32).unsqueeze(0).to(device)
    total = x.shape[1]
    out_o = torch.zeros((1, total, 88))
    out_f = torch.zeros((1, total, 88))
    out_v = torch.zeros((1, total, 88))
    out_of = torch.zeros((1, total, 88))
    out_onr = torch.zeros((1, total, 88))
    out_offr = torch.zeros((1, total, 88))
    count = torch.zeros((1, total, 1))
    for start in range(0, total, stride):
        end = min(start + window, total)
        chunk = x[:, start:end, :]
        if chunk.shape[1] == 0:
            continue
        with torch.autocast(device_type="cuda", enabled=(device == "cuda")):
            o, f, v, _p, _po, of, onr, offr = model(chunk)
            o = torch.sigmoid(o.float())
            f = torch.sigmoid(f.float())
            v = torch.sigmoid(v.float())
            of = torch.sigmoid(of.float())
            onr = torch.sigmoid(onr.float())
            offr = torch.sigmoid(offr.float())
        n = end - start
        out_o[:, start:end, :] += o[:, :n, :].cpu()
        out_f[:, start:end, :] += f[:, :n, :].cpu()
        out_v[:, start:end, :] += v[:, :n, :].cpu()
        out_of[:, start:end, :] += of[:, :n, :].cpu()
        out_onr[:, start:end, :] += onr[:, :n, :].cpu()
        out_offr[:, start:end, :] += offr[:, :n, :].cpu()
        count[:, start:end, :] += 1.0
    count[count == 0] = 1.0
    result = ((out_o / count).squeeze(0).numpy(),
              (out_f / count).squeeze(0).numpy(),
              (out_v / count).squeeze(0).numpy())
    if with_offset:
        result = result + ((out_of / count).squeeze(0).numpy(),)
    if with_regression:
        result = result + ((out_onr / count).squeeze(0).numpy(),
                           (out_offr / count).squeeze(0).numpy())
    return result
