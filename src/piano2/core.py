"""piano2.core：基础类型、常量和历史遗留的泛音抑制函数。

这里放的是多个模块共用的最小集合，避免模块之间互相 import 形成环。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from piano2.constants import FPS, HOP, SR

# 兼容旧代码里用的名字
HOP_LENGTH = HOP
FRAME_RATE = FPS

# 泛音偏移表（半音, 衰减系数）。注意：这是旧版全局抑制用的参数，
# 现在默认解码器（decode.py 的 conditional 模式）只在局部、非累乘地使用它。
HARMONIC_OFFSETS = (
    (12, 0.30),
    (19, 0.45),
    (24, 0.55),
    (28, 0.65),
    (31, 0.75),
    (34, 0.80),
    (36, 0.80),
)


@dataclass
class NoteEvent:
    """一个音符事件。

    属性:
        pitch (int): MIDI 音高（21–108）。
        start (float): 开始时间（秒）。
        end (float): 结束时间（秒）。
        velocity (int): 力度（1–127）。
    """

    pitch: int
    start: float
    end: float
    velocity: int


def frame_to_time(frame_index):
    """把帧号换成秒。

    参数:
        frame_index (int | float): 帧号。

    返回:
        float: 秒。
    """
    return float(frame_index) * HOP / SR


def suppress_harmonic_onsets(onset_probs, confirmed_pitches, strength=0.5):
    """旧版全局谐波抑制（保留作对照，**不要在新流程里使用**）。

    它的两个问题是 M0.5 查明的根因：
        1. 对整首曲子的时间轴一次性相乘，不区分时刻；
        2. 多个锚点会连乘，一个键可能被压到原概率的 7% 左右，导致大量漏音。

    参数:
        onset_probs (numpy.ndarray): (T, 88) onset 概率。
        confirmed_pitches (list[int]): 锚点音高（MIDI）。
        strength (float): 抑制强度 0–1。

    返回:
        numpy.ndarray: 抑制后的 (T, 88) 概率。
    """
    suppressed = onset_probs.copy()
    for pitch in confirmed_pitches:
        key_idx = pitch - 21
        if key_idx < 0:
            continue
        for offset, decay in HARMONIC_OFFSETS:
            harmonic_key = key_idx + offset
            if 0 <= harmonic_key < 88:
                suppressed[:, harmonic_key] *= (1.0 - decay * strength)
    return suppressed
