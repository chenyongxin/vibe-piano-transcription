"""M1：数据 v2 的 Dataset（memmap 随机切片，避免每步解压整首歌）。

数据布局见 `build_data_v2.py`：
    {split}/{name}.cqt.npy   (T, 264) float32
    {split}/{name}.lab.npy   (T, 6, 88) uint8
    {split}/{name}.pedal.npy (T, 2) uint8

用法：
    from dataset_v2 import SegmentDataset
    ds = SegmentDataset("train", augment=True)
"""

from __future__ import annotations

import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from piano2.constants import DATA_ROOT  # noqa: E402
SEG_FRAMES = 313  # 10 秒 @ 31.25 fps


class SegmentDataset(Dataset):
    """按 10 秒段索引读取 CQT 与标签 v2。

    参数:
        split (str): "train" / "validation" / "test"。
        label_channel (str): onset 标签变体，可选 "onset1" / "onset2" / "onset3"。
        augment (bool): 是否做简单数据增强（当前只有 dB 增益抖动）。
        limit (int | None): 只取前 N 个段（调试或固定验证集用）。
        data_root (str): 数据根目录。
    """

    LABEL_INDEX = {"onset1": 1, "onset2": 2, "onset3": 3}

    def __init__(self, split, label_channel="onset2", augment=False, limit=None,
                 data_root=DATA_ROOT, aug_gain=3.0, aug_freq_mask=0, aug_time_mask=0):
        """初始化。

        参数:
            split (str): "train" / "validation" / "test"。
            label_channel (str): onset 标签变体，可选 "onset1" / "onset2" / "onset3"。
            augment (bool): 是否做数据增强（增益抖动 + 可选频率/时间遮蔽）。
            limit (int | None): 只取前 N 个段（调试或固定验证集用）。
            data_root (str): 数据根目录。
            aug_gain (float): 增益抖动的最大幅度（dB，±aug_gain）。
            aug_freq_mask (int): 频率遮蔽的条数（每条随机宽 4–16 bins）。
            aug_time_mask (int): 时间遮蔽的条数（每条随机宽 4–16 帧）。
        """
        with open(os.path.join(data_root, split, "segments.json"), encoding="utf-8") as fh:
            self.segments = json.load(fh)
        if limit:
            self.segments = self.segments[:limit]
        self.root = os.path.join(data_root, split)
        self.augment = augment
        self.label_idx = self.LABEL_INDEX[label_channel]
        self.aug_gain = float(aug_gain)
        self.aug_freq_mask = int(aug_freq_mask)
        self.aug_time_mask = int(aug_time_mask)

    def __len__(self):
        return len(self.segments)

    def _load(self, name, start, n):
        """读取某个文件的一段（memmap 切片后转成内存数组）。"""
        arr = np.load(os.path.join(self.root, name), mmap_mode="r")
        seg = np.asarray(arr[start:start + n])
        return seg

    def __getitem__(self, index):
        seg = self.segments[index]
        name = seg["file"]
        base = name[: -len(".cqt.npy")]
        start, n = seg["start"], seg["n"]

        cqt = self._load(name, start, n).astype(np.float32)
        lab = self._load(base + ".lab.npy", start, n)
        pedal = self._load(base + ".pedal.npy", start, n)

        # 数据增强：增益抖动（模拟录音电平差异）+ 可选的频率/时间遮蔽（SpecAugment 风格）
        if self.augment:
            cqt = cqt + np.float32(np.random.uniform(-self.aug_gain, self.aug_gain))
        cqt = np.clip(cqt, -80.0, 0.0)
        # 输入归一化：把 dB 范围 [-80, 0] 映射到 [0, 1]
        x = (cqt + 80.0) / 80.0
        if self.augment:
            for _ in range(self.aug_freq_mask):
                width = int(np.random.randint(4, 17))
                start = int(np.random.randint(0, max(1, x.shape[1] - width)))
                x[:, start:start + width] = 0.0
            for _ in range(self.aug_time_mask):
                width = int(np.random.randint(4, 17))
                start = int(np.random.randint(0, max(1, x.shape[0] - width)))
                x[start:start + width, :] = 0.0

        pad = SEG_FRAMES - n
        if pad > 0:
            x = np.pad(x, ((0, pad), (0, 0)))
            lab = np.pad(lab, ((0, pad), (0, 0), (0, 0)))
            pedal = np.pad(pedal, ((0, pad), (0, 0)))

        onset = lab[:, self.label_idx, :].astype(np.float32)
        frame = lab[:, 0, :].astype(np.float32)
        velocity = lab[:, 4, :].astype(np.float32) / 127.0
        offset = (lab[:, 6, :] if lab.shape[1] >= 7
                  else np.zeros_like(lab[:, 0, :])).astype(np.float32)
        if lab.shape[1] >= 9:
            onset_reg = lab[:, 7, :].astype(np.float32) / 255.0
            offset_reg = lab[:, 8, :].astype(np.float32) / 255.0
        else:
            onset_reg = np.zeros_like(lab[:, 0, :], dtype=np.float32)
            offset_reg = np.zeros_like(lab[:, 0, :], dtype=np.float32)
        pedal_frame = pedal[:, 0:1].astype(np.float32)
        pedal_on = pedal[:, 1:2].astype(np.float32)

        return {
            "x": torch.from_numpy(x),
            "onset": torch.from_numpy(onset),
            "frame": torch.from_numpy(frame),
            "velocity": torch.from_numpy(velocity),
            "offset": torch.from_numpy(offset),
            "onset_reg": torch.from_numpy(onset_reg),
            "offset_reg": torch.from_numpy(offset_reg),
            "pedal": torch.from_numpy(pedal_frame),
            "pedal_on": torch.from_numpy(pedal_on),
            "n_frames": n,
        }


def collate(batch):
    """把一批样本拼成张量字典（形状已在 Dataset 里补齐到 SEG_FRAMES）。"""
    keys = batch[0].keys()
    out = {}
    for key in keys:
        if key == "n_frames":
            out[key] = torch.tensor([b[key] for b in batch])
        else:
            out[key] = torch.stack([b[key] for b in batch])
    return out
