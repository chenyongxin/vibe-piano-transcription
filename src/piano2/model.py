"""M1：模型 v2 —— 在原有 CNN + Transformer 上修正频率池化并增加踏板头。

相对旧模型（`model_industrial.py`）的改动：
    1. 频率池化从 MaxPool(1,2) 改成 MaxPool(1,3)：264 = 88 键 × 3 bins/半音，
       用 3 正好把每个琴键的 3 个 bin 合成一个，频率轴与输出头的 88 键对齐；
       旧的 (1,2) 会让 33% 的池化窗口跨半音。
    2. 输出头从 3 个增加到 5 个：onset / frame / velocity，外加 pedal（踏板状态）
       和 pedal_on（踩下瞬间）。
    3. 输入归一化在 Dataset 里做（dB 映射到 [0,1]），模型不再直接吃 dB 值。

用法：
    m = PianoModelV2()
    o, f, v, p, po = m(torch.randn(2, 100, 264))
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """标准正弦位置编码（给每一帧注入时间位置）。"""

    def __init__(self, d_model, max_len=2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1), :]


class PianoModelV2(nn.Module):
    """CNN + Transformer Encoder + 五个输出头。

    参数:
        input_features (int): 输入频率 bin 数，默认 264。
        hidden_size (int): Transformer 隐藏维度。
        output_features (int): 琴键数，默认 88。
        nhead (int): 注意力头数。
        num_layers (int): Transformer 层数。
        dropout (float): dropout 比例。
    """

    def __init__(self, input_features=264, hidden_size=512, output_features=88,
                 nhead=8, num_layers=4, dropout=0.1):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding="same"),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding="same"),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 3)),  # 264 -> 88，与琴键对齐
            nn.Dropout(0.25),
        )
        freq_after_pool = input_features // 3
        self.cnn_out_dim = 64 * freq_after_pool
        self.input_proj = nn.Linear(self.cnn_out_dim, hidden_size)
        self.pos_encoder = PositionalEncoding(hidden_size, max_len=2048)
        layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=nhead,
                                           dim_feedforward=hidden_size * 2,
                                           dropout=dropout, batch_first=True,
                                           norm_first=True)
        self.transformer_encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.fc_onset = nn.Linear(hidden_size, output_features)
        self.fc_frame = nn.Linear(hidden_size, output_features)
        self.fc_velocity = nn.Linear(hidden_size, output_features)
        self.fc_pedal = nn.Linear(hidden_size, 1)
        self.fc_pedal_on = nn.Linear(hidden_size, 1)
        # 第六个头：offset（键释放事件）。旧 5 头检查点没有这一层，
        # 加载时用 strict=False 兼容（见 inference.load_model）。
        self.fc_offset = nn.Linear(hidden_size, output_features)
        # 回归目标（数据 v4）：帧中心到最近 onset/offset 的时间差三角目标。
        # 旧检查点没有这两层，加载时 strict=False 兼容。
        self.fc_onset_reg = nn.Linear(hidden_size, output_features)
        self.fc_offset_reg = nn.Linear(hidden_size, output_features)

    def forward(self, x):
        """前向：x (B, T, 264) → 八个头的 logits（分类 6 个 + 回归 2 个）。"""
        x = x.unsqueeze(1)          # (B, 1, T, F)
        x = self.cnn(x)             # (B, 64, T, 88)
        b, c, t, f = x.size()
        x = x.permute(0, 2, 1, 3).reshape(b, t, c * f)
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        return (self.fc_onset(x), self.fc_frame(x), self.fc_velocity(x),
                self.fc_pedal(x), self.fc_pedal_on(x), self.fc_offset(x),
                self.fc_onset_reg(x), self.fc_offset_reg(x))


if __name__ == "__main__":
    model = PianoModelV2()
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"参数量: {n:,}")
    out = model(torch.randn(2, 313, 264))
    print([tuple(t.shape) for t in out])
