"""绘图风格：中文字体、配色、保存工具。

为什么单独成文件：容器默认没有 CJK 字体时，matplotlib 会把中文画成方块而且不报错。
画图前必须先调用 `use_cjk_font()`，并在找不到字体时直接抛异常，把问题暴露在配置阶段。

用法：
    import plot_style
    plot_style.use_cjk_font()
"""

from __future__ import annotations

import os
import shutil

# 先清掉 matplotlib 的字体缓存，否则新装的中文字体不会被发现。
# 必须在导入 pyplot 之前做。
import matplotlib  # noqa: E402

shutil.rmtree(matplotlib.get_cachedir(), ignore_errors=True)

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager, rcParams  # noqa: E402

# Debian/Ubuntu 的 fonts-noto-cjk 装进来的字体名，按优先级排列
CJK_FONT_CANDIDATES = (
    "Noto Sans CJK SC",
    "WenQuanYi Zen Hei",
    "Noto Sans CJK JP",
    "Noto Sans CJK",
    "Noto Serif CJK SC",
    "Noto Serif CJK JP",
)

# 统一配色（参考 daylight 项目的低饱和风格）
COLOR_DARK = "#2c3e50"
COLOR_BLUE = "#2980b9"
COLOR_LIGHT_BLUE = "#aed6f1"
COLOR_ORANGE = "#e67e22"
COLOR_LIGHT_ORANGE = "#fad7a0"
COLOR_GREEN = "#27ae60"
COLOR_LIGHT_GREEN = "#a9dfbf"
COLOR_RED = "#c0392b"
COLOR_LIGHT_RED = "#f5b7b1"
COLOR_GRAY = "#7f8c8d"
COLOR_BG_BLUE = "#eaf2f8"
COLOR_BG_ORANGE = "#fef5e7"
COLOR_BG_GREEN = "#eafaf1"
COLOR_BG_RED = "#fdedec"


def use_cjk_font() -> str:
    """把 matplotlib 配置成能显示中文。

    参数:
        无。

    返回:
        str: 实际使用的字体名。

    异常:
        RuntimeError: 系统里找不到任何 CJK 字体（此时画出来的中文是方块，必须显式失败）。
    """
    available = {font.name for font in font_manager.fontManager.ttflist}
    for candidate in CJK_FONT_CANDIDATES:
        if candidate in available:
            rcParams["font.sans-serif"] = [candidate, "DejaVu Sans"]
            rcParams["font.family"] = "sans-serif"
            # 中文字体多数没有 Unicode 负号，不关掉会让负号变方块
            rcParams["axes.unicode_minus"] = False
            rcParams["figure.dpi"] = 110
            rcParams["savefig.dpi"] = 150
            return candidate
    raise RuntimeError(
        "找不到 CJK 字体。安装方式：apt-get install -y fonts-noto-cjk"
    )


def save(fig, path: str, dpi: int = 150) -> None:
    """保存图片但不关闭 figure（便于同一张图存多个路径）。

    参数:
        fig (matplotlib.figure.Figure): 图对象。
        path (str): 输出路径。
        dpi (int): 保存分辨率。
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")


def save_and_close(fig, path: str, dpi: int = 150) -> None:
    """保存图片并关闭 figure。

    参数:
        fig (matplotlib.figure.Figure): 图对象。
        path (str): 输出路径。
        dpi (int): 保存分辨率。
    """
    save(fig, path, dpi=dpi)
    import matplotlib.pyplot as plt  # 局部导入，避免循环依赖
    plt.close(fig)
