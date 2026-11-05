"""画图自检工具：确认所有曲线、柱都在坐标范围内（没有被坐标轴裁剪）。

用法（在画图脚本末尾）：
    from fig_utils import check_clipping
    problems = check_clipping(fig)
    if problems:
        raise SystemExit("图形越界：\n" + "\n".join(problems))

为什么需要：matplotlib 默认会把超出 xlim/ylim 的部分裁掉且不报错。
我们曾因此把 55 Hz 滤波器的峰顶裁掉，图上看起来像"这根线没画出来"。
"""

from __future__ import annotations

import numpy as np


def _to_scale(values, scale):
    """把数据值转换到比较用的坐标（log 轴取对数）。"""
    values = np.asarray(values, dtype=float)
    if scale == "log":
        return np.log10(np.maximum(values, 1e-300))
    return values


def check_clipping(fig, verbose=True):
    """检查图中所有坐标轴的曲线和矩形是否超出显示范围。

    参数:
        fig (matplotlib.figure.Figure): 已绘制完成的图。
        verbose (bool): 是否打印统计信息。

    返回:
        list[str]: 越界描述；空列表表示全部在范围内。
    """
    problems = []
    checked = 0
    for index, ax in enumerate(fig.axes):
        xscale, yscale = ax.get_xscale(), ax.get_yscale()
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        bx0, bx1 = _to_scale([x0, x1], xscale)
        by0, by1 = _to_scale([y0, y1], yscale)

        def report(kind, xs, ys):
            nonlocal checked
            checked += 1
            xb = _to_scale(xs, xscale)
            yb = _to_scale(ys, yscale)
            if (np.nanmin(xb) < bx0 - 1e-9 or np.nanmax(xb) > bx1 + 1e-9
                    or np.nanmin(yb) < by0 - 1e-9 or np.nanmax(yb) > by1 + 1e-9):
                problems.append(
                    f"子图 {index + 1} 的{kind}越界: "
                    f"x∈[{np.nanmin(xb):.3g}, {np.nanmax(xb):.3g}] vs "
                    f"[{bx0:.3g}, {bx1:.3g}], "
                    f"y∈[{np.nanmin(yb):.3g}, {np.nanmax(yb):.3g}] vs "
                    f"[{by0:.3g}, {by1:.3g}]"
                )

        for line in ax.get_lines():
            xs, ys = line.get_xdata(), line.get_ydata()
            if len(xs) == 0:
                continue
            report("曲线", xs, ys)

        for patch in ax.patches:
            if not hasattr(patch, "get_x"):
                continue
            x = patch.get_x()
            y = patch.get_y()
            w = patch.get_width()
            h = patch.get_height()
            report("矩形", [x, x + w], [y, y + h])

    if verbose:
        print(f"图形自检：检查 {checked} 个元素，越界 {len(problems)} 个")
    return problems
