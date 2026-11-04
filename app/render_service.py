"""verovio 渲染服务。

为什么单独一层：verovio 的字体初始化**不是线程安全的**——如果第一个 toolkit 在
非主线程（FastAPI 的工作线程）里创建，会报「Bravura font could not be loaded」并导致
loadFile 返回 False。实测约：

    主线程创建 toolkit → 正常；工作线程复用同一个 toolkit → 正常；
    工作线程自己创建 toolkit → 失败。

所以这里在模块导入时（主线程）创建一个全局 toolkit，之后所有渲染请求都用锁串行复用它。
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import threading

import verovio

from piano2.render import _require_rsvg, _strip_smufl_text

_lock = threading.Lock()
# 导入时（主线程）就初始化：verovio 在非主线程首次创建 toolkit 会加载失败
_toolkit = verovio.toolkit()


def _get_toolkit():
    """取全局 toolkit（首次调用时在主线程创建；之后复用）。"""
    global _toolkit
    if _toolkit is None:
        _toolkit = verovio.toolkit()
    return _toolkit


def _apply_options(tk, page_width=2200, scale=48):
    """应用与渲染一致的页面选项（页数会随 pageWidth/scale 变化，必须统一）。"""
    tk.setOptions({
        "pageWidth": page_width,
        "scale": scale,
        "adjustPageHeight": True,
        "footer": "none",
        "header": "none",
        "breaks": "auto",
    })


def page_count(xml_path, page_width=2200, scale=48):
    """返回 MusicXML 的页数（用与渲染相同的页面选项）。"""
    tk = _get_toolkit()
    with _lock:
        # 必须先设选项再 loadFile：页数按选项重新布局后才准确
        _apply_options(tk, page_width=page_width, scale=scale)
        if not tk.loadFile(str(xml_path)):
            raise RuntimeError(f"verovio 无法加载: {xml_path}")
        return int(tk.getPageCount())


def render_png(xml_path, png_path, page=1, page_width=2200, scale=48, zoom=1.6):
    """把 MusicXML 的指定页渲染成 PNG（复用全局 toolkit）。

    参数:
        xml_path (str): MusicXML 路径。
        png_path (str): 输出 PNG 路径。
        page (int): 页码（从 1 开始）。
        page_width (int): verovio 页面宽度。
        scale (int): verovio 缩放。
        zoom (float): rsvg-convert 放大倍率。

    返回:
        tuple[int, int]: PNG 像素宽高。

    异常:
        RuntimeError: 加载失败、页码越界或渲染失败。
    """
    _require_rsvg()
    tk = _get_toolkit()
    with _lock:
        # 先设选项再加载，页数才与渲染一致
        _apply_options(tk, page_width=page_width, scale=scale)
        if not tk.loadFile(str(xml_path)):
            raise RuntimeError(f"verovio 无法加载: {xml_path}")
        _apply_options(tk, page_width=page_width, scale=scale)
        n_pages = tk.getPageCount()
        if page < 1 or page > n_pages:
            raise RuntimeError(f"页码 {page} 超出范围（共 {n_pages} 页）")
        svg = _strip_smufl_text(tk.renderToSVG(page))

    os.makedirs(os.path.dirname(os.path.abspath(png_path)) or ".", exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        svg_path = os.path.join(tmp, "page.svg")
        with open(svg_path, "w", encoding="utf-8") as fh:
            fh.write(svg)
        proc = subprocess.run(["rsvg-convert", "--zoom", str(zoom), "-o", str(png_path), svg_path],
                              capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"rsvg-convert 失败: {proc.stderr.strip()}")
    return _png_size(png_path)


def _png_size(png_path):
    """读取 PNG 宽高（避免依赖 PIL）。"""
    with open(png_path, "rb") as fh:
        head = fh.read(24)
    if head[:8] != b"\x89PNG\r\n\x1a\n":
        raise RuntimeError(f"不是合法 PNG: {png_path}")
    width = int.from_bytes(head[16:20], "big")
    height = int.from_bytes(head[20:24], "big")
    return width, height
