"""M0：把 MusicXML 渲染成 PNG，方便在对话里直接看排版。

技术路线：verovio（MusicXML → SVG）→ rsvg-convert（SVG → PNG）。
为什么不用 MuseScore / LilyPond：它们需要 GUI 或重量级依赖；verovio 是纯命令行，
在容器里就能跑。为什么不用 PyMuPDF 直接转：MuPDF 的 SVG 解析不支持 verovio 用的
CSS class（`<style>` + `class=`），会渲染成纯白。

用法（容器内）：
    python render_preview.py 输入.musicxml 输出.png [页码]
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile

import verovio

# verovio 把乐谱符号（如速度记号的四分音符）写成 SMuFL 私有区字符（U+E000–U+F8FF），
# 依赖内嵌的 Leipzig 字体；rsvg-convert 不加载 data-URI 字体，会渲染成一个巨大的乱码字形。
# 预览时把这些 SMuFL 文本字形去掉（音符头/谱号等本来就是矢量路径，不受影响）。
_PUA_TSPAN = re.compile(r"<tspan[^>]*>[^<]*[\ue000-\uf8ff][^<]*</tspan>")


def _strip_smufl_text(svg):
    """删除 SVG 里依赖内嵌音乐字体的私有区文本，避免渲染成乱码大字。"""
    return _PUA_TSPAN.sub("", svg)


def _require_rsvg():
    """检查 rsvg-convert 是否可用，否则给出可执行的修复提示。"""
    if shutil.which("rsvg-convert") is None:
        raise RuntimeError(
            "缺少 rsvg-convert。安装方式（Ubuntu 容器）：apt-get install -y librsvg2-bin"
        )


def render_png(
    xml_path,
    png_path,
    page=1,
    page_width=2200,
    scale=48,
    zoom=1.6,
    max_height_px=None,
):
    """把 MusicXML 的指定页渲染成 PNG，并校验不是白图。

    参数:
        xml_path (str): MusicXML 路径。
        png_path (str): 输出 PNG 路径。
        page (int): 页码，从 1 开始。
        page_width (int): verovio 页面宽度（内部单位）。
        scale (int): verovio 缩放百分比。
        zoom (float): rsvg-convert 的放大倍率，控制最终像素尺寸。
        max_height_px (int | None): 可选的高度上限（像素），超出则按比例降低 zoom。

    返回:
        tuple[int, int]: PNG 的 (宽, 高)。

    异常:
        RuntimeError: verovio 加载失败、页码越界、渲染结果为白图、或缺少 rsvg-convert。
    """
    _require_rsvg()
    tk = verovio.toolkit()
    ok = tk.loadFile(str(xml_path))
    if not ok:
        raise RuntimeError(f"verovio 无法加载: {xml_path}")
    n_pages = tk.getPageCount()
    if page < 1 or page > n_pages:
        raise RuntimeError(f"页码 {page} 超出范围（共 {n_pages} 页）")
    tk.setOptions(
        {
            "pageWidth": page_width,
            "scale": scale,
            "adjustPageHeight": True,
            "footer": "none",
            "header": "none",
            "breaks": "auto",
        }
    )
    svg = tk.renderToSVG(page)
    svg = _strip_smufl_text(svg)
    os.makedirs(os.path.dirname(os.path.abspath(png_path)) or ".", exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        svg_path = os.path.join(tmp, "page.svg")
        with open(svg_path, "w", encoding="utf-8") as fh:
            fh.write(svg)
        cmd = ["rsvg-convert", "--zoom", str(zoom), "-o", str(png_path), svg_path]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"rsvg-convert 失败: {proc.stderr.strip()}")

    width, height = _check_not_blank(png_path)
    if max_height_px and height > max_height_px:
        ratio = max_height_px / float(height)
        return render_png(
            xml_path, png_path, page=page, page_width=page_width, scale=scale,
            zoom=max(zoom * ratio, 0.3), max_height_px=None,
        )
    return width, height


def _check_not_blank(png_path):
    """用 PyMuPDF 打开 PNG 并检查像素方差，防止把白图当成功。"""
    import pymupdf

    doc = pymupdf.open(png_path)
    pix = doc[0].get_pixmap(dpi=72)
    samples = pix.samples
    if not samples:
        raise RuntimeError("渲染结果为空")
    n = len(samples)
    mean = sum(samples) / n
    lo, hi = min(samples), max(samples)
    if hi - lo < 10 or mean > 254.0:
        raise RuntimeError(
            f"渲染结果疑似白图（mean={mean:.1f}, min={lo}, max={hi}），拒绝交付"
        )
    return int(pix.width), int(pix.height)


def render_pages(xml_path, out_dir, prefix, pages=(1,), **kwargs):
    """批量渲染多页，返回 [(png_path, (宽, 高)), ...]。"""
    out = []
    for pg in pages:
        out_path = os.path.join(out_dir, f"{prefix}_p{pg}.png")
        wh = render_png(xml_path, out_path, page=pg, **kwargs)
        out.append((out_path, wh))
    return out


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("用法: python render_preview.py 输入.musicxml 输出.png [页码]")
        sys.exit(1)
    pg = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    print(render_png(sys.argv[1], sys.argv[2], page=pg))
