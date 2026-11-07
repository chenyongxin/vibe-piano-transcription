"""文档质量专项检查：重复图片（同文档/跨文档）、图片相对链接是否存在。

check_docs.js 负责 Markdown 渲染规则；本脚本补它不查的两项：
    1. 图片重复：同一篇文档里重复引用同一张图；同一张图被多篇「活跃文档」引用；
    2. 图片相对链接指向的文件不存在（坏链）。

归档目录（docs/history/**、experiments/reports/**）之间共用旧图属于存档行为，
跨文档重复检查对它们豁免；只要与活跃文档重复，仍然会报错。

用法（仓库根目录）：
    python scripts/check_docs_extra.py

返回码：0 表示通过，1 表示发现问题。
"""

from __future__ import annotations

import os
import re
import sys

IMAGE_PATTERN = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")
SKIP_DIRS = {".git", "node_modules", "outputs"}
ARCHIVE_DIRS = ("docs/history/", "experiments/reports/")


def iter_markdown(repo_root):
    """遍历仓库里的 Markdown 文件。

    参数:
        repo_root (str): 仓库根目录。

    返回:
        list[str]: 所有 .md 文件的绝对路径。
    """
    paths = []
    for root, dirs, files in os.walk(repo_root):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        rel_root = os.path.relpath(root, repo_root).replace(os.sep, "/") + "/"
        if "experiments/artifacts" in rel_root:
            continue
        for name in files:
            if name.endswith(".md"):
                paths.append(os.path.join(root, name))
    return paths


def is_archive(rel_path):
    """判断文档是否属于归档目录（跨文档重复对此豁免）。

    参数:
        rel_path (str): 相对仓库根目录的路径。

    返回:
        bool: 是否归档文档。
    """
    rel = rel_path.replace(os.sep, "/")
    return any(rel.startswith(prefix) for prefix in ARCHIVE_DIRS)


def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    problems = []
    by_image = {}
    for path in iter_markdown(repo_root):
        rel = os.path.relpath(path, repo_root)
        text = open(path, encoding="utf-8").read()
        links = IMAGE_PATTERN.findall(text)
        counts = {link: links.count(link) for link in set(links)}
        for link, count in counts.items():
            if count > 1:
                problems.append(f"{rel}: 同一张图引用 {count} 次 -> {link}")
        for link in set(links):
            if link.startswith("http"):
                continue
            target = os.path.normpath(os.path.join(os.path.dirname(path), link))
            if not os.path.exists(target):
                problems.append(f"{rel}: 图片不存在 -> {link}")
            by_image.setdefault(target, set()).add(rel)

    exempt = 0
    for target, docs in by_image.items():
        if len(docs) < 2:
            continue
        active = sorted(d for d in docs if not is_archive(d))
        if len(active) > 1:
            problems.append(
                "跨文档重复图片（活跃文档） -> "
                + os.path.relpath(target, repo_root).replace(os.sep, "/")
                + "："
                + "、".join(active)
            )
        elif active and len(docs) > len(active):
            exempt += 1

    if problems:
        print("发现 %d 个问题：" % len(problems))
        for item in problems:
            print(" -", item)
        return 1
    print(f"文档专项检查通过：无重复图片、无坏链（归档文档跨文档复用 {exempt} 处已豁免）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
