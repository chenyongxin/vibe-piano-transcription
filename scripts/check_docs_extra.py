"""文档质量专项检查：同文档重复图片、图片相对链接是否存在。

check_docs.js 负责 Markdown 渲染规则；本脚本补它不查的两项：
    1. 同一篇文档里重复引用同一张图（读者会以为内容重复）；
    2. 图片相对链接指向的文件不存在（坏链）。

用法（仓库根目录）：
    python scripts/check_docs_extra.py

返回码：0 表示没有问题，1 表示发现问题（信息打印在 stdout）。
"""

from __future__ import annotations

import os
import re
import sys

IMAGE_PATTERN = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")
SKIP_DIRS = {".git", "node_modules", "outputs", "experiments/artifacts"}


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
        for name in files:
            if name.endswith(".md"):
                paths.append(os.path.join(root, name))
    return paths


def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    problems = []
    for path in iter_markdown(repo_root):
        text = open(path, encoding="utf-8").read()
        links = IMAGE_PATTERN.findall(text)
        rel = os.path.relpath(path, repo_root)
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

    if problems:
        print("发现 %d 个问题：" % len(problems))
        for item in problems:
            print(" -", item)
        return 1
    print("文档专项检查通过：无重复图片、无坏链")
    return 0


if __name__ == "__main__":
    sys.exit(main())
