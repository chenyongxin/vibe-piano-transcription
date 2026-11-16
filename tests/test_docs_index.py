"""文档门禁：确保总览索引覆盖全部编号文档。"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = REPO_ROOT / "docs"
OVERVIEW = DOCS_DIR / "01_overview.md"


def test_overview_lists_every_numbered_doc():
    """docs/NN_*.md 必须出现在 docs/01_overview.md 的文档索引中。"""
    text = OVERVIEW.read_text(encoding="utf-8")
    numbered = sorted(p.name for p in DOCS_DIR.glob("[0-9][0-9]_*.md"))
    missing = [name for name in numbered if name not in text]
    assert not missing, f"docs/01_overview.md 缺少文档索引: {missing}"
