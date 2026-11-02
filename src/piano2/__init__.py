"""piano2：钢琴自动转录项目的最终版代码包。

模块划分（对应 `docs/01_overview.md` 的流水线）：
    core        基础类型与常量（NoteEvent、泛音表、采样参数）
    constants   路径与运行配置（可用环境变量覆盖）
    dataset     数据 v2 的段数据集（memmap）
    model       五头模型（CNN + Transformer）
    train       训练循环
    decode      概率图 → 音符事件（最终版解码器）
    notation    音符事件 → MusicXML
    render      MusicXML → PNG 预览
    eval        评测指标（note-level / 谱面可读性）
"""

__version__ = "0.1.0"
