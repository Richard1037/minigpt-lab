"""minigpt —— 从零搭一个能生成文本的小 Transformer。

分阶段学习路线见根目录 README.md；每阶段的原理与验收说明见 docs/stage-guide.md。
本包只依赖 numpy 与标准库。
"""

from .config import Config
from .toolkit import Report, attention_row, print_progress, scatter, sparkline, timer

__version__ = "0.1.0"

__all__ = [
    "Config",
    "Report",
    "attention_row",
    "print_progress",
    "scatter",
    "sparkline",
    "timer",
    "__version__",
]
