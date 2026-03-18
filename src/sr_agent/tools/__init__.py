"""工具模块。

所有工具都应继承自 BaseTool，并实现 execute 方法。
"""

from .base_tool import BaseTool, ToolMetadata
from .statistics import StatisticsTool
from .evaluate import EvaluateTool

__all__ = [
    "BaseTool",
    "ToolMetadata",
    "StatisticsTool",
    "EvaluateTool",
]
