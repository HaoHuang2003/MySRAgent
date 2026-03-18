"""工具基类定义。

所有工具都应继承自 BaseTool，并提供统一的接口。
"""

from logging import getLogger
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict

from ..utils.factory import FactoryMixin

_logger = getLogger(f'sr_agent.{__name__}')

@dataclass
class ToolMetadata:
    """工具元数据。

    Attributes:
        name: 工具名称，用于 LLM 识别和调用。
        description: 工具描述，说明工具的功能和适用场景。
        category: 工具类别，用于分类管理（如 "statistics", "regression"）。
    """

    name: str
    description: str
    category: str = "default"


class BaseTool(ABC, FactoryMixin):
    """工具基类。

    所有工具都应继承此类，并实现 execute 方法。
    工具的 docstring 会作为给 LLM 的说明。

    Example:
        @BaseTool.register_model('statistics_tool')
        class StatisticsTool(BaseTool):
            '''计算数据的统计量。'''
            def execute(self, *args, **kwargs):
                return {"mean": np.mean(kwargs.get('y'))}
    """

    metadata: ToolMetadata = None

    @abstractmethod
    def execute(self, *args, **kwargs) -> Dict[str, Any]:
        """执行工具。

        Args:
            *args: 传递给工具的参数。
            **kwargs: 传递给工具的关键词参数。

        Returns:
            执行结果字典。
        """
        pass

    def __call__(self, *args, **kwargs) -> Dict[str, Any]:
        """允许像函数一样调用工具。"""
        return self.execute(*args, **kwargs)
