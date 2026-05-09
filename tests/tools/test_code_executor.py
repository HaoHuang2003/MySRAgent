# Copyright (c) 2026-present, Yumeow. Licensed under the MIT License.
"""代码执行工具的测试。"""

import numpy as np

from src.sr_agent.tools.code_executor import CodeExecutorTool


class TestCodeExecutorTool:
    """测试代码执行工具。"""

    def setup_method(self):
        """设置测试夹具。"""
        self.tool = CodeExecutorTool()

    def test_basic_print(self):
        """测试基本打印输出。"""
        result = self.tool.execute('print("Hello, World!")')
        assert result["success"] is True
        assert "Hello, World!" in result["output"]
        assert result["error"] == ""

    def test_numpy_computation(self):
        """测试 numpy 计算。"""
        program = """
import numpy as np
arr = np.array([1, 2, 3, 4, 5])
print(f"Mean: {np.mean(arr)}")
print(f"Std: {np.std(arr):.4f}")
"""
        result = self.tool.execute(program)
        assert result["success"] is True
        assert "Mean: 3.0" in result["output"]

    def test_math_operations(self):
        """测试 math 模块操作。"""
        program = """
import math
print(math.sqrt(16))
print(math.sin(math.pi / 2))
"""
        result = self.tool.execute(program)
        assert result["success"] is True
        assert "4.0" in result["output"]

    def test_math_eval_literal_is_allowed(self):
        """测试允许安全的字面量数学表达式 eval。"""
        result = self.tool.execute(
            """
import math
x1 = 2.0
x2 = 0.5
print(eval("x1 + sin(x2)", {"__builtins__": {}}, {"x1": x1, "x2": x2, "sin": math.sin}))
"""
        )
        assert result["success"] is True
        assert result["error"] == ""

    def test_numpy_math_eval_literal_is_allowed(self):
        """测试允许 numpy 白名单数学函数 eval。"""
        result = self.tool.execute(
            """
import numpy as np
x1 = np.array([1.0, 2.0])
x2 = np.array([3.0, 4.0])
print(eval("x1 + np.sin(x2)", {"__builtins__": {}}, {"x1": x1, "x2": x2, "np": np}))
"""
        )
        assert result["success"] is True
        assert result["error"] == ""

    def test_dynamic_math_eval_is_allowed(self):
        """测试动态字符串 eval 会在运行时执行数学表达式校验。"""
        result = self.tool.execute('expr = "1+1"\nprint(eval(expr))')
        assert result["success"] is True
        assert "2" in result["output"]

    def test_dynamic_unsafe_eval_is_forbidden_at_runtime(self):
        """测试动态危险 eval 表达式会在运行时被拒绝。"""
        result = self.tool.execute('expr = "__import__(\'os\')"\nprint(eval(expr))')
        assert result["success"] is False
        assert "eval 数学表达式不安全" in result["error"]

    def test_indirect_eval_is_forbidden(self):
        """测试禁止把 eval 赋值后间接调用。"""
        result = self.tool.execute('e = eval\nprint(e("1+1"))')
        assert result["success"] is False
        assert "eval 只能直接调用静态字符串数学表达式" in result["error"]

    def test_nested_eval_is_forbidden(self):
        """测试 eval 字符串内部禁止嵌套 eval。"""
        result = self.tool.execute("""eval("eval('1+1')")""")
        assert result["success"] is False
        assert "禁止嵌套调用：eval" in result["error"]

    def test_nested_exec_is_forbidden(self):
        """测试 eval 字符串内部禁止嵌套 exec。"""
        result = self.tool.execute("""eval("exec('x=1')")""")
        assert result["success"] is False
        assert "禁止嵌套调用：exec" in result["error"]

    def test_eval_dunder_escape_is_forbidden(self):
        """测试 eval 字符串内部禁止对象模型逃逸。"""
        result = self.tool.execute("""eval("().__class__")""")
        assert result["success"] is False
        assert "禁止的表达式节点" in result["error"] or "禁止访问双下划线属性" in result["error"]

    def test_eval_non_math_expression_is_forbidden(self):
        """测试 eval 字符串内部只允许算术表达式。"""
        result = self.tool.execute("""eval("x1 > 0", {"__builtins__": {}}, {"x1": 1})""")
        assert result["success"] is False
        assert "禁止的表达式节点：Compare" in result["error"]

        result = self.tool.execute("""eval("'not math'")""")
        assert result["success"] is False
        assert "禁止的常量类型：str" in result["error"]

    def test_type_and_hasattr_builtins_are_allowed(self):
        """测试常用数据探测内置函数可用。"""
        result = self.tool.execute(
            """
value = [1, 2, 3]
print(type(value).__name__)
print(hasattr(value, "__len__"))
"""
        )
        assert result["success"] is True
        assert "list" in result["output"]
        assert "True" in result["output"]

    def test_forbidden_os_module(self):
        """测试禁止导入 os 模块。"""
        result = self.tool.execute('import os')
        assert result["success"] is False
        assert "禁止导入模块" in result["error"]

    def test_forbidden_subprocess(self):
        """测试禁止导入 subprocess 模块。"""
        result = self.tool.execute('import subprocess')
        assert result["success"] is False
        assert "禁止导入模块" in result["error"]

    def test_unauthorized_module(self):
        """测试未授权模块被拒绝。"""
        result = self.tool.execute('import pandas as pd')
        assert result["success"] is False
        assert "未授权的模块" in result["error"]

    def test_scipy_module_is_allowed(self):
        """测试 scipy 数值计算模块可用。"""
        result = self.tool.execute(
            """
from scipy import stats
print(f"{stats.pearsonr([1, 2, 3], [1, 2, 4]).statistic:.4f}")
"""
        )
        assert result["success"] is True
        assert "0.9820" in result["output"]

    def test_traceback_module_is_allowed(self):
        """测试 traceback 可用于用户代码内部调试。"""
        result = self.tool.execute(
            """
import traceback
try:
    1 / 0
except Exception:
    print(traceback.format_exc().splitlines()[-1])
"""
        )
        assert result["success"] is True
        assert "ZeroDivisionError" in result["output"]

    def test_syntax_error(self):
        """测试语法错误处理。"""
        result = self.tool.execute('print("missing quote')
        assert result["success"] is False
        assert "语法错误" in result["error"]

    def test_runtime_error(self):
        """测试运行时错误处理。"""
        result = self.tool.execute('print(1 / 0)')
        assert result["success"] is False
        assert "ZeroDivisionError" in result["error"]
        assert result["status"] == "runtime_error"

    def test_timeout(self):
        """测试死循环会被沙盒超时终止。"""
        result = self.tool.execute("while True:\n    pass", timeout=1)
        assert result["success"] is False
        assert result["status"] == "timeout"
        assert "超时" in result["error"]

    def test_forbidden_dunder_escape(self):
        """测试禁止通过双下划线属性枚举运行时对象。"""
        result = self.tool.execute("print((1).__class__)")
        assert result["success"] is False
        assert result["status"] == "security_error"

    def test_output_truncation(self):
        """测试超大输出会被截断而不是撑爆内存。"""
        result = self.tool.execute('print("x" * 70000)')
        assert result["success"] is True
        assert "[output truncated]" in result["output"]
        assert "输出超过限制" in result["error"]

    def test_context_data_is_available_from_stdin_as_dict(self):
        """测试工具上下文数据会作为 JSON dict 写入 stdin。"""
        tool = CodeExecutorTool(
            sandbox_data={
                "x1": np.array([1.0, 2.0, 3.0]),
                "x2": np.array([0.5, 1.5, 2.5]),
                "y": np.array([0.5, 0.5, 0.5]),
            }
        )
        program = """
import json
import sys

input_data_str = sys.stdin.read()
data_dict = json.loads(input_data_str)
print(sorted(data_dict))
print(data_dict["x1"][0])
print(data_dict["y"][-1])
"""
        result = tool.execute(program)
        assert result["success"] is True
        assert "['x1', 'x2', 'y']" in result["output"]
        assert "1.0" in result["output"]
        assert "0.5" in result["output"]
        assert result["stdin_data"] is True

    def test_stdin_loader_can_use_llm_chosen_variable_names(self):
        """测试 LLM 可以自行命名 stdin 解析后的变量。"""
        tool = CodeExecutorTool(sandbox_data={"x": [1, 2], "y": [3, 4]})
        program = """
import json
import sys

data_str = sys.stdin.read()
data_dict = json.loads(data_str)
data = {"sample": True}
print(data_dict["y"][-1])
print(data)
"""
        result = tool.execute(program)
        assert result["success"] is True
        assert "4" in result["output"]
        assert "{'sample': True}" in result["output"]

    def test_multiple_operations(self):
        """测试复杂计算。"""
        program = """
import numpy as np
import math

x = np.linspace(0, 2 * math.pi, 5)
y = np.sin(x)

print(f"x: {x}")
print(f"sin(x): {y}")
print(f"sum: {np.sum(y)}")
"""
        result = self.tool.execute(program)
        assert result["success"] is True
        assert "sin(x):" in result["output"]

    def test_builtin_functions(self):
        """测试内置函数可用性。"""
        program = """
data = [1, 2, 3, 4, 5]
print(f"Sum: {sum(data)}")
print(f"Max: {max(data)}")
print(f"Min: {min(data)}")
print(f"Length: {len(data)}")
"""
        result = self.tool.execute(program)
        assert result["success"] is True
        assert "Sum: 15" in result["output"]
        assert "Max: 5" in result["output"]

    def test_tool_metadata(self):
        """测试工具元数据。"""
        assert self.tool.metadata.name == "code_executor"
