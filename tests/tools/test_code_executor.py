# Copyright (c) 2026-present, Yumeow. Licensed under the MIT License.
"""代码执行工具的测试。"""

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

    def test_forbidden_eval(self):
        """测试禁止 eval 调用。"""
        result = self.tool.execute('eval("1+1")')
        assert result["success"] is False
        assert "禁止调用函数" in result["error"]

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

    def test_injected_data_list(self):
        """测试通过工具上下文注入数据并预置 data_list。"""
        tool = CodeExecutorTool(sandbox_data=[[1, 2], [3, 4]])
        result = tool.execute("print(data_list[0])\nprint(len(data))")
        assert result["success"] is True
        assert "[1, 2]" in result["output"]
        assert "2" in result["output"]
        assert result["injected_data"] is True

    def test_injected_data_stdin_compatible(self):
        """测试兼容 SR-Scientist 的 sys.stdin 读取模式。"""
        tool = CodeExecutorTool(sandbox_data=[[1, 2], [3, 4]])
        program = """
import json
import sys

input_data_str = sys.stdin.read()
rows = json.loads(input_data_str)
print(rows[-1])
"""
        result = tool.execute(program)
        assert result["success"] is True
        assert "[3, 4]" in result["output"]

    def test_injected_data_names_are_protected(self):
        """测试禁止 LLM 用样例数据覆盖注入数据变量。"""
        tool = CodeExecutorTool(sandbox_data=[[1, 2], [3, 4]])
        result = tool.execute("data = {'sample': True}\nprint(data)")
        assert result["success"] is False
        assert result["status"] == "security_error"
        assert "禁止覆盖沙盒注入变量" in result["error"]

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
        assert self.tool.metadata.category == "computation"
