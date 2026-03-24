# Symbolic Regression Skill

符号回归（Symbolic Regression）专家技能包，包含方法论、最佳实践和公式模式库。

## 技能包结构

```
symbolic-regression/
├── SKILL.md              # 本文档：核心方法论和使用指南
├── gotchas.md            # 常见错误和避坑指南
│
├── references/
│   └── operators.md      # nd2py 支持的运算符参考
│
├── patterns/
│   ├── linear.yaml       # 线性公式模式
│   ├── polynomial.yaml   # 多项式公式模式
│   └── trigonometric.yaml # 三角函数模式
│
└── scripts/
    └── residual_analyzer.py  # 残差分析脚本（待实现）
```

## 推荐工作流程

### 1. 数据探索阶段

首先调用 `statistics_analysis` 工具了解数据分布：

```
调用工具：statistics_analysis
目的：获取 min, max, mean, std, median 等统计量
关注点：
- 数据是否存在异常值？
- 分布是否对称？
- 量纲范围是什么？
```

### 2. 基础拟合阶段

使用 `polynomial_fit` 探索基础关系：

```
调用工具：polynomial_fit
参数建议：
- max_degree: 从 2 开始，逐步增加
- include_interactions: True（允许交叉项）
关注点：
- R² 是否达到可接受水平？
- 哪些项的 p_value < 0.05（显著）？
- 残差是否随机分布？
```

### 3. 假设提出阶段

根据数据特征提出公式假设：

| 数据特征 | 可能模式 | 参考文件 |
|----------|----------|----------|
| 单调递增/递减 | 线性、多项式 | `patterns/linear.yaml` |
| 周期性波动 | 三角函数 | `patterns/trigonometric.yaml` |
| 指数增长/衰减 | 指数、对数 | 参考 `references/operators.md` |
| S 型曲线 | sigmoid、tanh | 参考 `references/operators.md` |

### 4. 公式验证阶段

使用 `evaluate_formula` 验证假设：

```
调用工具：evaluate_formula
参数：
- eq: 公式字符串（使用 ** 表示幂运算！）
- fit: True（自动优化参数）
判断标准：
- R² > 0.99：优秀
- R² > 0.95：良好
- R² < 0.90：需要重新假设
```

### 5. 迭代优化

如果公式不够理想：
1. 分析残差模式（系统性偏差 → 缺少某项）
2. 参考 `patterns/` 中的其他模式
3. 尝试组合多个模式
4. 回到步骤 2 调整多项式拟合

## 工具调用指南

| 工具名 | 何时使用 | 关键参数 | 注意事项 |
|--------|----------|----------|----------|
| `statistics_analysis` | 任务开始时 | x_vars: 可选特征列表 | 关注异常值和分布形态 |
| `polynomial_fit` | 探索阶段 | max_degree, include_interactions | 从低阶开始，防止过拟合 |
| `evaluate_formula` | 验证阶段 | eq, fit=True | 公式用 `**` 不是 `^` |
| `code_executor` | 自定义计算 | program: Python 代码 | 只能使用白名单模块 |
| `skill_document` | 查阅文档 | operation: list/read | 本 Skill 文档也可通过这个工具查阅 |

## 可用资源

### 公式模式库 (`patterns/`)
- `linear.yaml` - 线性关系模式
- `polynomial.yaml` - 多项式关系模式
- `trigonometric.yaml` - 三角函数/周期性模式

### 运算符参考 (`references/`)
- `operators.md` - nd2py 支持的所有数学运算

### 常见错误 (`gotchas.md`)
- 10 个符号回归常见错误及解决方案

## 使用建议

1. **奥卡姆剃刀原则**：在拟合同样的情况下，优先选择更简单的公式
2. **量纲一致性**：检查公式的物理/数学意义
3. **参数可解释性**：避免过度复杂的嵌套结构
4. **交叉验证**：如果可能，用不同数据段验证公式泛化能力

## 关于本 Skill

本 Skill 包含符号回归领域的专家经验，通过 `skill_document` 工具可以随时查阅。

当你遇到具体问题时，可以使用：
```
skill_document(operation="read", skill_name="symbolic-regression", file_path="gotchas.md")
```
