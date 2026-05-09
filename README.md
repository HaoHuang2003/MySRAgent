# SR Agent

符号回归（Symbolic Regression）Agent，通过 LLM 调用工具分析数据并发现数学公式。

## 功能特点

- **数据探索**：自动计算数据的统计特征，帮助理解数据分布
- **公式评估**：使用 nd2py 符号引擎评估公式拟合能力，返回 MSE、R² 等指标
- **参数拟合**：支持 BFGS 算法自动优化公式中的参数
- **可扩展架构**：易于添加新的分析工具和 LLM 接口

## 安装

1. 创建虚拟环境：
```bash
conda create -p ./venv python=3.12 -y
conda activate ./venv  # Unix/Linux/MacOS
./venv/Scripts/activate  # Windows
```

2. 安装依赖：
```bash
pip install -e ".[dev]"
```

3. 配置环境变量：
```bash
cp .env.sample .env
# 编辑 .env，填入你的 API 密钥
```

## 运行测试

详见 [`tests/README.md`](tests/README.md)。

```bash
python -m pytest tests/ -v
```

## 项目结构

```
├── src/sr_agent/       # 核心代码，详见 src/sr_agent/README.md
│   ├── tools/          # 工具定义，详见 src/sr_agent/tools/README.md
│   ├── api/            # LLM 接口
│   ├── parser/         # 工具调用解析器
│   ├── prompts/        # Prompt 模板
│   ├── buffer/         # 消息管理
│   ├── skills/         # 技能文档
│   └── utils/          # 工具函数
├── src/llmsr_bench/    # LLM-SRBench 对接代码
├── tests/              # 单元测试，详见 tests/README.md
├── scripts/            # 临时性脚本和数据分析脚本
├── analysis/           # 数据分析 notebook
├── data/               # 数据文件
├── logs/               # 运行结果日志
└── playground/         # 实验性/临时代码
```

### 目录约定

- **根目录**只放具有明确功能的入口脚本（如 `run_sr_agent.py`、`bench_sr_agent.py`），避免根目录过于杂乱。
- **`scripts/`** 放临时性脚本和数据分析脚本，运行时需指定 `PYTHONPATH`：
  ```bash
  export PYTHONPATH=. && python ./scripts/xxx.py
  ```
- **`analysis/`** 放数据分析 notebook，命名格式为 `YYMMDD_xxx.ipynb`。注意控制 notebook 文件大小，避免撑爆 git 仓库。
- **`data/`** 放数据文件（已加入 `.gitignore`，不会被 git 跟踪）。
- **`logs/`** 放运行结果（已加入 `.gitignore`）。
- **`playground/`** 放不舍得删但代码中用不到的东西（已加入 `.gitignore`）。

## 许可证

MIT License
