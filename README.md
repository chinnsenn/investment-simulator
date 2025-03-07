# Investment Simulator

一个基于Python的投资模拟器，用于模拟不同投资策略和市场条件下的投资结果。

## 项目概述

Investment Simulator 允许用户模拟各种投资策略（如定投策略、价值平均策略）在不同市场条件下的表现。它支持多种统计分布模型来模拟市场波动，并提供详细的可视化和统计分析。

## 功能特点

- **多种投资策略**：支持定投策略和价值平均策略
- **多种分布模型**：支持正态分布、对数正态分布、t分布等多种收益率分布模型
- **蒙特卡洛模拟**：支持多轮模拟以获得更可靠的结果
- **并行计算**：大规模模拟时自动使用并行计算提高性能
- **可视化报告**：生成详细的投资结果可视化和统计报告
- **灵活配置**：通过配置文件轻松调整模拟参数

## 项目结构

```
investment-simulator/
├── config/                   # 配置模块
│   ├── __init__.py
│   └── investment_config.py  # 投资配置类
├── services/                 # 服务模块
│   ├── __init__.py
│   └── investment_service.py # 投资服务类
├── simulation/               # 模拟模块
│   ├── __init__.py
│   └── simulator.py          # 投资模拟器类
├── strategies/               # 策略模块
│   ├── __init__.py
│   └── investment_strategies.py # 投资策略实现
├── utils/                    # 工具模块
│   ├── __init__.py
│   ├── investment_utils.py   # 投资工具函数
│   ├── rate_distributions.py # 收益率分布模型
│   ├── summary_generator.py  # 摘要生成工具
│   └── visualization.py      # 可视化工具
├── classes.py                # 基础类定义
├── dca_calculator.py         # 定投计算器
├── main.py                   # 主程序入口
└── README.md                 # 项目文档
```

## 核心模块说明

### 配置模块 (config)

`investment_config.py` 定义了 `InvestmentConfig` 数据类，集中管理所有可配置参数：

- 最低允许收益率
- 并行计算设置
- 默认投资时机
- 默认t分布自由度
- 其他配置参数

### 服务模块 (services)

`investment_service.py` 提供了 `InvestmentService` 类，作为整个系统的主要接口：

- 整合模拟器、策略和工具功能
- 提供高级API进行投资计算和报告生成
- 管理计算结果和报告生成流程

### 模拟模块 (simulation)

`simulator.py` 提供了 `InvestmentSimulator` 类，负责执行投资模拟：

- 支持单次和多次模拟
- 自动选择并行或顺序执行
- 整合收益率生成和投资策略计算

### 策略模块 (strategies)

`investment_strategies.py` 实现了不同的投资策略：

- `DCAStrategy`：定投策略
- `ValueAveragingStrategy`：价值平均策略
- 支持不同的投资频率和投资时机

### 工具模块 (utils)

- `investment_utils.py`：基础工具函数
- `rate_distributions.py`：收益率分布模型实现
- `summary_generator.py`：生成投资结果摘要
- `visualization.py`：生成投资可视化图表

## 使用示例

```python
from config import InvestmentConfig
from services import InvestmentService
from classes import Currency

# 创建配置
config = InvestmentConfig()

# 创建投资服务
service = InvestmentService(config=config)

# 计算投资并生成报告
result = service.calculate_and_generate_report(
    investment_amount=1000,  # 每月投资1000元
    avg_rate=8,              # 平均年化收益率8%
    years=10,                # 投资10年
    volatility=15,           # 波动率15%
    frequency="每月",         # 每月投资
    currency=Currency.CNY,   # 使用人民币
    simulation_mode=True,    # 启用模拟模式
    simulation_rounds=1000,  # 模拟1000轮
    distribution_model="正态分布", # 使用正态分布
    strategy_name="定投策略",     # 使用定投策略
    include_visualizations=True  # 包含可视化
)

# 访问结果
final_amount = result["final_amount"]
total_profit = result["total_profit"]
annualized_return = result["annualized_return"]
summary_html = result["summary_html"]
visualizations = result["visualizations"]
```

## 扩展性

系统设计支持轻松扩展：

1. **添加新的投资策略**：继承 `InvestmentStrategy` 协议并实现 `calculate` 方法
2. **添加新的分布模型**：继承 `RateDistribution` 类并实现 `generate_yearly_rates` 方法
3. **自定义可视化**：修改 `visualization.py` 中的图表生成函数

## 依赖项

- Python 3.10+
- NumPy
- Pandas
- Matplotlib
- SciPy
- Gradio (UI)
- concurrent.futures (标准库，用于并行计算)

## 注意事项

- 本模拟器仅用于教育和研究目的，不构成投资建议
- 模拟结果基于历史数据和统计模型，不能保证未来表现
- 高波动率设置可能导致极端结果，请谨慎解释
