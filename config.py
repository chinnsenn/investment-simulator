"""
投资模拟器配置文件
包含默认参数和配置选项
"""
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class SimulationConfig:
    """模拟配置参数"""
    # 收益率相关参数
    min_allowed_rate: float = -50.0  # 允许的最小收益率(%)
    default_risk_free_rate: float = 2.0  # 默认无风险利率(%)
    default_autocorrelation: float = 0.0  # 默认自相关系数
    
    # T分布参数
    default_t_distribution_df: int = 3  # T分布默认自由度
    
    # 投资相关参数
    default_investment_timing: str = "期中"  # 默认投资时点
    default_currency: str = "CNY"  # 默认货币
    default_frequency: str = "月度"  # 默认投资频率
    
    # 模拟参数
    default_simulation_rounds: int = 1000  # 默认模拟轮数
    default_years: int = 10  # 默认投资年数
    default_avg_rate: float = 8.0  # 默认年化收益率(%)
    default_volatility: float = 15.0  # 默认波动率(%)
    
    # UI相关参数
    ui_port: int = 7860  # UI服务端口
    
    # 风险指标参数
    max_drawdown_warning_threshold: float = 30.0  # 最大回撤警告阈值(%)
    good_sharpe_ratio_threshold: float = 1.0  # 良好夏普比率阈值
    good_sortino_ratio_threshold: float = 1.5  # 良好索提诺比率阈值

# 全局配置实例
CONFIG = SimulationConfig()
