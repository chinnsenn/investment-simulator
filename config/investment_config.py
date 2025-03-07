"""
投资计算器配置模块
存储投资计算相关的配置参数
"""
from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class InvestmentConfig:
    """投资计算器配置类"""
    # 收益率相关配置
    min_allowed_rate: float = -50.0  # 最小允许收益率(%)
    default_t_distribution_df: int = 3  # t分布自由度
    
    # 并行计算配置
    use_parallel_computing: bool = True  # 是否使用并行计算
    min_simulations_for_parallel: int = 10  # 启用并行计算的最小模拟次数
    
    # 默认投资参数
    default_investment_timing: str = "期中"  # 默认投资时点
    
    # 风险分析配置
    confidence_intervals: Dict[str, float] = None  # 置信区间
    
    def __post_init__(self):
        if self.confidence_intervals is None:
            self.confidence_intervals = {
                "low": 5.0,   # 5%分位数
                "high": 95.0  # 95%分位数
            }
    
    def to_dict(self) -> Dict[str, Any]:
        """将配置转换为字典"""
        return {
            "min_allowed_rate": self.min_allowed_rate,
            "default_t_distribution_df": self.default_t_distribution_df,
            "use_parallel_computing": self.use_parallel_computing,
            "min_simulations_for_parallel": self.min_simulations_for_parallel,
            "default_investment_timing": self.default_investment_timing,
            "confidence_intervals": self.confidence_intervals
        }


# 默认配置实例
DEFAULT_INVESTMENT_CONFIG = InvestmentConfig()
