"""
投资策略模块
提供不同的投资策略实现
"""
from typing import List, Dict, Any, Optional, Protocol, Callable
from enum import Enum, auto
import numpy as np
from dataclasses import dataclass
from config.investment_config import InvestmentConfig


class InvestmentTiming(Enum):
    """投资时机枚举"""
    MID_PERIOD = "期中"  # 期中投资
    START_PERIOD = "期初"  # 期初投资
    END_PERIOD = "期末"  # 期末投资


class InvestmentFrequency(Enum):
    """投资频率枚举"""
    MONTHLY = "每月"
    QUARTERLY = "每季度"
    SEMI_ANNUALLY = "每半年"
    ANNUALLY = "每年"
    LUMP_SUM = "一次性"


class InvestmentStrategy(Protocol):
    """投资策略协议"""
    
    def calculate(
        self, 
        investment_amount: float,
        yearly_rates: List[float],
        years: int,
        frequency: InvestmentFrequency,
        timing: InvestmentTiming
    ) -> Dict[str, Any]:
        """计算投资结果
        
        Args:
            investment_amount: 投资金额
            yearly_rates: 年度收益率列表
            years: 投资年数
            frequency: 投资频率
            timing: 投资时机
            
        Returns:
            Dict[str, Any]: 投资结果
        """
        ...


@dataclass
class DCAStrategy:
    """定投策略实现"""
    
    config: InvestmentConfig
    
    def calculate(
        self, 
        investment_amount: float,
        yearly_rates: List[float],
        years: int,
        frequency: InvestmentFrequency,
        timing: InvestmentTiming = InvestmentTiming.MID_PERIOD
    ) -> Dict[str, Any]:
        """计算定投策略的投资结果
        
        Args:
            investment_amount: 投资金额
            yearly_rates: 年度收益率列表
            years: 投资年数
            frequency: 投资频率
            timing: 投资时机，默认为期中
            
        Returns:
            Dict[str, Any]: 投资结果
        """
        # 计算每期投资金额
        if frequency == InvestmentFrequency.MONTHLY:
            periods_per_year = 12
        elif frequency == InvestmentFrequency.QUARTERLY:
            periods_per_year = 4
        elif frequency == InvestmentFrequency.SEMI_ANNUALLY:
            periods_per_year = 2
        elif frequency == InvestmentFrequency.ANNUALLY:
            periods_per_year = 1
        elif frequency == InvestmentFrequency.LUMP_SUM:
            periods_per_year = 1
        else:
            raise ValueError(f"不支持的投资频率: {frequency}")
        
        total_periods = years * periods_per_year
        
        # 一次性投资特殊处理
        if frequency == InvestmentFrequency.LUMP_SUM:
            period_investment = investment_amount
            total_investment = investment_amount
        else:
            period_investment = investment_amount / total_periods
            total_investment = period_investment * total_periods
        
        # 计算每期收益率
        period_rates = []
        for year_rate in yearly_rates:
            # 计算每期收益率（复利）
            period_rate = (1 + year_rate) ** (1 / periods_per_year) - 1
            period_rates.extend([period_rate] * periods_per_year)
        
        # 根据投资时机调整收益率
        if timing == InvestmentTiming.MID_PERIOD:
            # 期中投资：每期收益率的一半
            adjusted_rates = [rate / 2 for rate in period_rates]
        elif timing == InvestmentTiming.START_PERIOD:
            # 期初投资：完整收益率
            adjusted_rates = period_rates
        elif timing == InvestmentTiming.END_PERIOD:
            # 期末投资：无当期收益
            adjusted_rates = [0] + period_rates[:-1]
        else:
            raise ValueError(f"不支持的投资时机: {timing}")
        
        # 计算最终金额
        final_amount = 0
        
        # 一次性投资特殊处理
        if frequency == InvestmentFrequency.LUMP_SUM:
            # 一次性投资的复利计算
            final_amount = period_investment
            for rate in period_rates:
                final_amount *= (1 + rate)
        else:
            # 定投的复利计算
            for i in range(total_periods):
                # 添加本期投资
                final_amount += period_investment
                
                # 应用本期收益率
                final_amount *= (1 + adjusted_rates[i])
        
        # 计算总收益和收益率
        total_profit = final_amount - total_investment
        return_rate = total_profit / total_investment
        
        # 计算年化收益率
        if years > 0:
            annualized_return = (final_amount / total_investment) ** (1 / years) - 1
        else:
            annualized_return = 0
        
        # 返回结果
        return {
            "最终金额": final_amount,
            "总投资": total_investment,
            "总收益": total_profit,
            "资产回报率": return_rate,
            "年化收益率": annualized_return
        }


@dataclass
class ValueAveragingStrategy:
    """价值平均策略实现"""
    
    config: InvestmentConfig
    
    def calculate(
        self, 
        investment_amount: float,
        yearly_rates: List[float],
        years: int,
        frequency: InvestmentFrequency,
        timing: InvestmentTiming = InvestmentTiming.MID_PERIOD
    ) -> Dict[str, Any]:
        """计算价值平均策略的投资结果
        
        Args:
            investment_amount: 投资金额
            yearly_rates: 年度收益率列表
            years: 投资年数
            frequency: 投资频率
            timing: 投资时机，默认为期中
            
        Returns:
            Dict[str, Any]: 投资结果
        """
        # 计算每期投资金额
        if frequency == InvestmentFrequency.MONTHLY:
            periods_per_year = 12
        elif frequency == InvestmentFrequency.QUARTERLY:
            periods_per_year = 4
        elif frequency == InvestmentFrequency.SEMI_ANNUALLY:
            periods_per_year = 2
        elif frequency == InvestmentFrequency.ANNUALLY:
            periods_per_year = 1
        elif frequency == InvestmentFrequency.LUMP_SUM:
            # 价值平均策略不支持一次性投资
            raise ValueError("价值平均策略不支持一次性投资")
        else:
            raise ValueError(f"不支持的投资频率: {frequency}")
        
        total_periods = years * periods_per_year
        
        # 计算每期目标价值增长
        period_value_increase = investment_amount / total_periods
        
        # 计算每期收益率
        period_rates = []
        for year_rate in yearly_rates:
            # 计算每期收益率（复利）
            period_rate = (1 + year_rate) ** (1 / periods_per_year) - 1
            period_rates.extend([period_rate] * periods_per_year)
        
        # 根据投资时机调整收益率
        if timing == InvestmentTiming.MID_PERIOD:
            # 期中投资：每期收益率的一半
            adjusted_rates = [rate / 2 for rate in period_rates]
        elif timing == InvestmentTiming.START_PERIOD:
            # 期初投资：完整收益率
            adjusted_rates = period_rates
        elif timing == InvestmentTiming.END_PERIOD:
            # 期末投资：无当期收益
            adjusted_rates = [0] + period_rates[:-1]
        else:
            raise ValueError(f"不支持的投资时机: {timing}")
        
        # 价值平均策略实现
        current_value = 0
        target_value = 0
        total_investment = 0
        
        for i in range(total_periods):
            # 更新目标价值
            target_value += period_value_increase
            
            # 计算当前价值（考虑收益率）
            current_value *= (1 + adjusted_rates[i])
            
            # 计算需要投资的金额
            period_investment = max(0, target_value - current_value)
            
            # 添加投资
            current_value += period_investment
            total_investment += period_investment
        
        final_amount = current_value
        
        # 计算总收益和收益率
        total_profit = final_amount - total_investment
        return_rate = total_profit / total_investment if total_investment > 0 else 0
        
        # 计算年化收益率
        if years > 0 and total_investment > 0:
            annualized_return = (final_amount / total_investment) ** (1 / years) - 1
        else:
            annualized_return = 0
        
        # 返回结果
        return {
            "最终金额": final_amount,
            "总投资": total_investment,
            "总收益": total_profit,
            "资产回报率": return_rate,
            "年化收益率": annualized_return
        }


def get_strategy(strategy_name: str, config: InvestmentConfig) -> InvestmentStrategy:
    """获取投资策略
    
    Args:
        strategy_name: 策略名称
        config: 投资配置
        
    Returns:
        InvestmentStrategy: 投资策略实例
    """
    strategies = {
        "定投策略": DCAStrategy(config),
        "价值平均策略": ValueAveragingStrategy(config)
    }
    
    if strategy_name not in strategies:
        raise ValueError(f"不支持的投资策略: {strategy_name}")
    
    return strategies[strategy_name]
