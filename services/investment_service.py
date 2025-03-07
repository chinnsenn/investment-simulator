"""
投资服务模块
提供完整的投资计算和模拟服务
"""
from typing import Dict, Any, List, Union, Optional
from dataclasses import dataclass

from config.investment_config import InvestmentConfig
from simulation.simulator import InvestmentSimulator
from utils.summary_generator import (
    generate_investment_summary,
    generate_simulation_statistics,
    generate_simulation_table
)
from utils.visualization import generate_investment_visualizations
from classes import Currency


@dataclass
class InvestmentService:
    """投资服务类，整合所有投资相关功能"""
    
    config: InvestmentConfig
    
    def __post_init__(self):
        """初始化投资模拟器"""
        self.simulator = InvestmentSimulator(config=self.config)
    
    def calculate_investment(
        self,
        investment_amount: float,
        avg_rate: float,
        years: int,
        volatility: float,
        frequency: str,
        currency: Union[str, Currency],
        simulation_mode: bool = False,
        simulation_rounds: int = 1000,
        distribution_model: str = "正态分布",
        strategy_name: str = "定投策略",
        investment_timing: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """计算投资结果
        
        Args:
            investment_amount: 投资金额
            avg_rate: 平均收益率
            years: 投资年数
            volatility: 波动率
            frequency: 投资频率
            currency: 货币类型
            simulation_mode: 是否为模拟模式
            simulation_rounds: 模拟轮数
            distribution_model: 分布模型
            strategy_name: 投资策略名称
            investment_timing: 投资时机，如果为None则使用配置默认值
            **kwargs: 其他参数
            
        Returns:
            Dict[str, Any]: 投资结果
        """
        # 使用默认投资时机（如果未指定）
        if investment_timing is None:
            investment_timing = self.config.default_investment_timing
        
        # 执行投资计算
        result = self.simulator.calculate_investment(
            investment_amount=investment_amount,
            avg_rate=avg_rate,
            years=years,
            volatility=volatility,
            frequency=frequency,
            currency=currency,
            simulation_mode=simulation_mode,
            simulation_rounds=simulation_rounds,
            distribution_model=distribution_model,
            strategy_name=strategy_name,
            investment_timing=investment_timing,
            **kwargs
        )
        
        return result
    
    def generate_investment_report(
        self,
        result: Dict[str, Any],
        include_visualizations: bool = True,
        include_summary: bool = True,
        max_table_rows: int = 10
    ) -> Dict[str, Any]:
        """生成投资报告
        
        Args:
            result: 投资计算结果
            include_visualizations: 是否包含可视化
            include_summary: 是否包含摘要
            max_table_rows: 表格最大行数
            
        Returns:
            Dict[str, Any]: 投资报告
        """
        report = {
            "final_amount": result["final_amount"],
            "total_investment": result["total_investment"],
            "total_profit": result["total_profit"],
            "annualized_return": result["annualized_return"],
            "return_rate": result["return_rate"],
            "currency": result["currency"]
        }
        
        # 生成统计数据
        if "simulation_results" in result and len(result["simulation_results"]) > 1:
            # 多次模拟的情况
            stats = generate_simulation_statistics(
                simulation_results=result["simulation_results"],
                total_investment=result["total_investment"],
                selected_currency=result["currency"]
            )
            report["statistics"] = stats
            
            # 生成表格
            table = generate_simulation_table(
                simulation_results=result["simulation_results"],
                max_rows=max_table_rows
            )
            report["table"] = table
        
        # 生成摘要
        if include_summary and "final_amounts" in result:
            summary_html = generate_investment_summary(
                total_investment=result["total_investment"],
                final_amounts=result["final_amounts"],
                total_profits=result["total_profits"],
                annualized_returns=result["annualized_returns"],
                return_rates=result["return_rates"],
                selected_currency=result["currency"]
            )
            report["summary_html"] = summary_html
        
        # 生成可视化
        if include_visualizations and "final_amounts" in result and len(result["final_amounts"]) > 1:
            # 提取投资年数（假设已知）
            years = len(result["simulation_results"][0].get("yearly_rates", [])) if "simulation_results" in result else 0
            
            # 如果yearly_rates不可用，尝试从其他参数推断
            if years == 0 and "kwargs" in result:
                years = result.get("kwargs", {}).get("years", 0)
            
            # 生成可视化
            visualizations = generate_investment_visualizations(
                years=years,
                investment_amount=result["total_investment"],
                final_amounts=result["final_amounts"],
                annualized_returns=result["annualized_returns"],
                selected_currency=result["currency"]
            )
            report["visualizations"] = visualizations
        
        return report
    
    def calculate_and_generate_report(
        self,
        investment_amount: float,
        avg_rate: float,
        years: int,
        volatility: float,
        frequency: str,
        currency: Union[str, Currency],
        simulation_mode: bool = False,
        simulation_rounds: int = 1000,
        distribution_model: str = "正态分布",
        strategy_name: str = "定投策略",
        investment_timing: Optional[str] = None,
        include_visualizations: bool = True,
        include_summary: bool = True,
        max_table_rows: int = 10,
        **kwargs
    ) -> Dict[str, Any]:
        """计算投资并生成报告
        
        Args:
            investment_amount: 投资金额
            avg_rate: 平均收益率
            years: 投资年数
            volatility: 波动率
            frequency: 投资频率
            currency: 货币类型
            simulation_mode: 是否为模拟模式
            simulation_rounds: 模拟轮数
            distribution_model: 分布模型
            strategy_name: 投资策略名称
            investment_timing: 投资时机
            include_visualizations: 是否包含可视化
            include_summary: 是否包含摘要
            max_table_rows: 表格最大行数
            **kwargs: 其他参数
            
        Returns:
            Dict[str, Any]: 投资报告
        """
        # 计算投资结果
        result = self.calculate_investment(
            investment_amount=investment_amount,
            avg_rate=avg_rate,
            years=years,
            volatility=volatility,
            frequency=frequency,
            currency=currency,
            simulation_mode=simulation_mode,
            simulation_rounds=simulation_rounds,
            distribution_model=distribution_model,
            strategy_name=strategy_name,
            investment_timing=investment_timing,
            **kwargs
        )
        
        # 生成报告
        report = self.generate_investment_report(
            result=result,
            include_visualizations=include_visualizations,
            include_summary=include_summary,
            max_table_rows=max_table_rows
        )
        
        # 合并结果和报告
        combined_result = {**result, **report}
        
        return combined_result
