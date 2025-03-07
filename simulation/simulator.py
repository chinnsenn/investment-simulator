"""
投资模拟器模块
提供执行投资模拟的功能
"""
import concurrent.futures
from typing import List, Dict, Any, Optional, Union, Callable
import numpy as np
from dataclasses import dataclass
from config.investment_config import InvestmentConfig
from utils.rate_distributions import generate_yearly_rates
from strategies.investment_strategies import (
    InvestmentStrategy,
    InvestmentTiming,
    InvestmentFrequency,
    get_strategy
)
from classes import Currency


@dataclass
class InvestmentSimulator:
    """投资模拟器类"""
    
    config: InvestmentConfig
    
    def run_single_simulation(
        self,
        investment_amount: float,
        avg_rate: float,
        years: int,
        volatility: float,
        frequency: str,
        distribution_model: str,
        strategy_name: str = "定投策略",
        investment_timing: str = "期中",
        **kwargs
    ) -> Dict[str, Any]:
        """运行单次投资模拟
        
        Args:
            investment_amount: 投资金额
            avg_rate: 平均收益率
            years: 投资年数
            volatility: 波动率
            frequency: 投资频率
            distribution_model: 分布模型
            strategy_name: 投资策略名称
            investment_timing: 投资时机
            **kwargs: 其他参数
            
        Returns:
            Dict[str, Any]: 模拟结果
        """
        # 生成年度收益率
        yearly_rates = generate_yearly_rates(
            avg_rate=avg_rate,
            years=years,
            volatility=volatility,
            distribution_model=distribution_model,
            config=self.config,
            **kwargs
        )
        
        # 获取投资策略
        strategy = get_strategy(strategy_name, self.config)
        
        # 转换投资频率和时机为枚举类型
        frequency_enum = InvestmentFrequency(frequency)
        timing_enum = InvestmentTiming(investment_timing)
        
        # 执行投资计算
        result = strategy.calculate(
            investment_amount=investment_amount,
            yearly_rates=yearly_rates,
            years=years,
            frequency=frequency_enum,
            timing=timing_enum
        )
        
        return result
    
    def run_multiple_simulations(
        self,
        investment_amount: float,
        avg_rate: float,
        years: int,
        volatility: float,
        frequency: str,
        distribution_model: str,
        simulation_rounds: int,
        strategy_name: str = "定投策略",
        investment_timing: str = "期中",
        **kwargs
    ) -> List[Dict[str, Any]]:
        """运行多次投资模拟
        
        Args:
            investment_amount: 投资金额
            avg_rate: 平均收益率
            years: 投资年数
            volatility: 波动率
            frequency: 投资频率
            distribution_model: 分布模型
            simulation_rounds: 模拟轮数
            strategy_name: 投资策略名称
            investment_timing: 投资时机
            **kwargs: 其他参数
            
        Returns:
            List[Dict[str, Any]]: 模拟结果列表
        """
        # 判断是否使用并行计算
        use_parallel = (
            self.config.use_parallel_computing and 
            simulation_rounds >= self.config.min_simulations_for_parallel
        )
        
        if use_parallel:
            # 并行执行模拟
            with concurrent.futures.ProcessPoolExecutor() as executor:
                futures = []
                for _ in range(simulation_rounds):
                    future = executor.submit(
                        self.run_single_simulation,
                        investment_amount=investment_amount,
                        avg_rate=avg_rate,
                        years=years,
                        volatility=volatility,
                        frequency=frequency,
                        distribution_model=distribution_model,
                        strategy_name=strategy_name,
                        investment_timing=investment_timing,
                        **kwargs
                    )
                    futures.append(future)
                
                # 收集结果
                results = [future.result() for future in futures]
        else:
            # 顺序执行模拟
            results = []
            for _ in range(simulation_rounds):
                result = self.run_single_simulation(
                    investment_amount=investment_amount,
                    avg_rate=avg_rate,
                    years=years,
                    volatility=volatility,
                    frequency=frequency,
                    distribution_model=distribution_model,
                    strategy_name=strategy_name,
                    investment_timing=investment_timing,
                    **kwargs
                )
                results.append(result)
        
        return results
    
    def calculate_investment(
        self,
        investment_amount: float,
        avg_rate: float,
        years: int,
        volatility: float,
        frequency: str,
        currency: Union[str, Currency],
        simulation_mode: bool,
        simulation_rounds: int,
        distribution_model: str,
        strategy_name: str = "定投策略",
        investment_timing: str = "期中",
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
            investment_timing: 投资时机
            **kwargs: 其他参数
            
        Returns:
            Dict[str, Any]: 投资结果
        """
        # 确保货币类型为Currency对象
        if isinstance(currency, str):
            currency = Currency(currency)
        
        # 根据模式执行计算
        if simulation_mode:
            # 模拟模式：执行多次模拟
            simulation_results = self.run_multiple_simulations(
                investment_amount=investment_amount,
                avg_rate=avg_rate,
                years=years,
                volatility=volatility,
                frequency=frequency,
                distribution_model=distribution_model,
                simulation_rounds=simulation_rounds,
                strategy_name=strategy_name,
                investment_timing=investment_timing,
                **kwargs
            )
            
            # 提取结果数据
            final_amounts = [sim["最终金额"] for sim in simulation_results]
            total_profits = [sim["总收益"] for sim in simulation_results]
            annualized_returns = [sim["年化收益率"] for sim in simulation_results]
            return_rates = [sim["资产回报率"] for sim in simulation_results]
            
            # 计算平均值作为主要结果
            avg_final_amount = np.mean(final_amounts)
            avg_total_profit = np.mean(total_profits)
            avg_annualized_return = np.mean(annualized_returns)
            avg_return_rate = np.mean(return_rates)
            
            # 计算总投资金额
            total_investment = simulation_results[0]["总投资"]
            
            # 返回结果
            return {
                "final_amount": avg_final_amount,
                "total_investment": total_investment,
                "total_profit": avg_total_profit,
                "annualized_return": avg_annualized_return,
                "return_rate": avg_return_rate,
                "simulation_results": simulation_results,
                "final_amounts": final_amounts,
                "total_profits": total_profits,
                "annualized_returns": annualized_returns,
                "return_rates": return_rates,
                "currency": currency
            }
        else:
            # 非模拟模式：执行单次计算
            result = self.run_single_simulation(
                investment_amount=investment_amount,
                avg_rate=avg_rate,
                years=years,
                volatility=volatility,
                frequency=frequency,
                distribution_model=distribution_model,
                strategy_name=strategy_name,
                investment_timing=investment_timing,
                **kwargs
            )
            
            # 返回结果
            return {
                "final_amount": result["最终金额"],
                "total_investment": result["总投资"],
                "total_profit": result["总收益"],
                "annualized_return": result["年化收益率"],
                "return_rate": result["资产回报率"],
                "simulation_results": [result],
                "final_amounts": [result["最终金额"]],
                "total_profits": [result["总收益"]],
                "annualized_returns": [result["年化收益率"]],
                "return_rates": [result["资产回报率"]],
                "currency": currency
            }
