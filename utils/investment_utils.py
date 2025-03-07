"""
投资计算工具模块
提供投资计算相关的工具函数
"""
import numpy as np
from typing import List, Dict, Any, Tuple
import concurrent.futures
import time

from classes import Currency, InvestmentTiming
from utils.rate_distributions import RateDistributionModel
from config import INVESTMENT_CONFIG


def format_currency(amount: float, currency: Currency) -> str:
    """格式化货币显示
    
    Args:
        amount: 金额
        currency: 货币类型
        
    Returns:
        str: 格式化后的货币字符串
    """
    if currency == Currency.JPY:
        # JPY通常不显示小数点
        return f"{currency.symbol}{int(amount):,}"
    return f"{currency.symbol}{amount:,.2f}"


def format_percentage(value: float) -> str:
    """格式化百分比显示
    
    Args:
        value: 百分比值
        
    Returns:
        str: 格式化后的百分比字符串
    """
    return f"{value:,.2f}%"


def get_enum_by_label(label: str, enum_class: type) -> Any:
    """根据标签获取枚举值
    
    Args:
        label: 枚举标签
        enum_class: 枚举类
        
    Returns:
        Any: 对应的枚举值
        
    Raises:
        ValueError: 如果找不到对应的枚举值
    """
    try:
        return next(item for item in enum_class if item.label == label)
    except StopIteration:
        raise ValueError(f"找不到标签为 '{label}' 的 {enum_class.__name__} 枚举值")


def get_currency_by_code(code: str) -> Currency:
    """根据代码获取货币枚举
    
    Args:
        code: 货币代码
        
    Returns:
        Currency: 对应的货币枚举
        
    Raises:
        ValueError: 如果找不到对应的货币枚举
    """
    try:
        return Currency[code]
    except KeyError:
        raise ValueError(f"不支持的货币代码: {code}")


def calculate_annualized_return(final_amount: float, total_investment: float, years: int) -> float:
    """计算年化收益率
    
    Args:
        final_amount: 最终金额
        total_investment: 总投资金额
        years: 投资年数
        
    Returns:
        float: 年化收益率(%)
    """
    if total_investment <= 0 or years <= 0:
        return 0.0
    
    return ((final_amount / total_investment) ** (1 / years) - 1) * 100


def calculate_return_rate(final_amount: float, total_investment: float) -> float:
    """计算资产回报率
    
    Args:
        final_amount: 最终金额
        total_investment: 总投资金额
        
    Returns:
        float: 资产回报率(%)
    """
    if total_investment <= 0:
        return 0.0
    
    return (final_amount / total_investment * 100) - 100


def calculate_period_growth(
    period_rate: float, 
    period_investment: float, 
    current_amount: float, 
    timing_type: str
) -> Tuple[float, float]:
    """计算单期增长
    
    Args:
        period_rate: 单期收益率
        period_investment: 单期投资金额
        current_amount: 当前金额
        timing_type: 投资时点类型
        
    Returns:
        Tuple[float, float]: (新金额, 投资金额)
    """
    if timing_type == "期初":
        # 期初投资：先投资，然后计算整期收益
        new_amount = (current_amount + period_investment) * (1 + period_rate)
        return new_amount, period_investment
    elif timing_type == "期末":
        # 期末投资：先计算现有资金收益，然后投资
        new_amount = current_amount * (1 + period_rate) + period_investment
        return new_amount, period_investment
    else:  # 期中投资（默认）
        # 期中投资：先计算半期收益，然后投资，再计算半期收益
        new_amount = current_amount * (1 + period_rate/2)
        new_amount = (new_amount + period_investment) * (1 + period_rate/2)
        return new_amount, period_investment


def generate_yearly_rates(
    avg_rate: float,
    years: int,
    volatility: float,
    distribution_model: RateDistributionModel = RateDistributionModel.NORMAL,
    **kwargs
) -> List[float]:
    """生成年度收益率
    
    Args:
        avg_rate: 平均年化收益率(%)
        years: 年数
        volatility: 波动率(%)
        distribution_model: 分布模型
        **kwargs: 额外参数
            - min_allowed_rate: 最小允许收益率(%)
            - df: t分布自由度
            - min_rate: 均匀分布最小值
            - max_rate: 均匀分布最大值
            
    Returns:
        List[float]: 年度收益率列表
    """
    if volatility == 0:
        return [avg_rate] * years
    
    mu = avg_rate / 100
    sigma = volatility / 100
    
    match distribution_model:
        case RateDistributionModel.NORMAL:
            rates = np.random.normal(mu, sigma, years)
        case RateDistributionModel.LOGNORMAL:
            mu_log = np.log((mu ** 2) / np.sqrt(sigma ** 2 + mu ** 2))
            sigma_log = np.sqrt(np.log(1 + (sigma ** 2) / (mu ** 2)))
            rates = np.random.lognormal(mu_log, sigma_log, years)
        case RateDistributionModel.STUDENT_T:
            df = kwargs.get('df', INVESTMENT_CONFIG.default_t_distribution_df)
            rates = mu + sigma * np.random.standard_t(df, years)
        case RateDistributionModel.UNIFORM:
            min_rate = kwargs.get('min_rate', mu - sigma * np.sqrt(3))
            max_rate = kwargs.get('max_rate', mu + sigma * np.sqrt(3))
            rates = np.random.uniform(min_rate, max_rate, years)
        case _:
            raise ValueError(f"不支持的分布模型: {distribution_model}")
    
    rates = rates * 100
    min_allowed_rate = kwargs.get('min_allowed_rate', INVESTMENT_CONFIG.min_allowed_rate)
    rates = np.maximum(rates, min_allowed_rate)
    
    return rates.tolist()


def calculate_year_investment_vectorized(
    yearly_rates: List[float], 
    yearly_investment: float, 
    periods_per_year: int, 
    selected_timing: InvestmentTiming, 
    selected_currency: Currency
) -> Tuple[List[Dict[str, Any]], float, float, float]:
    """向量化版本的年度投资计算函数
    
    Args:
        yearly_rates: 年度收益率数组
        yearly_investment: 年度投资总额
        periods_per_year: 每年投资期数
        selected_timing: 选择的投资时点
        selected_currency: 选择的货币类型
        
    Returns:
        Tuple[List[Dict[str, Any]], float, float, float]: 
            (结果列表, 最终金额, 总投资, 总收益)
    """
    results = []
    current_amount = 0
    period_investment = yearly_investment / periods_per_year
    total_investment = 0
    
    # 预计算每年的期间收益率
    period_rates = np.array([(1 + rate/100)**(1/periods_per_year) - 1 for rate in yearly_rates])
    
    for year, annual_rate in enumerate(yearly_rates, 1):
        period_rate = period_rates[year-1]
        year_start_amount = current_amount
        
        # 向量化计算一年内所有期间的投资和收益
        if selected_timing == InvestmentTiming.BEGINNING:
            # 期初投资向量化计算
            # 先一次性投入所有期间的资金
            year_investment = period_investment * periods_per_year
            total_investment += year_investment
            
            # 计算复利增长
            growth_factors = np.cumprod(np.full(periods_per_year, 1 + period_rate))
            current_amount = year_start_amount + year_investment
            current_amount = current_amount * growth_factors[-1]
            
        elif selected_timing == InvestmentTiming.END:
            # 期末投资向量化计算
            # 先计算现有资金的复利增长
            current_amount *= (1 + period_rate) ** periods_per_year
            
            # 然后添加投资（每期投资不产生当期收益）
            year_investment = period_investment * periods_per_year
            current_amount += year_investment
            total_investment += year_investment
            
        else:  # 期中投资（默认）
            # 期中投资向量化计算
            # 计算每期的半期复利因子
            half_period_factor = (1 + period_rate/2)
            
            # 初始金额先增长半期
            current_amount *= half_period_factor
            
            # 添加所有期间的投资
            year_investment = period_investment * periods_per_year
            current_amount += year_investment
            total_investment += year_investment
            
            # 再增长半期
            current_amount *= half_period_factor ** (periods_per_year - 1)
        
        year_profit = current_amount - year_start_amount - year_investment
        
        results.append({
            '年份': f"第{year}年",
            '年化收益率': format_percentage(annual_rate),
            '投资金额': format_currency(year_investment, selected_currency),
            '当年收益': format_currency(year_profit, selected_currency),
            '年末总额': format_currency(current_amount, selected_currency),
            '累计投入': format_currency(total_investment, selected_currency)
        })
        
    return results, current_amount, total_investment, current_amount - total_investment


def run_single_simulation(
    yearly_rates: List[float],
    yearly_investment: float,
    periods_per_year: int,
    selected_timing: InvestmentTiming,
    selected_currency: Currency,
    years: int
) -> Dict[str, Any]:
    """运行单次投资模拟
    
    Args:
        yearly_rates: 年度收益率列表
        yearly_investment: 年度投资总额
        periods_per_year: 每年投资期数
        selected_timing: 选择的投资时点
        selected_currency: 选择的货币类型
        years: 投资年数
        
    Returns:
        Dict[str, Any]: 模拟结果
    """
    sim_results, final_amt, total_inv, total_prof = calculate_year_investment_vectorized(
        yearly_rates, 
        yearly_investment, 
        periods_per_year, 
        selected_timing, 
        selected_currency
    )
    
    # 计算收益率指标
    annualized_return = calculate_annualized_return(final_amt, total_inv, years)
    return_rate = calculate_return_rate(final_amt, total_inv)
    
    return {
        '最终金额': final_amt,
        '总投资': total_inv,
        '总收益': total_prof,
        '年化收益率': annualized_return,
        '资产回报率': return_rate,
        '详细数据': sim_results
    }


def run_parallel_simulations(
    rates_reshaped: np.ndarray,
    yearly_investment: float,
    periods_per_year: int,
    selected_timing: InvestmentTiming,
    selected_currency: Currency,
    years: int
) -> List[Dict[str, Any]]:
    """并行运行多次投资模拟
    
    Args:
        rates_reshaped: 重塑后的收益率数组
        yearly_investment: 年度投资总额
        periods_per_year: 每年投资期数
        selected_timing: 选择的投资时点
        selected_currency: 选择的货币类型
        years: 投资年数
        
    Returns:
        List[Dict[str, Any]]: 模拟结果列表
    """
    all_simulations = []
    start_time = time.time()
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # 创建并行任务，传递所有必要参数
        futures = []
        for yearly_rates in rates_reshaped:
            futures.append(
                executor.submit(
                    calculate_year_investment_vectorized,
                    yearly_rates,
                    yearly_investment,
                    periods_per_year,
                    selected_timing,
                    selected_currency
                )
            )
        
        # 收集结果
        for future in concurrent.futures.as_completed(futures):
            sim_results, final_amt, total_inv, total_prof = future.result()
            
            # 计算收益率指标
            annualized_return = calculate_annualized_return(final_amt, total_inv, years)
            return_rate = calculate_return_rate(final_amt, total_inv)
            
            all_simulations.append({
                '最终金额': final_amt,
                '总投资': total_inv,
                '总收益': total_prof,
                '年化收益率': annualized_return,
                '资产回报率': return_rate,
                '详细数据': sim_results
            })
    
    print(f"并行计算完成，耗时: {time.time() - start_time:.2f}秒")
    return all_simulations


def run_sequential_simulations(
    rates_reshaped: np.ndarray,
    yearly_investment: float,
    periods_per_year: int,
    selected_timing: InvestmentTiming,
    selected_currency: Currency,
    years: int
) -> List[Dict[str, Any]]:
    """顺序运行多次投资模拟
    
    Args:
        rates_reshaped: 重塑后的收益率数组
        yearly_investment: 年度投资总额
        periods_per_year: 每年投资期数
        selected_timing: 选择的投资时点
        selected_currency: 选择的货币类型
        years: 投资年数
        
    Returns:
        List[Dict[str, Any]]: 模拟结果列表
    """
    all_simulations = []
    
    for yearly_rates in rates_reshaped:
        simulation = run_single_simulation(
            yearly_rates,
            yearly_investment,
            periods_per_year,
            selected_timing,
            selected_currency,
            years
        )
        all_simulations.append(simulation)
    
    return all_simulations


def extract_simulation_statistics(all_simulations: List[Dict[str, Any]]) -> Dict[str, List[float]]:
    """从模拟结果中提取统计数据
    
    Args:
        all_simulations: 模拟结果列表
        
    Returns:
        Dict[str, List[float]]: 统计数据
    """
    return {
        'final_amounts': [sim['最终金额'] for sim in all_simulations],
        'total_profits': [sim['总收益'] for sim in all_simulations],
        'annualized_returns': [sim['年化收益率'] for sim in all_simulations],
        'return_rates': [sim['资产回报率'] for sim in all_simulations],
        'total_investment': all_simulations[0]['总投资'] if all_simulations else 0
    }
