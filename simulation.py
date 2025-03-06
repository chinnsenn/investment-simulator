# -*- coding: utf-8 -*-

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from dataclasses import dataclass, field
from enum import Enum
from classes import *
import io
import base64
from config import CONFIG

class RateDistributionModel(Enum):
    NORMAL = "Normal"
    LOGNORMAL = "LogNormal"
    STUDENT_T = "Student-t"
    UNIFORM = "Uniform"

@dataclass
class RateSimulationResult:
    """收益率模拟结果数据类"""
    rates: np.ndarray  # 收益率数组
    stats: Dict[str, float]  # 统计指标
    distribution_params: Dict[str, float]  # 分布参数
    model: str  # 分布模型名称
    risk_metrics: Optional[Dict[str, float]] = None  # 风险指标

plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.size'] = '20'
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False  # Fix minus sign display

def simulate_rate_distribution(
    avg_rate: float,
    volatility: float,
    years: int,
    simulation_rounds: int,
    distribution_model: RateDistributionModel,
    risk_free_rate: float = CONFIG.default_risk_free_rate,
    autocorrelation: float = CONFIG.default_autocorrelation,
    **kwargs: Any
) -> RateSimulationResult:
    """
    生成年度收益率数据并进行统计分析
    
    Args:
        avg_rate (float): 预期平均年化收益率(%)
        volatility (float): 波动率(%)
        years (int): 模拟年数
        simulation_rounds (int): 模拟次数
        distribution_model (RateDistributionModel): 收益率分布模型
        risk_free_rate (float): 无风险利率(%)，用于计算夏普比率
        autocorrelation (float): 自相关系数，范围[-1,1]，用于模拟市场周期性波动
        **kwargs: 额外参数
            - df: t分布自由度
            - min_rate: 均匀分布最小值
            - max_rate: 均匀分布最大值
            - min_allowed_rate: 允许的最小收益率
    
    Returns:
        RateSimulationResult: 包含模拟结果的数据类
    """
    # 转换为小数
    mu = avg_rate / 100
    sigma = volatility / 100
    rf_rate = risk_free_rate / 100
    total_samples = years * simulation_rounds
    
    # 生成收益率数据
    rates = _generate_rates(
        mu, sigma, total_samples, distribution_model, 
        autocorrelation, years, simulation_rounds, **kwargs
    )
    
    # 应用最小收益率限制
    min_allowed_rate = kwargs.get('min_allowed_rate', CONFIG.min_allowed_rate) / 100
    rates = np.maximum(rates, min_allowed_rate)
    
    # 计算统计指标
    stats_dict = _calculate_statistics(rates)
    
    # 获取分布参数
    params = _get_distribution_params(mu, sigma, distribution_model, **kwargs)
    
    # 计算风险指标
    risk_metrics = calculate_risk_metrics(rates, rf_rate)
    
    return RateSimulationResult(
        rates=rates * 100,  # 转回百分比
        stats=stats_dict,
        distribution_params=params,
        model=distribution_model.value,
        risk_metrics=risk_metrics
    )

def _generate_rates(
    mu: float, 
    sigma: float, 
    total_samples: int, 
    distribution_model: RateDistributionModel,
    autocorrelation: float,
    years: int,
    simulation_rounds: int,
    **kwargs: Any
) -> np.ndarray:
    """
    根据指定的分布模型生成收益率数据
    
    Args:
        mu: 平均收益率(小数)
        sigma: 波动率(小数)
        total_samples: 总样本数
        distribution_model: 分布模型
        autocorrelation: 自相关系数
        years: 模拟年数
        simulation_rounds: 模拟轮数
        **kwargs: 额外参数
    
    Returns:
        np.ndarray: 生成的收益率数组
    """
    match distribution_model:
        case RateDistributionModel.NORMAL:
            rates = np.random.normal(mu, sigma, total_samples)
            
        case RateDistributionModel.LOGNORMAL:
            mu_log = np.log((mu ** 2) / np.sqrt(sigma ** 2 + mu ** 2))
            sigma_log = np.sqrt(np.log(1 + (sigma ** 2) / (mu ** 2)))
            rates = np.random.lognormal(mu_log, sigma_log, total_samples)
            
        case RateDistributionModel.STUDENT_T:
            df = kwargs.get('df', CONFIG.default_t_distribution_df)
            rates = mu + sigma * np.random.standard_t(df, total_samples)
            
        case RateDistributionModel.UNIFORM:
            min_rate = kwargs.get('min_rate', mu - sigma * np.sqrt(3))
            max_rate = kwargs.get('max_rate', mu + sigma * np.sqrt(3))
            rates = np.random.uniform(min_rate, max_rate, total_samples)
            
        case _:
            raise ValueError(f"不支持的分布模型: {distribution_model}")
    
    # 应用自相关性（如果指定）
    if autocorrelation != 0 and years > 1:
        rates = _apply_autocorrelation(rates, mu, sigma, autocorrelation, simulation_rounds, years)
    
    return rates

def _apply_autocorrelation(
    rates: np.ndarray, 
    mu: float, 
    sigma: float, 
    autocorrelation: float, 
    simulation_rounds: int, 
    years: int
) -> np.ndarray:
    """
    应用自相关性到收益率序列
    
    Args:
        rates: 原始收益率数组
        mu: 平均收益率
        sigma: 波动率
        autocorrelation: 自相关系数
        simulation_rounds: 模拟轮数
        years: 模拟年数
    
    Returns:
        np.ndarray: 应用自相关后的收益率数组
    """
    rates_reshaped = rates.reshape(simulation_rounds, years)
    for i in range(simulation_rounds):
        for j in range(1, years):
            # 应用自相关公式: r_t = μ + ρ(r_{t-1} - μ) + ε_t
            # 其中ε_t是随机噪声，μ是平均收益率，ρ是自相关系数
            noise = np.random.normal(0, sigma * np.sqrt(1 - autocorrelation**2))
            rates_reshaped[i, j] = mu + autocorrelation * (rates_reshaped[i, j-1] - mu) + noise
    return rates_reshaped.flatten()

def _get_distribution_params(
    mu: float, 
    sigma: float, 
    distribution_model: RateDistributionModel, 
    **kwargs: Any
) -> Dict[str, float]:
    """
    获取分布模型的参数
    
    Args:
        mu: 平均收益率
        sigma: 波动率
        distribution_model: 分布模型
        **kwargs: 额外参数
    
    Returns:
        Dict[str, float]: 分布参数字典
    """
    match distribution_model:
        case RateDistributionModel.NORMAL:
            return {'mu': mu, 'sigma': sigma}
            
        case RateDistributionModel.LOGNORMAL:
            mu_log = np.log((mu ** 2) / np.sqrt(sigma ** 2 + mu ** 2))
            sigma_log = np.sqrt(np.log(1 + (sigma ** 2) / (mu ** 2)))
            return {'mu_log': mu_log, 'sigma_log': sigma_log}
            
        case RateDistributionModel.STUDENT_T:
            df = kwargs.get('df', CONFIG.default_t_distribution_df)
            return {'mu': mu, 'sigma': sigma, 'df': df}
            
        case RateDistributionModel.UNIFORM:
            min_rate = kwargs.get('min_rate', mu - sigma * np.sqrt(3))
            max_rate = kwargs.get('max_rate', mu + sigma * np.sqrt(3))
            return {'min_rate': min_rate, 'max_rate': max_rate}
            
        case _:
            return {}

def _calculate_statistics(rates: np.ndarray) -> Dict[str, float]:
    """
    计算收益率的统计指标
    
    Args:
        rates: 收益率数组(小数形式)
    
    Returns:
        Dict[str, float]: 统计指标字典
    """
    stats_dict = {
        'mean': np.mean(rates) * 100,
        'median': np.median(rates) * 100,
        'std': np.std(rates) * 100,
        'skewness': stats.skew(rates),
        'kurtosis': stats.kurtosis(rates),
        'min': np.min(rates) * 100,
        'max': np.max(rates) * 100
    }
    
    # 添加分位数
    percentiles = [1, 5, 10, 25, 75, 90, 95, 99]
    for p in percentiles:
        stats_dict[f'percentile_{p}'] = np.percentile(rates, p) * 100
    
    return stats_dict

def calculate_risk_metrics(rates: np.ndarray, risk_free_rate: float) -> Dict[str, float]:
    """
    计算风险指标，包括最大回撤、夏普比率和索提诺比率
    
    Args:
        rates: 收益率数组（小数形式）
        risk_free_rate: 无风险利率（小数形式）
        
    Returns:
        Dict[str, float]: 包含风险指标的字典
    """
    # 计算最大回撤
    max_drawdown = calculate_maximum_drawdown(rates)
    
    # 计算夏普比率 = (平均收益率 - 无风险利率) / 收益率标准差
    excess_return = np.mean(rates) - risk_free_rate
    sharpe_ratio = excess_return / np.std(rates) if np.std(rates) > 0 else 0
    
    # 计算索提诺比率 = (平均收益率 - 无风险利率) / 下行标准差
    # 下行标准差只考虑低于目标收益率（通常是无风险利率）的收益率
    downside_returns = rates[rates < risk_free_rate] - risk_free_rate
    downside_deviation = np.std(downside_returns) if len(downside_returns) > 0 else 0
    sortino_ratio = excess_return / downside_deviation if downside_deviation > 0 else 0
    
    return {
        'max_drawdown': max_drawdown * 100,  # 转为百分比
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio
    }

def calculate_maximum_drawdown(returns: np.ndarray) -> float:
    """
    计算最大回撤
    
    最大回撤是投资组合从峰值到谷值的最大跌幅，用于衡量投资风险
    
    Args:
        returns: 收益率数组（小数形式）
        
    Returns:
        float: 最大回撤值（小数形式）
    """
    # 将收益率转换为累积收益
    cumulative_returns = np.cumprod(1 + returns)
    
    # 计算运行最大值
    running_max = np.maximum.accumulate(cumulative_returns)
    
    # 计算每个点的回撤
    drawdowns = (running_max - cumulative_returns) / running_max
    
    # 返回最大回撤
    return np.max(drawdowns) if len(drawdowns) > 0 else 0

def plot_rate_distribution(result: RateSimulationResult) -> str:
    """
    绘制收益率分布图
    
    Args:
        result: 收益率模拟结果
        
    Returns:
        str: 包含图像的HTML字符串
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # 直方图和核密度估计
    sns.histplot(result.rates, kde=True, ax=ax1)
    ax1.set_title(f'收益率分布 ({result.model})')
    ax1.set_xlabel('收益率 (%)')
    ax1.set_ylabel('频率')
    
    # 添加统计信息
    stats_text = (
        f"均值: {result.stats['mean']:.2f}%\n"
        f"中位数: {result.stats['median']:.2f}%\n"
        f"标准差: {result.stats['std']:.2f}%\n"
    )
    ax1.text(0.95, 0.95, stats_text,
             transform=ax1.transAxes,
             verticalalignment='top',
             horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # QQ图
    stats.probplot(result.rates, dist="norm", plot=ax2)
    ax2.set_title('正态Q-Q图')
    
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.getvalue()).decode()
    plt.close()
    
    return f'<img src="data:image/png;base64,{img_str}" style="width:100%">'

def generate_rate_summary(result: RateSimulationResult) -> str:
    """
    生成收益率分析的HTML摘要
    
    Args:
        result: RateSimulationResult 对象
    
    Returns:
        str: HTML格式的摘要
    """
    html = f"""
    <div style="background-color: #f8f9fa; padding: 20px; border-radius: 10px; margin: 20px 0;">
        <h3 style="color: #2c3e50;">📊 收益率分布分析 ({result.model})</h3>
        
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
            <div style="background-color: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <h4 style="color: #3498db;">基本统计</h4>
                <ul style="list-style-type: none; padding-left: 0;">
                    <li>均值: {result.stats['mean']:.2f}%</li>
                    <li>中位数: {result.stats['median']:.2f}%</li>
                    <li>标准差: {result.stats['std']:.2f}%</li>
                    <li>偏度: {result.stats['skewness']:.2f}</li>
                    <li>峰度: {result.stats['kurtosis']:.2f}</li>
                </ul>
            </div>
            
            <div style="background-color: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <h4 style="color: #3498db;">分位数分析</h4>
                <ul style="list-style-type: none; padding-left: 0;">
                    <li>最小值: {result.stats['min']:.2f}%</li>
                    <li>5%分位: {result.stats['percentile_5']:.2f}%</li>
                    <li>25%分位: {result.stats['percentile_25']:.2f}%</li>
                    <li>75%分位: {result.stats['percentile_75']:.2f}%</li>
                    <li>95%分位: {result.stats['percentile_95']:.2f}%</li>
                    <li>最大值: {result.stats['max']:.2f}%</li>
                </ul>
            </div>
        </div>
        
        {_generate_risk_metrics_html(result) if result.risk_metrics else ""}
    </div>
    """
    return html

def _generate_risk_metrics_html(result: RateSimulationResult) -> str:
    """
    生成风险指标的HTML
    
    Args:
        result: 包含风险指标的模拟结果
        
    Returns:
        str: 风险指标HTML
    """
    return f"""
    <div style="background-color: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-top: 20px;">
        <h4 style="color: #e74c3c;">风险评估指标</h4>
        <ul style="list-style-type: none; padding-left: 0;">
            <li>最大回撤: {result.risk_metrics['max_drawdown']:.2f}%</li>
            <li>夏普比率: {result.risk_metrics['sharpe_ratio']:.2f}</li>
            <li>索提诺比率: {result.risk_metrics['sortino_ratio']:.2f}</li>
        </ul>
        <div style="font-size: 0.9em; color: #7f8c8d; margin-top: 10px;">
            <p>• 最大回撤：衡量投资过程中可能遭受的最大损失百分比</p>
            <p>• 夏普比率：每承担一单位总风险，能获得的超额收益</p>
            <p>• 索提诺比率：每承担一单位下行风险，能获得的超额收益</p>
        </div>
    </div>
    """
