# -*- coding: utf-8 -*-

import numpy as np
from typing import Dict, List
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from dataclasses import dataclass
from enum import Enum
from classes import *
import io
import base64

class RateDistributionModel(Enum):
    NORMAL = "Normal"
    LOGNORMAL = "LogNormal"
    STUDENT_T = "Student-t"
    UNIFORM = "Uniform"

@dataclass
class RateSimulationResult:
    rates: np.ndarray
    stats: Dict[str, float]
    distribution_params: Dict[str, float]
    model: str

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
    **kwargs
) -> RateSimulationResult:
    """
    生成年度收益率数据并进行统计分析
    
    Args:
        avg_rate (float): 预期平均年化收益率(%)
        volatility (float): 波动率(%)
        years (int): 模拟年数
        simulation_rounds (int): 模拟次数
        distribution_model (RateDistributionModel): 收益率分布模型
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
    total_samples = years * simulation_rounds
    
    # 生成收益率数据
    match distribution_model:
        case RateDistributionModel.NORMAL:
            rates = np.random.normal(mu, sigma, total_samples)
            params = {'mu': mu, 'sigma': sigma}
            
        case RateDistributionModel.LOGNORMAL:
            mu_log = np.log((mu ** 2) / np.sqrt(sigma ** 2 + mu ** 2))
            sigma_log = np.sqrt(np.log(1 + (sigma ** 2) / (mu ** 2)))
            rates = np.random.lognormal(mu_log, sigma_log, total_samples)
            params = {'mu_log': mu_log, 'sigma_log': sigma_log}
            
        case RateDistributionModel.STUDENT_T:
            df = kwargs.get('df', 3)
            rates = mu + sigma * np.random.standard_t(df, total_samples)
            params = {'mu': mu, 'sigma': sigma, 'df': df}
            
        case RateDistributionModel.UNIFORM:
            min_rate = kwargs.get('min_rate', mu - sigma * np.sqrt(3))
            max_rate = kwargs.get('max_rate', mu + sigma * np.sqrt(3))
            rates = np.random.uniform(min_rate, max_rate, total_samples)
            params = {'min_rate': min_rate, 'max_rate': max_rate}
            
        case _:
            raise ValueError(f"不支持的分布模型: {distribution_model}")
    
    # 应用最小收益率限制
    min_allowed_rate = kwargs.get('min_allowed_rate', -50) / 100
    rates = np.maximum(rates, min_allowed_rate)
    
    # 计算统计指标
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
    
    return RateSimulationResult(
        rates=rates * 100,  # 转回百分比
        stats=stats_dict,
        distribution_params=params,
        model=distribution_model.value
    )

def plot_rate_distribution(result: RateSimulationResult) -> str:
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
    </div>
    """
    return html
