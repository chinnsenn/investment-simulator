"""
模拟工具模块
提供投资模拟相关的辅助函数
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import io
import base64
from typing import List, Dict, Any, Tuple, Optional
from utils.rate_distributions import RateDistributionModel, generate_yearly_rates
from config.investment_config import InvestmentConfig


def simulate_rate_distribution(
    avg_rate: float,
    volatility: float,
    years: int,
    simulation_rounds: int,
    distribution_model: str,
    config: InvestmentConfig
) -> np.ndarray:
    """模拟收益率分布
    
    Args:
        avg_rate: 平均收益率(%)
        volatility: 波动率(%)
        years: 年数
        simulation_rounds: 模拟轮数
        distribution_model: 分布模型
        config: 投资配置
        
    Returns:
        np.ndarray: 模拟的收益率数组，形状为(simulation_rounds, years)
    """
    # 创建结果数组
    results = np.zeros((simulation_rounds, years))
    
    # 执行多次模拟
    for i in range(simulation_rounds):
        yearly_rates = generate_yearly_rates(
            avg_rate=avg_rate,
            years=years,
            volatility=volatility,
            distribution_model=distribution_model,
            config=config
        )
        results[i] = yearly_rates
    
    return results


def generate_rate_summary(
    rates: np.ndarray,
    avg_rate: float,
    volatility: float,
    distribution_model: str
) -> str:
    """生成收益率摘要
    
    Args:
        rates: 收益率数组
        avg_rate: 平均收益率(%)
        volatility: 波动率(%)
        distribution_model: 分布模型
        
    Returns:
        str: 收益率摘要HTML
    """
    # 计算统计数据
    mean_rate = np.mean(rates) * 100
    median_rate = np.median(rates) * 100
    std_rate = np.std(rates) * 100
    min_rate = np.min(rates) * 100
    max_rate = np.max(rates) * 100
    p5_rate = np.percentile(rates, 5) * 100
    p95_rate = np.percentile(rates, 95) * 100
    
    # 生成HTML摘要
    html = f"""
    <div style="font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto;">
        <h2 style="color: #333; text-align: center;">收益率分布摘要</h2>
        
        <div style="background-color: #f5f5f5; padding: 15px; border-radius: 5px; margin-bottom: 20px;">
            <h3 style="margin-top: 0; color: #444;">基本参数</h3>
            <p><strong>平均收益率:</strong> {avg_rate:.2f}%</p>
            <p><strong>波动率:</strong> {volatility:.2f}%</p>
            <p><strong>分布模型:</strong> {distribution_model}</p>
        </div>
        
        <div style="background-color: #e8f4f8; padding: 15px; border-radius: 5px; margin-bottom: 20px;">
            <h3 style="margin-top: 0; color: #444;">统计数据</h3>
            <table style="width: 100%; border-collapse: collapse;">
                <tr>
                    <td style="padding: 8px; border-bottom: 1px solid #ddd;"><strong>平均值:</strong></td>
                    <td style="padding: 8px; border-bottom: 1px solid #ddd; text-align: right;">{mean_rate:.2f}%</td>
                </tr>
                <tr>
                    <td style="padding: 8px; border-bottom: 1px solid #ddd;"><strong>中位数:</strong></td>
                    <td style="padding: 8px; border-bottom: 1px solid #ddd; text-align: right;">{median_rate:.2f}%</td>
                </tr>
                <tr>
                    <td style="padding: 8px; border-bottom: 1px solid #ddd;"><strong>标准差:</strong></td>
                    <td style="padding: 8px; border-bottom: 1px solid #ddd; text-align: right;">{std_rate:.2f}%</td>
                </tr>
                <tr>
                    <td style="padding: 8px; border-bottom: 1px solid #ddd;"><strong>最小值:</strong></td>
                    <td style="padding: 8px; border-bottom: 1px solid #ddd; text-align: right;">{min_rate:.2f}%</td>
                </tr>
                <tr>
                    <td style="padding: 8px; border-bottom: 1px solid #ddd;"><strong>最大值:</strong></td>
                    <td style="padding: 8px; border-bottom: 1px solid #ddd; text-align: right;">{max_rate:.2f}%</td>
                </tr>
                <tr>
                    <td style="padding: 8px; border-bottom: 1px solid #ddd;"><strong>5%分位数:</strong></td>
                    <td style="padding: 8px; border-bottom: 1px solid #ddd; text-align: right;">{p5_rate:.2f}%</td>
                </tr>
                <tr>
                    <td style="padding: 8px; border-bottom: 1px solid #ddd;"><strong>95%分位数:</strong></td>
                    <td style="padding: 8px; border-bottom: 1px solid #ddd; text-align: right;">{p95_rate:.2f}%</td>
                </tr>
            </table>
        </div>
        
        <div style="font-size: 0.8em; color: #666; text-align: center; margin-top: 20px;">
            <p>注意：以上结果基于模拟数据，实际投资结果可能有所不同。</p>
        </div>
    </div>
    """
    
    return html


def plot_rate_distribution(
    rates: np.ndarray,
    distribution_model: str,
    bins: int = 30
) -> str:
    """绘制收益率分布图并返回base64编码的图像
    
    Args:
        rates: 收益率数组
        distribution_model: 分布模型
        bins: 直方图的箱数
        
    Returns:
        str: base64编码的图像
    """
    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 将收益率转换为百分比
    rates_percent = rates.flatten() * 100
    
    # 绘制直方图
    n, bins, patches = ax.hist(
        rates_percent, 
        bins=bins, 
        color='skyblue', 
        edgecolor='black', 
        alpha=0.7
    )
    
    # 设置标题和标签
    ax.set_title(f'收益率分布 ({distribution_model})', fontsize=14)
    ax.set_xlabel('收益率 (%)', fontsize=12)
    ax.set_ylabel('频率', fontsize=12)
    
    # 添加网格线
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # 计算并显示统计数据
    mean_val = np.mean(rates_percent)
    median_val = np.median(rates_percent)
    std_val = np.std(rates_percent)
    
    # 添加统计数据文本
    stats_text = f"均值: {mean_val:.2f}%\n中位数: {median_val:.2f}%\n标准差: {std_val:.2f}%"
    ax.text(0.95, 0.95, stats_text,
            transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 绘制均值和中位数线
    ax.axvline(mean_val, color='r', linestyle='dashed', linewidth=1, label=f'均值: {mean_val:.2f}%')
    ax.axvline(median_val, color='g', linestyle='dashed', linewidth=1, label=f'中位数: {median_val:.2f}%')
    
    # 添加图例
    ax.legend()
    
    # 调整布局
    plt.tight_layout()
    
    # 将图形转换为base64编码
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    
    return img_str
