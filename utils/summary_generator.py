"""
投资结果摘要生成模块
提供生成投资结果摘要的功能
"""
import numpy as np
from typing import List, Dict, Any, Optional
import pandas as pd
from classes import Currency
from utils.investment_utils import format_currency, format_percentage


def generate_investment_summary(
    total_investment: float,
    final_amounts: List[float],
    total_profits: List[float],
    annualized_returns: List[float],
    return_rates: List[float],
    selected_currency: Currency
) -> str:
    """生成投资结果摘要的HTML
    
    Args:
        total_investment: 总投资金额
        final_amounts: 最终金额列表
        total_profits: 总收益列表
        annualized_returns: 年化收益率列表
        return_rates: 资产回报率列表
        selected_currency: 选择的货币类型
        
    Returns:
        str: HTML格式的投资结果摘要
    """
    # 计算统计数据
    simulation_count = len(final_amounts)
    
    # 计算均值
    avg_final_amount = np.mean(final_amounts)
    avg_total_profit = np.mean(total_profits)
    avg_annualized_return = np.mean(annualized_returns)
    avg_return_rate = np.mean(return_rates)
    
    # 计算中位数
    median_final_amount = np.median(final_amounts)
    median_total_profit = np.median(total_profits)
    median_annualized_return = np.median(annualized_returns)
    median_return_rate = np.median(return_rates)
    
    # 计算最大值和最小值
    min_final_amount = np.min(final_amounts)
    max_final_amount = np.max(final_amounts)
    min_annualized_return = np.min(annualized_returns)
    max_annualized_return = np.max(annualized_returns)
    
    # 计算分位数
    p5_final_amount = np.percentile(final_amounts, 5)
    p95_final_amount = np.percentile(final_amounts, 95)
    p5_annualized_return = np.percentile(annualized_returns, 5)
    p95_annualized_return = np.percentile(annualized_returns, 95)
    
    # 计算标准差
    std_final_amount = np.std(final_amounts)
    std_annualized_return = np.std(annualized_returns)
    
    # 计算成功率（最终金额大于总投资的比例）
    success_rate = np.mean(np.array(final_amounts) > total_investment) * 100
    
    # 生成HTML摘要
    html = f"""
    <div style="font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto;">
        <h2 style="color: #333; text-align: center;">投资结果摘要</h2>
        
        <div style="background-color: #f5f5f5; padding: 15px; border-radius: 5px; margin-bottom: 20px;">
            <h3 style="margin-top: 0; color: #444;">基本信息</h3>
            <p><strong>总投资金额:</strong> {format_currency(total_investment, selected_currency)}</p>
            <p><strong>模拟次数:</strong> {simulation_count}</p>
            <p><strong>投资成功率:</strong> {format_percentage(success_rate)}</p>
        </div>
        
        <div style="display: flex; flex-wrap: wrap; gap: 20px; margin-bottom: 20px;">
            <div style="flex: 1; min-width: 300px; background-color: #e8f4f8; padding: 15px; border-radius: 5px;">
                <h3 style="margin-top: 0; color: #444;">最终金额统计</h3>
                <table style="width: 100%; border-collapse: collapse;">
                    <tr>
                        <td style="padding: 8px; border-bottom: 1px solid #ddd;"><strong>平均值:</strong></td>
                        <td style="padding: 8px; border-bottom: 1px solid #ddd; text-align: right;">{format_currency(avg_final_amount, selected_currency)}</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px; border-bottom: 1px solid #ddd;"><strong>中位数:</strong></td>
                        <td style="padding: 8px; border-bottom: 1px solid #ddd; text-align: right;">{format_currency(median_final_amount, selected_currency)}</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px; border-bottom: 1px solid #ddd;"><strong>最小值:</strong></td>
                        <td style="padding: 8px; border-bottom: 1px solid #ddd; text-align: right;">{format_currency(min_final_amount, selected_currency)}</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px; border-bottom: 1px solid #ddd;"><strong>最大值:</strong></td>
                        <td style="padding: 8px; border-bottom: 1px solid #ddd; text-align: right;">{format_currency(max_final_amount, selected_currency)}</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px; border-bottom: 1px solid #ddd;"><strong>标准差:</strong></td>
                        <td style="padding: 8px; border-bottom: 1px solid #ddd; text-align: right;">{format_currency(std_final_amount, selected_currency)}</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px; border-bottom: 1px solid #ddd;"><strong>5%分位数:</strong></td>
                        <td style="padding: 8px; border-bottom: 1px solid #ddd; text-align: right;">{format_currency(p5_final_amount, selected_currency)}</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px; border-bottom: 1px solid #ddd;"><strong>95%分位数:</strong></td>
                        <td style="padding: 8px; border-bottom: 1px solid #ddd; text-align: right;">{format_currency(p95_final_amount, selected_currency)}</td>
                    </tr>
                </table>
            </div>
            
            <div style="flex: 1; min-width: 300px; background-color: #f0f8e8; padding: 15px; border-radius: 5px;">
                <h3 style="margin-top: 0; color: #444;">收益率统计</h3>
                <table style="width: 100%; border-collapse: collapse;">
                    <tr>
                        <td style="padding: 8px; border-bottom: 1px solid #ddd;"><strong>平均年化收益率:</strong></td>
                        <td style="padding: 8px; border-bottom: 1px solid #ddd; text-align: right;">{format_percentage(avg_annualized_return)}</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px; border-bottom: 1px solid #ddd;"><strong>中位数年化收益率:</strong></td>
                        <td style="padding: 8px; border-bottom: 1px solid #ddd; text-align: right;">{format_percentage(median_annualized_return)}</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px; border-bottom: 1px solid #ddd;"><strong>最小年化收益率:</strong></td>
                        <td style="padding: 8px; border-bottom: 1px solid #ddd; text-align: right;">{format_percentage(min_annualized_return)}</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px; border-bottom: 1px solid #ddd;"><strong>最大年化收益率:</strong></td>
                        <td style="padding: 8px; border-bottom: 1px solid #ddd; text-align: right;">{format_percentage(max_annualized_return)}</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px; border-bottom: 1px solid #ddd;"><strong>标准差:</strong></td>
                        <td style="padding: 8px; border-bottom: 1px solid #ddd; text-align: right;">{format_percentage(std_annualized_return)}</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px; border-bottom: 1px solid #ddd;"><strong>5%分位数:</strong></td>
                        <td style="padding: 8px; border-bottom: 1px solid #ddd; text-align: right;">{format_percentage(p5_annualized_return)}</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px; border-bottom: 1px solid #ddd;"><strong>95%分位数:</strong></td>
                        <td style="padding: 8px; border-bottom: 1px solid #ddd; text-align: right;">{format_percentage(p95_annualized_return)}</td>
                    </tr>
                </table>
            </div>
        </div>
        
        <div style="background-color: #f8f0e8; padding: 15px; border-radius: 5px; margin-bottom: 20px;">
            <h3 style="margin-top: 0; color: #444;">收益概况</h3>
            <p><strong>平均总收益:</strong> {format_currency(avg_total_profit, selected_currency)} ({format_percentage(avg_return_rate)})</p>
            <p><strong>中位数总收益:</strong> {format_currency(median_total_profit, selected_currency)} ({format_percentage(median_return_rate)})</p>
            <p><strong>收益区间 (90% 置信区间):</strong> {format_currency(p5_final_amount - total_investment, selected_currency)} 至 {format_currency(p95_final_amount - total_investment, selected_currency)}</p>
        </div>
        
        <div style="font-size: 0.8em; color: #666; text-align: center; margin-top: 20px;">
            <p>注意：以上结果基于历史数据和模拟，不代表未来表现。投资有风险，请谨慎决策。</p>
        </div>
    </div>
    """
    
    return html


def generate_simulation_statistics(
    simulation_results: List[Dict[str, Any]],
    total_investment: float,
    selected_currency: Currency
) -> Dict[str, Any]:
    """生成模拟统计数据
    
    Args:
        simulation_results: 模拟结果列表
        total_investment: 总投资金额
        selected_currency: 选择的货币类型
        
    Returns:
        Dict[str, Any]: 统计数据
    """
    # 提取数据
    final_amounts = [sim['最终金额'] for sim in simulation_results]
    total_profits = [sim['总收益'] for sim in simulation_results]
    annualized_returns = [sim['年化收益率'] for sim in simulation_results]
    return_rates = [sim['资产回报率'] for sim in simulation_results]
    
    # 计算统计数据
    stats = {
        'simulation_count': len(simulation_results),
        'total_investment': total_investment,
        'avg_final_amount': np.mean(final_amounts),
        'median_final_amount': np.median(final_amounts),
        'min_final_amount': np.min(final_amounts),
        'max_final_amount': np.max(final_amounts),
        'std_final_amount': np.std(final_amounts),
        'p5_final_amount': np.percentile(final_amounts, 5),
        'p95_final_amount': np.percentile(final_amounts, 95),
        'avg_total_profit': np.mean(total_profits),
        'median_total_profit': np.median(total_profits),
        'avg_annualized_return': np.mean(annualized_returns),
        'median_annualized_return': np.median(annualized_returns),
        'min_annualized_return': np.min(annualized_returns),
        'max_annualized_return': np.max(annualized_returns),
        'std_annualized_return': np.std(annualized_returns),
        'p5_annualized_return': np.percentile(annualized_returns, 5),
        'p95_annualized_return': np.percentile(annualized_returns, 95),
        'avg_return_rate': np.mean(return_rates),
        'median_return_rate': np.median(return_rates),
        'success_rate': np.mean(np.array(final_amounts) > total_investment) * 100,
        'currency': selected_currency
    }
    
    return stats


def generate_simulation_table(
    simulation_results: List[Dict[str, Any]],
    max_rows: int = 10
) -> pd.DataFrame:
    """生成模拟结果表格
    
    Args:
        simulation_results: 模拟结果列表
        max_rows: 最大行数
        
    Returns:
        pd.DataFrame: 模拟结果表格
    """
    # 提取数据
    data = []
    for i, sim in enumerate(simulation_results[:max_rows], 1):
        data.append({
            '模拟序号': i,
            '最终金额': sim['最终金额'],
            '总收益': sim['总收益'],
            '年化收益率': sim['年化收益率'],
            '资产回报率': sim['资产回报率']
        })
    
    # 创建DataFrame
    df = pd.DataFrame(data)
    
    return df
