"""
投资可视化模块
提供生成投资结果可视化图表的功能
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from typing import List, Dict, Any, Optional, Tuple
import io
import base64
from classes import Currency
from utils.investment_utils import format_currency, format_percentage


def create_histogram_plot(
    data: List[float],
    title: str,
    xlabel: str,
    ylabel: str = "频率",
    bins: int = 30,
    color: str = "skyblue",
    edgecolor: str = "black",
    alpha: float = 0.7,
    figsize: Tuple[int, int] = (10, 6),
    show_stats: bool = True,
    formatter=None
) -> plt.Figure:
    """创建直方图
    
    Args:
        data: 数据列表
        title: 图表标题
        xlabel: x轴标签
        ylabel: y轴标签
        bins: 直方图的箱数
        color: 填充颜色
        edgecolor: 边缘颜色
        alpha: 透明度
        figsize: 图表大小
        show_stats: 是否显示统计数据
        formatter: 格式化函数，用于格式化统计数据
        
    Returns:
        plt.Figure: matplotlib图形对象
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # 绘制直方图
    n, bins, patches = ax.hist(data, bins=bins, color=color, edgecolor=edgecolor, alpha=alpha)
    
    # 设置标题和标签
    ax.set_title(title, fontsize=14)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    
    # 添加网格线
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # 计算并显示统计数据
    if show_stats:
        mean_val = np.mean(data)
        median_val = np.median(data)
        std_val = np.std(data)
        
        # 格式化统计数据
        if formatter:
            mean_str = formatter(mean_val)
            median_str = formatter(median_val)
            std_str = formatter(std_val)
        else:
            mean_str = f"{mean_val:.2f}"
            median_str = f"{median_val:.2f}"
            std_str = f"{std_val:.2f}"
        
        # 添加统计数据文本
        stats_text = f"均值: {mean_str}\n中位数: {median_str}\n标准差: {std_str}"
        ax.text(0.95, 0.95, stats_text,
                transform=ax.transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 绘制均值和中位数线
        ax.axvline(mean_val, color='r', linestyle='dashed', linewidth=1, label=f'均值: {mean_str}')
        ax.axvline(median_val, color='g', linestyle='dashed', linewidth=1, label=f'中位数: {median_str}')
        
        # 添加图例
        ax.legend()
    
    # 调整布局
    plt.tight_layout()
    
    return fig


def create_investment_growth_chart(
    years: int,
    investment_amount: float,
    final_amounts: List[float],
    selected_currency: Currency,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """创建投资增长图表
    
    Args:
        years: 投资年数
        investment_amount: 投资金额
        final_amounts: 最终金额列表
        selected_currency: 选择的货币类型
        figsize: 图表大小
        
    Returns:
        plt.Figure: matplotlib图形对象
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # 计算百分位数
    p5 = np.percentile(final_amounts, 5)
    p25 = np.percentile(final_amounts, 25)
    p50 = np.percentile(final_amounts, 50)
    p75 = np.percentile(final_amounts, 75)
    p95 = np.percentile(final_amounts, 95)
    
    # 创建时间点
    time_points = np.linspace(0, years, 100)
    
    # 假设线性增长来简化可视化
    def growth_curve(final_value):
        return investment_amount + (final_value - investment_amount) * (time_points / years) ** 1.5
    
    # 绘制不同百分位数的增长曲线
    ax.fill_between(time_points, growth_curve(p5), growth_curve(p95), alpha=0.2, color='blue', label='5%-95%区间')
    ax.fill_between(time_points, growth_curve(p25), growth_curve(p75), alpha=0.3, color='blue', label='25%-75%区间')
    ax.plot(time_points, growth_curve(p50), 'b-', linewidth=2, label='中位数增长')
    ax.plot([0, years], [investment_amount, investment_amount], 'r--', label='初始投资')
    
    # 设置标题和标签
    ax.set_title('投资增长预测', fontsize=14)
    ax.set_xlabel('年数', fontsize=12)
    ax.set_ylabel('投资价值', fontsize=12)
    
    # 格式化y轴标签为货币格式
    def currency_formatter(x, pos):
        return format_currency(x, selected_currency, include_symbol=False)
    
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(currency_formatter))
    
    # 添加网格线
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # 添加图例
    ax.legend(loc='upper left')
    
    # 添加注释
    median_final = p50
    median_profit = median_final - investment_amount
    median_return_rate = (median_final / investment_amount - 1) * 100
    
    annotation_text = (
        f"中位数最终金额: {format_currency(median_final, selected_currency)}\n"
        f"中位数总收益: {format_currency(median_profit, selected_currency)}\n"
        f"中位数回报率: {format_percentage(median_return_rate/100)}"
    )
    
    ax.text(0.95, 0.05, annotation_text,
            transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 调整布局
    plt.tight_layout()
    
    return fig


def create_return_distribution_chart(
    annualized_returns: List[float],
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """创建收益率分布图表
    
    Args:
        annualized_returns: 年化收益率列表
        figsize: 图表大小
        
    Returns:
        plt.Figure: matplotlib图形对象
    """
    # 创建格式化函数
    def percentage_formatter(x):
        return format_percentage(x)
    
    # 创建直方图
    fig = create_histogram_plot(
        data=annualized_returns,
        title="年化收益率分布",
        xlabel="年化收益率",
        bins=30,
        color="lightgreen",
        figsize=figsize,
        formatter=percentage_formatter
    )
    
    return fig


def create_final_amount_distribution_chart(
    final_amounts: List[float],
    selected_currency: Currency,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """创建最终金额分布图表
    
    Args:
        final_amounts: 最终金额列表
        selected_currency: 选择的货币类型
        figsize: 图表大小
        
    Returns:
        plt.Figure: matplotlib图形对象
    """
    # 创建格式化函数
    def currency_formatter(x):
        return format_currency(x, selected_currency, include_symbol=True)
    
    # 创建直方图
    fig = create_histogram_plot(
        data=final_amounts,
        title="最终金额分布",
        xlabel="最终金额",
        bins=30,
        color="lightblue",
        figsize=figsize,
        formatter=currency_formatter
    )
    
    return fig


def fig_to_base64(fig: plt.Figure) -> str:
    """将matplotlib图形转换为base64编码的字符串
    
    Args:
        fig: matplotlib图形对象
        
    Returns:
        str: base64编码的图像字符串
    """
    # 创建内存中的字节流
    buf = io.BytesIO()
    
    # 保存图形到字节流
    fig.savefig(buf, format='png', dpi=100)
    
    # 重置缓冲区位置
    buf.seek(0)
    
    # 将图像转换为base64编码
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    
    # 关闭图形以释放内存
    plt.close(fig)
    
    return img_str


def generate_investment_visualizations(
    years: int,
    investment_amount: float,
    final_amounts: List[float],
    annualized_returns: List[float],
    selected_currency: Currency
) -> Dict[str, str]:
    """生成投资可视化图表
    
    Args:
        years: 投资年数
        investment_amount: 投资金额
        final_amounts: 最终金额列表
        annualized_returns: 年化收益率列表
        selected_currency: 选择的货币类型
        
    Returns:
        Dict[str, str]: 包含base64编码图像的字典
    """
    # 创建图表
    growth_chart = create_investment_growth_chart(
        years=years,
        investment_amount=investment_amount,
        final_amounts=final_amounts,
        selected_currency=selected_currency
    )
    
    final_amount_chart = create_final_amount_distribution_chart(
        final_amounts=final_amounts,
        selected_currency=selected_currency
    )
    
    return_chart = create_return_distribution_chart(
        annualized_returns=annualized_returns
    )
    
    # 转换为base64
    growth_chart_b64 = fig_to_base64(growth_chart)
    final_amount_chart_b64 = fig_to_base64(final_amount_chart)
    return_chart_b64 = fig_to_base64(return_chart)
    
    # 返回结果
    return {
        'growth_chart': growth_chart_b64,
        'final_amount_chart': final_amount_chart_b64,
        'return_chart': return_chart_b64
    }
