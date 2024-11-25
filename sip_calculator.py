import numpy as np
import gradio as gr
import pandas as pd
import yfinance as yf
from typing import List
from dataclasses import dataclass
from classes import *
from simulation import simulate_rate_distribution, RateDistributionModel, generate_rate_summary ,plot_rate_distribution

def format_currency(amount: float, currency: Currency) -> str:
    """格式化货币显示"""
    if currency == Currency.JPY:
        # JPY通常不显示小数点
        return f"{currency.symbol}{int(amount):,}"
    return f"{currency.symbol}{amount:,.2f}"

def format_percentage(value: float) -> str:
    """格式化百分比显示"""
    return f"{value:,.2f}%"

def generate_yearly_rates(
    avg_rate: float,
    years: int,
    volatility: float,
    distribution_model: RateDistributionModel = RateDistributionModel.NORMAL,
    **kwargs
) -> List[float]:
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
            df = kwargs.get('df', 3)
            rates = mu + sigma * np.random.standard_t(df, years)
        case RateDistributionModel.UNIFORM:
            min_rate = kwargs.get('min_rate', mu - sigma * np.sqrt(3))
            max_rate = kwargs.get('max_rate', mu + sigma * np.sqrt(3))
            rates = np.random.uniform(min_rate, max_rate, years)
        case _:
            raise ValueError(f"Unsupported distribution model: {distribution_model}")
    
    rates = rates * 100
    min_allowed_rate = kwargs.get('min_allowed_rate', -50)
    rates = np.maximum(rates, min_allowed_rate)
    
    return rates.tolist()

def calculate_investment(
    investment_amount: float,
    avg_rate: float,
    years: int,
    volatility: float,
    frequency: str,
    currency: str,
    simulation_mode: bool,
    simulation_rounds: int,
    distribution_model: str
) -> tuple:
    """投资计算主函数"""
    if not simulation_mode:
        volatility = 0
        simulation_rounds = 1
    
    selected_currency = Currency[currency]
    selected_frequency = next(f for f in InvestmentFrequency if f.label == frequency)
    periods_per_year = selected_frequency.periods_per_year
    yearly_investment = investment_amount * periods_per_year
    
    # 使用选择的分布模型生成收益率
    result = simulate_rate_distribution(
        avg_rate=avg_rate,
        volatility=volatility,
        years=years,
        simulation_rounds=simulation_rounds,
        distribution_model=RateDistributionModel[distribution_model]
    )
    
    def calculate_year_investment(yearly_rates):
        results = []
        current_amount = 0
        period_investment = yearly_investment / periods_per_year
        total_investment = 0
        
        for year, rate in enumerate(yearly_rates, 1):
            period_rate = rate / 100 / periods_per_year
            year_start_amount = current_amount
            
            for _ in range(periods_per_year):
                current_amount += period_investment
                total_investment += period_investment
                current_amount *= (1 + period_rate)
            
            year_investment = period_investment * periods_per_year
            year_profit = current_amount - year_start_amount - year_investment
            
            results.append({
                '年份': f"第{year}年",
                '年化收益率': format_percentage(rate),
                '投资金额': format_currency(year_investment, selected_currency),
                '当年收益': format_currency(year_profit, selected_currency),
                '年末总额': format_currency(current_amount, selected_currency),
                '累计投入': format_currency(total_investment, selected_currency)
            })
            
        return results, current_amount, total_investment, current_amount - total_investment

    # 存储所有模拟结果
    all_simulations = []
    
    # 使用生成的收益率进行模拟
    rates_reshaped = result.rates.reshape(simulation_rounds, years)
    for yearly_rates in rates_reshaped:
        sim_results, final_amt, total_inv, total_prof = calculate_year_investment(yearly_rates)
        return_rate = (final_amt / total_inv * 100) - 100
        all_simulations.append({
            '最终金额': final_amt,
            '总投资': total_inv,
            '总收益': total_prof,
            '年化收益率': ((final_amt/total_inv)**(1/years) - 1) * 100,
            '资产回报率': return_rate,
            '详细数据': sim_results
        })

    # 计算统计结果
    final_amounts = [sim['最终金额'] for sim in all_simulations]
    total_profits = [sim['总收益'] for sim in all_simulations]
    annualized_returns = [sim['年化收益率'] for sim in all_simulations]
    return_rates = [sim['资产回报率'] for sim in all_simulations]
    total_investment = all_simulations[0]['总投资']

    # 生成投资结果摘要
    summary_html = generate_investment_summary(
        total_investment, final_amounts, total_profits, 
        annualized_returns, return_rates, selected_currency
    )
    
    # 生成收益率分布图
    distribution_plot = plot_rate_distribution(result)
    
    # 合并HTML结果
    final_html = summary_html + distribution_plot
    
    return final_html

def generate_investment_summary(
    total_investment: float,
    final_amounts: List[float],
    total_profits: List[float],
    annualized_returns: List[float],
    return_rates: List[float],
    selected_currency: Currency
) -> str:
    """
    生成投资结果摘要的HTML
    
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
    summary_html = f"""
    <div style="background-color: #e3f2fd; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
        <h3 style="color: #1565c0; margin-top: 0;">💰 投资结果摘要</h3>
        <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;">
            <div style="background-color: #ffffff; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <h4 style="color: #1976d2; margin-top: 0;">总投入资金</h4>
                <p style="font-size: 1.2em; color: #2196f3;">{format_currency(total_investment, selected_currency)}</p>
            </div>
            <div style="background-color: #ffffff; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <h4 style="color: #1976d2; margin-top: 0;">账户总额</h4>
                <p style="font-size: 1.2em; color: #2196f3;">{format_currency(np.mean(final_amounts), selected_currency)}</p>
            </div>
        </div>
        <div style="background-color: #ffffff; padding: 15px; border-radius: 8px; margin-top: 15px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <h4 style="color: #1976d2; margin-top: 0;">投资回报率</h4>
            <p style="font-size: 1.2em; color: #2196f3;">{format_percentage(np.mean(return_rates))}</p>
        </div>
    </div>
    """

    if len(final_amounts) > 2:
        summary_html += f"""
        <div style="background-color: #f5f5f5; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
            <h3 style="color: #2c3e50;">📊 模拟统计结果</h3>
            <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 15px;">
                <div style="background-color: #c8e6c9; padding: 15px; border-radius: 8px;">
                    <h4 style="color: #2e7d32; margin-top: 0;">最优情况</h4>
                    <p>最终金额: {format_currency(max(final_amounts), selected_currency)}</p>
                    <p>总收益: {format_currency(max(total_profits), selected_currency)}</p>
                    <p>年化收益率: {format_percentage(max(annualized_returns))}</p>
                    <p>资产回报率: {format_percentage(max(return_rates))}</p>
                </div>
                <div style="background-color: #ffcdd2; padding: 15px; border-radius: 8px;">
                    <h4 style="color: #c62828; margin-top: 0;">最差情况</h4>
                    <p>最终金额: {format_currency(min(final_amounts), selected_currency)}</p>
                    <p>总收益: {format_currency(min(total_profits), selected_currency)}</p>
                    <p>年化收益率: {format_percentage(min(annualized_returns))}</p>
                    <p>资产回报率: {format_percentage(min(return_rates))}</p>
                </div>
                <div style="background-color: #bbdefb; padding: 15px; border-radius: 8px;">
                    <h4 style="color: #1565c0; margin-top: 0;">平均情况</h4>
                    <p>最终金额: {format_currency(np.mean(final_amounts), selected_currency)}</p>
                    <p>总收益: {format_currency(np.mean(total_profits), selected_currency)}</p>
                    <p>年化收益率: {format_percentage(np.mean(annualized_returns))}</p>
                    <p>资产回报率: {format_percentage(np.mean(return_rates))}</p>
                </div>
            </div>
        </div>
        """

    return summary_html

def display_simulation_results(
    avg_rate, volatility, years, simulation_rounds, distribution_model
):
    # 调用模拟函数
    result = simulate_rate_distribution(
        avg_rate=avg_rate,
        volatility=volatility,
        years=years,
        simulation_rounds=simulation_rounds,
        distribution_model=RateDistributionModel[distribution_model]
    )
    # 生成HTML摘要
    html_summary = generate_rate_summary(result)
    return html_summary

def get_nasdaq100_stats(years=40):
    """获取纳斯达克100指数历史数据统计"""
    try:
        # 使用纳斯达克100 ETF (QQQ)的数据
        ticker = "QQQ"
        
        # 获取历史数据
        end_date = pd.Timestamp.now()
        start_date = end_date - pd.DateOffset(years=years)
        
        qqq = yf.Ticker(ticker)
        hist = qqq.history(start=start_date, end=end_date)
        
        # 计算年化收益率
        annual_returns = hist['Close'].pct_change().dropna()
        avg_annual_return = (1 + annual_returns.mean()) ** 252 - 1  # 252个交易日
        
        # 计算标准差（波动率）
        volatility = annual_returns.std() * np.sqrt(252)
        
        # 保留两位小数并确保返回float类型
        return (
            float(round(avg_annual_return * 100, 2)),
            float(round(volatility * 100, 2))
        )
    except Exception as e:
        print(f"获取数据失败: {str(e)}")
        return 10.0, 8.0  # 返回默认值

def create_interface():
    with gr.Blocks(theme=gr.themes.Soft(), title="多币种 DCA 收益计算器") as demo:
        gr.Markdown("# 📈 多币种 DCA 收益计算器")
        
        with gr.Row():
            with gr.Column():
                investment_amount = gr.Number(
                    label="每次定投金额",
                    value=1000.0,
                    minimum=0.0
                )
                avg_rate = gr.Number(
                    label="预期平均年化收益率（%）",
                    value=10.0
                )
                years = gr.Number(
                    label="投资年限（年）",
                    value=5.0,
                    minimum=1.0
                )
            
            with gr.Column():
                volatility = gr.Number(
                    label="收益率波动率",
                    value=8.0,
                    minimum=0.0
                )
                # 新增导入纳斯达克100数据按钮
                import_nasdaq_btn = gr.Button("📊 导入纳斯达克100历史数据", variant="secondary")
                frequency = gr.Radio(
                    label="定投周期",
                    choices=[f.label for f in InvestmentFrequency],
                    value=InvestmentFrequency.MONTHLY.label
                )
                currency = gr.Radio(
                    label="货币类型",
                    choices=[c.code for c in Currency],
                    value=Currency.CNY.code
                )
                simulation_mode = gr.Checkbox(
                    label="真实模拟模式",
                    value=True
                )
                simulation_rounds = gr.Slider(
                    label="模拟轮数",
                    minimum=1,
                    maximum=20,
                    value=5,
                    step=1,
                    visible=True
                )
                distribution_model = gr.Radio(
                    label="收益率分布模型",
                    choices=[model.name for model in RateDistributionModel],
                    value=RateDistributionModel.NORMAL.name
                )

        calculate_btn = gr.Button("开始计算", variant="primary")
        
        output_html = gr.HTML(label="计算结果")
        
        # 添加导入数据的处理函数
        def import_nasdaq_data():
            avg_return, vol = get_nasdaq100_stats()
            return [avg_return, vol]
        
        calculate_btn.click(
            calculate_investment,
            inputs=[
                investment_amount,
                avg_rate,
                years,
                volatility,
                frequency,
                currency,
                simulation_mode,
                simulation_rounds,
                distribution_model
            ],
            outputs=[output_html]
        )
        
        import_nasdaq_btn.click(
            import_nasdaq_data,
            outputs=[avg_rate, volatility]
        )
    
    return demo

if __name__ == "__main__":
    demo = create_interface()
    demo.launch()
