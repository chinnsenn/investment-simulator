import numpy as np
import statistics
import gradio as gr
from typing import Dict, List
import pandas as pd
from decimal import Decimal
from dataclasses import dataclass
from enum import Enum

class Currency(Enum):
    CNY = ("CNY", "¥")
    USD = ("USD", "$")
    EUR = ("EUR", "€")
    GBP = ("GBP", "£")
    JPY = ("JPY", "¥")

    def __init__(self, code: str, symbol: str):
        self.code = code
        self.symbol = symbol

class InvestmentFrequency(Enum):
    HALF_MONTHLY = ("半月度", 24)
    MONTHLY = ("月度", 12)
    QUARTERLY = ("季度", 4)
    HALF_YEARLY = ("半年度", 2)
    YEARLY = ("年度", 1)

    def __init__(self, label: str, periods_per_year: int):
        self.label = label
        self.periods_per_year = periods_per_year

@dataclass
class InvestmentResult:
    year: int
    annual_rate: float
    investment_amount: float
    yearly_profit: float
    year_end_total: float
    cumulative_investment: float
    currency: Currency

def format_currency(amount: float, currency: Currency) -> str:
    """格式化货币显示"""
    if currency == Currency.JPY:
        # JPY通常不显示小数点
        return f"{currency.symbol}{int(amount):,}"
    return f"{currency.symbol}{amount:,.2f}"

def format_percentage(value: float) -> str:
    """格式化百分比显示"""
    return f"{value:,.2f}%"

def calculate_investment(
    investment_amount: float,
    avg_rate: float,
    years: int,
    volatility: float,
    frequency: str,
    currency: str,
    simulation_mode: bool,
    simulation_rounds: int
) -> tuple:
    """投资计算主函数"""
    if not simulation_mode:
        volatility = 0
        simulation_rounds = 1
    
    # 获取货币设置
    selected_currency = Currency[currency]
    
    # 获取频率设置
    selected_frequency = next(f for f in InvestmentFrequency if f.label == frequency)
    periods_per_year = selected_frequency.periods_per_year
    
    # 计算年度投资金额
    yearly_investment = investment_amount * periods_per_year
    
    def generate_yearly_rates():
        """生成年化收益率"""
        if volatility == 0:
            return [avg_rate] * years
        return np.random.normal(avg_rate, volatility, years).tolist()

    def calculate_year_investment(yearly_rates):
        """计算年度投资结果"""
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
            
            result = InvestmentResult(
                year=year,
                annual_rate=rate,
                investment_amount=year_investment,
                yearly_profit=year_profit,
                year_end_total=current_amount,
                cumulative_investment=total_investment,
                currency=selected_currency
            )
            
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
    
    for _ in range(simulation_rounds):
        yearly_rates = generate_yearly_rates()
        results, final_amt, total_inv, total_prof = calculate_year_investment(yearly_rates)
        return_rate = (final_amt / total_inv * 100) - 100
        all_simulations.append({
            '最终金额': final_amt,
            '总投资': total_inv,
            '总收益': total_prof,
            '年化收益率': ((final_amt/total_inv)**(1/years) - 1) * 100,
            '资产回报率': return_rate,
            '详细数据': results
        })

    # 计算统计结果
    final_amounts = [sim['最终金额'] for sim in all_simulations]
    total_profits = [sim['总收益'] for sim in all_simulations]
    annualized_returns = [sim['年化收益率'] for sim in all_simulations]
    return_rates = [sim['资产回报率'] for sim in all_simulations]
    total_investment = all_simulations[0]['总投资']

    # 构建结果摘要HTML
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
                <p style="font-size: 1.2em; color: #2196f3;">{format_currency(statistics.mean(final_amounts), selected_currency)}</p>
            </div>
        </div>
        <div style="background-color: #ffffff; padding: 15px; border-radius: 8px; margin-top: 15px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <h4 style="color: #1976d2; margin-top: 0;">投资回报率</h4>
            <p style="font-size: 1.2em; color: #2196f3;">{format_percentage(statistics.mean(return_rates))}</p>
        </div>
    </div>
    """

    if simulation_mode and simulation_rounds > 2:
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
                    <p>最终金额: {format_currency(statistics.mean(final_amounts), selected_currency)}</p>
                    <p>总收益: {format_currency(statistics.mean(total_profits), selected_currency)}</p>
                    <p>年化收益率: {format_percentage(statistics.mean(annualized_returns))}</p>
                    <p>资产回报率: {format_percentage(statistics.mean(return_rates))}</p>
                </div>
            </div>
        </div>
        """

    # 构建基本信息HTML
    output_html = f"""
    <div style="background-color: #f5f5f5; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
        <h3 style="color: #2c3e50;">投资参数</h3>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
            <div style="background-color: #ffffff; padding: 15px; border-radius: 8px;">
                <h4 style="color: #3498db;">基本信息</h4>
                <p>投资周期: {frequency}</p>
                <p>每次投资: {format_currency(investment_amount, selected_currency)}</p>
                <p>年度投资: {format_currency(yearly_investment, selected_currency)}</p>
                <p>投资年限: {years}年</p>
            </div>
            <div style="background-color: #ffffff; padding: 15px; border-radius: 8px;">
                <h4 style="color: #3498db;">收益参数</h4>
                <p>目标收益率: {format_percentage(avg_rate)}</p>
                <p>波动率: {format_percentage(volatility)}</p>
                <p>模拟轮数: {simulation_rounds}次</p>
                <p>货币类型: {selected_currency.code}</p>
            </div>
        </div>
    </div>
    """ + summary_html

    return output_html, pd.DataFrame(all_simulations[0]['详细数据'])

def create_interface():
    with gr.Blocks(theme=gr.themes.Soft(), title="多币种 DCA 收益计算器") as demo:
        gr.Markdown("# 📈 多币种 DCA 收益计算器")
        
        with gr.Row():
            with gr.Column():
                investment_amount = gr.Number(
                    label="每次定投金额",
                    value=1000,
                    minimum=0
                )
                avg_rate = gr.Number(
                    label="预期平均年化收益率（%）",
                    value=10
                )
                years = gr.Number(
                    label="投资年限（年）",
                    value=5,
                    minimum=1
                )
            
            with gr.Column():
                volatility = gr.Number(
                    label="收益率波动率",
                    value=8,
                    minimum=0
                )
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

        calculate_btn = gr.Button("开始计算", variant="primary")
        
        output_html = gr.HTML(label="计算结果")
        output_table = gr.DataFrame(label="年度详细数据")
        
        def update_simulation_settings(simulation_mode):
            return [
                gr.Slider(visible=simulation_mode),
                gr.Number(value=0 if not simulation_mode else 8)
            ]
        
        simulation_mode.change(
            update_simulation_settings,
            inputs=[simulation_mode],
            outputs=[simulation_rounds, volatility]
        )
        
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
                simulation_rounds
            ],
            outputs=[output_html, output_table]
        )
    
    return demo

if __name__ == "__main__":
    demo = create_interface()
    demo.launch()
