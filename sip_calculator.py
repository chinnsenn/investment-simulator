import numpy as np
import statistics
import gradio as gr
from typing import Dict, List
import pandas as pd

def format_currency(amount: float) -> str:
    """格式化金额显示"""
    return f"{amount:,.2f}元"

def format_percentage(value: float) -> str:
    """格式化百分比显示"""
    return f"{value:,.2f}%"

def calculate_investment(
    investment_amount: float,
    avg_rate: float,
    years: int,
    volatility: float,
    frequency: str,
    simulation_mode: bool,
    simulation_rounds: int
) -> tuple:
    """投资计算主函数"""
    # 如果不是模拟模式，将波动率设为0，模拟轮数设为1
    if not simulation_mode:
        volatility = 0
        simulation_rounds = 1
    
    # 转换频率显示
    frequency_map = {
        "月度": "monthly",
        "季度": "quarterly",
        "年度": "yearly"
    }
    freq = frequency_map[frequency]
    
    # 计算年度投资金额
    yearly_investment = investment_amount * (12 if freq == 'monthly' else 4 if freq == 'quarterly' else 1)
    
    def generate_yearly_rates():
        """生成年化收益率"""
        if volatility == 0:
            return [avg_rate] * years
        return np.random.normal(avg_rate, volatility, years).tolist()

    def calculate_year_investment(yearly_rates):
        """计算年度投资结果"""
        results = []
        current_amount = 0
        periods_per_year = 12 if freq == 'monthly' else 4 if freq == 'quarterly' else 1
        period_investment = yearly_investment / periods_per_year
        total_investment = 0
        
        for year, rate in enumerate(yearly_rates, 1):
            period_rate = rate / 100 / periods_per_year
            year_start_amount = current_amount
            for _ in range(periods_per_year):
                # 在每个周期开始时进行投资
                current_amount += period_investment
                total_investment += period_investment
                # 投资后立即计算收益
                current_amount *= (1 + period_rate)
            
            year_investment = period_investment * periods_per_year
            year_profit = current_amount - year_start_amount - year_investment
            
            results.append({
                '年份': f"第{year}年",
                '年化收益率': format_percentage(rate),
                '投资金额': format_currency(year_investment),
                '当年收益': format_currency(year_profit),
                '年末总额': format_currency(current_amount),
                '累计投入': format_currency(total_investment)
            })
        return results, current_amount, total_investment, current_amount - total_investment

    # 存储所有模拟结果
    all_simulations = []
    
    for i in range(simulation_rounds):
        yearly_rates = generate_yearly_rates()
        results, final_amt, total_inv, total_prof = calculate_year_investment(yearly_rates)
        return_rate = (final_amt / total_inv * 100) - 100  # 计算资产回报率
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
    total_investment = all_simulations[0]['总投资']  # 总投资金额对所有模拟都是相同的

    # 构建结果摘要HTML
    summary_html = f"""
    <div style="background-color: #e3f2fd; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
        <h3 style="color: #1565c0; margin-top: 0;">💰 投资结果摘要</h3>
        <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;">
            <div style="background-color: #ffffff; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <h4 style="color: #1976d2; margin-top: 0;">总投入资金</h4>
                <p style="font-size: 1.2em; color: #2196f3;">{format_currency(total_investment)}</p>
            </div>
            <div style="background-color: #ffffff; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <h4 style="color: #1976d2; margin-top: 0;">账户总额</h4>
                <p style="font-size: 1.2em; color: #2196f3;">{format_currency(statistics.mean(final_amounts))}</p>
            </div>
        </div>
        <div style="background-color: #ffffff; padding: 15px; border-radius: 8px; margin-top: 15px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <h4 style="color: #1976d2; margin-top: 0;">投资回报率</h4>
            <p style="font-size: 1.2em; color: #2196f3;">{format_percentage(statistics.mean(return_rates))}</p>
        </div>
    </div>
    """

    # 如果是模拟模式且轮数大于2，添加统计信息
    if simulation_mode and simulation_rounds > 2:
        summary_html += f"""
        <div style="background-color: #f5f5f5; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
            <h3 style="color: #2c3e50;">📊 模拟统计结果</h3>
            <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 15px;">
                <div style="background-color: #c8e6c9; padding: 15px; border-radius: 8px;">
                    <h4 style="color: #2e7d32; margin-top: 0;">最优情况</h4>
                    <p>最终金额: {format_currency(max(final_amounts))}</p>
                    <p>总收益: {format_currency(max(total_profits))}</p>
                    <p>年化收益率: {format_percentage(max(annualized_returns))}</p>
                    <p>资产回报率: {format_percentage(max(return_rates))}</p>
                </div>
                <div style="background-color: #ffcdd2; padding: 15px; border-radius: 8px;">
                    <h4 style="color: #c62828; margin-top: 0;">最差情况</h4>
                    <p>最终金额: {format_currency(min(final_amounts))}</p>
                    <p>总收益: {format_currency(min(total_profits))}</p>
                    <p>年化收益率: {format_percentage(min(annualized_returns))}</p>
                    <p>资产回报率: {format_percentage(min(return_rates))}</p>
                </div>
                <div style="background-color: #bbdefb; padding: 15px; border-radius: 8px;">
                    <h4 style="color: #1565c0; margin-top: 0;">平均情况</h4>
                    <p>最终金额: {format_currency(statistics.mean(final_amounts))}</p>
                    <p>总收益: {format_currency(statistics.mean(total_profits))}</p>
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
                <p>每次投资: {format_currency(investment_amount)}</p>
                <p>年度投资: {format_currency(yearly_investment)}</p>
                <p>投资年限: {years}年</p>
            </div>
            <div style="background-color: #ffffff; padding: 15px; border-radius: 8px;">
                <h4 style="color: #3498db;">收益参数</h4>
                <p>目标收益率: {format_percentage(avg_rate)}</p>
                <p>波动率: {format_percentage(volatility)}</p>
                <p>计算模式: {'模拟模式' if simulation_mode else '固定收益率模式'}</p>
                <p>模拟轮数: {simulation_rounds if simulation_mode else 1}轮</p>
            </div>
        </div>
    </div>
    """

    # 添加使用说明
    output_html += """
    <div style="background-color: #fff3e0; padding: 20px; border-radius: 10px;">
        <h3 style="color: #2c3e50;">使用说明</h3>
        <ul style="list-style-type: none; padding-left: 0;">
            <li>🎯 波动率建议值：保守(5-8)、中等(8-12)、激进(12-15)</li>
            <li>📊 模拟模式下会进行多轮计算，结果更接近真实情况</li>
            <li>💰 所有计算结果仅供参考，实际投资收益受市场影响</li>
            <li>⚠️ 投资有风险，入市需谨慎</li>
        </ul>
    </div>
    """

    # 组合最终HTML（摘要在前，基本信息在后）
    final_html = summary_html + output_html

    # 返回年度详细数据（使用最后一次模拟或平均结果）
    if simulation_mode and simulation_rounds > 1:
        avg_results = []
        for year in range(years):
            avg_results.append({
                '年份': f"第{year+1}年",
                '年化收益率': format_percentage(statistics.mean([float(sim['详细数据'][year]['年化收益率'].rstrip('%')) for sim in all_simulations])),
                '投资金额': all_simulations[0]['详细数据'][year]['投资金额'],
                '当年收益': format_currency(statistics.mean([float(sim['详细数据'][year]['当年收益'].rstrip('元').replace(',', '')) for sim in all_simulations])),
                '年末总额': format_currency(statistics.mean([float(sim['详细数据'][year]['年末总额'].rstrip('元').replace(',', '')) for sim in all_simulations])),
                '累计投入': all_simulations[0]['详细数据'][year]['累计投入']
            })
        yearly_details = avg_results
    else:
        yearly_details = all_simulations[0]['详细数据']

    return final_html, pd.DataFrame(yearly_details)

def create_interface():
    with gr.Blocks(theme=gr.themes.Soft(), title="模拟 DCA 收益计算器") as demo:
        gr.Markdown("# 📈 模拟 DCA 收益计算器")
        
        with gr.Row():
            with gr.Column():
                investment_amount = gr.Number(
                    label="每次定投金额（元）",
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
                    choices=["月度", "季度", "年度"],
                    value="月度"
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
        
        # 更新模拟轮数滑块和波动率的可见性与值
        def update_simulation_settings(simulation_mode):
            return [
                gr.Slider(visible=simulation_mode),  # 模拟轮数滑块可见性
                gr.Number(value=0 if not simulation_mode else 8)  # 波动率值
            ]
        
        simulation_mode.change(
            update_simulation_settings,
            inputs=[simulation_mode],
            outputs=[simulation_rounds, volatility]
        )
        
        # 计算按钮点击事件
        calculate_btn.click(
            calculate_investment,
            inputs=[
                investment_amount,
                avg_rate,
                years,
                volatility,
                frequency,
                simulation_mode,
                simulation_rounds
            ],
            outputs=[output_html, output_table]
        )
    
    return demo

if __name__ == "__main__":
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        favicon_path="stocks.svg"
    )