import numpy as np
import gradio as gr
import pandas as pd
import yfinance as yf
from typing import List
from classes import *
from utils.rate_distributions import RateDistributionModel
from utils.simulation_utils import simulate_rate_distribution, generate_rate_summary
from config import INVESTMENT_CONFIG
from utils.investment_utils import (
    format_currency, format_percentage, get_enum_by_label, get_currency_by_code,
    run_parallel_simulations, run_sequential_simulations,
    extract_simulation_statistics
)

def get_symbol_from_label(label: str) -> str:
    """根据标签获取股票代码
    
    Args:
        label: 股票标签
        
    Returns:
        str: 股票代码
    """
    try:
        select_symbol = next(s for s in IndexStock if s.label == label).symbol
    except Exception:
        select_symbol = label
    return select_symbol

def calculate_investment(
    investment_amount: float,
    avg_rate: float,
    years: int,
    volatility: float,
    frequency: str,
    currency: str,
    simulation_mode: bool,
    simulation_rounds: int,
    distribution_model: str,
    investment_timing: str = INVESTMENT_CONFIG.default_investment_timing
) -> str:
    """投资计算主函数
    
    Args:
        investment_amount: 年度投资金额
        avg_rate: 平均年化收益率(%)
        years: 投资年数
        volatility: 波动率(%)
        frequency: 投资频率
        currency: 货币代码
        simulation_mode: 是否启用模拟模式
        simulation_rounds: 模拟轮数
        distribution_model: 分布模型
        investment_timing: 投资时点
        
    Returns:
        str: 投资结果摘要HTML
    """
    # 如果不是模拟模式，则波动率为0，只进行一次模拟
    if not simulation_mode:
        volatility = 0
        simulation_rounds = 1
    
    # 获取枚举值
    selected_currency = get_currency_by_code(currency)
    selected_frequency = get_enum_by_label(frequency, InvestmentFrequency)
    selected_timing = get_enum_by_label(investment_timing, InvestmentTiming)
    
    # 计算投资参数
    periods_per_year = selected_frequency.periods_per_year
    yearly_investment = investment_amount * periods_per_year
    
    # 使用选择的分布模型生成收益率
    result = simulate_rate_distribution(
        avg_rate=avg_rate,
        volatility=volatility,
        years=years,
        simulation_rounds=simulation_rounds,
        distribution_model=RateDistributionModel[distribution_model],
        config=INVESTMENT_CONFIG
    )
    
    # 重塑收益率数组以便于模拟
    rates_reshaped = result.reshape(simulation_rounds, years)
    
    # 根据模拟轮数决定是否使用并行计算
    if simulation_rounds > INVESTMENT_CONFIG.min_simulations_for_parallel and INVESTMENT_CONFIG.use_parallel_computing:
        all_simulations = run_parallel_simulations(
            rates_reshaped,
            yearly_investment,
            periods_per_year,
            selected_timing,
            selected_currency,
            years
        )
    else:
        all_simulations = run_sequential_simulations(
            rates_reshaped,
            yearly_investment,
            periods_per_year,
            selected_timing,
            selected_currency,
            years
        )
    
    # 提取统计数据
    stats = extract_simulation_statistics(all_simulations)
    
    # 生成投资结果摘要
    summary_html = generate_investment_summary(
        stats['total_investment'], 
        stats['final_amounts'], 
        stats['total_profits'], 
        stats['annualized_returns'], 
        stats['return_rates'], 
        selected_currency
    )
    
    return summary_html

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
                    <p style="color: #000000;">最终金额: {format_currency(max(final_amounts), selected_currency)}</p>
                    <p style="color: #000000;">总收益: {format_currency(max(total_profits), selected_currency)}</p>
                    <p style="color: #000000;">年化收益率: {format_percentage(max(annualized_returns))}</p>
                    <p style="color: #000000;">资产回报率: {format_percentage(max(return_rates))}</p>
                </div>
                <div style="background-color: #ffcdd2; padding: 15px; border-radius: 8px;">
                    <h4 style="color: #c62828; margin-top: 0;">最差情况</h4>
                    <p style="color: #000000;">最终金额: {format_currency(min(final_amounts), selected_currency)}</p>
                    <p style="color: #000000;">总收益: {format_currency(min(total_profits), selected_currency)}</p>
                    <p style="color: #000000;">年化收益率: {format_percentage(min(annualized_returns))}</p>
                    <p style="color: #000000;">资产回报率: {format_percentage(min(return_rates))}</p>
                </div>
                <div style="background-color: #bbdefb; padding: 15px; border-radius: 8px;">
                    <h4 style="color: #1565c0; margin-top: 0;">平均情况</h4>
                    <p style="color: #000000;">最终金额: {format_currency(np.mean(final_amounts), selected_currency)}</p>
                    <p style="color: #000000;">总收益: {format_currency(np.mean(total_profits), selected_currency)}</p>
                    <p style="color: #000000;">年化收益率: {format_percentage(np.mean(annualized_returns))}</p>
                    <p style="color: #000000;">资产回报率: {format_percentage(np.mean(return_rates))}</p>
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
        distribution_model=RateDistributionModel[distribution_model],
        config=INVESTMENT_CONFIG
    )
    # 生成HTML摘要
    html_summary = generate_rate_summary(
        rates=result,
        avg_rate=avg_rate,
        volatility=volatility,
        distribution_model=RateDistributionModel[distribution_model].value
    )
    return html_summary

def get_nasdaq100_stats(symbol:str, years):
    try:
        ticker = symbol
        
        # 获取历史数据
        end_date = pd.Timestamp.now()
        start_date = end_date - pd.DateOffset(years=years)
        
        symbol = yf.Ticker(ticker)
        hist = symbol.history(start=start_date, end=end_date)
        
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
    choices = [f.label for f in IndexStock]
    with gr.Blocks(theme=gr.themes.Soft(), title="DCA 收益模拟计算器") as demo:
        gr.Markdown("# 📈 DCA 收益模拟计算器")
        gr.Markdown("本计算器旨在利用历史数据模拟定投复利收益的回测结果，其结论仅供参考，不构成对未来收益的保证。")
        gr.Markdown("## **投资市场具有风险，请投资者谨慎决策，理性参与。**")
        with gr.Row():
            with gr.Column():
                with gr.Column():
                    with gr.Row():
                        investment_amount = gr.Number(
                            label="每次定投金额",
                            value=1000.0,
                            minimum=0.0
                        )
                        years = gr.Number(
                            label="投资年限（年）",
                            value=5.0,
                            minimum=1.0
                        )
                    with gr.Row():
                        avg_rate = gr.Number(
                            label="预期平均年化收益率（%）",
                            value=10.0
                        )
                        volatility = gr.Number(
                            label="收益率波动率（标准差）",
                            value=8.0,
                            minimum=0.0
                        )
                with gr.Column():
                    with gr.Row():
                        symbollabel = gr.Dropdown(
                            choices=choices,
                            label="回测指标",
                            value=choices[0],
                            filterable=False,
                            allow_custom_value=False,
                            info="选择'自定义...'可输入新的指标"
                        )
                        data_years = gr.Slider(
                            label="回测年数",
                            minimum=2,
                            maximum=40,
                            value=20,
                            step=1,
                            visible=True,
                            info="选择回测年数"
                        )
                # 添加数据来源标签
                data_source_label = gr.Markdown(f"**[「{choices[0]}」数据来源](https://finance.yahoo.com/quote/{get_symbol_from_label(choices[0])})**")
                # 新增导入纳斯达克100数据按钮
                import_nasdaq_btn = gr.Button(f"📊 导入「{choices[0]}」的历史数据", variant="secondary")
            
            with gr.Column():
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
                    value=RateDistributionModel.LOGNORMAL.name,
                    visible=False
                )
                investment_timing = gr.Radio(
                    label="投资时点",
                    choices=[t.label for t in InvestmentTiming],
                    value=InvestmentTiming.MIDDLE.label
                )
                

        calculate_btn = gr.Button("开始计算", variant="primary")
        
        output_html = gr.HTML(label="计算结果")
        
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
                distribution_model,
                investment_timing
            ],
            outputs=[output_html]
        )
        
        def on_select(choice):
            """当选择改变时的处理函数"""
            if choice == IndexStock.CUSTOM.label:
                return gr.update(value="", filterable=True, allow_custom_value=True)
            else:
                return gr.update(value=choice, filterable=False, allow_custom_value=False)
        
        symbollabel.select(
            fn=on_select,
            inputs=[symbollabel],
            outputs=[symbollabel]
        )
        
        def on_dropdown_change(symbol):
            """
            When the dropdown list is changed, return a button with the text "  {symbol}  " and a radio button with the selected distribution model.
            
            Args:
                symbol (str): The selected symbol.
            
            Returns:
                tuple[gr.Button, gr.Radio]: A tuple of a button and a radio button.
            """
            model = RateDistributionModel.LOGNORMAL.name
            if(symbol == IndexStock.BTCF.label):
                model = RateDistributionModel.STUDENT_T.name
            else:
                model = RateDistributionModel.LOGNORMAL.name
            return gr.update(value=f"📊 导入「{symbol}」的历史回测数据"), gr.update(value=model), gr.Markdown(f"**[「{symbol}」数据来源](https://finance.yahoo.com/quote/{get_symbol_from_label(symbol)})**")
        
        # 监听下拉框的变化
        symbollabel.change(
            fn=on_dropdown_change,  # 处理函数
            inputs=[symbollabel],      # 输入组件
            outputs=[import_nasdaq_btn, distribution_model, data_source_label]        # 输出组件
        )
        
        # 添加导入数据的处理函数
        def import_nasdaq_data(symbol, years):
            """
            导入纳斯达克100指数的历史回测数据

            Parameters:
                symbol (str): 选择的股票代码
                years (int): 回测的年数

            Returns:
                list: [平均年化收益率 (%), 波动率 (%)]
            """

            avg_return, vol = get_nasdaq100_stats(get_symbol_from_label(symbol), years)
            return [avg_return, vol]
        
        import_nasdaq_btn.click(
            import_nasdaq_data,
            inputs=[
                symbollabel,
                data_years
                ],
            outputs=[avg_rate, volatility]
        )
    
    return demo

if __name__ == "__main__":
    demo = create_interface()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
