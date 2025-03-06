import numpy as np
import gradio as gr
import pandas as pd
import yfinance as yf
from typing import List, Dict, Any, Tuple, Callable
from dataclasses import dataclass
from classes import *
from simulation import simulate_rate_distribution, RateDistributionModel, generate_rate_summary, plot_rate_distribution
import concurrent.futures
from functools import lru_cache
import time

def format_currency(amount: float, currency: Currency) -> str:
    """格式化货币显示"""
    if currency == Currency.JPY:
        # JPY通常不显示小数点
        return f"{currency.symbol}{int(amount):,}"
    return f"{currency.symbol}{amount:,.2f}"

def format_percentage(value: float) -> str:
    """格式化百分比显示"""
    return f"{value:,.2f}%"
        
def get_symbol_from_label(label):
    try:
        select_symbol = next(s for s in IndexStock if s.label == label).symbol
    except Exception:
        select_symbol = label
    return select_symbol

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

def calculate_period_growth(period_rate: float, period_investment: float, current_amount: float, timing_type: str) -> Tuple[float, float]:
    """计算单期增长，使用缓存减少重复计算
    
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

def calculate_year_investment_vectorized(yearly_rates, yearly_investment, periods_per_year, selected_timing, selected_currency):
    """向量化版本的年度投资计算函数
    
    Args:
        yearly_rates: 年度收益率数组
        yearly_investment: 年度投资总额
        periods_per_year: 每年投资期数
        selected_timing: 选择的投资时点
        selected_currency: 选择的货币类型
        
    Returns:
        tuple: (结果列表, 最终金额, 总投资, 总收益)
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
    investment_timing: str = "期中"  # 默认为期中投资
) -> tuple:
    """投资计算主函数"""
    if not simulation_mode:
        volatility = 0
        simulation_rounds = 1
    
    selected_currency = Currency[currency]
    selected_frequency = next(f for f in InvestmentFrequency if f.label == frequency)
    selected_timing = next(t for t in InvestmentTiming if t.label == investment_timing)
    
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
    
    # 存储所有模拟结果
    all_simulations = []
    
    # 使用生成的收益率进行模拟
    rates_reshaped = result.rates.reshape(simulation_rounds, years)
    
    # 使用并行计算加速多次模拟
    if simulation_rounds > 1:
        start_time = time.time()
        with concurrent.futures.ProcessPoolExecutor() as executor:
            # 创建并行任务，传递所有必要参数
            futures = []
            for i, yearly_rates in enumerate(rates_reshaped):
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
                # 使用复合收益率公式计算年化收益率
                annualized_return = ((final_amt/total_inv)**(1/years) - 1) * 100
                return_rate = (final_amt / total_inv * 100) - 100
                all_simulations.append({
                    '最终金额': final_amt,
                    '总投资': total_inv,
                    '总收益': total_prof,
                    '年化收益率': annualized_return,
                    '资产回报率': return_rate,
                    '详细数据': sim_results
                })
        print(f"并行计算完成，耗时: {time.time() - start_time:.2f}秒")
    else:
        # 单次模拟不需要并行
        for yearly_rates in rates_reshaped:
            sim_results, final_amt, total_inv, total_prof = calculate_year_investment_vectorized(
                yearly_rates, 
                yearly_investment, 
                periods_per_year, 
                selected_timing, 
                selected_currency
            )
            # 使用复合收益率公式计算年化收益率
            annualized_return = ((final_amt/total_inv)**(1/years) - 1) * 100
            return_rate = (final_amt / total_inv * 100) - 100
            all_simulations.append({
                '最终金额': final_amt,
                '总投资': total_inv,
                '总收益': total_prof,
                '年化收益率': annualized_return,
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
        distribution_model=RateDistributionModel[distribution_model]
    )
    # 生成HTML摘要
    html_summary = generate_rate_summary(result)
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
    demo.launch(server_port=7860, share=True)
