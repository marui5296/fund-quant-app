import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 设置专业级页面配置
st.set_page_config(
    page_title="AlphaFund Pro - 投资模拟与策略回测系统",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 专业CSS样式
st.markdown("""
<style>
    .professional-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
        text-align: center;
    }
    .strategy-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #dee2e6;
        margin-bottom: 1rem;
        transition: all 0.3s ease;
    }
    .strategy-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #007bff;
        margin-bottom: 1rem;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    .performance-good { color: #28a745; font-weight: bold; }
    .performance-neutral { color: #ffc107; font-weight: bold; }
    .performance-bad { color: #dc3545; font-weight: bold; }
    .tab-content {
        padding: 1.5rem;
        background-color: white;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

class InvestmentSimulator:
    """投资模拟引擎"""
    
    def __init__(self, risk_free_rate=0.015):
        self.risk_free_rate = risk_free_rate
        self.strategies = {}
        
    def generate_realistic_fund_data(self, fund_info, start_date='2020-01-01'):
        """生成真实感基金数据"""
        np.random.seed(hash(fund_info['code']) % 10000)
        
        # 基于基金特性设置参数
        risk_level = fund_info['risk']
        if risk_level == "高风险":
            base_volatility = 0.025
            base_return = 0.0012
        elif risk_level == "中高风险":
            base_volatility = 0.018
            base_return = 0.0009
        else:
            base_volatility = 0.012
            base_return = 0.0006
        
        # 创建日期范围（仅工作日）
        dates = pd.date_range(start=start_date, end=datetime.now(), freq='B')
        
        # 生成更真实的收益率序列
        returns = np.random.normal(base_return, base_volatility, len(dates))
        
        # 添加市场相关性（如果多只基金）
        market_factor = np.random.normal(0.0005, 0.01, len(dates))
        returns = returns * 0.7 + market_factor * 0.3
        
        # 添加季节性效应
        seasonal_factor = np.sin(np.arange(len(dates)) / 252 * 2 * np.pi) * 0.0005
        returns = returns + seasonal_factor
        
        # 生成净值序列
        nav = 1.0 * (1 + pd.Series(returns)).cumprod()
        
        return pd.DataFrame({
            'date': dates,
            'nav': nav.values,
            'return': returns
        }).set_index('date')
    
    def execute_strategy(self, strategy_type, fund_data, initial_capital, **params):
        """执行投资策略"""
        
        if strategy_type == "一次性买入":
            return self._lump_sum_investment(fund_data, initial_capital)
        elif strategy_type == "定期定额":
            return self._dollar_cost_averaging(fund_data, initial_capital, params)
        elif strategy_type == "价值平均":
            return self._value_averaging(fund_data, initial_capital, params)
        elif strategy_type == "金字塔买入":
            return self._pyramid_buying(fund_data, initial_capital, params)
        elif strategy_type == "网格交易":
            return self._grid_trading(fund_data, initial_capital, params)
        elif strategy_type == "均线策略":
            return self._moving_average_strategy(fund_data, initial_capital, params)
        elif strategy_type == "动态平衡":
            return self._dynamic_balance(fund_data, initial_capital, params)
        else:
            raise ValueError(f"未知策略: {strategy_type}")
    
    def _lump_sum_investment(self, fund_data, initial_capital):
        """一次性买入策略"""
        nav = fund_data['nav']
        shares = initial_capital / nav.iloc[0]
        portfolio_value = shares * nav
        trades = [{'date': nav.index[0], 'action': 'BUY', 'shares': shares, 'price': nav.iloc[0]}]
        
        return {
            'portfolio_value': portfolio_value,
            'shares': pd.Series(shares, index=nav.index),
            'trades': trades,
            'cash': pd.Series(0, index=nav.index)
        }
    
    def _dollar_cost_averaging(self, fund_data, initial_capital, params):
        """定期定额投资策略"""
        nav = fund_data['nav']
        interval = params.get('interval', 30)  # 天
        amount = params.get('amount', 1000)    # 每次投入金额
        
        cash = initial_capital
        shares = 0
        portfolio_value = []
        trades = []
        
        for i, (date, price) in enumerate(nav.items()):
            # 定期投入
            if i % interval == 0 and cash >= amount:
                buy_shares = amount / price
                shares += buy_shares
                cash -= amount
                trades.append({'date': date, 'action': 'BUY', 'shares': buy_shares, 'price': price})
            
            portfolio_value.append(shares * price + cash)
        
        return {
            'portfolio_value': pd.Series(portfolio_value, index=nav.index),
            'shares': pd.Series(shares, index=nav.index),
            'trades': trades,
            'cash': pd.Series(cash, index=nav.index)
        }
    
    def _pyramid_buying(self, fund_data, initial_capital, params):
        """金字塔买入策略"""
        nav = fund_data['nav']
        buy_levels = params.get('buy_levels', [0, -0.05, -0.10, -0.15])  # 买入触发点
        buy_amounts = params.get('buy_amounts', [0.2, 0.3, 0.3, 0.2])    # 各层买入比例
        
        # 初始买入
        initial_buy_amount = initial_capital * buy_amounts[0]
        shares = initial_buy_amount / nav.iloc[0]
        cash = initial_capital - initial_buy_amount
        
        portfolio_value = []
        trades = []
        trigger_points = []  # 记录触发点
        
        # 计算参考价格（初始价格）
        reference_price = nav.iloc[0]
        
        for i, (date, price) in enumerate(nav.items()):
            # 检查是否需要加仓
            drawdown = (price - reference_price) / reference_price
            
            for level_idx, level in enumerate(buy_levels[1:], 1):
                if drawdown <= level and level not in trigger_points:
                    # 计算买入金额
                    buy_amount = initial_capital * buy_amounts[level_idx]
                    if cash >= buy_amount:
                        buy_shares = buy_amount / price
                        shares += buy_shares
                        cash -= buy_amount
                        trades.append({
                            'date': date, 
                            'action': 'BUY', 
                            'shares': buy_shares, 
                            'price': price,
                            'level': f"第{level_idx}层"
                        })
                        trigger_points.append(level)
            
            portfolio_value.append(shares * price + cash)
        
        return {
            'portfolio_value': pd.Series(portfolio_value, index=nav.index),
            'shares': pd.Series(shares, index=nav.index),
            'trades': trades,
            'cash': pd.Series(cash, index=nav.index)
        }
    
    def _moving_average_strategy(self, fund_data, initial_capital, params):
        """均线策略"""
        nav = fund_data['nav']
        short_window = params.get('short_window', 20)
        long_window = params.get('long_window', 50)
        
        # 计算移动平均
        short_ma = nav.rolling(window=short_window).mean()
        long_ma = nav.rolling(window=long_window).mean()
        
        cash = initial_capital
        shares = 0
        portfolio_value = []
        trades = []
        position = 0  # 0:空仓, 1:持仓
        
        for i in range(max(short_window, long_window), len(nav)):
            date = nav.index[i]
            price = nav.iloc[i]
            
            # 金叉买入，死叉卖出
            if short_ma.iloc[i] > long_ma.iloc[i] and position == 0:
                # 买入
                shares = cash / price
                cash = 0
                position = 1
                trades.append({
                    'date': date, 
                    'action': 'BUY', 
                    'shares': shares, 
                    'price': price,
                    'signal': '金叉'
                })
            elif short_ma.iloc[i] < long_ma.iloc[i] and position == 1:
                # 卖出
                cash = shares * price
                trades.append({
                    'date': date, 
                    'action': 'SELL', 
                    'shares': shares, 
                    'price': price,
                    'signal': '死叉'
                })
                shares = 0
                position = 0
            
            portfolio_value.append(shares * price + cash)
        
        # 填充前期的空值
        for i in range(max(short_window, long_window)):
            portfolio_value.insert(0, initial_capital)
        
        return {
            'portfolio_value': pd.Series(portfolio_value, index=nav.index),
            'shares': pd.Series(shares, index=nav.index),
            'trades': trades,
            'cash': pd.Series(cash, index=nav.index),
            'signals': pd.DataFrame({
                'price': nav,
                'short_ma': short_ma,
                'long_ma': long_ma
            })
        }
    
    def calculate_performance_metrics(self, portfolio_value, benchmark_value=None):
        """计算投资组合绩效指标"""
        returns = portfolio_value.pct_change().dropna()
        
        metrics = {}
        
        # 基础收益指标
        total_return = (portfolio_value.iloc[-1] / portfolio_value.iloc[0]) - 1
        annual_return = (1 + total_return) ** (252 / len(portfolio_value)) - 1
        
        # 风险指标
        volatility = returns.std() * np.sqrt(252)
        downside_returns = returns[returns < 0]
        downside_volatility = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        
        # 风险调整收益指标
        sharpe_ratio = (annual_return - self.risk_free_rate) / volatility if volatility > 0 else 0
        sortino_ratio = (annual_return - self.risk_free_rate) / downside_volatility if downside_volatility > 0 else 0
        
        # 最大回撤
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Calmar比率
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # 胜率
        winning_days = (returns > 0).sum()
        total_days = len(returns)
        win_rate = winning_days / total_days if total_days > 0 else 0
        
        # 盈亏比
        avg_win = returns[returns > 0].mean() if len(returns[returns > 0]) > 0 else 0
        avg_loss = abs(returns[returns < 0].mean()) if len(returns[returns < 0]) > 0 else 0
        profit_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0
        
        metrics.update({
            '累计收益率': total_return,
            '年化收益率': annual_return,
            '年化波动率': volatility,
            '最大回撤': max_drawdown,
            '夏普比率': sharpe_ratio,
            '索提诺比率': sortino_ratio,
            '卡玛比率': calmar_ratio,
            '胜率': win_rate,
            '盈亏比': profit_loss_ratio,
            '交易天数': total_days
        })
        
        # 如果提供了基准，计算超额收益
        if benchmark_value is not None:
            benchmark_return = (benchmark_value.iloc[-1] / benchmark_value.iloc[0]) - 1
            excess_return = total_return - benchmark_return
            
            # 计算信息比率
            excess_returns = portfolio_value.pct_change() - benchmark_value.pct_change()
            tracking_error = excess_returns.std() * np.sqrt(252)
            information_ratio = excess_return / tracking_error if tracking_error > 0 else 0
            
            metrics.update({
                '基准收益率': benchmark_return,
                '超额收益率': excess_return,
                '信息比率': information_ratio
            })
        
        return metrics

def main():
    # 专业标题
    st.markdown("""
    <div class="professional-header">
        <h1>🚀 AlphaFund Pro - 投资模拟与策略回测系统</h1>
        <p>专业的基金投资策略模拟与回测平台</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 初始化投资模拟引擎
    simulator = InvestmentSimulator()
    
    # 侧边栏 - 配置区域
    st.sidebar.header("🔧 投资配置")
    
    # 基金数据库
    FUND_UNIVERSE = {
        "022365": {"name": "永赢科技智选混合C", "category": "科技主题", "risk": "高风险"},
        "001618": {"name": "天弘中证电子ETF联接A", "category": "科技主题", "risk": "高风险"},
        "110022": {"name": "易方达消费行业股票", "category": "消费主题", "risk": "中高风险"},
        "161725": {"name": "招商中证白酒指数", "category": "消费主题", "risk": "高风险"},
        "005827": {"name": "易方达蓝筹精选混合", "category": "均衡配置", "risk": "中高风险"},
        "000961": {"name": "天弘沪深300ETF联接A", "category": "宽基指数", "risk": "中风险"},
        "519697": {"name": "交银优势行业混合", "category": "灵活配置", "risk": "中高风险"},
        "002190": {"name": "农银新能源主题", "category": "新能源主题", "risk": "高风险"},
    }
    
    # 投资配置
    selected_funds = st.sidebar.multiselect(
        "选择投资基金",
        options=list(FUND_UNIVERSE.keys()),
        format_func=lambda x: f"{x} - {FUND_UNIVERSE[x]['name']}",
        default=["022365"]
    )
    
    initial_capital = st.sidebar.number_input("初始资金 (元)", value=100000, min_value=1000, step=1000)
    
    # 选择策略
    strategy_options = {
        "一次性买入": "最简单的投资方式，一次性投入全部资金",
        "定期定额": "定期投入固定金额，适合长期投资",
        "金字塔买入": "价格下跌时逐步加仓，越跌买得越多",
        "均线策略": "基于移动平均线的趋势跟踪策略"
    }
    
    selected_strategy = st.sidebar.selectbox(
        "选择投资策略",
        options=list(strategy_options.keys()),
        format_func=lambda x: f"{x} - {strategy_options[x]}"
    )
    
    # 策略参数配置
    st.sidebar.header("⚙️ 策略参数")
    
    if selected_strategy == "定期定额":
        interval = st.sidebar.slider("定投周期 (天)", 7, 90, 30)
        amount = st.sidebar.number_input("每次定投金额 (元)", value=2000, min_value=100, step=100)
        strategy_params = {'interval': interval, 'amount': amount}
        
    elif selected_strategy == "金字塔买入":
        st.sidebar.markdown("**金字塔买入策略配置**")
        levels = st.sidebar.slider("买入层级", 2, 5, 3)
        
        buy_levels = [0]
        buy_amounts = []
        
        for i in range(1, levels + 1):
            col1, col2 = st.sidebar.columns(2)
            with col1:
                level = st.number_input(f"第{i}层触发跌幅 (%)", 
                                       value=5*i, min_value=1, max_value=50, key=f"level_{i}")
            with col2:
                amount_pct = st.number_input(f"第{i}层仓位 (%)", 
                                           value=int(100/levels), min_value=1, max_value=100, key=f"amount_{i}")
            
            buy_levels.append(-level/100)
            buy_amounts.append(amount_pct/100)
        
        strategy_params = {'buy_levels': buy_levels, 'buy_amounts': buy_amounts}
        
    elif selected_strategy == "均线策略":
        col1, col2 = st.sidebar.columns(2)
        with col1:
            short_window = st.slider("短期均线周期", 5, 60, 20)
        with col2:
            long_window = st.slider("长期均线周期", 20, 200, 50)
        
        strategy_params = {'short_window': short_window, 'long_window': long_window}
        
    else:  # 一次性买入
        strategy_params = {}
    
    # 回测时间范围
    st.sidebar.header("📅 回测设置")
    backtest_period = st.sidebar.selectbox("回测时间范围", 
                                          ["3个月", "6个月", "1年", "2年", "3年"], 
                                          index=2)
    
    # 主内容区域
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 策略回测", "📊 绩效分析", "📈 对比分析", "💡 策略建议"])
    
    with tab1:
        st.markdown('<div class="tab-content">', unsafe_allow_html=True)
        
        if st.button("🚀 开始模拟投资", type="primary"):
            if not selected_funds:
                st.warning("请至少选择一只基金")
                st.stop()
            
            with st.spinner("正在执行策略回测..."):
                # 生成基金数据
                period_mapping = {
                    "3个月": 90,
                    "6个月": 180,
                    "1年": 365,
                    "2年": 730,
                    "3年": 1095
                }
                
                days = period_mapping[backtest_period]
                start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
                
                fund_data_dict = {}
                for fund_code in selected_funds:
                    fund_info = {'code': fund_code, **FUND_UNIVERSE[fund_code]}
                    fund_data = simulator.generate_realistic_fund_data(fund_info, start_date)
                    fund_data_dict[fund_code] = fund_data
                
                # 执行策略
                results = {}
                for fund_code, fund_data in fund_data_dict.items():
                    result = simulator.execute_strategy(
                        selected_strategy, fund_data, initial_capital, **strategy_params
                    )
                    results[fund_code] = result
                    
                    # 存储到session state以便其他标签页使用
                    if 'results' not in st.session_state:
                        st.session_state.results = {}
                    st.session_state.results[fund_code] = result
                    st.session_state.fund_data = fund_data_dict
                
                # 显示投资概况
                st.subheader("📋 投资概况")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("初始资金", f"¥{initial_capital:,.0f}")
                with col2:
                    st.metric("投资策略", selected_strategy)
                with col3:
                    st.metric("回测周期", backtest_period)
                
                # 显示净值曲线
                st.subheader("📈 投资组合净值曲线")
                
                fig = go.Figure()
                
                for fund_code, result in results.items():
                    fund_name = FUND_UNIVERSE[fund_code]['name']
                    portfolio_value = result['portfolio_value']
                    
                    fig.add_trace(go.Scatter(
                        x=portfolio_value.index,
                        y=portfolio_value,
                        name=f"{fund_name} - 策略",
                        line=dict(width=2)
                    ))
                    
                    # 添加基金净值作为基准
                    fund_nav = fund_data_dict[fund_code]['nav']
                    benchmark_value = initial_capital * (fund_nav / fund_nav.iloc[0])
                    
                    fig.add_trace(go.Scatter(
                        x=fund_nav.index,
                        y=benchmark_value,
                        name=f"{fund_name} - 买入持有",
                        line=dict(dash='dash', width=1),
                        opacity=0.7
                    ))
                
                fig.update_layout(
                    title="投资组合净值 vs 买入持有基准",
                    xaxis_title="日期",
                    yaxis_title="组合价值 (元)",
                    hovermode='x unified',
                    height=500,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # 显示交易记录
                st.subheader("📝 交易记录")
                
                for fund_code, result in results.items():
                    if result['trades']:
                        fund_name = FUND_UNIVERSE[fund_code]['name']
                        st.markdown(f"**{fund_name} 交易记录**")
                        
                        trades_df = pd.DataFrame(result['trades'])
                        trades_df['金额'] = trades_df['shares'] * trades_df['price']
                        trades_df = trades_df.round(4)
                        
                        st.dataframe(trades_df, use_container_width=True)
                    else:
                        st.info("该策略在此期间无交易记录")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        st.markdown('<div class="tab-content">', unsafe_allow_html=True)
        
        if 'results' in st.session_state and st.session_state.results:
            st.subheader("📊 绩效指标分析")
            
            # 计算并显示绩效指标
            metrics_list = []
            
            for fund_code, result in st.session_state.results.items():
                fund_name = FUND_UNIVERSE[fund_code]['name']
                portfolio_value = result['portfolio_value']
                
                # 基准净值（买入持有）
                fund_nav = st.session_state.fund_data[fund_code]['nav']
                benchmark_value = initial_capital * (fund_nav / fund_nav.iloc[0])
                
                # 计算绩效指标
                metrics = simulator.calculate_performance_metrics(portfolio_value, benchmark_value)
                metrics['基金名称'] = fund_name
                metrics_list.append(metrics)
            
            metrics_df = pd.DataFrame(metrics_list)
            
            # 选择要显示的指标
            display_columns = ['基金名称', '累计收益率', '年化收益率', '年化波动率', 
                             '最大回撤', '夏普比率', '胜率', '盈亏比']
            
            if '基准收益率' in metrics_df.columns:
                display_columns.insert(2, '基准收益率')
                display_columns.insert(3, '超额收益率')
            
            display_df = metrics_df[display_columns].copy()
            
            # 格式化显示
            percent_cols = ['累计收益率', '年化收益率', '年化波动率', '最大回撤', '胜率']
            if '基准收益率' in display_df.columns:
                percent_cols.extend(['基准收益率', '超额收益率'])
            
            for col in percent_cols:
                if col in display_df.columns:
                    display_df[col] = display_df[col].apply(lambda x: f"{x:.2%}")
            
            # 条件格式化函数
            def color_metrics(val_str):
                if '%' in str(val_str):
                    val = float(val_str.strip('%')) / 100
                    if '收益率' in val_str or '胜率' in val_str:
                        if val > 0:
                            return f'<span class="performance-good">{val_str}</span>'
                        elif val < 0:
                            return f'<span class="performance-bad">{val_str}</span>'
                        else:
                            return f'<span class="performance-neutral">{val_str}</span>'
                    elif '最大回撤' in val_str:
                        if val > -0.1:
                            return f'<span class="performance-good">{val_str}</span>'
                        elif val < -0.2:
                            return f'<span class="performance-bad">{val_str}</span>'
                        else:
                            return f'<span class="performance-neutral">{val_str}</span>'
                return val_str
            
            # 应用条件格式化
            html_table = display_df.to_html(escape=False, index=False)
            for col in percent_cols:
                if col in display_df.columns:
                    html_table = html_table.replace(f'<th>{col}</th>', 
                                                  f'<th style="text-align:center">{col}</th>')
            
            # 渲染HTML表格
            st.markdown(html_table, unsafe_allow_html=True)
            
            # 绩效对比雷达图
            st.subheader("🎯 策略绩效雷达图")
            
            if len(metrics_df) > 0:
                # 选择关键指标进行雷达图展示
                radar_metrics = ['年化收益率', '夏普比率', '胜率', '盈亏比', '最大回撤']
                
                fig_radar = go.Figure()
                
                for idx, row in metrics_df.iterrows():
                    values = []
                    for metric in radar_metrics:
                        if metric == '最大回撤':
                            # 最大回撤为负值，取绝对值并反转
                            values.append(abs(row[metric]) * 10)  # 缩放
                        else:
                            values.append(row[metric] * 100 if row[metric] < 1 else row[metric])
                    
                    fig_radar.add_trace(go.Scatterpolar(
                        r=values,
                        theta=radar_metrics,
                        fill='toself',
                        name=row['基金名称']
                    ))
                
                fig_radar.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, max([max(values) for values in 
                                          [abs(metrics_df[m])*10 if m=='最大回撤' else 
                                           metrics_df[m]*100 if metrics_df[m].max()<1 else metrics_df[m]
                                           for m in radar_metrics]]) * 1.2]
                        )),
                    showlegend=True,
                    title="策略绩效多维对比"
                )
                
                st.plotly_chart(fig_radar, use_container_width=True)
            
        else:
            st.info("请先在'策略回测'标签页运行模拟投资")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab3:
        st.markdown('<div class="tab-content">', unsafe_allow_html=True)
        st.subheader("📈 策略对比分析")
        
        if 'results' in st.session_state and st.session_state.results:
            # 策略对比分析
            comparison_data = []
            
            for fund_code in selected_funds:
                fund_name = FUND_UNIVERSE[fund_code]['name']
                
                # 当前策略
                result = st.session_state.results[fund_code]
                portfolio_value = result['portfolio_value']
                
                # 一次性买入策略（作为基准）
                fund_data = st.session_state.fund_data[fund_code]
                lump_sum_result = simulator.execute_strategy(
                    "一次性买入", fund_data, initial_capital
                )
                
                # 定期定额策略
                dca_result = simulator.execute_strategy(
                    "定期定额", fund_data, initial_capital, 
                    interval=30, amount=initial_capital/12
                )
                
                # 计算各种策略的最终收益
                strategies = {
                    "当前策略": portfolio_value.iloc[-1],
                    "一次性买入": lump_sum_result['portfolio_value'].iloc[-1],
                    "定期定额": dca_result['portfolio_value'].iloc[-1]
                }
                
                for strategy_name, final_value in strategies.items():
                    return_pct = (final_value - initial_capital) / initial_capital
                    comparison_data.append({
                        '基金': fund_name,
                        '策略': strategy_name,
                        '最终价值': final_value,
                        '收益率': return_pct
                    })
            
            comparison_df = pd.DataFrame(comparison_data)
            
            # 创建对比柱状图
            fig_comparison = px.bar(
                comparison_df,
                x='策略',
                y='收益率',
                color='基金',
                barmode='group',
                title="不同策略收益率对比",
                text=comparison_df['收益率'].apply(lambda x: f"{x:.2%}")
            )
            
            fig_comparison.update_layout(
                yaxis_tickformat='.2%',
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_comparison, use_container_width=True)
            
            # 显示详细对比表格
            st.subheader("详细对比数据")
            
            pivot_df = comparison_df.pivot_table(
                index='基金', 
                columns='策略', 
                values='收益率'
            ).style.format("{:.2%}")
            
            st.dataframe(pivot_df, use_container_width=True)
            
        else:
            st.info("请先在'策略回测'标签页运行模拟投资")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab4:
        st.markdown('<div class="tab-content">', unsafe_allow_html=True)
        st.subheader("💡 专业策略建议")
        
        if 'results' in st.session_state and st.session_state.results:
            # 基于回测结果生成建议
            recommendations = []
            
            for fund_code in selected_funds:
                fund_name = FUND_UNIVERSE[fund_code]['name']
                fund_risk = FUND_UNIVERSE[fund_code]['risk']
                result = st.session_state.results[fund_code]
                
                portfolio_value = result['portfolio_value']
                fund_data = st.session_state.fund_data[fund_code]
                
                # 计算关键指标
                returns = portfolio_value.pct_change().dropna()
                total_return = (portfolio_value.iloc[-1] / portfolio_value.iloc[0]) - 1
                volatility = returns.std() * np.sqrt(252)
                sharpe_ratio = (total_return * 252/len(portfolio_value) - simulator.risk_free_rate) / volatility if volatility > 0 else 0
                
                # 分析最大回撤
                cumulative = (1 + returns).cumprod()
                rolling_max = cumulative.expanding().max()
                drawdown = (cumulative - rolling_max) / rolling_max
                max_drawdown = drawdown.min()
                
                # 生成建议
                recommendation = {
                    '基金': fund_name,
                    '风险评估': fund_risk,
                    '总体评价': '',
                    '具体建议': []
                }
                
                # 评估策略表现
                if sharpe_ratio > 1.0:
                    recommendation['总体评价'] = "优秀"
                    recommendation['具体建议'].append("✅ 策略表现优异，夏普比率较高")
                elif sharpe_ratio > 0.5:
                    recommendation['总体评价'] = "良好"
                    recommendation['具体建议'].append("✅ 策略表现良好，风险收益比较合理")
                else:
                    recommendation['总体评价'] = "一般"
                    recommendation['具体建议'].append("⚠️ 策略表现一般，建议优化参数或更换策略")
                
                # 风险评估
                if abs(max_drawdown) > 0.2:
                    recommendation['具体建议'].append("⚠️ 最大回撤较大，需注意风险控制")
                elif abs(max_drawdown) < 0.1:
                    recommendation['具体建议'].append("✅ 回撤控制良好，风险相对较低")
                
                # 基于基金类型和策略的建议
                if "科技" in fund_name or "新能源" in fund_name:
                    recommendation['具体建议'].append("📱 科技/新能源基金波动较大，建议采用金字塔买入或定期定额策略")
                elif "消费" in fund_name or "白酒" in fund_name:
                    recommendation['具体建议'].append("🍶 消费主题基金适合长期持有，建议结合定投策略")
                elif "均衡" in fund_name or "沪深300" in fund_name:
                    recommendation['具体建议'].append("⚖️ 均衡型/宽基基金适合作为核心持仓")
                
                # 基于策略类型的建议
                if selected_strategy == "一次性买入":
                    recommendation['具体建议'].append("💰 一次性买入策略适合市场低位时使用")
                elif selected_strategy == "定期定额":
                    recommendation['具体建议'].append("📅 定投策略适合长期投资，能有效平滑成本")
                elif selected_strategy == "金字塔买入":
                    recommendation['具体建议'].append("🏗️ 金字塔买入策略适合高波动基金，能在下跌中积累低成本份额")
                elif selected_strategy == "均线策略":
                    recommendation['具体建议'].append("📈 均线策略适合趋势明显的市场环境")
                
                recommendations.append(recommendation)
            
            # 显示建议
            for rec in recommendations:
                st.markdown(f"""
                <div class="strategy-card">
                    <h4>{rec['基金']} <span style="float:right; color:{'#28a745' if rec['总体评价']=='优秀' else '#ffc107' if rec['总体评价']=='良好' else '#dc3545'}">
                        {rec['总体评价']}
                    </span></h4>
                    <p><strong>风险评估:</strong> {rec['风险评估']}</p>
                    <ul>
                        {''.join([f'<li>{item}</li>' for item in rec['具体建议']])}
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            
            # 通用投资建议
            st.markdown("""
            ### 📚 通用投资原则
            
            1. **分散投资**：不要把所有资金投入单一基金
            2. **长期视角**：基金投资应以年为单位，避免频繁交易
            3. **风险匹配**：选择与自身风险承受能力匹配的基金
            4. **定期检视**：每季度检视投资组合，根据市场环境适当调整
            5. **纪律投资**：严格执行既定策略，避免情绪化交易
            
            ### 🔄 策略调整建议
            
            如果您发现当前策略表现不佳，可以考虑：
            - 调整策略参数（如定投频率、金字塔层级）
            - 更换更适合当前市场的策略
            - 增加对冲或风险控制措施
            - 调整不同策略的组合比例
            """)
            
        else:
            st.info("请先在'策略回测'标签页运行模拟投资，获取个性化建议")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # 底部免责声明
    st.sidebar.markdown("---")
    st.sidebar.warning("""
    **免责声明**
    
    本系统基于历史数据回测，结果仅供参考。
    投资有风险，过往业绩不代表未来表现。
    投资决策需谨慎，建议咨询专业投资顾问。
    
    *AlphaFund Pro v3.0 - 专业投资模拟系统*
    """)

if __name__ == "__main__":
    main()
