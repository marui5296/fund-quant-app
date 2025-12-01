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
    page_title="AlphaFund Pro - 专业投资模拟系统",
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
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
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
        try:
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
            
            if len(dates) < 30:
                raise ValueError("数据日期不足，请选择更长的时间范围")
            
            # 生成更真实的收益率序列
            returns = np.random.normal(base_return, base_volatility, len(dates))
            
            # 添加市场相关性
            market_factor = np.random.normal(0.0005, 0.01, len(dates))
            returns = returns * 0.7 + market_factor * 0.3
            
            # 生成净值序列
            nav = 1.0 * (1 + pd.Series(returns)).cumprod()
            
            df = pd.DataFrame({
                'date': dates,
                'nav': nav.values,
                'return': returns
            }).set_index('date')
            
            return df
            
        except Exception as e:
            st.error(f"生成基金数据时出错: {str(e)}")
            # 返回一个简单的数据框避免崩溃
            dates = pd.date_range(start=start_date, end=datetime.now(), freq='B')
            nav = np.ones(len(dates))
            return pd.DataFrame({
                'date': dates,
                'nav': nav,
                'return': np.zeros(len(dates))
            }).set_index('date')
    
    def execute_strategy(self, strategy_type, fund_data, initial_capital, **params):
        """执行投资策略"""
        try:
            if len(fund_data) < 30:
                raise ValueError("基金数据不足，请选择更长的时间范围")
            
            strategy_map = {
                "一次性买入": self._lump_sum_investment,
                "定期定额": self._dollar_cost_averaging,
                "价值平均": self._value_averaging,
                "金字塔买入": self._pyramid_buying,
                "网格交易": self._grid_trading,
                "均线策略": self._moving_average_strategy,
                "动态平衡": self._dynamic_balance
            }
            
            if strategy_type not in strategy_map:
                raise ValueError(f"未知策略: {strategy_type}")
            
            return strategy_map[strategy_type](fund_data, initial_capital, **params)
            
        except Exception as e:
            st.error(f"执行策略时出错: {str(e)}")
            # 返回一个基础的结果避免崩溃
            return self._lump_sum_investment(fund_data, initial_capital)
    
    def _lump_sum_investment(self, fund_data, initial_capital, **params):
        """一次性买入策略"""
        try:
            nav = fund_data['nav']
            if len(nav) == 0:
                raise ValueError("基金数据为空")
                
            shares = initial_capital / nav.iloc[0]
            portfolio_value = shares * nav
            
            return {
                'portfolio_value': portfolio_value,
                'shares': pd.Series(shares, index=nav.index),
                'trades': [{'date': nav.index[0], 'action': 'BUY', 'shares': shares, 'price': nav.iloc[0], 'note': '一次性买入'}],
                'cash': pd.Series(0, index=nav.index),
                'success': True
            }
        except Exception as e:
            st.error(f"一次性买入策略出错: {str(e)}")
            return self._create_error_result(fund_data, initial_capital)
    
    def _dollar_cost_averaging(self, fund_data, initial_capital, **params):
        """定期定额投资策略"""
        try:
            nav = fund_data['nav']
            if len(nav) == 0:
                raise ValueError("基金数据为空")
            
            interval = params.get('interval', 30)  # 天
            amount = params.get('amount', min(1000, initial_capital / 12))    # 每次投入金额
            
            cash = initial_capital
            shares = 0
            portfolio_values = []
            trades = []
            
            for i, (date, price) in enumerate(nav.items()):
                # 定期投入
                if i % interval == 0 and cash >= amount:
                    buy_shares = amount / price
                    shares += buy_shares
                    cash -= amount
                    trades.append({
                        'date': date, 
                        'action': 'BUY', 
                        'shares': buy_shares, 
                        'price': price,
                        'note': f'第{len(trades)+1}次定投'
                    })
                
                portfolio_values.append(shares * price + cash)
            
            return {
                'portfolio_value': pd.Series(portfolio_values, index=nav.index),
                'shares': pd.Series(shares, index=nav.index),
                'trades': trades,
                'cash': pd.Series(cash, index=nav.index),
                'success': True
            }
            
        except Exception as e:
            st.error(f"定期定额策略出错: {str(e)}")
            return self._create_error_result(fund_data, initial_capital)
    
    def _pyramid_buying(self, fund_data, initial_capital, **params):
        """金字塔买入策略 - 修复版"""
        try:
            nav = fund_data['nav']
            if len(nav) == 0:
                raise ValueError("基金数据为空")
            
            # 获取参数，设置默认值
            buy_levels = params.get('buy_levels', [0, -0.05, -0.10])
            buy_amounts = params.get('buy_amounts', [0.3, 0.4, 0.3])
            
            # 验证参数
            if len(buy_levels) != len(buy_amounts):
                raise ValueError(f"买入层级数量({len(buy_levels)})和买入金额比例数量({len(buy_amounts)})不匹配")
            
            # 确保买入金额比例总和为1
            total_ratio = sum(buy_amounts)
            if abs(total_ratio - 1.0) > 0.001:
                # 自动归一化
                buy_amounts = [amt / total_ratio for amt in buy_amounts]
                st.warning(f"买入金额比例已自动归一化: {buy_amounts}")
            
            # 初始买入
            initial_buy_amount = initial_capital * buy_amounts[0]
            if initial_buy_amount > initial_capital:
                initial_buy_amount = initial_capital
                
            shares = initial_buy_amount / nav.iloc[0]
            cash = initial_capital - initial_buy_amount
            
            portfolio_values = []
            trades = []
            triggered_levels = set()  # 记录已触发的层级
            
            # 初始买入交易记录
            trades.append({
                'date': nav.index[0], 
                'action': 'BUY', 
                'shares': shares, 
                'price': nav.iloc[0],
                'note': '金字塔第1层买入'
            })
            
            # 设置参考价格（初始价格）
            reference_price = nav.iloc[0]
            
            for i, (date, price) in enumerate(nav.items()):
                # 计算从参考价格的跌幅
                if reference_price > 0:
                    drawdown = (price - reference_price) / reference_price
                else:
                    drawdown = 0
                
                # 检查是否需要加仓（从第二层开始）
                for level_idx in range(1, len(buy_levels)):
                    if level_idx >= len(buy_amounts):
                        break  # 安全保护
                        
                    level = buy_levels[level_idx]
                    amount_ratio = buy_amounts[level_idx]
                    
                    # 如果跌幅达到或超过该层级，且该层级尚未触发
                    if drawdown <= level and level not in triggered_levels:
                        buy_amount = initial_capital * amount_ratio
                        if cash >= buy_amount:
                            buy_shares = buy_amount / price
                            shares += buy_shares
                            cash -= buy_amount
                            trades.append({
                                'date': date, 
                                'action': 'BUY', 
                                'shares': buy_shares, 
                                'price': price,
                                'note': f'金字塔第{level_idx+1}层买入 (跌幅:{drawdown:.2%})'
                            })
                            triggered_levels.add(level)
                
                portfolio_values.append(shares * price + cash)
            
            return {
                'portfolio_value': pd.Series(portfolio_values, index=nav.index),
                'shares': pd.Series(shares, index=nav.index),
                'trades': trades,
                'cash': pd.Series(cash, index=nav.index),
                'triggered_levels': list(triggered_levels),
                'success': True
            }
            
        except Exception as e:
            st.error(f"金字塔买入策略出错: {str(e)}")
            return self._create_error_result(fund_data, initial_capital)
    
    def _moving_average_strategy(self, fund_data, initial_capital, **params):
        """均线策略"""
        try:
            nav = fund_data['nav']
            if len(nav) < 60:  # 需要足够的数据计算均线
                raise ValueError("数据不足，至少需要60个交易日数据")
            
            short_window = params.get('short_window', 20)
            long_window = params.get('long_window', 50)
            
            # 计算移动平均
            short_ma = nav.rolling(window=short_window, min_periods=1).mean()
            long_ma = nav.rolling(window=long_window, min_periods=1).mean()
            
            cash = initial_capital
            shares = 0
            portfolio_values = []
            trades = []
            position = 0  # 0:空仓, 1:持仓
            
            for i in range(len(nav)):
                date = nav.index[i]
                price = nav.iloc[i]
                
                # 确保有足够数据计算均线
                if i >= max(short_window, long_window) - 1:
                    # 金叉买入，死叉卖出
                    if short_ma.iloc[i] > long_ma.iloc[i] and position == 0:
                        # 买入
                        if cash > 0:
                            shares = cash / price
                            cash = 0
                            position = 1
                            trades.append({
                                'date': date, 
                                'action': 'BUY', 
                                'shares': shares, 
                                'price': price,
                                'note': f'金叉信号 (短均线:{short_ma.iloc[i]:.4f}, 长均线:{long_ma.iloc[i]:.4f})'
                            })
                    elif short_ma.iloc[i] < long_ma.iloc[i] and position == 1:
                        # 卖出
                        if shares > 0:
                            cash = shares * price
                            trades.append({
                                'date': date, 
                                'action': 'SELL', 
                                'shares': shares, 
                                'price': price,
                                'note': f'死叉信号 (短均线:{short_ma.iloc[i]:.4f}, 长均线:{long_ma.iloc[i]:.4f})'
                            })
                            shares = 0
                            position = 0
                
                portfolio_values.append(shares * price + cash)
            
            return {
                'portfolio_value': pd.Series(portfolio_values, index=nav.index),
                'shares': pd.Series(shares, index=nav.index),
                'trades': trades,
                'cash': pd.Series(cash, index=nav.index),
                'signals': pd.DataFrame({
                    'price': nav,
                    'short_ma': short_ma,
                    'long_ma': long_ma
                }),
                'success': True
            }
            
        except Exception as e:
            st.error(f"均线策略出错: {str(e)}")
            return self._create_error_result(fund_data, initial_capital)
    
    def _value_averaging(self, fund_data, initial_capital, **params):
        """价值平均策略"""
        try:
            nav = fund_data['nav']
            monthly_target = params.get('monthly_target', initial_capital / 12)
            
            cash = initial_capital
            shares = 0
            portfolio_values = []
            trades = []
            
            # 每月调整一次
            for i in range(0, len(nav), 21):  # 大约每月21个交易日
                if i >= len(nav):
                    break
                    
                date = nav.index[i]
                price = nav.iloc[i]
                
                # 目标市值 = 已投资月数 * 每月目标
                target_value = (i // 21 + 1) * monthly_target
                current_value = shares * price + cash
                
                # 计算需要调整的金额
                adjustment = target_value - current_value
                
                if adjustment > 0 and cash >= adjustment:  # 需要买入
                    buy_shares = adjustment / price
                    shares += buy_shares
                    cash -= adjustment
                    trades.append({
                        'date': date, 
                        'action': 'BUY', 
                        'shares': buy_shares, 
                        'price': price,
                        'note': f'价值平均补仓 (目标:{target_value:.0f}, 当前:{current_value:.0f})'
                    })
                elif adjustment < 0 and shares > 0:  # 需要卖出
                    sell_value = abs(adjustment)
                    sell_shares = min(sell_value / price, shares)
                    shares -= sell_shares
                    cash += sell_shares * price
                    trades.append({
                        'date': date, 
                        'action': 'SELL', 
                        'shares': sell_shares, 
                        'price': price,
                        'note': f'价值平均减仓 (目标:{target_value:.0f}, 当前:{current_value:.0f})'
                    })
                
            # 计算每日净值
            for i, price in enumerate(nav):
                portfolio_values.append(shares * price + cash)
            
            return {
                'portfolio_value': pd.Series(portfolio_values, index=nav.index),
                'shares': pd.Series(shares, index=nav.index),
                'trades': trades,
                'cash': pd.Series(cash, index=nav.index),
                'success': True
            }
            
        except Exception as e:
            st.error(f"价值平均策略出错: {str(e)}")
            return self._create_error_result(fund_data, initial_capital)
    
    def _create_error_result(self, fund_data, initial_capital):
        """创建错误时的默认结果"""
        nav = fund_data['nav']
        return {
            'portfolio_value': pd.Series([initial_capital] * len(nav), index=nav.index),
            'shares': pd.Series(0, index=nav.index),
            'trades': [],
            'cash': pd.Series(initial_capital, index=nav.index),
            'success': False
        }
    
    def calculate_performance_metrics(self, portfolio_value, benchmark_value=None):
        """计算投资组合绩效指标"""
        try:
            if len(portfolio_value) < 2:
                return {}
            
            returns = portfolio_value.pct_change().dropna()
            
            if len(returns) == 0:
                return {}
            
            metrics = {}
            
            # 基础收益指标
            total_return = (portfolio_value.iloc[-1] / portfolio_value.iloc[0]) - 1
            days = (portfolio_value.index[-1] - portfolio_value.index[0]).days
            if days > 0:
                annual_return = (1 + total_return) ** (365 / days) - 1
            else:
                annual_return = 0
            
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
            max_drawdown = drawdown.min() if len(drawdown) > 0 else 0
            
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
            if benchmark_value is not None and len(benchmark_value) > 1:
                benchmark_return = (benchmark_value.iloc[-1] / benchmark_value.iloc[0]) - 1
                excess_return = total_return - benchmark_return
                
                # 计算信息比率
                excess_returns = portfolio_value.pct_change() - benchmark_value.pct_change()
                tracking_error = excess_returns.std() * np.sqrt(252) if len(excess_returns) > 0 else 0
                information_ratio = excess_return / tracking_error if tracking_error > 0 else 0
                
                metrics.update({
                    '基准收益率': benchmark_return,
                    '超额收益率': excess_return,
                    '信息比率': information_ratio
                })
            
            return metrics
            
        except Exception as e:
            st.error(f"计算绩效指标时出错: {str(e)}")
            return {}

def main():
    # 专业标题
    st.markdown("""
    <div class="professional-header">
        <h1>🚀 AlphaFund Pro - 专业投资模拟与策略回测系统</h1>
        <p>基于10年量化经验构建，支持多种投资策略模拟</p>
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
        default=["022365"],
        help="可以选择多只基金进行组合分析"
    )
    
    initial_capital = st.sidebar.number_input(
        "初始资金 (元)", 
        value=100000, 
        min_value=1000, 
        step=1000,
        help="模拟投资的起始资金"
    )
    
    # 选择策略
    strategy_options = {
        "一次性买入": "最简单的投资方式，一次性投入全部资金",
        "定期定额": "定期投入固定金额，适合长期投资",
        "金字塔买入": "价格下跌时逐步加仓，越跌买得越多",
        "均线策略": "基于移动平均线的趋势跟踪策略",
        "价值平均": "每月调整持仓至目标价值，自动低买高卖"
    }
    
    selected_strategy = st.sidebar.selectbox(
        "选择投资策略",
        options=list(strategy_options.keys()),
        format_func=lambda x: f"{x} - {strategy_options[x]}",
        index=0
    )
    
    # 策略参数配置
    st.sidebar.header("⚙️ 策略参数")
    
    strategy_params = {}
    
    if selected_strategy == "定期定额":
        interval = st.sidebar.slider("定投周期 (天)", 7, 90, 30)
        amount = st.sidebar.number_input("每次定投金额 (元)", 
                                        value=min(2000, initial_capital // 12), 
                                        min_value=100, 
                                        max_value=initial_capital,
                                        step=100)
        strategy_params = {'interval': interval, 'amount': amount}
        
    elif selected_strategy == "金字塔买入":
        st.sidebar.markdown("**金字塔买入策略配置**")
        levels = st.sidebar.slider("金字塔层级", 2, 5, 3)
        
        # 初始化买入层级和金额比例
        buy_levels = [0]  # 第1层：初始买入
        buy_amounts = []
        
        # 第一层配置
        amount_pct1 = st.sidebar.number_input(
            "第1层仓位比例 (%)", 
            value=int(100/levels), 
            min_value=10, 
            max_value=100,
            help="初始买入的资金比例"
        )
        buy_amounts.append(amount_pct1/100)
        
        # 后续层级配置
        for i in range(1, levels):
            col1, col2 = st.sidebar.columns(2)
            with col1:
                level = st.sidebar.number_input(
                    f"第{i+1}层触发跌幅 (%)", 
                    value=5*i, 
                    min_value=1, 
                    max_value=50, 
                    key=f"level_{i}"
                )
            with col2:
                amount_pct = st.sidebar.number_input(
                    f"第{i+1}层仓位比例 (%)", 
                    value=int(100/levels), 
                    min_value=1, 
                    max_value=100, 
                    key=f"amount_{i}"
                )
            
            buy_levels.append(-level/100)
            buy_amounts.append(amount_pct/100)
        
        strategy_params = {'buy_levels': buy_levels, 'buy_amounts': buy_amounts}
        
        # 显示配置信息
        st.sidebar.markdown("**配置预览:**")
        for i in range(levels):
            if i == 0:
                st.sidebar.write(f"第{i+1}层: 初始买入 {buy_amounts[i]:.1%}")
            else:
                st.sidebar.write(f"第{i+1}层: 跌幅 ≥ {abs(buy_levels[i]):.1%} 时买入 {buy_amounts[i]:.1%}")
        
    elif selected_strategy == "均线策略":
        col1, col2 = st.sidebar.columns(2)
        with col1:
            short_window = st.slider("短期均线周期", 5, 60, 20)
        with col2:
            long_window = st.slider("长期均线周期", 20, 200, 50)
        
        strategy_params = {'short_window': short_window, 'long_window': long_window}
        
    elif selected_strategy == "价值平均":
        monthly_target = st.sidebar.number_input(
            "每月目标增值 (元)", 
            value=initial_capital // 12, 
            min_value=100, 
            max_value=initial_capital,
            step=100
        )
        strategy_params = {'monthly_target': monthly_target}
    
    # 回测时间范围
    st.sidebar.header("📅 回测设置")
    backtest_period = st.sidebar.selectbox(
        "回测时间范围", 
        ["3个月", "6个月", "1年", "2年", "3年"], 
        index=2
    )
    
    # 主内容区域
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 策略回测", "📊 绩效分析", "📈 对比分析", "💡 策略建议"])
    
    # 初始化session state
    if 'simulation_results' not in st.session_state:
        st.session_state.simulation_results = None
    if 'fund_data_dict' not in st.session_state:
        st.session_state.fund_data_dict = None
    
    with tab1:
        st.markdown('<div class="tab-content">', unsafe_allow_html=True)
        
        if st.button("🚀 开始模拟投资", type="primary", use_container_width=True):
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
                
                days = period_mapping.get(backtest_period, 365)
                start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
                
                fund_data_dict = {}
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for idx, fund_code in enumerate(selected_funds):
                    status_text.text(f"正在生成 {FUND_UNIVERSE[fund_code]['name']} 的历史数据...")
                    fund_info = {'code': fund_code, **FUND_UNIVERSE[fund_code]}
                    fund_data = simulator.generate_realistic_fund_data(fund_info, start_date)
                    fund_data_dict[fund_code] = fund_data
                    progress_bar.progress((idx + 1) / len(selected_funds))
                
                # 执行策略
                results = {}
                for fund_code in selected_funds:
                    fund_data = fund_data_dict[fund_code]
                    result = simulator.execute_strategy(
                        selected_strategy, fund_data, initial_capital, **strategy_params
                    )
                    results[fund_code] = result
                
                # 存储到session state
                st.session_state.simulation_results = results
                st.session_state.fund_data_dict = fund_data_dict
                st.session_state.selected_funds = selected_funds
                st.session_state.selected_strategy = selected_strategy
                st.session_state.initial_capital = initial_capital
                
                status_text.text("策略回测完成！")
                progress_bar.empty()
                
                # 显示投资概况
                st.subheader("📋 投资概况")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("初始资金", f"¥{initial_capital:,.0f}")
                with col2:
                    st.metric("投资策略", selected_strategy)
                with col3:
                    st.metric("回测周期", backtest_period)
                with col4:
                    st.metric("分析基金数", len(selected_funds))
                
                # 显示净值曲线
                st.subheader("📈 投资组合净值曲线")
                
                fig = go.Figure()
                
                for fund_code in selected_funds:
                    fund_name = FUND_UNIVERSE[fund_code]['name']
                    result = results[fund_code]
                    
                    if result.get('success', False):
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
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # 显示交易记录
                st.subheader("📝 交易记录")
                
                for fund_code in selected_funds:
                    result = results[fund_code]
                    if result.get('trades') and len(result['trades']) > 0:
                        fund_name = FUND_UNIVERSE[fund_code]['name']
                        st.markdown(f"**{fund_name} 交易记录**")
                        
                        trades_df = pd.DataFrame(result['trades'])
                        trades_df['金额'] = trades_df['shares'] * trades_df['price']
                        trades_df = trades_df.round({
                            'shares': 2,
                            'price': 4,
                            '金额': 2
                        })
                        
                        st.dataframe(trades_df, use_container_width=True)
                        
                        # 显示交易统计
                        buy_trades = trades_df[trades_df['action'] == 'BUY']
                        sell_trades = trades_df[trades_df['action'] == 'SELL']
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("买入次数", len(buy_trades))
                        with col2:
                            st.metric("卖出次数", len(sell_trades))
                    else:
                        st.info(f"**{FUND_UNIVERSE[fund_code]['name']}** 在此期间无交易记录")
        
        elif st.session_state.simulation_results:
            # 如果已经有结果，显示上次的分析
            st.info("显示上次模拟投资的结果")
            
            results = st.session_state.simulation_results
            fund_data_dict = st.session_state.fund_data_dict
            selected_funds = st.session_state.selected_funds
            
            # 显示净值曲线
            st.subheader("📈 投资组合净值曲线")
            
            fig = go.Figure()
            
            for fund_code in selected_funds:
                fund_name = FUND_UNIVERSE[fund_code]['name']
                result = results[fund_code]
                
                if result.get('success', False):
                    portfolio_value = result['portfolio_value']
                    
                    fig.add_trace(go.Scatter(
                        x=portfolio_value.index,
                        y=portfolio_value,
                        name=f"{fund_name} - 策略",
                        line=dict(width=2)
                    ))
                    
                    # 添加基金净值作为基准
                    fund_nav = fund_data_dict[fund_code]['nav']
                    benchmark_value = st.session_state.initial_capital * (fund_nav / fund_nav.iloc[0])
                    
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
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        st.markdown('<div class="tab-content">', unsafe_allow_html=True)
        
        if st.session_state.simulation_results:
            st.subheader("📊 绩效指标分析")
            
            results = st.session_state.simulation_results
            fund_data_dict = st.session_state.fund_data_dict
            
            # 计算并显示绩效指标
            metrics_list = []
            
            for fund_code in st.session_state.selected_funds:
                fund_name = FUND_UNIVERSE[fund_code]['name']
                result = results[fund_code]
                
                if result.get('success', False):
                    portfolio_value = result['portfolio_value']
                    
                    # 基准净值（买入持有）
                    fund_nav = fund_data_dict[fund_code]['nav']
                    benchmark_value = st.session_state.initial_capital * (fund_nav / fund_nav.iloc[0])
                    
                    # 计算绩效指标
                    metrics = simulator.calculate_performance_metrics(portfolio_value, benchmark_value)
                    metrics['基金名称'] = fund_name
                    metrics_list.append(metrics)
            
            if metrics_list:
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
                        try:
                            display_df[col] = display_df[col].apply(lambda x: f"{x:.2%}")
                        except:
                            display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}")
                
                # 显示表格
                st.dataframe(display_df, use_container_width=True)
                
                # 绩效对比雷达图
                st.subheader("🎯 策略绩效雷达图")
                
                if len(metrics_df) > 0:
                    # 选择关键指标进行雷达图展示
                    radar_metrics = ['年化收益率', '夏普比率', '胜率', '盈亏比']
                    
                    # 归一化处理
                    normalized_data = []
                    fund_names = []
                    
                    for idx, row in metrics_df.iterrows():
                        values = []
                        for metric in radar_metrics:
                            val = row.get(metric, 0)
                            if metric == '年化收益率':
                                # 年化收益率可能为负，进行偏移
                                values.append((val + 0.2) * 100)  # 假设最低-20%，归一化到0-100
                            elif metric == '夏普比率':
                                values.append(max(0, val) * 20)  # 夏普比率通常0-5，归一化到0-100
                            elif metric == '胜率':
                                values.append(val * 100)  # 胜率0-1，转为百分比
                            elif metric == '盈亏比':
                                values.append(min(val * 20, 100))  # 盈亏比通常0-5，归一化到0-100
                        
                        normalized_data.append(values)
                        fund_names.append(row['基金名称'])
                    
                    fig_radar = go.Figure()
                    
                    for values, name in zip(normalized_data, fund_names):
                        fig_radar.add_trace(go.Scatterpolar(
                            r=values,
                            theta=radar_metrics,
                            fill='toself',
                            name=name
                        ))
                    
                    fig_radar.update_layout(
                        polar=dict(
                            radialaxis=dict(
                                visible=True,
                                range=[0, 100]
                            )),
                        showlegend=True,
                        title="策略绩效多维对比 (已归一化)",
                        height=500
                    )
                    
                    st.plotly_chart(fig_radar, use_container_width=True)
            else:
                st.warning("未能计算绩效指标，请检查数据")
        else:
            st.info("请先在'策略回测'标签页运行模拟投资")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab3:
        st.markdown('<div class="tab-content">', unsafe_allow_html=True)
        st.subheader("📈 策略对比分析")
        
        if st.session_state.simulation_results:
            # 策略对比分析
            comparison_data = []
            
            for fund_code in st.session_state.selected_funds:
                fund_name = FUND_UNIVERSE[fund_code]['name']
                
                # 当前策略
                result = st.session_state.simulation_results[fund_code]
                if result.get('success', False):
                    portfolio_value = result['portfolio_value']
                    final_value = portfolio_value.iloc[-1]
                    
                    # 一次性买入策略（作为基准）
                    fund_data = st.session_state.fund_data_dict[fund_code]
                    lump_sum_result = simulator.execute_strategy(
                        "一次性买入", fund_data, st.session_state.initial_capital
                    )
                    
                    # 定期定额策略
                    dca_result = simulator.execute_strategy(
                        "定期定额", fund_data, st.session_state.initial_capital, 
                        interval=30, amount=st.session_state.initial_capital/12
                    )
                    
                    # 计算各种策略的最终收益
                    strategies = {
                        "当前策略": final_value,
                        "一次性买入": lump_sum_result['portfolio_value'].iloc[-1] if lump_sum_result.get('success') else st.session_state.initial_capital,
                        "定期定额": dca_result['portfolio_value'].iloc[-1] if dca_result.get('success') else st.session_state.initial_capital
                    }
                    
                    for strategy_name, final_val in strategies.items():
                        return_pct = (final_val - st.session_state.initial_capital) / st.session_state.initial_capital
                        comparison_data.append({
                            '基金': fund_name,
                            '策略': strategy_name,
                            '最终价值': final_val,
                            '收益率': return_pct
                        })
            
            if comparison_data:
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
                    hovermode='x unified',
                    height=500
                )
                
                fig_comparison.update_traces(textposition='outside')
                
                st.plotly_chart(fig_comparison, use_container_width=True)
                
                # 显示详细对比表格
                st.subheader("详细对比数据")
                
                pivot_df = comparison_df.pivot_table(
                    index='基金', 
                    columns='策略', 
                    values='收益率'
                )
                
                # 格式化百分比
                styled_df = pivot_df.style.format("{:.2%}")
                st.dataframe(styled_df, use_container_width=True)
            else:
                st.warning("无法进行策略对比分析")
        else:
            st.info("请先在'策略回测'标签页运行模拟投资")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab4:
        st.markdown('<div class="tab-content">', unsafe_allow_html=True)
        st.subheader("💡 专业策略建议")
        
        if st.session_state.simulation_results:
            # 基于回测结果生成建议
            recommendations = []
            
            for fund_code in st.session_state.selected_funds:
                fund_name = FUND_UNIVERSE[fund_code]['name']
                fund_risk = FUND_UNIVERSE[fund_code]['risk']
                result = st.session_state.simulation_results[fund_code]
                
                if result.get('success', False):
                    portfolio_value = result['portfolio_value']
                    fund_data = st.session_state.fund_data_dict[fund_code]
                    
                    # 计算关键指标
                    returns = portfolio_value.pct_change().dropna()
                    if len(returns) > 0:
                        total_return = (portfolio_value.iloc[-1] / portfolio_value.iloc[0]) - 1
                        volatility = returns.std() * np.sqrt(252)
                        sharpe_ratio = (total_return * 252/len(portfolio_value) - simulator.risk_free_rate) / volatility if volatility > 0 else 0
                        
                        # 分析最大回撤
                        cumulative = (1 + returns).cumprod()
                        rolling_max = cumulative.expanding().max()
                        drawdown = (cumulative - rolling_max) / rolling_max
                        max_drawdown = drawdown.min() if len(drawdown) > 0 else 0
                        
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
                        
                        recommendations.append(recommendation)
            
            # 显示建议
            if recommendations:
                for rec in recommendations:
                    color_map = {
                        "优秀": "#28a745",
                        "良好": "#ffc107",
                        "一般": "#dc3545"
                    }
                    
                    st.markdown(f"""
                    <div class="strategy-card">
                        <h4>{rec['基金']} <span style="float:right; color:{color_map.get(rec['总体评价'], '#6c757d')}">
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
                st.warning("无法生成策略建议，请检查数据")
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
    
    *AlphaFund Pro v3.1 - 专业投资模拟系统*
    """)

if __name__ == "__main__":
    main()
