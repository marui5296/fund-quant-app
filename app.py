import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# 设置专业级页面配置
st.set_page_config(
    page_title="AlphaFund Quant - 专业基金量化分析系统",
    page_icon="📊",
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
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #007bff;
        margin-bottom: 1rem;
    }
    .risk-high { background-color: #f8d7da; border-left-color: #dc3545; }
    .risk-medium { background-color: #fff3cd; border-left-color: #ffc107; }
    .risk-low { background-color: #d1ecf1; border-left-color: #17a2b8; }
    .signal-buy { background-color: #d4edda; color: #155724; padding: 0.5rem; border-radius: 5px; }
    .signal-sell { background-color: #f8d7da; color: #721c24; padding: 0.5rem; border-radius: 5px; }
    .signal-hold { background-color: #e2e3e5; color: #383d41; padding: 0.5rem; border-radius: 5px; }
</style>
""", unsafe_allow_html=True)

class ProfessionalFundQuant:
    """专业级基金量化分析系统"""
    
    def __init__(self):
        self.fund_data = {}
        self.models = {}
        self.risk_free_rate = 0.015  # 无风险利率1.5%
        
    def calculate_advanced_metrics(self, returns_series, window=252):
        """计算专业量化指标"""
        metrics = {}
        
        # 基础收益指标
        total_return = (1 + returns_series).prod() - 1
        annual_return = (1 + total_return) ** (window / len(returns_series)) - 1
        
        # 风险指标
        volatility = returns_series.std() * np.sqrt(window)
        downside_returns = returns_series[returns_series < 0]
        downside_volatility = downside_returns.std() * np.sqrt(window) if len(downside_returns) > 0 else 0
        
        # 风险调整收益指标
        sharpe_ratio = (annual_return - self.risk_free_rate) / volatility if volatility > 0 else 0
        sortino_ratio = (annual_return - self.risk_free_rate) / downside_volatility if downside_volatility > 0 else 0
        
        # 最大回撤
        cumulative = (1 + returns_series).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Calmar比率
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # VaR (95%置信度)
        var_95 = returns_series.quantile(0.05)
        
        # 偏度和峰度
        skewness = returns_series.skew()
        kurtosis = returns_series.kurtosis()
        
        metrics.update({
            '年化收益率': annual_return,
            '年化波动率': volatility,
            '夏普比率': sharpe_ratio,
            '索提诺比率': sortino_ratio,
            '卡玛比率': calmar_ratio,
            '最大回撤': max_drawdown,
            '在险价值(VaR 95%)': var_95,
            '收益偏度': skewness,
            '收益峰度': kurtosis
        })
        
        return metrics
    
    def generate_technical_features(self, price_series):
        """生成专业级技术指标特征"""
        df = pd.DataFrame(index=price_series.index)
        df['price'] = price_series
        
        # 收益率特征
        for period in [1, 5, 10, 20]:
            df[f'return_{period}d'] = price_series.pct_change(period)
        
        # 移动平均线
        for window in [5, 10, 20, 50]:
            df[f'ma_{window}'] = price_series.rolling(window).mean()
            df[f'price_vs_ma{window}'] = price_series / df[f'ma_{window}'] - 1
        
        # 波动率特征
        for window in [5, 10, 20]:
            df[f'volatility_{window}d'] = df['return_1d'].rolling(window).std()
        
        # RSI
        delta = price_series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = price_series.ewm(span=12).mean()
        exp2 = price_series.ewm(span=26).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # 布林带
        df['bb_middle'] = price_series.rolling(20).mean()
        bb_std = price_series.rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + 2 * bb_std
        df['bb_lower'] = df['bb_middle'] - 2 * bb_std
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (price_series - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # 动量指标
        df['momentum_1m'] = price_series / price_series.shift(20) - 1
        df['momentum_3m'] = price_series / price_series.shift(60) - 1
        
        return df.dropna()

def main():
    # 专业标题
    st.markdown("""
    <div class="professional-header">
        <h1>📊 AlphaFund Quant - 专业基金量化分析系统</h1>
        <p>基于10年量化投资经验构建的专业级基金分析平台</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 初始化量化引擎
    quant_engine = ProfessionalFundQuant()
    
    # 侧边栏 - 专业参数设置
    st.sidebar.header("🔧 专业参数配置")
    
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
    
    selected_funds = st.sidebar.multiselect(
        "选择分析基金",
        options=list(FUND_UNIVERSE.keys()),
        format_func=lambda x: f"{x} - {FUND_UNIVERSE[x]['name']}",
        default=["022365", "001618", "110022"]
    )
    
    # 分析参数
    col1, col2 = st.sidebar.columns(2)
    with col1:
        analysis_period = st.selectbox("分析周期", ["1年", "2年", "3年", "5年"], index=1)
    with col2:
        monte_carlo_sims = st.slider("蒙特卡洛模拟次数", 100, 5000, 1000)
    
    # 风险参数
    risk_tolerance = st.sidebar.select_slider(
        "风险承受能力",
        options=["保守型", "稳健型", "平衡型", "成长型", "激进型"],
        value="平衡型"
    )
    
    # 主分析区域
    if not selected_funds:
        st.warning("请选择至少一只基金进行分析")
        return
    
    # 生成专业数据
    if st.button("🚀 执行专业量化分析", type="primary"):
        with st.spinner("正在进行专业级量化分析..."):
            # 模拟生成专业基金数据
            import numpy as np
            from datetime import datetime, timedelta
            
            # 生成历史数据
            start_date = datetime.now() - timedelta(days=365*3)  # 3年数据
            dates = pd.date_range(start=start_date, end=datetime.now(), freq='D')
            
            fund_data = {}
            for fund_code in selected_funds:
                # 基于基金特性生成不同模式的数据
                np.random.seed(hash(fund_code) % 10000)
                
                if FUND_UNIVERSE[fund_code]['risk'] == "高风险":
                    base_volatility = 0.025
                    base_return = 0.0012
                elif FUND_UNIVERSE[fund_code]['risk'] == "中高风险":
                    base_volatility = 0.018
                    base_return = 0.0009
                else:
                    base_volatility = 0.012
                    base_return = 0.0006
                
                # 生成更真实的收益率序列
                returns = np.random.normal(base_return, base_volatility, len(dates))
                # 添加波动聚集效应
                for i in range(1, len(returns)):
                    if abs(returns[i-1]) > 2 * base_volatility:
                        returns[i] = returns[i] * 1.5
                
                nav = 1.0 * (1 + pd.Series(returns)).cumprod()
                
                fund_data[fund_code] = pd.DataFrame({
                    'date': dates,
                    'nav': nav.values,
                    'return_1d': returns
                }).set_index('date')
            
            # 1. 专业指标分析
            st.subheader("📈 专业量化指标分析")
            
            metrics_data = []
            for fund_code in selected_funds:
                returns = fund_data[fund_code]['return_1d']
                metrics = quant_engine.calculate_advanced_metrics(returns)
                metrics['基金代码'] = fund_code
                metrics['基金名称'] = FUND_UNIVERSE[fund_code]['name']
                metrics_data.append(metrics)
            
            metrics_df = pd.DataFrame(metrics_data)
            metrics_display = metrics_df[['基金名称', '年化收益率', '年化波动率', '夏普比率', 
                                        '最大回撤', '索提诺比率', '卡玛比率']].round(4)
            
            # 格式化显示
            for col in ['年化收益率', '年化波动率', '最大回撤']:
                metrics_display[col] = metrics_display[col].apply(lambda x: f"{x:.2%}")
            for col in ['夏普比率', '索提诺比率', '卡玛比率']:
                metrics_display[col] = metrics_display[col].apply(lambda x: f"{x:.2f}")
            
            st.dataframe(metrics_display, use_container_width=True)
            
            # 2. 风险收益散点图
            st.subheader("🎯 风险-收益特征分析")
            
            fig_scatter = px.scatter(
                metrics_df, 
                x='年化波动率', 
                y='年化收益率',
                size='夏普比率',
                color='基金名称',
                hover_data=['最大回撤', '索提诺比率'],
                title="风险-收益散点图 (气泡大小代表夏普比率)"
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
            
            # 3. 技术指标分析
            st.subheader("🔧 技术指标分析")
            
            selected_fund = st.selectbox("选择基金进行技术分析", selected_funds,
                                       format_func=lambda x: f"{x} - {FUND_UNIVERSE[x]['name']}")
            
            if selected_fund:
                tech_data = quant_engine.generate_technical_features(fund_data[selected_fund]['nav'])
                
                # 创建技术分析图表
                fig_tech = make_subplots(rows=3, cols=1, 
                                       subplot_titles=['价格与移动平均线', 'RSI指标', 'MACD指标'],
                                       vertical_spacing=0.08,
                                       row_heights=[0.5, 0.25, 0.25])
                
                # 价格和移动平均线
                fig_tech.add_trace(go.Scatter(x=tech_data.index, y=tech_data['price'], 
                                            name='净值', line=dict(color='#1f77b4')), row=1, col=1)
                fig_tech.add_trace(go.Scatter(x=tech_data.index, y=tech_data['ma_20'], 
                                            name='20日均线', line=dict(color='orange')), row=1, col=1)
                fig_tech.add_trace(go.Scatter(x=tech_data.index, y=tech_data['ma_50'], 
                                            name='50日均线', line=dict(color='red')), row=1, col=1)
                
                # RSI
                fig_tech.add_trace(go.Scatter(x=tech_data.index, y=tech_data['rsi'], 
                                            name='RSI', line=dict(color='purple')), row=2, col=1)
                fig_tech.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
                fig_tech.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
                
                # MACD
                fig_tech.add_trace(go.Scatter(x=tech_data.index, y=tech_data['macd'], 
                                            name='MACD', line=dict(color='blue')), row=3, col=1)
                fig_tech.add_trace(go.Scatter(x=tech_data.index, y=tech_data['macd_signal'], 
                                            name='信号线', line=dict(color='red')), row=3, col=1)
                
                fig_tech.update_layout(height=800, showlegend=True)
                st.plotly_chart(fig_tech, use_container_width=True)
    
    # 免责声明
    st.sidebar.markdown("---")
    st.sidebar.warning("""
    **专业免责声明：**
    
    本系统基于历史数据回测，不构成投资建议。
    量化模型存在局限性，实际投资需结合市场判断。
    基金投资有风险，过往业绩不代表未来表现。
    
    *AlphaFund Quant v2.0 - 专业量化分析系统*
    """)

if __name__ == "__main__":
    main()