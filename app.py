import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# 设置专业级页面配置
st.set_page_config(
    page_title="QuantMaster Pro - 专业量化模型系统",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 专业CSS样式
st.markdown("""
<style>
    .professional-header {
        background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
        text-align: center;
    }
    .factor-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #007bff;
        margin-bottom: 1rem;
        transition: all 0.3s ease;
    }
    .factor-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .model-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #dee2e6;
        margin-bottom: 1rem;
    }
    .metric-badge {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        border-radius: 20px;
        font-size: 0.85rem;
        margin: 0.1rem;
    }
    .metric-good { background-color: #d4edda; color: #155724; }
    .metric-neutral { background-color: #fff3cd; color: #856404; }
    .metric-bad { background-color: #f8d7da; color: #721c24; }
    .tab-content {
        padding: 1.5rem;
        background-color: white;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

class QuantModelSystem:
    """专业量化模型系统"""
    
    def __init__(self):
        self.factors = {}
        self.models = {}
        self.scaler = StandardScaler()
        self.risk_free_rate = 0.015
        
    def generate_factor_data(self, fund_code, start_date='2020-01-01'):
        """生成多因子数据"""
        try:
            np.random.seed(hash(fund_code) % 10000)
            
            # 创建日期范围
            dates = pd.date_range(start=start_date, end=datetime.now(), freq='B')
            
            # 生成基础收益率序列
            base_return = 0.0008
            base_volatility = 0.02
            returns = np.random.normal(base_return, base_volatility, len(dates))
            
            # 添加市场因子
            market_factor = np.random.normal(0.0005, 0.015, len(dates))
            returns = returns * 0.6 + market_factor * 0.4
            
            # 生成价格序列
            price = 1.0 * (1 + pd.Series(returns)).cumprod()
            
            # 计算各类因子
            factor_data = pd.DataFrame(index=dates)
            factor_data['price'] = price.values
            factor_data['returns'] = returns
            
            # 动量因子
            factor_data['momentum_1m'] = price / price.shift(20) - 1
            factor_data['momentum_3m'] = price / price.shift(60) - 1
            factor_data['momentum_6m'] = price / price.shift(120) - 1
            
            # 估值因子（模拟）
            factor_data['pe_ratio'] = np.random.uniform(10, 30, len(dates))
            factor_data['pb_ratio'] = np.random.uniform(1, 5, len(dates))
            
            # 质量因子
            factor_data['roe'] = np.random.uniform(0.05, 0.25, len(dates))
            factor_data['roa'] = np.random.uniform(0.02, 0.15, len(dates))
            
            # 波动率因子
            factor_data['volatility_1m'] = pd.Series(returns).rolling(20).std()
            factor_data['volatility_3m'] = pd.Series(returns).rolling(60).std()
            
            # 流动性因子
            factor_data['volume'] = np.random.lognormal(10, 1, len(dates))
            factor_data['turnover'] = np.random.uniform(0.01, 0.1, len(dates))
            
            # 技术因子
            factor_data['rsi'] = self._calculate_rsi(price, 14)
            factor_data['macd'] = self._calculate_macd(price)
            factor_data['bollinger_position'] = self._calculate_bollinger_position(price, 20)
            
            # 规模因子（模拟）
            factor_data['market_cap'] = np.random.lognormal(20, 2, len(dates))
            factor_data['float_market_cap'] = factor_data['market_cap'] * 0.7
            
            # 删除NaN值
            factor_data = factor_data.dropna()
            
            return factor_data
            
        except Exception as e:
            st.error(f"生成因子数据时出错: {str(e)}")
            return pd.DataFrame()
    
    def _calculate_rsi(self, prices, period=14):
        """计算RSI指标"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices):
        """计算MACD"""
        exp1 = prices.ewm(span=12).mean()
        exp2 = prices.ewm(span=26).mean()
        macd = exp1 - exp2
        return macd
    
    def _calculate_bollinger_position(self, prices, window=20):
        """计算布林带位置"""
        sma = prices.rolling(window).mean()
        std = prices.rolling(window).std()
        upper = sma + 2 * std
        lower = sma - 2 * std
        position = (prices - lower) / (upper - lower)
        return position
    
    def calculate_factor_returns(self, factor_data, forward_period=5):
        """计算因子收益"""
        # 目标变量：未来N天的收益率
        factor_data = factor_data.copy()
        factor_data['target_return'] = factor_data['price'].shift(-forward_period) / factor_data['price'] - 1
        
        # 删除包含NaN的行
        factor_data = factor_data.dropna()
        
        # 计算因子与未来收益的相关性
        factor_cols = [col for col in factor_data.columns if col not in ['price', 'returns', 'target_return']]
        correlations = {}
        
        for factor in factor_cols:
            corr = factor_data[factor].corr(factor_data['target_return'])
            correlations[factor] = corr
        
        # 排序相关性
        sorted_correlations = dict(sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True))
        
        return sorted_correlations, factor_data
    
    def build_factor_model(self, factor_data, top_n=10):
        """构建多因子模型"""
        try:
            # 获取因子和目标变量
            factor_data = factor_data.copy()
            factor_cols = [col for col in factor_data.columns if col not in ['price', 'returns', 'target_return']]
            
            # 选择相关性最高的因子
            correlations, _ = self.calculate_factor_returns(factor_data)
            selected_factors = list(correlations.keys())[:top_n]
            
            # 准备数据
            X = factor_data[selected_factors]
            y = factor_data['target_return']
            
            # 标准化特征
            X_scaled = self.scaler.fit_transform(X)
            
            # 划分训练集和测试集（时间序列分割）
            tscv = TimeSeriesSplit(n_splits=5)
            
            # 训练线性回归模型
            model = Ridge(alpha=1.0)
            
            # 交叉验证
            cv_scores = []
            for train_idx, test_idx in tscv.split(X_scaled):
                X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                score = r2_score(y_test, y_pred)
                cv_scores.append(score)
            
            # 最终模型
            model.fit(X_scaled, y)
            
            # 获取因子权重
            factor_weights = dict(zip(selected_factors, model.coef_))
            sorted_weights = dict(sorted(factor_weights.items(), key=lambda x: abs(x[1]), reverse=True))
            
            return {
                'model': model,
                'selected_factors': selected_factors,
                'factor_weights': sorted_weights,
                'cv_mean_score': np.mean(cv_scores),
                'cv_std_score': np.std(cv_scores),
                'feature_importance': dict(zip(selected_factors, abs(model.coef_)))
            }
            
        except Exception as e:
            st.error(f"构建因子模型时出错: {str(e)}")
            return None
    
    def build_ml_model(self, factor_data, model_type='random_forest'):
        """构建机器学习模型"""
        try:
            # 准备特征和目标
            factor_cols = [col for col in factor_data.columns if col not in ['price', 'returns', 'target_return']]
            X = factor_data[factor_cols]
            y = factor_data['target_return']
            
            # 标准化特征
            X_scaled = self.scaler.fit_transform(X)
            
            # 时间序列分割
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
            
            # 选择模型
            if model_type == 'random_forest':
                model = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=20,
                    random_state=42
                )
            elif model_type == 'gradient_boosting':
                model = GradientBoostingRegressor(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=5,
                    random_state=42
                )
            else:
                model = LinearRegression()
            
            # 训练模型
            model.fit(X_train, y_train)
            
            # 预测
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # 评估指标
            metrics = {
                'train_r2': r2_score(y_train, y_pred_train),
                'test_r2': r2_score(y_test, y_pred_test),
                'train_mse': mean_squared_error(y_train, y_pred_train),
                'test_mse': mean_squared_error(y_test, y_pred_test),
                'train_mae': mean_absolute_error(y_train, y_pred_train),
                'test_mae': mean_absolute_error(y_test, y_pred_test)
            }
            
            # 特征重要性
            if hasattr(model, 'feature_importances_'):
                feature_importance = dict(zip(factor_cols, model.feature_importances_))
            elif hasattr(model, 'coef_'):
                feature_importance = dict(zip(factor_cols, abs(model.coef_)))
            else:
                feature_importance = {}
            
            return {
                'model': model,
                'metrics': metrics,
                'feature_importance': feature_importance,
                'predictions': {
                    'train': y_pred_train,
                    'test': y_pred_test,
                    'actual': y_test
                }
            }
            
        except Exception as e:
            st.error(f"构建机器学习模型时出错: {str(e)}")
            return None
    
    def generate_signals(self, factor_model, current_factors, threshold=0.02):
        """生成交易信号"""
        try:
            # 获取选中的因子
            selected_factors = factor_model['selected_factors']
            
            # 准备当前数据
            X_current = current_factors[selected_factors].values.reshape(1, -1)
            X_scaled = self.scaler.transform(X_current)
            
            # 预测未来收益
            predicted_return = factor_model['model'].predict(X_scaled)[0]
            
            # 生成信号
            if predicted_return > threshold:
                signal = "强烈买入"
                signal_strength = min(predicted_return / threshold, 3.0)
            elif predicted_return > threshold * 0.5:
                signal = "买入"
                signal_strength = predicted_return / threshold
            elif predicted_return > -threshold * 0.5:
                signal = "持有"
                signal_strength = 0
            elif predicted_return > -threshold:
                signal = "卖出"
                signal_strength = abs(predicted_return / threshold)
            else:
                signal = "强烈卖出"
                signal_strength = min(abs(predicted_return / threshold), 3.0)
            
            # 因子贡献分析
            factor_contributions = {}
            model_coef = factor_model['model'].coef_
            
            for i, factor in enumerate(selected_factors):
                contribution = model_coef[i] * X_current[0][i]
                factor_contributions[factor] = contribution
            
            return {
                'predicted_return': predicted_return,
                'signal': signal,
                'signal_strength': signal_strength,
                'factor_contributions': factor_contributions,
                'confidence': min(abs(predicted_return) / threshold, 1.0)
            }
            
        except Exception as e:
            st.error(f"生成信号时出错: {str(e)}")
            return None
    
    def portfolio_optimization(self, funds_data, target_return=None, risk_aversion=1.0):
        """投资组合优化"""
        try:
            # 收集所有基金的收益率
            returns_data = {}
            for fund_name, data in funds_data.items():
                if 'returns' in data.columns:
                    returns_data[fund_name] = data['returns']
            
            if len(returns_data) < 2:
                raise ValueError("至少需要2只基金进行组合优化")
            
            # 创建收益率矩阵
            returns_df = pd.DataFrame(returns_data).dropna()
            
            if len(returns_df) < 30:
                raise ValueError("数据不足，至少需要30个交易日数据")
            
            # 计算预期收益和协方差矩阵
            expected_returns = returns_df.mean() * 252
            cov_matrix = returns_df.cov() * 252
            
            # 马科维茨优化
            from scipy.optimize import minimize
            
            n_assets = len(expected_returns)
            
            def portfolio_return(weights):
                return weights.T @ expected_returns
            
            def portfolio_volatility(weights):
                return np.sqrt(weights.T @ cov_matrix @ weights)
            
            def objective(weights):
                return - (portfolio_return(weights) - 0.5 * risk_aversion * portfolio_volatility(weights) ** 2)
            
            # 约束条件
            constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
            bounds = tuple((0, 1) for _ in range(n_assets))
            
            # 初始权重
            initial_weights = np.array([1/n_assets] * n_assets)
            
            # 优化
            result = minimize(
                objective,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
            
            optimal_weights = result.x
            
            return {
                'weights': optimal_weights,
                'expected_return': portfolio_return(optimal_weights),
                'expected_volatility': portfolio_volatility(optimal_weights),
                'sharpe_ratio': (portfolio_return(optimal_weights) - self.risk_free_rate) / portfolio_volatility(optimal_weights)
            }
            
        except Exception as e:
            st.error(f"投资组合优化时出错: {str(e)}")
            return None

def main():
    # 专业标题
    st.markdown("""
    <div class="professional-header">
        <h1>🧠 QuantMaster Pro - 专业量化模型系统</h1>
        <p>基于多因子模型和机器学习的专业量化分析平台</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 初始化量化系统
    quant_system = QuantModelSystem()
    
    # 侧边栏 - 配置区域
    st.sidebar.header("🔧 系统配置")
    
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
    
    # 选择基金
    selected_funds = st.sidebar.multiselect(
        "选择分析基金",
        options=list(FUND_UNIVERSE.keys()),
        format_func=lambda x: f"{x} - {FUND_UNIVERSE[x]['name']}",
        default=["022365"],
        help="选择要进行量化分析的基金"
    )
    
    # 主内容区域
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 因子分析", 
        "🤖 机器学习", 
        "📈 模型回测", 
        "💼 组合优化", 
        "🎯 实时信号"
    ])
    
    # 初始化session state
    if 'factor_data' not in st.session_state:
        st.session_state.factor_data = {}
    if 'factor_models' not in st.session_state:
        st.session_state.factor_models = {}
    
    with tab1:
        st.markdown('<div class="tab-content">', unsafe_allow_html=True)
        st.subheader("📊 多因子分析")
        
        if selected_funds:
            if st.button("生成因子数据", type="primary"):
                with st.spinner("正在生成因子数据..."):
                    for fund_code in selected_funds:
                        fund_name = FUND_UNIVERSE[fund_code]['name']
                        st.write(f"**正在分析 {fund_name}**")
                        
                        # 生成因子数据
                        factor_data = quant_system.generate_factor_data(fund_code)
                        if not factor_data.empty:
                            st.session_state.factor_data[fund_code] = factor_data
                            
                            # 计算因子收益相关性
                            correlations, _ = quant_system.calculate_factor_returns(factor_data)
                            
                            # 显示相关性分析
                            st.write(f"**因子收益相关性 (前10个)**")
                            corr_df = pd.DataFrame({
                                '因子': list(correlations.keys())[:10],
                                '相关性': list(correlations.values())[:10]
                            })
                            
                            # 创建相关性图表
                            fig_corr = px.bar(
                                corr_df,
                                x='因子',
                                y='相关性',
                                title=f"{fund_name} - 因子收益相关性",
                                color='相关性',
                                color_continuous_scale='RdYlGn',
                                range_color=[-1, 1]
                            )
                            fig_corr.update_layout(height=400)
                            st.plotly_chart(fig_corr, use_container_width=True)
                            
                            # 显示因子数据预览
                            with st.expander("查看因子数据详情"):
                                st.dataframe(factor_data.describe(), use_container_width=True)
                        else:
                            st.error(f"无法生成 {fund_name} 的因子数据")
        else:
            st.info("请选择至少一只基金进行分析")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        st.markdown('<div class="tab-content">', unsafe_allow_html=True)
        st.subheader("🤖 机器学习模型")
        
        if st.session_state.factor_data:
            selected_fund = st.selectbox(
                "选择要建模的基金",
                options=list(st.session_state.factor_data.keys()),
                format_func=lambda x: FUND_UNIVERSE[x]['name']
            )
            
            if selected_fund:
                factor_data = st.session_state.factor_data[selected_fund]
                
                col1, col2 = st.columns(2)
                with col1:
                    model_type = st.selectbox(
                        "选择模型类型",
                        ["线性回归", "随机森林", "梯度提升"],
                        index=0
                    )
                
                with col2:
                    top_n_factors = st.slider("使用因子数量", 5, 30, 10)
                
                if st.button("训练机器学习模型", type="primary"):
                    with st.spinner("正在训练模型..."):
                        # 训练因子模型
                        factor_model = quant_system.build_factor_model(factor_data, top_n_factors)
                        
                        if factor_model:
                            st.session_state.factor_models[selected_fund] = factor_model
                            
                            # 显示模型结果
                            st.subheader("模型性能")
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric(
                                    "交叉验证R²均值",
                                    f"{factor_model['cv_mean_score']:.4f}"
                                )
                            with col2:
                                st.metric(
                                    "交叉验证R²标准差",
                                    f"{factor_model['cv_std_score']:.4f}"
                                )
                            
                            # 显示因子权重
                            st.subheader("因子权重分析")
                            
                            weights_df = pd.DataFrame({
                                '因子': list(factor_model['factor_weights'].keys()),
                                '权重': list(factor_model['factor_weights'].values())
                            })
                            
                            fig_weights = px.bar(
                                weights_df,
                                x='因子',
                                y='权重',
                                title="因子权重",
                                color='权重',
                                color_continuous_scale='RdBu'
                            )
                            fig_weights.update_layout(height=400)
                            st.plotly_chart(fig_weights, use_container_width=True)
                            
                            # 训练机器学习模型
                            ml_model_type = {
                                "线性回归": "linear",
                                "随机森林": "random_forest",
                                "梯度提升": "gradient_boosting"
                            }[model_type]
                            
                            ml_model = quant_system.build_ml_model(factor_data, ml_model_type)
                            
                            if ml_model:
                                # 显示ML模型结果
                                st.subheader("机器学习模型性能")
                                
                                metrics_df = pd.DataFrame({
                                    '指标': ['R²分数', '均方误差', '平均绝对误差'],
                                    '训练集': [
                                        ml_model['metrics']['train_r2'],
                                        ml_model['metrics']['train_mse'],
                                        ml_model['metrics']['train_mae']
                                    ],
                                    '测试集': [
                                        ml_model['metrics']['test_r2'],
                                        ml_model['metrics']['test_mse'],
                                        ml_model['metrics']['test_mae']
                                    ]
                                })
                                
                                st.dataframe(metrics_df, use_container_width=True)
                                
                                # 特征重要性
                                if ml_model['feature_importance']:
                                    st.subheader("特征重要性")
                                    
                                    importance_df = pd.DataFrame({
                                        '特征': list(ml_model['feature_importance'].keys())[:15],
                                        '重要性': list(ml_model['feature_importance'].values())[:15]
                                    }).sort_values('重要性', ascending=False)
                                    
                                    fig_importance = px.bar(
                                        importance_df,
                                        x='特征',
                                        y='重要性',
                                        title="特征重要性排名",
                                        color='重要性',
                                        color_continuous_scale='Blues'
                                    )
                                    fig_importance.update_layout(height=400)
                                    st.plotly_chart(fig_importance, use_container_width=True)
                                
                                # 预测 vs 实际对比
                                st.subheader("预测 vs 实际对比")
                                
                                fig_predictions = go.Figure()
                                fig_predictions.add_trace(go.Scatter(
                                    x=np.arange(len(ml_model['predictions']['test'])),
                                    y=ml_model['predictions']['test'],
                                    name='预测值',
                                    mode='lines'
                                ))
                                fig_predictions.add_trace(go.Scatter(
                                    x=np.arange(len(ml_model['predictions']['actual'])),
                                    y=ml_model['predictions']['actual'],
                                    name='实际值',
                                    mode='lines'
                                ))
                                
                                fig_predictions.update_layout(
                                    title="测试集预测 vs 实际",
                                    xaxis_title="样本",
                                    yaxis_title="收益率",
                                    height=400
                                )
                                st.plotly_chart(fig_predictions, use_container_width=True)
                        else:
                            st.error("模型训练失败")
        else:
            st.info("请先在'因子分析'标签页生成因子数据")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab3:
        st.markdown('<div class="tab-content">', unsafe_allow_html=True)
        st.subheader("📈 模型回测")
        
        if st.session_state.factor_models:
            selected_fund = st.selectbox(
                "选择要回测的模型",
                options=list(st.session_state.factor_models.keys()),
                format_func=lambda x: FUND_UNIVERSE[x]['name'],
                key="backtest_fund"
            )
            
            if selected_fund:
                factor_model = st.session_state.factor_models[selected_fund]
                factor_data = st.session_state.factor_data[selected_fund]
                
                # 回测参数
                col1, col2 = st.columns(2)
                with col1:
                    initial_capital = st.number_input("初始资金", value=100000, min_value=1000, step=1000)
                with col2:
                    threshold = st.slider("交易阈值 (%)", 0.1, 5.0, 2.0) / 100
                
                if st.button("执行模型回测", type="primary"):
                    with st.spinner("正在执行回测..."):
                        # 模拟交易回测
                        cash = initial_capital
                        shares = 0
                        portfolio_values = []
                        trades = []
                        
                        for i in range(len(factor_data)):
                            if i >= 100:  # 从第100天开始，确保有足够的历史数据
                                current_factors = factor_data.iloc[i]
                                
                                # 准备特征数据
                                selected_factors = factor_model['selected_factors']
                                if set(selected_factors).issubset(factor_data.columns):
                                    X_current = factor_data[selected_factors].iloc[i].values.reshape(1, -1)
                                    X_scaled = quant_system.scaler.transform(X_current)
                                    
                                    # 预测收益
                                    predicted_return = factor_model['model'].predict(X_scaled)[0]
                                    
                                    current_price = factor_data['price'].iloc[i]
                                    
                                    # 交易逻辑
                                    if predicted_return > threshold and cash > 0:
                                        # 买入
                                        buy_amount = cash * 0.5  # 使用50%现金买入
                                        buy_shares = buy_amount / current_price
                                        shares += buy_shares
                                        cash -= buy_amount
                                        trades.append({
                                            'date': factor_data.index[i],
                                            'action': 'BUY',
                                            'price': current_price,
                                            'shares': buy_shares,
                                            'predicted_return': predicted_return
                                        })
                                    elif predicted_return < -threshold and shares > 0:
                                        # 卖出
                                        sell_shares = shares * 0.5  # 卖出50%持仓
                                        cash += sell_shares * current_price
                                        shares -= sell_shares
                                        trades.append({
                                            'date': factor_data.index[i],
                                            'action': 'SELL',
                                            'price': current_price,
                                            'shares': sell_shares,
                                            'predicted_return': predicted_return
                                        })
                            
                            portfolio_values.append(shares * factor_data['price'].iloc[i] + cash)
                        
                        # 计算回测结果
                        portfolio_series = pd.Series(portfolio_values, index=factor_data.index)
                        benchmark_series = initial_capital * (factor_data['price'] / factor_data['price'].iloc[0])
                        
                        # 计算绩效指标
                        portfolio_returns = portfolio_series.pct_change().dropna()
                        benchmark_returns = benchmark_series.pct_change().dropna()
                        
                        total_return = (portfolio_series.iloc[-1] / portfolio_series.iloc[0]) - 1
                        benchmark_return = (benchmark_series.iloc[-1] / benchmark_series.iloc[0]) - 1
                        
                        volatility = portfolio_returns.std() * np.sqrt(252)
                        sharpe_ratio = (total_return * 252/len(portfolio_series) - quant_system.risk_free_rate) / volatility if volatility > 0 else 0
                        
                        # 最大回撤
                        cumulative = (1 + portfolio_returns).cumprod()
                        rolling_max = cumulative.expanding().max()
                        drawdown = (cumulative - rolling_max) / rolling_max
                        max_drawdown = drawdown.min()
                        
                        # 显示回测结果
                        st.subheader("回测结果")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("策略总收益", f"{total_return:.2%}")
                        with col2:
                            st.metric("基准收益", f"{benchmark_return:.2%}")
                        with col3:
                            st.metric("超额收益", f"{(total_return - benchmark_return):.2%}")
                        with col4:
                            st.metric("夏普比率", f"{sharpe_ratio:.2f}")
                        
                        # 净值曲线
                        st.subheader("净值曲线对比")
                        
                        fig_backtest = go.Figure()
                        fig_backtest.add_trace(go.Scatter(
                            x=portfolio_series.index,
                            y=portfolio_series,
                            name='策略净值',
                            line=dict(width=2, color='blue')
                        ))
                        fig_backtest.add_trace(go.Scatter(
                            x=benchmark_series.index,
                            y=benchmark_series,
                            name='基准净值',
                            line=dict(width=1, color='gray', dash='dash')
                        ))
                        
                        fig_backtest.update_layout(
                            title="策略净值 vs 基准净值",
                            xaxis_title="日期",
                            yaxis_title="净值",
                            height=500
                        )
                        st.plotly_chart(fig_backtest, use_container_width=True)
                        
                        # 交易记录
                        if trades:
                            st.subheader("交易记录")
                            trades_df = pd.DataFrame(trades)
                            st.dataframe(trades_df, use_container_width=True)
                        else:
                            st.info("在回测期间没有产生交易")
        else:
            st.info("请先在'机器学习'标签页训练模型")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab4:
        st.markdown('<div class="tab-content">', unsafe_allow_html=True)
        st.subheader("💼 投资组合优化")
        
        if len(selected_funds) >= 2:
            if st.button("执行组合优化", type="primary"):
                with st.spinner("正在优化投资组合..."):
                    # 收集所有基金的因子数据
                    funds_data = {}
                    for fund_code in selected_funds:
                        if fund_code in st.session_state.factor_data:
                            funds_data[FUND_UNIVERSE[fund_code]['name']] = st.session_state.factor_data[fund_code]
                    
                    if len(funds_data) >= 2:
                        # 执行组合优化
                        optimization_result = quant_system.portfolio_optimization(funds_data)
                        
                        if optimization_result:
                            # 显示优化结果
                            st.subheader("优化结果")
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric(
                                    "预期年化收益",
                                    f"{optimization_result['expected_return']:.2%}"
                                )
                            with col2:
                                st.metric(
                                    "预期波动率",
                                    f"{optimization_result['expected_volatility']:.2%}"
                                )
                            with col3:
                                st.metric(
                                    "夏普比率",
                                    f"{optimization_result['sharpe_ratio']:.2f}"
                                )
                            
                            # 显示权重分配
                            st.subheader("最优权重分配")
                            
                            weights_df = pd.DataFrame({
                                '基金': list(funds_data.keys()),
                                '权重': optimization_result['weights'],
                                '建议': ['超配' if w > 1/len(funds_data) else '低配' for w in optimization_result['weights']]
                            }).sort_values('权重', ascending=False)
                            
                            fig_weights = px.pie(
                                weights_df,
                                values='权重',
                                names='基金',
                                title="最优投资组合权重分配",
                                hole=0.3
                            )
                            st.plotly_chart(fig_weights, use_container_width=True)
                            
                            # 显示权重表格
                            st.dataframe(weights_df, use_container_width=True)
                            
                            # 有效前沿分析
                            st.subheader("有效前沿")
                            
                            # 生成随机权重组合
                            n_portfolios = 1000
                            portfolio_returns = []
                            portfolio_volatilities = []
                            
                            for _ in range(n_portfolios):
                                weights = np.random.random(len(funds_data))
                                weights /= weights.sum()
                                
                                # 收集所有基金的收益率
                                returns_list = []
                                for fund_name, data in funds_data.items():
                                    if 'returns' in data.columns:
                                        returns_list.append(data['returns'])
                                
                                if returns_list:
                                    returns_df = pd.concat(returns_list, axis=1).dropna()
                                    if len(returns_df) > 0:
                                        cov_matrix = returns_df.cov() * 252
                                        expected_returns = returns_df.mean() * 252
                                        
                                        port_return = weights.T @ expected_returns
                                        port_volatility = np.sqrt(weights.T @ cov_matrix @ weights)
                                        
                                        portfolio_returns.append(port_return)
                                        portfolio_volatilities.append(port_volatility)
                            
                            if portfolio_returns and portfolio_volatilities:
                                # 创建有效前沿图表
                                frontier_df = pd.DataFrame({
                                    '收益率': portfolio_returns,
                                    '波动率': portfolio_volatilities,
                                    '夏普比率': [(r - quant_system.risk_free_rate) / v if v > 0 else 0 
                                                for r, v in zip(portfolio_returns, portfolio_volatilities)]
                                })
                                
                                fig_frontier = px.scatter(
                                    frontier_df,
                                    x='波动率',
                                    y='收益率',
                                    color='夏普比率',
                                    title="有效前沿",
                                    color_continuous_scale='Viridis'
                                )
                                
                                # 添加最优组合点
                                fig_frontier.add_trace(go.Scatter(
                                    x=[optimization_result['expected_volatility']],
                                    y=[optimization_result['expected_return']],
                                    mode='markers',
                                    marker=dict(size=15, color='red', symbol='star'),
                                    name='最优组合'
                                ))
                                
                                st.plotly_chart(fig_frontier, use_container_width=True)
                        else:
                            st.error("组合优化失败")
                    else:
                        st.warning("需要至少2只基金的数据进行组合优化")
        else:
            st.info("请选择至少2只基金进行组合优化")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab5:
        st.markdown('<div class="tab-content">', unsafe_allow_html=True)
        st.subheader("🎯 实时交易信号")
        
        if st.session_state.factor_models and st.session_state.factor_data:
            # 选择基金和模型
            col1, col2 = st.columns(2)
            with col1:
                signal_fund = st.selectbox(
                    "选择基金",
                    options=list(st.session_state.factor_models.keys()),
                    format_func=lambda x: FUND_UNIVERSE[x]['name'],
                    key="signal_fund"
                )
            
            with col2:
                signal_threshold = st.slider("信号阈值 (%)", 0.5, 10.0, 2.0) / 100
            
            if st.button("生成实时信号", type="primary"):
                with st.spinner("正在分析..."):
                    factor_model = st.session_state.factor_models[signal_fund]
                    factor_data = st.session_state.factor_data[signal_fund]
                    
                    # 获取最新数据
                    latest_factors = factor_data.iloc[-1]
                    
                    # 生成信号
                    signal_result = quant_system.generate_signals(
                        factor_model, 
                        latest_factors.to_frame().T, 
                        signal_threshold
                    )
                    
                    if signal_result:
                        # 显示信号
                        st.subheader("📢 交易信号")
                        
                        # 信号强度指示器
                        signal_strength = signal_result['signal_strength']
                        signal_color = {
                            "强烈买入": "green",
                            "买入": "lightgreen", 
                            "持有": "gray",
                            "卖出": "lightcoral",
                            "强烈卖出": "red"
                        }.get(signal_result['signal'], "gray")
                        
                        # 创建信号卡片
                        st.markdown(f"""
                        <div style="
                            background-color: {signal_color};
                            color: white;
                            padding: 2rem;
                            border-radius: 10px;
                            text-align: center;
                            margin: 1rem 0;
                            font-size: 1.5rem;
                            font-weight: bold;
                        ">
                            {signal_result['signal']}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # 信号详情
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric(
                                "预测收益率",
                                f"{signal_result['predicted_return']:.2%}",
                                delta=f"{signal_result['predicted_return'] - signal_threshold:.2%}"
                            )
                        with col2:
                            st.metric(
                                "信号强度",
                                f"{signal_result['signal_strength']:.2f}"
                            )
                        with col3:
                            st.metric(
                                "置信度",
                                f"{signal_result['confidence']:.2%}"
                            )
                        
                        # 因子贡献分析
                        st.subheader("📊 因子贡献分析")
                        
                        contributions_df = pd.DataFrame({
                            '因子': list(signal_result['factor_contributions'].keys()),
                            '贡献度': list(signal_result['factor_contributions'].values())
                        }).sort_values('贡献度', key=abs, ascending=False)
                        
                        fig_contributions = px.bar(
                            contributions_df.head(10),
                            x='因子',
                            y='贡献度',
                            title="前10大因子贡献度",
                            color='贡献度',
                            color_continuous_scale='RdYlBu'
                        )
                        fig_contributions.update_layout(height=400)
                        st.plotly_chart(fig_contributions, use_container_width=True)
                        
                        # 历史信号表现
                        st.subheader("📈 历史信号表现")
                        
                        # 分析过去一段时间的信号准确性
                        history_days = 100
                        history_signals = []
                        history_actual = []
                        
                        for i in range(len(factor_data) - history_days, len(factor_data)):
                            if i >= 100:
                                current_factors = factor_data.iloc[i]
                                X_current = factor_data[factor_model['selected_factors']].iloc[i].values.reshape(1, -1)
                                X_scaled = quant_system.scaler.transform(X_current)
                                
                                predicted = factor_model['model'].predict(X_scaled)[0]
                                
                                # 计算实际收益
                                if i + 5 < len(factor_data):
                                    actual_return = factor_data['price'].iloc[i+5] / factor_data['price'].iloc[i] - 1
                                    
                                    history_signals.append(predicted)
                                    history_actual.append(actual_return)
                        
                        if history_signals and history_actual:
                            # 计算信号准确性
                            correct_predictions = 0
                            for pred, actual in zip(history_signals, history_actual):
                                if (pred > signal_threshold and actual > 0) or \
                                   (pred < -signal_threshold and actual < 0) or \
                                   (abs(pred) <= signal_threshold and abs(actual) < signal_threshold):
                                    correct_predictions += 1
                            
                            accuracy = correct_predictions / len(history_signals) if history_signals else 0
                            
                            st.metric("历史信号准确率", f"{accuracy:.2%}")
                            
                            # 创建历史信号图表
                            fig_history = go.Figure()
                            fig_history.add_trace(go.Scatter(
                                x=np.arange(len(history_signals)),
                                y=history_signals,
                                name='预测信号',
                                mode='lines+markers'
                            ))
                            fig_history.add_trace(go.Scatter(
                                x=np.arange(len(history_actual)),
                                y=history_actual,
                                name='实际收益',
                                mode='lines+markers'
                            ))
                            
                            fig_history.update_layout(
                                title="历史信号 vs 实际收益",
                                xaxis_title="时间点",
                                yaxis_title="收益率",
                                height=400
                            )
                            st.plotly_chart(fig_history, use_container_width=True)
                    else:
                        st.error("生成信号失败")
        else:
            st.info("请先在'机器学习'标签页训练模型")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # 底部信息
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **📚 量化模型说明**
    
    1. **因子模型**: 基于多因子线性回归
    2. **机器学习**: 随机森林、梯度提升等
    3. **组合优化**: 马科维茨最优组合
    4. **风险控制**: 夏普比率、最大回撤等
    
    *QuantMaster Pro v1.0*
    """)

if __name__ == "__main__":
    main()
