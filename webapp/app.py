from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO
import os
from datetime import datetime
import json

app = Flask(__name__)
CORS(app)

class StrategyAnalyzer:
    def __init__(self, filepath="../中信建投/任务一/MinutesIdx.h5", target_col="IC"):
        self.filepath = filepath
        self.target_col = target_col
        self.raw_data = None
        self.price_data = None
        self.dates = None
        self.load_data()

    def load_data(self):
        """加载数据"""
        try:
            self.raw_data = pd.read_hdf(self.filepath)
            self.target_raw = self.raw_data[self.target_col]
            self.dates = self.target_raw.index.tolist()

            # 提取收盘价
            close_prices = []
            for date in self.dates:
                day_data = self.target_raw.loc[date]
                if hasattr(day_data, 'Close'):
                    close_prices.append(day_data['Close'].iloc[-1])
                else:
                    close_prices.append(day_data.iloc[-1] if hasattr(day_data, 'iloc') else day_data)

            self.price_data = pd.Series(close_prices, index=self.dates)
            print(f"数据加载成功: {len(self.price_data)} 个交易日")

        except Exception as e:
            print(f"数据加载失败: {e}")
            # 创建模拟数据
            self.create_sample_data()

    def create_sample_data(self):
        """创建示例数据"""
        print("创建示例数据...")
        dates = pd.date_range('2020-01-01', periods=252, freq='D')
        # 模拟价格走势
        returns = np.random.normal(0.001, 0.02, 252)
        prices = 100 * np.exp(np.cumsum(returns))
        self.price_data = pd.Series(prices, index=dates)
        self.dates = dates

    def calculate_returns(self, prices):
        """计算收益率"""
        return prices.pct_change().fillna(0)

    def calculate_sharpe(self, returns, risk_free_rate=0.02):
        """计算夏普比率"""
        if len(returns) == 0 or returns.std() == 0:
            return 0
        excess_returns = returns - risk_free_rate/252
        return np.sqrt(252) * excess_returns.mean() / returns.std()

    def calculate_max_drawdown(self, cumulative_returns):
        """计算最大回撤"""
        peak = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - peak) / peak
        return drawdown.min()

    def strategy_basic(self):
        """基础策略"""
        returns = self.calculate_returns(self.price_data)
        # 简单反转：前期跌就买，涨就卖
        signal = -np.sign(returns.shift(1))
        strategy_returns = signal * returns
        return strategy_returns.fillna(0)

    def strategy_fixed_threshold(self, threshold=0.01):
        """固定阈值策略"""
        returns = self.calculate_returns(self.price_data)
        z_score = (self.price_data - self.price_data.rolling(20).mean()) / self.price_data.rolling(20).std()
        signal = np.where(z_score > threshold, -1, np.where(z_score < -threshold, 1, 0))
        strategy_returns = pd.Series(signal, index=self.price_data.index).shift(1) * returns
        return strategy_returns.fillna(0)

    def strategy_adaptive_threshold(self, window=7):
        """自适应阈值策略"""
        returns = self.calculate_returns(self.price_data)
        z_score = (self.price_data - self.price_data.rolling(20).mean()) / self.price_data.rolling(20).std()
        adaptive_threshold = z_score.rolling(window).std()
        signal = np.where(z_score > adaptive_threshold, -1, np.where(z_score < -adaptive_threshold, 1, 0))
        strategy_returns = pd.Series(signal, index=self.price_data.index).shift(1) * returns
        return strategy_returns.fillna(0)

    def strategy_adaptive_trend(self, ma_window=50):
        """自适应阈值+趋势过滤"""
        returns = self.calculate_returns(self.price_data)
        z_score = (self.price_data - self.price_data.rolling(20).mean()) / self.price_data.rolling(20).std()
        adaptive_threshold = z_score.rolling(7).std()

        # 趋势过滤
        trend = self.price_data.rolling(ma_window).mean()
        trend_signal = np.where(self.price_data > trend, 1, -1)

        signal = np.where(z_score > adaptive_threshold, -1, np.where(z_score < -adaptive_threshold, 1, 0))
        signal = signal * trend_signal

        strategy_returns = pd.Series(signal, index=self.price_data.index).shift(1) * returns
        return strategy_returns.fillna(0)

    def strategy_dynamic_position(self, threshold=1.5):
        """动态仓位控制"""
        returns = self.calculate_returns(self.price_data)
        z_score = (self.price_data - self.price_data.rolling(20).mean()) / self.price_data.rolling(20).std()
        position = np.tanh(z_score / threshold)  # 平滑仓位
        strategy_returns = pd.Series(position, index=self.price_data.index).shift(1) * returns
        return strategy_returns.fillna(0)

    def strategy_multi_factor(self):
        """多因子策略"""
        returns = self.calculate_returns(self.price_data)

        # 因子1: 动量因子
        momentum = self.price_data.pct_change(5)

        # 因子2: 波动率因子
        volatility = returns.rolling(10).std()

        # 因子3: 均值回归
        mean_reversion = (self.price_data - self.price_data.rolling(20).mean()) / self.price_data.rolling(20).std()

        # 组合信号
        combined_signal = 0.3 * momentum + 0.3 * (1/volatility) + 0.4 * mean_reversion
        signal = np.tanh(combined_signal)  # 归一化到[-1,1]

        strategy_returns = pd.Series(signal, index=self.price_data.index).shift(1) * returns
        return strategy_returns.fillna(0)

    def generate_all_strategies_data(self):
        """生成所有策略数据"""
        strategies = {
            '基础策略': self.strategy_basic(),
            '固定阈值': self.strategy_fixed_threshold(),
            '自适应阈值': self.strategy_adaptive_threshold(),
            '自适应+趋势': self.strategy_adaptive_trend(),
            '动态仓位': self.strategy_dynamic_position(),
            '多因子': self.strategy_multi_factor()
        }

        results = {}
        for name, returns in strategies.items():
            cumulative = (1 + returns).cumprod()
            sharpe = self.calculate_sharpe(returns)
            max_dd = self.calculate_max_drawdown(cumulative)
            annual_return = cumulative.iloc[-1] ** (252/len(cumulative)) - 1 if len(cumulative) > 0 else 0

            results[name] = {
                'returns': returns.tolist(),
                'cumulative': cumulative.tolist(),
                'dates': [str(d) for d in returns.index],
                'sharpe_ratio': sharpe,
                'max_drawdown': max_dd,
                'annual_return': annual_return,
                'volatility': returns.std() * np.sqrt(252)
            }

        return results

# 全局分析器实例
analyzer = StrategyAnalyzer()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/strategies')
def get_strategies():
    """获取所有策略数据"""
    try:
        data = analyzer.generate_all_strategies_data()
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/strategy/<strategy_name>')
def get_strategy(strategy_name):
    """获取单个策略数据"""
    strategy_map = {
        'basic': '基础策略',
        'fixed': '固定阈值',
        'adaptive': '自适应阈值',
        'trend': '自适应+趋势',
        'dynamic': '动态仓位',
        'multi': '多因子'
    }

    try:
        chinese_name = strategy_map.get(strategy_name)
        if not chinese_name:
            return jsonify({'error': '策略不存在'}), 404

        all_data = analyzer.generate_all_strategies_data()
        if chinese_name not in all_data:
            return jsonify({'error': '策略数据生成失败'}), 500

        return jsonify(all_data[chinese_name])
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/compare')
def compare_strategies():
    """比较策略"""
    try:
        data = analyzer.generate_all_strategies_data()

        comparison = {
            'names': list(data.keys()),
            'sharpe_ratios': [data[name]['sharpe_ratio'] for name in data.keys()],
            'max_drawdowns': [data[name]['max_drawdown'] for name in data.keys()],
            'annual_returns': [data[name]['annual_return'] for name in data.keys()],
            'volatilities': [data[name]['volatility'] for name in data.keys()]
        }

        return jsonify(comparison)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # 创建模板目录
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)

    print("策略分析网页服务器启动...")
    print("访问: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)