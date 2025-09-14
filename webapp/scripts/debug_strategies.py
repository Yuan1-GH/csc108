import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from app import StrategyAnalyzer
import numpy as np

# 创建分析器实例
analyzer = StrategyAnalyzer()

# 分析动态仓位策略
print("=== 动态仓位策略分析 ===")

# 获取策略收益
dynamic_returns = analyzer.strategy_dynamic_position()
print(f"动态仓位策略收益统计:")
print(f"均值: {dynamic_returns.mean():.6f}")
print(f"标准差: {dynamic_returns.std():.6f}")
print(f"最大值: {dynamic_returns.max():.6f}")
print(f"最小值: {dynamic_returns.min():.6f}")
print(f"非零收益数量: {(dynamic_returns != 0).sum()}")

# 检查仓位计算
returns = analyzer.calculate_returns(analyzer.price_data)
z_score = (analyzer.price_data - analyzer.price_data.rolling(20).mean()) / analyzer.price_data.rolling(20).std()
position = np.tanh(z_score / 1.5)

print(f"\n仓位分析:")
print(f"仓位均值: {position.mean():.6f}")
print(f"仓位标准差: {position.std():.6f}")
print(f"最大仓位: {position.max():.6f}")
print(f"最小仓位: {position.min():.6f}")

# 检查是否存在前视偏差
print(f"\n前视偏差检查:")
print(f"收益数据长度: {len(returns)}")
print(f"仓位数据长度: {len(position)}")
print(f"策略收益长度: {len(dynamic_returns)}")

# 比较其他策略
print(f"\n=== 所有策略对比 ===")
strategies = {
    '基础策略': analyzer.strategy_basic(),
    '固定阈值': analyzer.strategy_fixed_threshold(),
    '自适应阈值': analyzer.strategy_adaptive_threshold(),
    '自适应+趋势': analyzer.strategy_adaptive_trend(),
    '动态仓位': analyzer.strategy_dynamic_position(),
    '多因子': analyzer.strategy_multi_factor()
}

for name, ret in strategies.items():
    sharpe = analyzer.calculate_sharpe(ret)
    print(f"{name}: 夏普比率 = {sharpe:.4f}, 收益均值 = {ret.mean():.6f}, 收益标准差 = {ret.std():.6f}")