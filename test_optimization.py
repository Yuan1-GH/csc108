#!/usr/bin/env python3
"""
参数优化效果测试脚本
对比优化前后的策略表现
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'webapp'))

from app import StrategyAnalyzer
import json

def test_optimization_results():
    """测试参数优化效果"""
    print("=" * 60)
    print("参数优化效果测试")
    print("=" * 60)

    # 创建分析器实例
    analyzer = StrategyAnalyzer()

    # 获取优化后的参数
    print("正在优化参数...")
    optimized_params = analyzer.optimize_strategy_parameters()

    print("\n📊 优化结果:")
    print("-" * 40)

    total_sharpe_before = 0
    total_sharpe_after = 0
    strategy_count = 0

    for strategy_name, params in optimized_params.items():
        sharpe_before = 0
        sharpe_after = params['sharpe']

        # 估算优化前的夏普比率（使用默认参数）
        if strategy_name == '固定阈值':
            sharpe_before = analyzer.calculate_sharpe(analyzer.strategy_fixed_threshold(0.001))
        elif strategy_name == '自适应阈值':
            sharpe_before = analyzer.calculate_sharpe(analyzer.strategy_adaptive_threshold(7))
        elif strategy_name == '自适应+趋势':
            sharpe_before = analyzer.calculate_sharpe(analyzer.strategy_adaptive_trend(50))
        elif strategy_name == '动态仓位':
            sharpe_before = analyzer.calculate_sharpe(analyzer.strategy_dynamic_position(1.5))
        elif strategy_name == '多因子':
            sharpe_before = analyzer.calculate_sharpe(analyzer.strategy_multi_factor())

        print(f"{strategy_name}:")
        print(f"  优化前夏普比率: {sharpe_before:.4f}")
        print(f"  优化后夏普比率: {sharpe_after:.4f}")
        print(f"  改进幅度: {((sharpe_after - sharpe_before) / abs(sharpe_before) * 100):.1f}%")

        # 显示优化后的参数
        for param_name, param_value in params.items():
            if param_name != 'sharpe':
                if isinstance(param_value, list):
                    print(f"  {param_name}: [{', '.join(f'{v:.3f}' for v in param_value)}]")
                else:
                    print(f"  {param_name}: {param_value:.4f}")
        print()

        total_sharpe_before += sharpe_before
        total_sharpe_after += sharpe_after
        strategy_count += 1

    # 总体改进效果
    avg_improvement = ((total_sharpe_after - total_sharpe_before) / abs(total_sharpe_before) * 100)

    print("=" * 60)
    print("🎯 总体优化效果:")
    print(f"平均夏普比率改进: {avg_improvement:.1f}%")
    print(f"优化前平均夏普比率: {total_sharpe_before/strategy_count:.4f}")
    print(f"优化后平均夏普比率: {total_sharpe_after/strategy_count:.4f}")
    print("=" * 60)

    # 返回结果
    return {
        'optimized_params': optimized_params,
        'avg_improvement': avg_improvement,
        'sharpe_before': total_sharpe_before/strategy_count,
        'sharpe_after': total_sharpe_after/strategy_count
    }

def test_web_api():
    """测试Web API功能"""
    print("\n🌐 测试Web API功能...")

    import requests

    try:
        # 测试策略数据API
        response = requests.get('http://localhost:5000/api/strategies')
        if response.status_code == 200:
            data = response.json()
            print("✅ 策略数据API正常工作")

            # 检查参数是否包含在响应中
            has_params = any('parameters' in strategy_data and strategy_data['parameters']
                           for strategy_data in data.values())
            if has_params:
                print("✅ 参数信息已正确包含在API响应中")
            else:
                print("❌ 参数信息未包含在API响应中")
        else:
            print(f"❌ 策略数据API错误: {response.status_code}")

        # 测试优化参数API
        response = requests.get('http://localhost:5000/api/optimized-parameters')
        if response.status_code == 200:
            print("✅ 优化参数API正常工作")
        else:
            print(f"❌ 优化参数API错误: {response.status_code}")

    except Exception as e:
        print(f"❌ API测试失败: {e}")

if __name__ == "__main__":
    try:
        # 测试优化效果
        results = test_optimization_results()

        # 测试Web API
        test_web_api()

        print("\n🎉 测试完成！")
        print("\n📋 使用说明:")
        print("1. 访问 http://localhost:5000 查看网页界面")
        print("2. 点击'优化参数'按钮进行实时参数优化")
        print("3. 查看参数显示区域的优化结果")
        print("4. 对比优化前后的策略表现")

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()