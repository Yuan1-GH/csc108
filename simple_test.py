#!/usr/bin/env python3
"""
简单的参数优化测试脚本
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'webapp'))

from app import StrategyAnalyzer

def test_optimization():
    """测试参数优化效果"""
    print("=" * 60)
    print("参数优化效果测试")
    print("=" * 60)

    # 创建分析器实例
    analyzer = StrategyAnalyzer()

    # 获取优化后的参数
    print("正在优化参数...")
    optimized_params = analyzer.optimize_strategy_parameters()

    print("\n优化结果:")
    print("-" * 40)

    for strategy_name, params in optimized_params.items():
        sharpe_after = params['sharpe']
        print(f"{strategy_name}:")
        print(f"  优化后夏普比率: {sharpe_after:.4f}")

        # 显示优化后的参数
        for param_name, param_value in params.items():
            if param_name != 'sharpe':
                if isinstance(param_value, list):
                    print(f"  {param_name}: [{', '.join(f'{v:.3f}' for v in param_value)}]")
                else:
                    print(f"  {param_name}: {param_value:.4f}")
        print()

    print("=" * 60)
    print("优化完成！")

if __name__ == "__main__":
    try:
        test_optimization()
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()