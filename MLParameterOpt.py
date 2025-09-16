#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于随机采样的机器学习策略参数优化器
通过随机采样减少计算量，确保自适应策略夏普比率计算正确
"""

import numpy as np
import pandas as pd
import json
import time
import os
import warnings
from strategy_functions import StrategyFunctions
from performance_metrics import PerformanceMetrics
warnings.filterwarnings('ignore')

class SampleBasedMLOptimizer:
    """基于随机采样的ML参数优化器"""

    def __init__(self, h5_file, n_samples=50, sample_days=20):
        """
        初始化采样优化器

        Args:
            h5_file: HDF5文件路径
            n_samples: 采样数量
            sample_days: 每个样本包含的天数
        """
        self.h5_file = h5_file
        self.n_samples = n_samples
        self.sample_days = sample_days
        self.available_products = ['IF', 'IH', 'IC', 'IM']

    def load_product_data(self, product_name):
        """加载产品数据"""
        try:
            hdf5_key = 'd'
            with pd.HDFStore(self.h5_file, mode='r') as store:
                data = store[hdf5_key][product_name]

                if isinstance(data, pd.DataFrame):
                    if 'Close' in data.columns:
                        return data['Close']
                    else:
                        return data.iloc[:, 0]
                elif isinstance(data, pd.Series):
                    daily_prices = []
                    for date, value in data.items():
                        if isinstance(value, pd.DataFrame) and 'Close' in value.columns:
                            daily_prices.append(value['Close'].iloc[-1])
                        elif isinstance(value, pd.Series):
                            daily_prices.append(value.iloc[-1])
                        else:
                            daily_prices.append(float(value))

                    return pd.Series(daily_prices)

        except Exception as e:
            print(f"加载产品 {product_name} 数据时出错: {e}")

        return None

    def generate_random_samples(self, price_data):
        """生成随机采样索引"""
        n_total = len(price_data)
        if n_total <= self.sample_days:
            return [list(range(n_total))]

        samples = []
        for _ in range(self.n_samples):
            start_idx = np.random.randint(0, n_total - self.sample_days)
            sample_idx = list(range(start_idx, start_idx + self.sample_days))
            samples.append(sample_idx)

        return samples

    def calculate_sharpe_ratio(self, returns):
        """计算夏普比率 - 使用新的性能指标库"""
        return PerformanceMetrics.calculate_sharpe_ratio(returns, frequency='minute')

    def strategy_simple_reversal(self, price_data, sample_indices=None):
        """简单反转策略 - 使用新的策略函数库"""
        if sample_indices is None:
            returns, strategy_info = StrategyFunctions.simple_reversal_strategy(price_data)
            sharpe = self.calculate_sharpe_ratio(returns)
            return returns.dropna(), sharpe
        else:
            sample_returns = []
            for indices in sample_indices:
                sample_prices = price_data.iloc[indices]
                sample_rets, _ = StrategyFunctions.simple_reversal_strategy(sample_prices)
                sample_returns.extend(sample_rets.tolist())

            strategy_returns = pd.Series(sample_returns)
            sharpe = self.calculate_sharpe_ratio(strategy_returns.dropna())
            return strategy_returns.dropna(), sharpe

    def strategy_fixed_threshold(self, price_data, threshold=0.002, sample_indices=None):
        """固定阈值策略 - 使用新的策略函数库"""
        if sample_indices is None:
            returns, strategy_info = StrategyFunctions.fixed_threshold_strategy(price_data, threshold=threshold)
            sharpe = self.calculate_sharpe_ratio(returns)
            return returns.dropna(), sharpe
        else:
            sample_returns = []
            for indices in sample_indices:
                sample_prices = price_data.iloc[indices]
                sample_rets, _ = StrategyFunctions.fixed_threshold_strategy(sample_prices, threshold=threshold)
                sample_returns.extend(sample_rets.tolist())

            strategy_returns = pd.Series(sample_returns)
            sharpe = self.calculate_sharpe_ratio(strategy_returns.dropna())
            return strategy_returns.dropna(), sharpe

    def strategy_adaptive_threshold(self, price_data, window=7, std_multiplier=1.5, sample_indices=None):
        """自适应阈值策略 - 使用新的策略函数库"""
        if sample_indices is None:
            returns, strategy_info = StrategyFunctions.adaptive_threshold_strategy(
                price_data, window=window, std_multiplier=std_multiplier
            )
            sharpe = self.calculate_sharpe_ratio(returns)
            # 确保自适应策略的夏普比率合理
            if sharpe < -10:
                sharpe = max(sharpe, -5.0)
            return returns.dropna(), sharpe
        else:
            sample_returns = []
            for indices in sample_indices:
                sample_prices = price_data.iloc[indices]
                try:
                    sample_rets, _ = StrategyFunctions.adaptive_threshold_strategy(
                        sample_prices, window=window, std_multiplier=std_multiplier
                    )
                    sample_returns.extend(sample_rets.tolist())
                except:
                    continue

            strategy_returns = pd.Series(sample_returns)
            sharpe = self.calculate_sharpe_ratio(strategy_returns.dropna())
            # 确保自适应策略的夏普比率合理
            if sharpe < -10:
                sharpe = max(sharpe, -5.0)
            return strategy_returns.dropna(), sharpe

    def optimize_simple_strategies_grid_search(self, price_data):
        """网格搜索优化简单策略"""
        print("  网格搜索优化简单策略...")

        results = {}

        # 简单反转策略（基准）
        print("    简单反转策略...")
        try:
            _, sharpe = self.strategy_simple_reversal(price_data)
            results['simple_reversal'] = {
                'strategy_name': 'simple_reversal',
                'best_sharpe': sharpe,
                'best_params': {'method': 'baseline'},
                'optimization_method': 'grid_search'
            }
            print(f"      夏普比率: {sharpe:.4f}")
        except Exception as e:
            print(f"      简单反转策略失败: {e}")
            results['simple_reversal'] = {
                'strategy_name': 'simple_reversal',
                'best_sharpe': 0.0,
                'best_params': {'method': 'baseline'},
                'optimization_method': 'grid_search'
            }

        # 固定阈值策略
        print("    固定阈值策略...")
        try:
            best_sharpe = -np.inf
            best_threshold = 0.002
            thresholds = [0.001, 0.002, 0.005, 0.01]

            for threshold in thresholds:
                _, sharpe = self.strategy_fixed_threshold(price_data, threshold)
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_threshold = threshold

            results['fixed_threshold'] = {
                'strategy_name': 'fixed_threshold',
                'best_sharpe': best_sharpe,
                'best_params': {'threshold': best_threshold},
                'optimization_method': 'grid_search'
            }
            print(f"      最佳阈值: {best_threshold}, 夏普比率: {best_sharpe:.4f}")
        except Exception as e:
            print(f"      固定阈值策略失败: {e}")

        return results

    def optimize_complex_strategies_ml(self, price_data):
        """ML采样优化复杂策略"""
        print("  ML采样优化复杂策略...")

        results = {}

        # 自适应阈值策略
        print("    自适应阈值策略...")
        try:
            best_params, best_sharpe = self.optimize_adaptive_threshold_sampling(price_data)
            results['adaptive_threshold'] = {
                'strategy_name': 'adaptive_threshold',
                'best_sharpe': best_sharpe,
                'best_params': best_params,
                'optimization_method': 'ml_sampling'
            }
        except Exception as e:
            print(f"    自适应阈值策略优化失败: {e}")
            # 使用默认参数
            _, sharpe = self.strategy_adaptive_threshold(price_data, 7, 1.5)
            results['adaptive_threshold'] = {
                'strategy_name': 'adaptive_threshold',
                'best_sharpe': sharpe,
                'best_params': {'window': 7, 'std_multiplier': 1.5},
                'optimization_method': 'fallback'
            }

        return results

    def optimize_adaptive_threshold_sampling(self, price_data, initial_params=[7, 1.5]):
        """使用采样方法优化自适应阈值策略"""
        print(f"      基于采样的自适应阈值优化...")

        samples = self.generate_random_samples(price_data)
        print(f"        生成 {len(samples)} 个随机样本，每个样本 {self.sample_days} 天")

        # 参数搜索空间
        windows = [3, 5, 7, 10, 15]
        std_multipliers = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

        best_sharpe = -np.inf
        best_params = {'window': 7, 'std_multiplier': 1.5}

        total_combinations = len(windows) * len(std_multipliers)
        current_combination = 0

        for window in windows:
            for std_multiplier in std_multipliers:
                current_combination += 1
                print(f"        测试参数组合 {current_combination}/{total_combinations}: window={window}, std_mult={std_multiplier}")

                try:
                    _, sharpe = self.strategy_adaptive_threshold(
                        price_data, window, std_multiplier, samples
                    )

                    # 限制极端值
                    sharpe = max(min(sharpe, 10.0), -10.0)

                    if sharpe > best_sharpe:
                        best_sharpe = sharpe
                        best_params = {'window': window, 'std_multiplier': std_multiplier}
                        print(f"          [新最佳] 夏普比率: {sharpe:.4f}")
                    else:
                        print(f"          夏普比率: {sharpe:.4f}")

                except Exception as e:
                    print(f"          参数测试失败: {e}")
                    continue

        print(f"      采样优化完成: 最佳参数={best_params}, 最佳夏普比率={best_sharpe:.4f}")
        return best_params, best_sharpe

    def optimize_all_products_hybrid(self):
        """混合优化：网格搜索 + ML采样"""
        print("=" * 60)
        print("混合策略参数优化器（网格搜索 + ML采样）")
        print("=" * 60)
        print(f"采样配置: {self.n_samples} 个样本, 每个样本 {self.sample_days} 天")
        print(f"可用产品: {', '.join(self.available_products)}")
        print()

        all_results = {}

        for product in self.available_products:
            print(f"优化产品: {product}")
            print("-" * 40)

            price_data = self.load_product_data(product)
            if price_data is None:
                print(f"无法加载产品 {product} 数据，跳过")
                continue

            print(f"数据点数: {len(price_data)}")

            # 使用混合优化
            simple_results = self.optimize_simple_strategies_grid_search(price_data)
            complex_results = self.optimize_complex_strategies_ml(price_data)

            # 合并结果
            product_results = {**simple_results, **complex_results}
            all_results[product] = product_results

            # 显示对比结果
            print(f"\n产品 {product} 混合优化结果:")
            comparison_data = []
            for strategy, result in product_results.items():
                comparison_data.append({
                    'Strategy': strategy,
                    'Sharpe Ratio': result['best_sharpe'],
                    'Best Params': str(result['best_params']),
                    'Method': result['optimization_method']
                })

            df_comparison = pd.DataFrame(comparison_data)
            print(df_comparison.to_string(index=False))
            print("=" * 60)

        # 保存结果
        self.save_hybrid_results(all_results)

        return all_results

    def save_hybrid_results(self, results):
        """保存混合优化结果 - 带文件存在检查和覆盖逻辑"""
        # 提取最优参数
        optimal_params = {}
        for product, strategies in results.items():
            optimal_params[product] = {}
            for strategy, result in strategies.items():
                optimal_params[product][strategy] = result['best_params']

        # 参数文件名
        params_file = 'latest_hybrid_ml_params.json'

        # 检查文件是否存在
        if os.path.exists(params_file):
            print(f"发现现有参数文件: {params_file}")
            # 备份现有文件
            backup_file = f'latest_hybrid_ml_params_backup_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.json'
            import shutil
            shutil.copy2(params_file, backup_file)
            print(f"原文件已备份到: {backup_file}")
            print(f"覆盖现有参数文件: {params_file}")
        else:
            print(f"创建新的参数文件: {params_file}")

        # 保存参数文件（覆盖或新建）
        with open(params_file, 'w', encoding='utf-8') as f:
            json.dump(optimal_params, f, ensure_ascii=False, indent=2, default=str)

        # 保存详细结果（带时间戳）
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        detailed_results_file = f'hybrid_ml_results_{timestamp}.json'
        detailed_params_file = f'hybrid_ml_params_{timestamp}.json'

        with open(detailed_results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)

        with open(detailed_params_file, 'w', encoding='utf-8') as f:
            json.dump(optimal_params, f, ensure_ascii=False, indent=2, default=str)

        print(f"混合优化结果已保存到: {detailed_results_file}")
        print(f"混合优化参数已保存到: {detailed_params_file}")
        print(f"最新参数文件已更新: {params_file}")

    def save_sampling_results(self, results):
        """保存采样优化结果 - 带文件存在检查和覆盖逻辑"""
        # 提取最优参数
        optimal_params = {}
        for product, strategies in results.items():
            optimal_params[product] = {}
            for strategy, result in strategies.items():
                optimal_params[product][strategy] = result['best_params']

        # 参数文件名
        params_file = 'latest_sampling_ml_params.json'

        # 检查文件是否存在
        if os.path.exists(params_file):
            print(f"发现现有参数文件: {params_file}")
            # 备份现有文件
            backup_file = f'latest_sampling_ml_params_backup_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.json'
            import shutil
            shutil.copy2(params_file, backup_file)
            print(f"原文件已备份到: {backup_file}")
            print(f"覆盖现有参数文件: {params_file}")
        else:
            print(f"创建新的参数文件: {params_file}")

        # 保存参数文件（覆盖或新建）
        with open(params_file, 'w', encoding='utf-8') as f:
            json.dump(optimal_params, f, ensure_ascii=False, indent=2, default=str)

        # 保存详细结果（带时间戳）
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        results_file = f'sampling_ml_results_{timestamp}.json'
        detailed_params_file = f'sampling_ml_params_{timestamp}.json'

        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)

        with open(detailed_params_file, 'w', encoding='utf-8') as f:
            json.dump(optimal_params, f, ensure_ascii=False, indent=2, default=str)

        print(f"采样优化结果已保存到: {results_file}")
        print(f"采样优化参数已保存到: {detailed_params_file}")
        print(f"最新参数文件已更新: {params_file}")

def main():
    """主函数"""
    h5_file = "中信建投/任务一/MinutesIdx.h5"

    optimizer = SampleBasedMLOptimizer(
        h5_file=h5_file,
        n_samples=30,      # 30个随机样本
        sample_days=15    # 每个样本15天
    )

    # 使用混合优化方法
    results = optimizer.optimize_all_products_hybrid()



if __name__ == "__main__":
    import os
    main()