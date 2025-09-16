#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
产品可视化分析器
读取最优参数配置，展示选定产品的完整图表信息
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import warnings
from performance_metrics import PerformanceMetrics
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class DataAnalyzer:
    """数据分析器"""

    def __init__(self, h5_file):
        """
        初始化数据分析器

        Args:
            h5_file: HDF5文件路径
        """
        self.h5_file = h5_file
        self.available_products = self._get_available_products()
        self.raw_data = None
        self.current_product = None
        self.dates = None
        self.daily_close_prices = None
        self.volume_data = None
        self.price_data = None

    def _get_available_products(self):
        """获取可用的产品列表"""
        try:
            with pd.HDFStore(self.h5_file, mode='r') as store:
                keys = store.keys()
                if keys:
                    key = keys[0]
                    data = store[key]
                    if hasattr(data, 'columns'):
                        return list(data.columns)
        except Exception as e:
            print(f"读取HDF5文件时出错: {e}")

        return ['IF', 'IH', 'IC', 'IM']

    def load_product_data(self, product_name):
        """
        加载产品数据

        Args:
            product_name: 产品名称
        """
        print(f"正在加载产品: {product_name}")

        try:
            hdf5_key = 'd'
            with pd.HDFStore(self.h5_file, mode='r') as store:
                self.raw_data = store[hdf5_key][product_name]

            self.current_product = product_name
            self.dates = self.raw_data.index

            # 提取每日收盘价和成交量数据
            daily_close_prices = []
            daily_volumes = []
            daily_prices = []

            for date, daily_data in self.raw_data.items():
                if isinstance(daily_data, pd.DataFrame) and 'Close' in daily_data.columns:
                    daily_close_prices.append(daily_data['Close'].iloc[-1])
                    if 'Volume' in daily_data.columns:
                        daily_volumes.append(daily_data['Volume'].sum())
                    else:
                        daily_volumes.append(1.0)
                    daily_prices.append(daily_data['Close'].mean())
                elif isinstance(daily_data, pd.Series):
                    daily_close_prices.append(daily_data.iloc[-1])
                    daily_volumes.append(1.0)
                    daily_prices.append(daily_data.mean())
                else:
                    daily_close_prices.append(float(daily_data))
                    daily_volumes.append(1.0)
                    daily_prices.append(float(daily_data))

            self.daily_close_prices = pd.Series(daily_close_prices, index=self.dates)
            self.volume_data = pd.Series(daily_volumes, index=self.dates)
            self.price_data = pd.Series(daily_prices, index=self.dates)

            print(f"成功加载产品 {product_name}，共 {len(self.dates)} 个交易日")
            print(f"每日数据点数: {len(self.raw_data.iloc[0]) if len(self.raw_data) > 0 else 'N/A'}")

        except Exception as e:
            print(f"加载产品 {product_name} 数据时出错: {e}")

    def calculate_product_characteristics(self):
        """
        计算产品特征
        """
        if self.daily_close_prices is None:
            return None

        characteristics = {
            'price': self.daily_close_prices,
            'volume': self.volume_data,
            'returns': self.daily_close_prices.pct_change().dropna(),
            'volatility': self.daily_close_prices.pct_change().rolling(window=20).std(),
            'stats': {
                'mean_return': self.daily_close_prices.pct_change().mean(),
                'volatility_annual': self.daily_close_prices.pct_change().std() * np.sqrt(252),
                'sharpe_ratio': self.daily_close_prices.pct_change().mean() / self.daily_close_prices.pct_change().std() * np.sqrt(252),
                'max_drawdown': self._calculate_max_drawdown(self.daily_close_prices)
            }
        }

        return characteristics

    def _calculate_max_drawdown(self, prices):
        """
        计算最大回撤

        Args:
            prices: 价格序列

        Returns:
            float: 最大回撤
        """
        cumulative = (1 + prices.pct_change()).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()

    def show_product_characteristics(self):
        """展示产品特征走势"""
        if self.daily_close_prices is None:
            print("请先加载产品数据")
            return

        characteristics = self.calculate_product_characteristics()
        if characteristics is None:
            return

        print("展示产品特征走势...")

        # 创建子图
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()

        # 1. 价格走势
        ax1 = axes[0]
        characteristics['price'].plot(ax=ax1, title='价格走势', linewidth=1)
        ax1.set_ylabel('价格')
        ax1.grid(True, alpha=0.3)

        # 2. 成交量
        ax2 = axes[1]
        characteristics['volume'].plot(ax=ax2, title='成交量', color='orange', alpha=0.7)
        ax2.set_ylabel('成交量')
        ax2.grid(True, alpha=0.3)

        # 3. 收益率分布
        ax3 = axes[2]
        characteristics['returns'].hist(ax=ax3, bins=50, alpha=0.7, color='green', edgecolor='black')
        ax3.set_title('收益率分布')
        ax3.set_xlabel('收益率')
        ax3.set_ylabel('频数')
        ax3.grid(True, alpha=0.3)

        # 4. 波动率走势
        ax4 = axes[3]
        characteristics['volatility'].plot(ax=ax4, title='波动率走势', color='red', linewidth=1)
        ax4.set_ylabel('波动率')
        ax4.grid(True, alpha=0.3)

        # 添加统计信息文本
        stats_text = f"""
        产品统计特征：

        平均日收益率: {characteristics['stats']['mean_return']:.6f}
        年化波动率: {characteristics['stats']['volatility_annual']:.4f}
        夏普比率: {characteristics['stats']['sharpe_ratio']:.4f}
        最大回撤: {characteristics['stats']['max_drawdown']:.4f}

        数据点数: {len(characteristics['price'])}
        数据时间范围: {str(characteristics['price'].index.min())} 至 {str(characteristics['price'].index.max())}
        """

        fig.text(0.02, 0.02, stats_text, fontsize=10, verticalalignment='bottom',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))

        plt.suptitle(f'产品 {self.current_product} 特征分析', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)
        plt.show()

class StrategyAnalyzer:
    """策略分析器"""

    def __init__(self, h5_file, optimal_params_file='latest_optimal_params.json'):
        """
        初始化策略分析器

        Args:
            h5_file: HDF5文件路径
            optimal_params_file: 最优参数文件路径
        """
        self.data_analyzer = DataAnalyzer(h5_file)
        self.optimal_params = self._load_optimal_params(optimal_params_file)
        self.fibonacci_times = [0, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 239]

    def _load_optimal_params(self, params_file):
        """
        加载最优参数

        Args:
            params_file: 参数文件路径

        Returns:
            dict: 最优参数
        """
        try:
            with open(params_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"警告: 找不到参数文件 {params_file}，将使用默认参数")
            return {}
        except Exception as e:
            print(f"加载参数文件时出错: {e}")
            return {}

    def calculate_sharpe_ratio(self, returns, frequency='daily'):
        """
        计算夏普比率 - 使用新的性能指标库

        Args:
            returns: 收益率序列
            frequency: 数据频率

        Returns:
            float: 夏普比率
        """
        return PerformanceMetrics.calculate_sharpe_ratio(returns, frequency=frequency)

    def _calculate_execution_prices(self, t_exec):
        """
        计算在不同执行时间下的价格

        Args:
            t_exec: 执行时间（分钟）

        Returns:
            pd.Series: 执行价格序列
        """
        execution_prices = []

        for date in self.data_analyzer.dates:
            daily_data = self.data_analyzer.raw_data.loc[date]
            if isinstance(daily_data, pd.DataFrame) and 'Close' in daily_data.columns:
                if len(daily_data) > t_exec:
                    execution_price = daily_data['Close'].iloc[t_exec]
                else:
                    execution_price = daily_data['Close'].iloc[-1]
            elif isinstance(daily_data, pd.Series):
                if len(daily_data) > t_exec:
                    execution_price = daily_data.iloc[t_exec]
                else:
                    execution_price = daily_data.iloc[-1]
            else:
                execution_price = float(daily_data)

            execution_prices.append(execution_price)

        return pd.Series(execution_prices, index=self.data_analyzer.dates)

    def strategy_simple_reversal_time_analysis(self):
        """
        简单反转策略的时间敏感度分析
        """
        if self.data_analyzer.raw_data is None:
            raise ValueError("请先加载数据")

        print("执行简单反转策略时间敏感度分析...")

        sharpe_results = []
        returns_results = []

        for t_exec in self.fibonacci_times:
            try:
                # 计算执行价格
                execution_prices = self._calculate_execution_prices(t_exec)

                # 计算VWAP
                vwap_series = self.data_analyzer.price_data.rolling(window=5).mean()

                # 生成交易信号
                returns = execution_prices.pct_change().dropna()
                signals = -np.sign(returns.shift(1))
                strategy_returns = signals * returns

                sharpe = self.calculate_sharpe_ratio(strategy_returns)
                sharpe_results.append(sharpe)
                returns_results.append(strategy_returns)

            except Exception as e:
                print(f"执行时间 {t_exec} 分钟计算失败: {e}")
                sharpe_results.append(0)
                returns_results.append(pd.Series())

        return {
            'strategy_name': 'Simple Reversal',
            'execution_times': self.fibonacci_times,
            'sharpe_ratios': sharpe_results,
            'returns_list': returns_results
        }

    def analyze_time_sensitivity(self):
        """
        分析所有策略的时间敏感度
        """
        print("=" * 60)
        print("时间敏感度分析")
        print("=" * 60)

        if self.data_analyzer.current_product not in self.optimal_params:
            print(f"警告: 找不到产品 {self.data_analyzer.current_product} 的最优参数")
            return None

        product_params = self.optimal_params[self.data_analyzer.current_product]
        all_results = []

        # 分析每个策略的时间敏感度
        strategies = ['simple_reversal']
        strategy_names = ['Simple Reversal']

        # 只分析有参数的策略
        for strategy in ['fixed_threshold', 'adaptive_threshold', 'adaptive_trend']:
            if strategy in product_params and product_params[strategy]:
                strategies.append(strategy)
                strategy_names.append(strategy.replace('_', ' ').title())

        for i, strategy in enumerate(strategies):
            print(f"执行 {strategy_names[i]} 时间敏感度分析...")
            result = self.strategy_simple_reversal_time_analysis()
            all_results.append(result)

        # 绘制时间敏感度图表
        self._plot_time_sensitivity(all_results)

        # 生成报告
        self._generate_time_report(all_results)

        return all_results

    def _plot_time_sensitivity(self, results):
        """
        绘制时间敏感度图表

        Args:
            results: 分析结果
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # 1. 夏普比率随执行时间变化
        ax1 = axes[0, 0]
        for result in results:
            ax1.plot(result['execution_times'], result['sharpe_ratios'],
                    marker='o', label=result['strategy_name'], linewidth=2)

        ax1.set_xlabel('执行时间（分钟）')
        ax1.set_ylabel('夏普比率')
        ax1.set_title('不同执行时间的夏普比率')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. 最优执行时间推荐
        ax2 = axes[0, 1]
        best_times = []
        best_sharpes = []

        for result in results:
            sharpe_array = np.array(result['sharpe_ratios'])
            best_idx = np.argmax(sharpe_array)
            best_times.append(result['execution_times'][best_idx])
            best_sharpes.append(sharpe_array[best_idx])

        strategy_names = [result['strategy_name'] for result in results]
        colors = plt.cm.Set3(np.linspace(0, 1, len(strategy_names)))

        bars = ax2.bar(strategy_names, best_sharpes, color=colors)
        ax2.set_ylabel('最佳夏普比率')
        ax2.set_title('各策略最佳表现')

        # 在柱状图上标注执行时间
        for i, (bar, time_val) in enumerate(zip(bars, best_times)):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{time_val}min', ha='center', va='bottom')

        plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)

        # 3. 累积收益率对比
        ax3 = axes[1, 0]
        for result in results:
            if result['returns_list']:
                # 使用最佳执行时间的收益率序列
                sharpe_array = np.array(result['sharpe_ratios'])
                best_idx = np.argmax(sharpe_array)
                best_returns = result['returns_list'][best_idx]
                if len(best_returns) > 0:
                    cumulative_returns = (1 + best_returns).cumprod()
                    ax3.plot(cumulative_returns.index, cumulative_returns,
                            label=result['strategy_name'], linewidth=2)

        ax3.set_xlabel('日期')
        ax3.set_ylabel('累积收益率')
        ax3.set_title('最佳执行时间的累积收益率')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. 执行时间稳定性分析
        ax4 = axes[1, 1]
        stability_scores = []
        for result in results:
            sharpe_array = np.array(result['sharpe_ratios'])
            # 计算夏普比率的标准差作为稳定性指标
            stability = np.std(sharpe_array[sharpe_array != 0]) if np.any(sharpe_array != 0) else 0
            stability_scores.append(stability)

        bars = ax4.bar(strategy_names, stability_scores, color=colors)
        ax4.set_ylabel('稳定性得分（标准差）')
        ax4.set_title('执行时间稳定性分析')

        # 在柱状图上标注稳定性得分
        for i, (bar, score) in enumerate(zip(bars, stability_scores)):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{score:.3f}', ha='center', va='bottom')

        plt.setp(ax4.get_xticklabels(), rotation=45, ha='right')
        ax4.grid(True, alpha=0.3)

        plt.suptitle(f'产品 {self.data_analyzer.current_product} 时间敏感度分析', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()

    def _generate_time_report(self, results):
        """
        生成时间敏感度分析报告

        Args:
            results: 分析结果
        """
        report_content = f"""
        时间敏感度分析报告
        产品: {self.data_analyzer.current_product}
        分析时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
        数据时间范围: {str(self.data_analyzer.dates.min())} 至 {str(self.data_analyzer.dates.max())}

        === 分析结果摘要 ===

        """

        for result in results:
            sharpe_array = np.array(result['sharpe_ratios'])
            best_idx = np.argmax(sharpe_array)
            best_time = result['execution_times'][best_idx]
            best_sharpe = sharpe_array[best_idx]
            worst_sharpe = np.min(sharpe_array)

            # 计算时间敏感度（最佳与最差夏普比率的差异）
            time_sensitivity = (best_sharpe - worst_sharpe) / abs(best_sharpe) if best_sharpe != 0 else 0

            report_content += f"""
        {result['strategy_name']}:
          最佳夏普比率: {best_sharpe:.4f} (执行时间: {best_time}分钟)
          最差夏普比率: {worst_sharpe:.4f}
          时间敏感度: {time_sensitivity:.1%}

        """

        # 稳定性分析
        report_content += f"""
        === 时间稳定性分析 ===

        时间稳定性衡量策略在不同执行时间下表现的一致性。
        稳定性越高，策略对执行时间的选择越不敏感。

        """

        for result in results:
            sharpe_array = np.array(result['sharpe_ratios'])
            valid_sharpes = sharpe_array[sharpe_array != 0]
            if len(valid_sharpes) > 0:
                stability = 1 - (np.std(valid_sharpes) / abs(np.mean(valid_sharpes)))
                stability = max(0, stability)  # 确保非负
            else:
                stability = 0

            report_content += f"        {result['strategy_name']} 稳定性: {stability:.2%}\n"

        print(report_content)

        # 保存报告
        report_file = f'time_sensitivity_report_{self.data_analyzer.current_product}_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.txt'
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        print(f"报告已保存到: {report_file}")

def main():
    """主函数"""
    h5_file = "中信建投/任务一/MinutesIdx.h5"

    if not os.path.exists(h5_file):
        print(f"错误: 找不到HDF5文件 {h5_file}")
        return

    # 初始化分析器
    analyzer = StrategyAnalyzer(h5_file)

    # 显示可用产品
    print("可用产品:")
    for i, product in enumerate(analyzer.data_analyzer.available_products, 1):
        print(f"{i}. {product}")

    # 选择产品
    try:
        choice = int(input("\n请选择产品编号: ")) - 1
        if 0 <= choice < len(analyzer.data_analyzer.available_products):
            selected_product = analyzer.data_analyzer.available_products[choice]
        else:
            print("无效选择，使用默认产品")
            selected_product = analyzer.data_analyzer.available_products[0]
    except (ValueError, KeyboardInterrupt):
        print("使用默认产品")
        selected_product = analyzer.data_analyzer.available_products[0]

    # 加载产品数据
    analyzer.data_analyzer.load_product_data(selected_product)

    # 展示产品特征
    analyzer.data_analyzer.show_product_characteristics()

    # 分析时间敏感度
    analyzer.analyze_time_sensitivity()

    print("\n分析完成！")

if __name__ == "__main__":
    # 导入必要的模块
    import os
    main()