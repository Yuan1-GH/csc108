import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib import rcParams

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']  # 优先使用的中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


class stock_info:
    def __init__(self, filepath, target_col):
        # 从HDF文件加载数据
        self.raw = pd.read_hdf(filepath)
        
        # 提取目标列，即"IC"
        self.target_raw = self.raw[target_col]
        
        # 获取日期列表
        self.dates = self.target_raw.index.tolist()
        
        # 初始化一个新的数据框记录每日收益率
        self.readouts = pd.DataFrame(self.dates, columns=["dates"])
        self.readouts.set_index("dates", inplace=True)
        
        # 计算T_last_close和T_vap（成交量加权平均价格）
        self.readouts["T_last_close"] = [
            self.target_raw.loc[date]["Close"].iloc[-1] for date in self.dates
        ]
        self.readouts["T_vap"] = [
            (self.target_raw.loc[date]["Close"] * self.target_raw.loc[date]["Volume"]).sum() /
            self.target_raw.loc[date]["Volume"].sum()
            for date in self.dates
        ]
        
        # 计算固定阈值策略，参数为a
        # 如果最后收盘价与VWAP的比值超过阈值(a)，
        # 当比值 > 1+a时卖出(-1)，当比值 < 1-a时买入(+1)，否则持有(0)
        ratio = np.array(self.readouts["T_last_close"]) / np.array(self.readouts["T_vap"])
        threshold = 0.001  # 默认阈值
        self.readouts["fixed_thres"] = np.where(ratio > 1 + threshold, -1, 
                                               np.where(ratio < 1 - threshold, 1, 0))
    
    def calculate_strategy_with_threshold(self, threshold_a):
        """
        根据给定的阈值参数a计算策略信号
        
        参数:
        threshold_a: 阈值参数
        """
        ratio = np.array(self.readouts["T_last_close"]) / np.array(self.readouts["T_vap"])
        strategy_signals = np.where(ratio > 1 + threshold_a, -1, 
                                   np.where(ratio < 1 - threshold_a, 1, 0))
        return strategy_signals
    
    def evaluate_threshold_performance(self, threshold_a, t_exec=59):
        """
        评估特定阈值参数的收益率表现
        
        参数:
        threshold_a: 阈值参数
        t_exec: 执行时间点
        
        返回:
        total_return: 总收益率
        """
        # 计算策略信号
        strategy_signals = self.calculate_strategy_with_threshold(threshold_a)
        
        # 计算收益率
        strategy_ = strategy_signals[:-2]
        short_price_T_p1 = np.array([self.target_raw.loc[day]["Close"].iloc[t_exec] for day in self.dates[1:-1]])
        long_price_T_p2 = np.array([self.target_raw.loc[day]["Close"].iloc[t_exec] for day in self.dates[2:]])
        
        return_rates = ((long_price_T_p2/short_price_T_p1)-1.) * strategy_
        total_return = np.sum(return_rates)
        
        return total_return
    
    def optimize_threshold(self, threshold_range=(0.0001, 0.01), num_points=100, t_exec=59):
        """
        优化阈值参数，寻找收益率之和最高的参数
        
        参数:
        threshold_range: 阈值搜索范围 (最小值, 最大值)
        num_points: 搜索点数
        t_exec: 执行时间点
        
        返回:
        best_threshold: 最优阈值
        best_return: 最优收益率
        results: 所有测试结果
        """
        thresholds = np.linspace(threshold_range[0], threshold_range[1], num_points)
        results = []
        
        print(f"开始优化阈值参数，搜索范围: {threshold_range}, 搜索点数: {num_points}")
        
        for i, threshold in enumerate(thresholds):
            total_return = self.evaluate_threshold_performance(threshold, t_exec)
            results.append((threshold, total_return))
            
            if (i + 1) % 20 == 0:
                print(f"已完成 {i + 1}/{num_points} 个参数测试")
        
        # 找到最优参数
        best_threshold, best_return = max(results, key=lambda x: x[1])
        
        print(f"\n优化完成！")
        print(f"最优阈值: {best_threshold:.6f}")
        print(f"最优总收益率: {best_return:.6f}")
        
        return best_threshold, best_return, results
    
    def plot_threshold_optimization(self, results):
        """
        绘制阈值优化结果图表
        
        参数:
        results: 优化结果列表 [(threshold, return), ...]
        """
        thresholds = [r[0] for r in results]
        returns = [r[1] for r in results]
        
        plt.figure(figsize=(12, 6))
        plt.plot(thresholds, returns, 'b-', linewidth=2, label='总收益率')
        
        # 标记最优点
        best_idx = np.argmax(returns)
        plt.scatter(thresholds[best_idx], returns[best_idx], 
                   color='red', s=100, zorder=5, label=f'最优点 (a={thresholds[best_idx]:.6f})')
        
        plt.xlabel('阈值参数 a')
        plt.ylabel('总收益率')
        plt.title('固定阈值策略参数优化结果')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()
        
    def update_strategy_with_optimal_threshold(self, optimal_threshold):
        """
        使用最优阈值更新策略
        
        参数:
        optimal_threshold: 最优阈值参数
        """
        ratio = np.array(self.readouts["T_last_close"]) / np.array(self.readouts["T_vap"])
        self.readouts["fixed_thres"] = np.where(ratio > 1 + optimal_threshold, -1, 
                                               np.where(ratio < 1 - optimal_threshold, 1, 0))
        print(f"策略已更新为最优阈值: {optimal_threshold:.6f}")

    def evaluate_return_T_p1(self, t_exec, strategy_array_name="fixed_thres"):
        # 读取T+1的策略
        strategy_ = np.array(self.readouts[strategy_array_name].tolist()[:-2])
        
        # 分别读取T+1和T+2指定时间的收盘价
        short_price_T_p1 = np.array([self.target_raw.loc[day]["Close"].iloc[t_exec] for day in self.dates[1:-1]])
        long_price_T_p2 = np.array([self.target_raw.loc[day]["Close"].iloc[t_exec] for day in self.dates[2:]])
        
        # 计算T+1的收益率
        return_rate_T_p1 = pd.Series(((long_price_T_p2/short_price_T_p1)-1.)*strategy_, index=self.dates[:-2])
        
        # 添加到self.readouts
        self.readouts["return_"+strategy_array_name+"_"+str(t_exec)] = return_rate_T_p1

    def get_sharpe_ratio(self, strategy_return):
        # 获取指定策略下的收益率
        return_arr = self.readouts[strategy_return].tolist()[:-2]
        
        # 计算传统夏普比率
        sharpe_ratio = np.sqrt(250.)*np.mean(return_arr)/np.std(return_arr)
        return sharpe_ratio

    def generate_time_dependency(self, strategy_array_name):
        fig, ax1 = plt.subplots(1, 1)
        
        # 交易时间数组
        fibonnacci = [0, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 239]
        return_arr = []
        total_return_arr = []
        
        for t_exec in fibonnacci:
            # 计算策略与不同交易时间组合的收益率
            self.evaluate_return_T_p1(t_exec, strategy_array_name=strategy_array_name)
            # 记录夏普比率
            sharpe_ratio = self.get_sharpe_ratio(strategy_return="return_"+strategy_array_name+"_"+str(t_exec))
            return_arr.append(sharpe_ratio)
            
            # 计算总收益率
            return_column = "return_" + strategy_array_name + "_" + str(t_exec)
            total_return = self.readouts[return_column].sum()
            total_return_arr.append(total_return)
            
        # 找到最大夏普比率及对应时间
        max_sharpe_idx = np.argmax(return_arr)
        max_sharpe_ratio = return_arr[max_sharpe_idx]
        max_sharpe_time = fibonnacci[max_sharpe_idx]
        
        # 找到最大总收益率及对应时间
        max_return_idx = np.argmax(total_return_arr)
        max_total_return = total_return_arr[max_return_idx]
        max_return_time = fibonnacci[max_return_idx]
        
        # 绘制敏感度图
        ax1.plot(fibonnacci, return_arr, label="夏普比率", c="blue")
        ax1.scatter(fibonnacci, return_arr, c="blue")
        
        # 突出显示最大值点
        ax1.scatter(max_sharpe_time, max_sharpe_ratio, c="red", s=100, 
                   label=f"最高夏普比率: {max_sharpe_ratio:.4f} (时间: {max_sharpe_time}分钟)")
        
        title = strategy_array_name + " 的时间敏感度"
        ax1.set_title(title)
        ax1.set_xlabel("交易时间 (分钟)")
        ax1.set_ylabel("夏普比率")
        plt.legend()
        plt.show()
        
        return max_sharpe_ratio, max_sharpe_time, max_total_return, max_return_time
    
    def calculate_max_cumulative_return(self, strategy_array_name="fixed_thres"):
        """
        计算策略能达到的最高累计收益率（与时间敏感度无关）
        """
        # 使用默认执行时间59分钟计算收益率
        self.evaluate_return_T_p1(59, strategy_array_name=strategy_array_name)
        return_column = "return_" + strategy_array_name + "_59"
        
        # 计算累计收益率
        returns = self.readouts[return_column].dropna()
        cumulative_returns = (1 + returns).cumprod() - 1
        max_cumulative_return = cumulative_returns.max()
        
        return max_cumulative_return
    
    def calculate_return_std(self, strategy_array_name="fixed_thres"):
        """
        计算收益率变化的标准差
        """
        # 使用默认执行时间59分钟计算收益率
        self.evaluate_return_T_p1(59, strategy_array_name=strategy_array_name)
        return_column = "return_" + strategy_array_name + "_59"
        
        returns = self.readouts[return_column].dropna()
        return_std = returns.std()
        
        return return_std
        
    def plot_return_rate(self, strategy_array_name="fixed_thres", t_exec=59):
        """
        显示指定策略在特定执行时间点的T+1收益率图表
        
        参数:
        strategy_array_name: 策略名称
        t_exec: 执行时间点
        """
        # 确保已经计算了该策略在该时间点的收益率
        return_column = "return_" + strategy_array_name + "_" + str(t_exec)
        if return_column not in self.readouts.columns:
            self.evaluate_return_T_p1(t_exec, strategy_array_name=strategy_array_name)
            
        # 获取收益率数据
        returns = self.readouts[return_column].dropna()
        
        # 将日期索引转换为datetime格式
        returns.index = pd.to_datetime(returns.index)
        
        # 计算累积收益率
        cumulative_returns = (1 + returns).cumprod() - 1
        
        # 创建子图
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # 定义收益率区间
        bins = [-float('inf'), -0.03, -0.025, -0.02, -0.015, -0.01, -0.005, 0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, float('inf')]
        bin_labels = ['<-0.03', '-0.03~-0.025', '-0.025~-0.02', '-0.02~-0.015', '-0.015~-0.01', '-0.01~-0.005', '-0.005~0', '0~0.005', '0.005~0.01', '0.01~0.015', '0.015~0.02', '0.02~0.025', '0.025~0.03', '>0.03']
        
        # 统计各区间的天数
        return_counts = pd.cut(returns, bins=bins, labels=bin_labels).value_counts().sort_index()
        
        # 绘制收益率分布柱状图
        bars = ax1.bar(return_counts.index, return_counts.values, color='lightgreen')
        ax1.set_title(f"{strategy_array_name} 策略在时间点 {t_exec} 的T+1收益率分布")
        ax1.set_xlabel("收益率区间")
        ax1.set_ylabel("天数")
        
        # 在柱状图上显示具体数值
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{int(height)}', ha='center', va='bottom')
        
        # 绘制累积收益率曲线
        ax2.plot(cumulative_returns.index, cumulative_returns.values, 
                 label=f"{strategy_array_name} 累积收益率", color='green')
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax2.set_title(f"{strategy_array_name} 策略在时间点 {t_exec} 的累积收益率")
        ax2.set_xlabel("日期")
        ax2.set_ylabel("累积收益率")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 设置x轴只显示每月一号
        from matplotlib.dates import MonthLocator, DateFormatter
        ax2.xaxis.set_major_locator(MonthLocator(interval=1))  # 每月一号
        ax2.xaxis.set_major_formatter(DateFormatter('%Y-%m'))  # 显示年-月格式
        
        # 调整布局
        plt.tight_layout()
        plt.show()


# 主程序
if __name__ == "__main__":
    # 创建股票信息实例
    stock = stock_info("src/MinutesIdx.h5", "IC")

    # 阈值参数优化
    best_threshold, best_return, optimization_results = stock.optimize_threshold(
        threshold_range=(0.0001, 0.02),  # 搜索范围
        num_points=200,                   # 搜索精度
        t_exec=59                        # 执行时间点
    )

    # 更新策略为最优参数
    stock.update_strategy_with_optimal_threshold(best_threshold)

    # 绘制参数优化结果图表
    stock.plot_threshold_optimization(optimization_results)

    # 生成时间敏感度图表并获取最高夏普比率
    max_sharpe_ratio, max_sharpe_time, _, _ = stock.generate_time_dependency("fixed_thres")

    # 计算最高累计收益率（与时间敏感度无关）
    max_cumulative_return = stock.calculate_max_cumulative_return("fixed_thres")

    # 计算收益率变化的标准差
    return_std = stock.calculate_return_std("fixed_thres")

    # 输出优化后的结果
    print(f"\n=== Fixed Threshold Strategy 分析结果 ===")
    print(f"时间敏感度图中的最高夏普比率: {max_sharpe_ratio:.4f}")
    print(f"最高累计收益率: {max_cumulative_return:.4f}")
    print(f"收益率变化的标准差: {return_std:.4f}")

    # 绘制优化后策略的累积收益率图表
    stock.plot_return_rate("fixed_thres", 59)