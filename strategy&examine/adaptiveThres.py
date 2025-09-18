import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

class stock_info:
    def __init__(self, filepath, target_col, window_size=7):
        """
        初始化股票信息类，实现自适应阈值策略
        
        参数:
        filepath: HDF文件路径
        target_col: 目标列名
        window_size: 滚动窗口大小，默认为7天
        """
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
        
        # 存储窗口大小
        self.window_size = window_size
        
        # 使用滚动窗口计算自适应阈值策略
        self.calculate_adaptive_threshold_strategy(window_size)
    
    def calculate_adaptive_threshold_strategy(self, window_size):
        """
        计算自适应阈值策略
        
        参数:
        window_size: 滚动窗口大小
        """
        # 计算每日收盘价相对于VWAP的偏离度
        deviation = (self.readouts["T_last_close"] - self.readouts["T_vap"]) / self.readouts["T_vap"]
        
        # 计算滚动窗口内偏离度的标准差作为自适应阈值
        adaptive_threshold = deviation.rolling(window=window_size, min_periods=1).std()
        
        # 应用自适应阈值策略
        # 如果偏离度 > 阈值，卖出(-1)；如果偏离度 < -阈值，买入(+1)；否则持有(0)
        self.readouts["adaptive_thres"] = np.where(
            deviation > adaptive_threshold, -1,
            np.where(deviation < -adaptive_threshold, 1, 0)
        )
        
        # 存储阈值用于分析
        self.readouts["threshold_values"] = adaptive_threshold
    
    def calculate_strategy_with_window(self, window_size):
        """
        根据给定的窗口大小计算策略信号
        
        参数:
        window_size: 滚动窗口大小
        """
        # 计算每日收盘价相对于VWAP的偏离度
        deviation = (self.readouts["T_last_close"] - self.readouts["T_vap"]) / self.readouts["T_vap"]
        
        # 计算滚动窗口内偏离度的标准差作为自适应阈值
        adaptive_threshold = deviation.rolling(window=window_size, min_periods=1).std()
        
        # 应用自适应阈值策略
        strategy_signals = np.where(
            deviation > adaptive_threshold, -1,
            np.where(deviation < -adaptive_threshold, 1, 0)
        )
        
        return strategy_signals
    
    def evaluate_window_performance(self, window_size, t_exec=59):
        """
        评估特定窗口大小的收益率表现
        
        参数:
        window_size: 滚动窗口大小
        t_exec: 执行时间点
        
        返回:
        total_return: 总收益率
        """
        # 计算策略信号
        strategy_signals = self.calculate_strategy_with_window(window_size)
        
        # 计算收益率
        strategy_ = strategy_signals[:-2]
        short_price_T_p1 = np.array([self.target_raw.loc[day]["Close"].iloc[t_exec] for day in self.dates[1:-1]])
        long_price_T_p2 = np.array([self.target_raw.loc[day]["Close"].iloc[t_exec] for day in self.dates[2:]])
        
        return_rates = ((long_price_T_p2/short_price_T_p1)-1.) * strategy_
        total_return = np.sum(return_rates)
        
        return total_return
    
    def optimize_window_size(self, window_range=(3, 30), t_exec=59):
        """
        优化滚动窗口大小，寻找收益率之和最高的参数
        
        参数:
        window_range: 窗口大小搜索范围 (最小值, 最大值)
        t_exec: 执行时间点
        
        返回:
        best_window: 最优窗口大小
        best_return: 最优收益率
        results: 所有测试结果
        """
        window_sizes = range(window_range[0], window_range[1] + 1)
        results = []
        
        print(f"开始优化滚动窗口参数，搜索范围: {window_range}")
        
        for i, window_size in enumerate(window_sizes):
            total_return = self.evaluate_window_performance(window_size, t_exec)
            results.append((window_size, total_return))
            
            if (i + 1) % 5 == 0:
                print(f"已完成 {i + 1}/{len(window_sizes)} 个参数测试")
        
        # 找到最优参数
        best_window, best_return = max(results, key=lambda x: x[1])
        
        print(f"\n优化完成！")
        print(f"最优窗口大小: {best_window}")
        print(f"最优总收益率: {best_return:.6f}")
        
        return best_window, best_return, results
    
    def plot_window_optimization(self, results):
        """
        绘制窗口大小优化结果图表
        
        参数:
        results: 优化结果列表 [(window_size, return), ...]
        """
        window_sizes = [r[0] for r in results]
        returns = [r[1] for r in results]
        
        plt.figure(figsize=(12, 6))
        plt.plot(window_sizes, returns, 'b-', linewidth=2, label='总收益率')
        
        # 标记最优点
        best_idx = np.argmax(returns)
        plt.scatter(window_sizes[best_idx], returns[best_idx], 
                   color='red', s=100, zorder=5, label=f'最优点 (窗口={window_sizes[best_idx]})')
        
        plt.xlabel('滚动窗口大小')
        plt.ylabel('总收益率')
        plt.title('自适应阈值策略窗口大小优化结果')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    def update_strategy_with_optimal_window(self, optimal_window):
        """
        使用最优窗口大小更新策略
        
        参数:
        optimal_window: 最优窗口大小
        """
        self.window_size = optimal_window
        self.calculate_adaptive_threshold_strategy(optimal_window)
        print(f"策略已更新为最优窗口大小: {optimal_window}")
    
    def plot_threshold_evolution(self):
        """
        绘制自适应阈值的演化图表
        """
        plt.figure(figsize=(12, 8))
        
        # 计算偏离度
        deviation = (self.readouts["T_last_close"] - self.readouts["T_vap"]) / self.readouts["T_vap"]
        
        # 转换日期索引
        dates = pd.to_datetime(self.readouts.index)
        
        plt.subplot(2, 1, 1)
        plt.plot(dates, deviation, label='价格偏离度', alpha=0.7)
        plt.plot(dates, self.readouts["threshold_values"], label=f'自适应阈值 (窗口={self.window_size})', color='red')
        plt.plot(dates, -self.readouts["threshold_values"], label='负阈值', color='red', linestyle='--')
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.title('自适应阈值演化')
        plt.ylabel('偏离度/阈值')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 1, 2)
        plt.plot(dates, self.readouts["adaptive_thres"], label='策略信号', drawstyle='steps-post')
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.title('策略信号')
        plt.xlabel('日期')
        plt.ylabel('信号 (-1: 卖出, 0: 持有, 1: 买入)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 设置x轴格式
        from matplotlib.dates import MonthLocator, DateFormatter
        for ax in plt.gcf().axes:
            ax.xaxis.set_major_locator(MonthLocator(interval=3))
            ax.xaxis.set_major_formatter(DateFormatter('%Y-%m'))
        
        plt.tight_layout()
        plt.show()

    def evaluate_return_T_p1(self, t_exec, strategy_array_name="adaptive_thres"):
        """
        评估T+1收益率
        
        参数:
        t_exec: 执行时间点
        strategy_array_name: 策略名称
        
        返回:
        (mean_return, std_return): 平均收益率和标准差
        """
        # 读取T+1的策略
        strategy_ = np.array(self.readouts[strategy_array_name].tolist()[:-2])
        
        # 分别读取T+1和T+2指定时间的收盘价
        short_price_T_p1 = np.array([self.target_raw.loc[day]["Close"].iloc[t_exec] for day in self.dates[1:-1]])
        long_price_T_p2 = np.array([self.target_raw.loc[day]["Close"].iloc[t_exec] for day in self.dates[2:]])
        
        # 计算T+1的收益率
        return_rate_T_p1 = pd.Series(((long_price_T_p2/short_price_T_p1)-1.)*strategy_, index=self.dates[:-2])
        
        # 添加到self.readouts
        self.readouts["return_"+strategy_array_name+"_"+str(t_exec)] = return_rate_T_p1
        
        # 计算统计量
        mean_return = np.mean(return_rate_T_p1)
        std_return = np.std(return_rate_T_p1)
        
        return mean_return, std_return

    def calculate_sharpe_ratio(self, strategy_array_name="adaptive_thres", t_exec=59):
        """
        计算夏普比率
        
        参数:
        strategy_array_name: 策略名称
        t_exec: 执行时间点
        
        返回:
        sharpe_ratio: 夏普比率
        """
        return_column = "return_" + strategy_array_name + "_" + str(t_exec)
        
        # 如果还没有计算收益率，先计算
        if return_column not in self.readouts.columns:
            self.evaluate_return_T_p1(t_exec, strategy_array_name)
        
        # 获取指定策略下的收益率
        return_arr = self.readouts[return_column].dropna().tolist()
        
        # 计算传统夏普比率
        if len(return_arr) > 0 and np.std(return_arr) != 0:
            sharpe_ratio = np.sqrt(250.) * np.mean(return_arr) / np.std(return_arr)
        else:
            sharpe_ratio = 0
            
        return sharpe_ratio

    def plot_time_sensitivity(self, strategy_array_name="adaptive_thres"):
        """
        绘制时间敏感度分析图表
        
        参数:
        strategy_array_name: 策略名称
        """
        fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))
        
        # array of trading time
        fibonacci = [0, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 239]
        return_arr = []
        total_return_arr = []
        
        for t_exec in fibonacci:
            # calculate return rate of combinations of strategy with different trading times
            self.evaluate_return_T_p1(t_exec, strategy_array_name=strategy_array_name)
            # record sharpe ratio
            sharpe_ratio = self.calculate_sharpe_ratio(strategy_array_name, t_exec)
            return_arr.append(sharpe_ratio)
            
            # calculate total return
            return_column = "return_" + strategy_array_name + "_" + str(t_exec)
            total_return = self.readouts[return_column].sum()
            total_return_arr.append(total_return)
            
        # find maximum sharpe ratio and corresponding time
        max_sharpe_idx = np.argmax(return_arr)
        max_sharpe_ratio = return_arr[max_sharpe_idx]
        max_sharpe_time = fibonacci[max_sharpe_idx]
        
        # find maximum total return and corresponding time
        max_return_idx = np.argmax(total_return_arr)
        max_total_return = total_return_arr[max_return_idx]
        max_return_time = fibonacci[max_return_idx]
        
        # plot sensitivity    
        ax1.plot(fibonacci, return_arr, label="夏普比率", c="blue", marker='o')
        
        # highlight maximum points
        ax1.scatter(max_sharpe_time, max_sharpe_ratio, c="red", s=100, 
                   label=f"最高夏普比率: {max_sharpe_ratio:.4f} (时间: {max_sharpe_time}分钟)")
        
        title = f"{strategy_array_name} 策略的时间敏感度分析"
        ax1.set_title(title)
        ax1.set_xlabel("交易时间 (分钟)")
        ax1.set_ylabel("夏普比率")
        ax1.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()
        
        return max_sharpe_ratio, max_sharpe_time, max_total_return, max_return_time
    
    def calculate_max_cumulative_return(self, strategy_array_name="adaptive_thres"):
        """
        计算策略能达到的最高累计收益率（与时间敏感度无关）
        """
        # 使用默认执行时间59分钟计算收益率
        self.evaluate_return_T_p1(59, strategy_array_name=strategy_array_name)
        return_column = "return_" + strategy_array_name + "_59"
        
        # 计算累计收益率
        returns = self.readouts[return_column].dropna()
        cumulative_returns = (1 + returns).cumprod()
        
        # 返回最高累计收益率
        return cumulative_returns.max()
    
    def calculate_return_std(self, strategy_array_name="adaptive_thres"):
        """
        计算收益率变化的标准差
        """
        # 使用默认执行时间59分钟计算收益率
        self.evaluate_return_T_p1(59, strategy_array_name=strategy_array_name)
        return_column = "return_" + strategy_array_name + "_59"
        
        # 获取指定策略下的收益率
        return_arr = self.readouts[return_column].dropna()
        
        # 返回收益率的标准差
        return return_arr.std()
        
    def plot_return_rate(self, strategy_array_name="adaptive_thres", t_exec=59):
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
        bins = [-float('inf'), -0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15, float('inf')]
        bin_labels = ['<-0.15', '-0.15~-0.1', '-0.1~-0.05', '-0.05~0', '0~0.05', '0.05~0.1', '0.1~0.15', '>0.15']
        
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
            if height > 0:  # 只在有数据的柱子上显示数值
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
    stock = stock_info("src/MinutesIdx.h5", "IC", window_size=7)

    # 滚动窗口参数优化
    best_window, best_return, optimization_results = stock.optimize_window_size(
        window_range=(3, 20),  # 搜索范围
        t_exec=59              # 执行时间点
    )

    # 更新策略为最优参数
    stock.update_strategy_with_optimal_window(best_window)

    # 绘制窗口大小优化结果图表
    stock.plot_window_optimization(optimization_results)

    # 绘制阈值演化图表
    # stock.plot_threshold_evolution()

    # 生成时间敏感度图表并获取最高夏普比率
    max_sharpe_ratio, max_sharpe_time, _, _ = stock.plot_time_sensitivity("adaptive_thres")

    # 计算最高累计收益率（与时间敏感度无关）
    max_cumulative_return = stock.calculate_max_cumulative_return("adaptive_thres")

    # 计算收益率变化的标准差
    return_std = stock.calculate_return_std("adaptive_thres")

    # 输出优化后的结果
    print(f"\n=== Adaptive Threshold Strategy 分析结果 ===")
    print(f"时间敏感度图中的最高夏普比率: {max_sharpe_ratio:.4f}")
    print(f"最高累计收益率: {max_cumulative_return:.4f}")
    print(f"收益率变化的标准差: {return_std:.4f}")

    # 绘制优化后策略的累积收益率图表
    stock.plot_return_rate("adaptive_thres")