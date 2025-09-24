import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
rcParams['axes.unicode_minus'] = False

class ZBolinger:
    """
    固定布林带策略（中轨50MA，k=1）
    
    策略逻辑：
    1. 计算n日移动平均线作为中轨（MAn）
    2. 计算m日标准差作为波动率
    3. 上轨 = 中轨 + k*标准差，下轨 = 中轨 - k*标准差
    4. 当前一天收盘价高于上轨时卖出，低于下轨时买入
    """
    
    def __init__(self, filepath, product):
        """
        初始化策略类
        
        参数:
        filepath: H5文件路径
        product: 产品名称（如'IC'）
        """
        self.filepath = filepath
        self.product = product
        self.raw = pd.read_hdf(filepath)
        self.target_raw = self.raw[product]
        self.dates = sorted(self.target_raw.keys())
        self.readouts = pd.DataFrame()
        
        # 布林带参数
        self.ma_period =  10 # 中轨周期
        self.std_period = 10 # 标准差周期
        self.k = 1.38  # 布林带系数
        
        # 初始化数据
        self.initialize_data()
    
    def initialize_data(self):
        """
        初始化数据，计算布林带
        """
        # 提取每日收盘价
        daily_close = []
        dates_list = []
        
        for date in self.dates:
            day_data = self.target_raw[date]
            if len(day_data) > 0:
                close_price = day_data["Close"].iloc[-1]  # 使用收盘价
                daily_close.append(close_price)
                dates_list.append(date)
        
        # 创建DataFrame
        self.readouts = pd.DataFrame({
            'close_price': daily_close,
            'date': dates_list
        })
        self.readouts.set_index('date', inplace=True)
        
        # 计算布林带
        self.calculate_bollinger_bands()
    
    def calculate_bollinger_bands(self):
        """
        计算布林带指标
        """
        # 计算中轨（n日移动平均线）
        self.readouts[f'MA{self.ma_period}'] = self.readouts['close_price'].rolling(window=self.ma_period, min_periods=1).mean()
        
        # 计算标准差（m日）
        self.readouts[f'std{self.std_period}'] = self.readouts['close_price'].rolling(window=self.std_period, min_periods=1).std()
        
        # 计算上轨和下轨
        self.readouts[f'upper_band{self.ma_period}'] = self.readouts[f'MA{self.ma_period}'] + self.k * self.readouts[f'std{self.std_period}']    
        self.readouts[f'lower_band{self.ma_period}'] = self.readouts[f'MA{self.ma_period}'] - self.k * self.readouts[f'std{self.std_period}']           
        
        # 计算策略信号（基于前一天收盘价与布林带的关系）
        self.calculate_signals()
    
    def calculate_signals(self):
        """
        计算策略信号
        信号基于前一天收盘价与布林带的关系
        """
        # 获取前一天收盘价（向后移位）
        prev_close = self.readouts['close_price'].shift(1)
        
        signals = np.where(
            prev_close > self.readouts[f'upper_band{self.ma_period}'], -1,  # 高于上轨，卖出
            np.where(prev_close < self.readouts[f'lower_band{self.ma_period}'], +1, 0)  # 低于下轨，买入；其他情况持有  
        )
        
        self.readouts['signal'] = signals
        
        # 统计信号分布
        buy_signals = np.sum(signals == 1)
        sell_signals = np.sum(signals == -1)
        hold_signals = np.sum(signals == 0)
        
        print(f"布林带策略信号计算完成")
        print(f"信号分布: 买入={buy_signals}, 卖出={sell_signals}, 持有={hold_signals}")
    
    def evaluate_return_T_p1(self, t_exec=59, strategy_array_name="signal"):
        """
        评估T+1收益率
        
        参数:
        t_exec: 执行时间点
        strategy_array_name: 策略名称
        
        返回:
        (mean_return, std_return): 平均收益率和标准差
        """
        # 读取T+1的策略（跳过第一天因为没有前一天信号）
        strategy_ = np.array(self.readouts[strategy_array_name].tolist()[1:-1])
        
        # 分别读取T+1和T+2指定时间的收盘价
        short_price_T_p1 = np.array([self.target_raw.loc[day]["Close"].iloc[t_exec] for day in self.dates[1:-1]])
        long_price_T_p2 = np.array([self.target_raw.loc[day]["Close"].iloc[t_exec] for day in self.dates[2:]])
        
        # 确保数组长度一致
        min_length = min(len(strategy_), len(long_price_T_p2))
        strategy_ = strategy_[:min_length]
        short_price_T_p1 = short_price_T_p1[:min_length]
        long_price_T_p2 = long_price_T_p2[:min_length]
        
        # 计算T+1的收益率
        return_rate_T_p1 = pd.Series(((long_price_T_p2/short_price_T_p1)-1.)*strategy_, index=self.dates[1:1+min_length])
        
        # 添加到self.readouts
        self.readouts["return_"+strategy_array_name+"_"+str(t_exec)] = return_rate_T_p1
        
        # 计算统计量
        mean_return = np.mean(return_rate_T_p1)
        std_return = np.std(return_rate_T_p1)
        
        return mean_return, std_return
    
    def calculate_sharpe_ratio(self, strategy_array_name="signal", t_exec=59):
        """
        计算夏普比率
        
        参数:
        strategy_array_name: 策略名称
        t_exec: 执行时间点
        
        返回:
        sharpe_ratio: 夏普比率
        """
        return_column = "return_" + strategy_array_name + "_" + str(t_exec)
        
        if return_column not in self.readouts.columns:
            self.evaluate_return_T_p1(t_exec, strategy_array_name)
        
        returns = self.readouts[return_column].dropna()
        
        if len(returns) == 0:
            return 0
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        # 假设无风险利率为0
        sharpe_ratio = np.sqrt(250.) * (mean_return / std_return if std_return != 0 else 0)
        
        return sharpe_ratio
    
    def calculate_max_cumulative_return(self, strategy_array_name="signal"):
        """
        计算策略能达到的最高累计收益率（从2010年开始到数据提供日期为止）
        
        参数:
        strategy_array_name: 策略名称
        
        返回:
        max_cumulative_return: 最高累计收益率
        """
        # 使用默认执行时间59分钟计算收益率
        self.evaluate_return_T_p1(59, strategy_array_name=strategy_array_name)
        return_column = "return_" + strategy_array_name + "_59"
        
        # 获取收益率数据
        returns = self.readouts[return_column].dropna()
        
        if len(returns) == 0:
            return 0
        
        # 计算累计收益率（从2010年开始到数据提供日期为止）
        cumulative_returns = (1 + returns).cumprod()
        
        # 返回最高累计收益率
        return cumulative_returns.max()
    
    def calculate_return_std(self, strategy_array_name="signal"):
        """
        计算收益率变化的标准差
        
        参数:
        strategy_array_name: 策略名称
        
        返回:
        return_std: 收益率标准差
        """
        # 使用默认执行时间59分钟计算收益率
        self.evaluate_return_T_p1(59, strategy_array_name=strategy_array_name)
        return_column = "return_" + strategy_array_name + "_59"
        
        # 获取指定策略下的收益率
        return_arr = self.readouts[return_column].dropna()
        
        # 返回收益率的标准差
        return return_arr.std()

    def run_comprehensive_analysis(self):
        """
        运行布林带策略的综合分析，包括斐波那契时间敏感度、最高累计收益率等
        
        返回:
        analysis_results: 综合分析结果字典
        """
        print("=== 布林带策略综合分析 ===")
        print(f"参数: MA周期={self.ma_period}, 标准差周期={self.std_period}, K值={self.k}")
        
        # 1. 运行斐波那契时间敏感度分析
        print("\n1. 正在运行斐波那契时间敏感度分析...")
        fib_results = self.fibonacci_time_sensitivity_analysis()
        
        # 2. 计算最高累计收益率（从2010年开始到数据提供日期为止）
        print("\n2. 正在计算最高累计收益率...")
        max_cumulative_return = self.calculate_max_cumulative_return()
        
        # 3. 计算收益率变化的标准差
        print("\n3. 正在计算收益率标准差...")
        return_std = self.calculate_return_std()
        
        # 4. 获取最佳执行时间（从斐波那契分析结果中）
        valid_results = {k: v for k, v in fib_results.items() if v is not None}
        if valid_results:
            best_time = max(valid_results.keys(), key=lambda x: valid_results[x]['sharpe_ratio'])
            best_sharpe = valid_results[best_time]['sharpe_ratio']
        else:
            best_time = 59  # 默认值
            best_sharpe = 0
        
        # 5. 生成综合分析结果
        analysis_results = {
            'fibonacci_sensitivity': fib_results,
            'max_cumulative_return': max_cumulative_return,
            'return_std': return_std,
            'best_execution_time': best_time,
            'best_sharpe_ratio': best_sharpe,
            'strategy_parameters': {
                'ma_period': self.ma_period,
                'std_period': self.std_period,
                'k_value': self.k
            }
        }
        
        # 6. 输出分析结果
        print(f"\n=== 布林带策略综合分析结果 ===")
        print(f"最高累计收益率: {max_cumulative_return:.4f}")
        print(f"收益率标准差: {return_std:.4f}")
        print(f"最佳执行时间点: {best_time} (夏普比率: {best_sharpe:.4f})")
        
        if valid_results:
            # 计算一些统计指标
            sharpe_ratios = [valid_results[t]['sharpe_ratio'] for t in valid_results.keys()]
            mean_sharpe = np.mean(sharpe_ratios)
            std_sharpe = np.std(sharpe_ratios)
            print(f"平均夏普比率: {mean_sharpe:.4f}")
            print(f"夏普比率标准差: {std_sharpe:.4f}")
            print(f"策略时间敏感度: {std_sharpe:.4f}")
        
        return analysis_results

    def plot_cumulative_return_chart(self, strategy_array_name="signal", save_path=None):
        """
        绘制累计收益率图表（从2010年开始到数据提供日期为止）
        
        参数:
        strategy_array_name: 策略名称
        save_path: 保存路径（可选）
        """
        # 确保已经计算了收益率
        return_column = "return_" + strategy_array_name + "_59"
        if return_column not in self.readouts.columns:
            self.evaluate_return_T_p1(59, strategy_array_name)
        
        # 获取收益率数据
        returns = self.readouts[return_column].dropna()
        
        if len(returns) == 0:
            print("没有足够的数据来绘制累计收益率图表")
            return
        
        # 计算累计收益率
        cumulative_returns = (1 + returns).cumprod() - 1
        
        # 转换日期索引
        dates = pd.to_datetime(returns.index)
        
        # 创建图表
        plt.figure(figsize=(14, 8))
        
        # 绘制累计收益率曲线
        plt.plot(dates, cumulative_returns * 100, linewidth=3, color='darkgreen', 
                label='累计收益率', alpha=0.8)
        
        # 添加水平参考线
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=1)
        
        # 找出最高点和最低点
        max_return = cumulative_returns.max()
        min_return = cumulative_returns.min()
        max_idx = cumulative_returns.idxmax()
        min_idx = cumulative_returns.idxmin()
        
        # 标注最高点和最低点
        plt.scatter(pd.to_datetime(max_idx), max_return * 100, color='red', s=150, 
                   marker='*', zorder=5, label=f'最高点: {max_return:.2%}')
        plt.scatter(pd.to_datetime(min_idx), min_return * 100, color='blue', s=150, 
                   marker='v', zorder=5, label=f'最低点: {min_return:.2%}')
        
        # 添加统计信息
        final_return = cumulative_returns.iloc[-1]
        total_days = len(cumulative_returns)
        annualized_return = (1 + final_return) ** (252 / total_days) - 1 if total_days > 0 else 0
        
        stats_text = f"""累计收益率统计:
最终收益率: {final_return:.2%}
最高收益率: {max_return:.2%}
最低收益率: {min_return:.2%}
年化收益率: {annualized_return:.2%}
交易天数: {total_days}"""
        
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', 
                facecolor='lightgreen', alpha=0.8), fontsize=10)
        
        # 设置图表属性
        plt.xlabel('日期', fontsize=12, fontweight='bold')
        plt.ylabel('累计收益率 (%)', fontsize=12, fontweight='bold')
        plt.title(f'布林带策略累计收益率走势\n(MA周期={self.ma_period}, 标准差周期={self.std_period}, K值={self.k})', 
                 fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best', framealpha=0.8)
        
        # 设置x轴格式
        from matplotlib.dates import MonthLocator, DateFormatter
        plt.gca().xaxis.set_major_locator(MonthLocator(interval=3))
        plt.gca().xaxis.set_major_formatter(DateFormatter('%Y-%m'))
        
        plt.tight_layout()
        
        # 保存图表（如果指定了路径）
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"累计收益率图表已保存到: {save_path}")
        
        plt.show()
        
        # 打印总结信息
        print(f"\n=== 累计收益率分析总结 ===")
        print(f"最终累计收益率: {final_return:.4f}")
        print(f"最高累计收益率: {max_return:.4f}")
        print(f"最低累计收益率: {min_return:.4f}")
        print(f"年化收益率: {annualized_return:.4f}")
        print(f"交易总天数: {total_days}")
    
    def plot_bollinger_bands(self):
        """
        绘制布林带图表
        """
        plt.figure(figsize=(15, 10))
        
        # 转换日期索引
        dates = pd.to_datetime(self.readouts.index)
        
        # 子图1：价格与布林带
        plt.subplot(2, 1, 1)
        plt.plot(dates, self.readouts['close_price'], label='收盘价', alpha=0.8, linewidth=1.5)
        plt.plot(dates, self.readouts['MA50'], label='MA50 (中轨)', color='orange', linewidth=2)
        plt.plot(dates, self.readouts['upper_band'], label='上轨', color='red', linestyle='--', alpha=0.7)
        plt.plot(dates, self.readouts['lower_band'], label='下轨', color='green', linestyle='--', alpha=0.7)
        
        # 标记买卖信号
        buy_signals = self.readouts['signal'] == 1
        sell_signals = self.readouts['signal'] == -1
        
        plt.scatter(dates[buy_signals], self.readouts['close_price'][buy_signals], 
                   color='green', marker='^', s=100, label='买入信号', zorder=5)
        plt.scatter(dates[sell_signals], self.readouts['close_price'][sell_signals], 
                   color='red', marker='v', s=100, label='卖出信号', zorder=5)
        
        plt.title('布林带策略 - 价格与信号')
        plt.ylabel('价格')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 子图2：策略信号
        plt.subplot(2, 1, 2)
        plt.plot(dates, self.readouts['signal'], label='策略信号', drawstyle='steps-post', color='blue', linewidth=2)
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.axhline(y=1, color='green', linestyle='--', alpha=0.5, label='买入信号')
        plt.axhline(y=-1, color='red', linestyle='--', alpha=0.5, label='卖出信号')
        plt.title('布林带策略信号')
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
    
    def plot_return_distribution(self, strategy_array_name="signal", t_exec=59):
        """
        绘制收益率分布图
        
        参数:
        strategy_array_name: 策略名称
        t_exec: 执行时间点
        """
        return_column = "return_" + strategy_array_name + "_" + str(t_exec)
        
        if return_column not in self.readouts.columns:
            self.evaluate_return_T_p1(t_exec, strategy_array_name)
        
        returns = self.readouts[return_column].dropna()
        
        if len(returns) == 0:
            print("没有足够的数据来绘制收益率分布图")
            return
        
        # 定义收益率区间（0.005间隔）
        bins = [-float('inf'), -0.03, -0.025, -0.02, -0.015, -0.01, -0.005, 0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, float('inf')]
        bin_labels = ['<-0.03', '-0.03~-0.025', '-0.025~-0.02', '-0.02~-0.015', '-0.015~-0.01', '-0.01~-0.005', '-0.005~0', '0~0.005', '0.005~0.01', '0.01~0.015', '0.015~0.02', '0.02~0.025', '0.025~0.03', '>0.03']
        
        # 统计各区间的天数
        return_categories = pd.cut(returns, bins=bins, labels=bin_labels)
        return_counts = return_categories.value_counts().sort_index()
        
        # 绘制柱状图
        plt.figure(figsize=(14, 8))
        bars = plt.bar(range(len(return_counts)), return_counts.values, alpha=0.7, color='skyblue', edgecolor='black')
        
        # 在柱子上添加数值标签
        for bar, count in zip(bars, return_counts.values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, str(count), 
                    ha='center', va='bottom', fontsize=10)
        
        plt.xlabel('收益率区间')
        plt.ylabel('天数')
        plt.title(f'布林带策略收益率分布 (执行时间: {t_exec})')
        plt.xticks(range(len(return_counts)), return_counts.index, rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        
        # 添加统计信息
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        sharpe_ratio = self.calculate_sharpe_ratio(strategy_array_name, t_exec)
        
        plt.text(0.02, 0.98, f'平均收益率: {mean_return:.4f}\n标准差: {std_return:.4f}\n夏普比率: {sharpe_ratio:.4f}', 
                transform=plt.gca().transAxes, verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.show()
    
    def fibonacci_time_sensitivity_analysis(self, fib_numbers=None):
        """
        斐波那契数列时间敏感度分析
        
        参数:
        fib_numbers: 斐波那契数列，默认为[1, 2, 3, 5, 8, 13, 21, 34, 55, 89]
        
        返回:
        sensitivity_results: 各时间点的策略表现字典
        """
        if fib_numbers is None:
            fib_numbers = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89]
        
        print("=== 斐波那契数列时间敏感度分析 ===")
        print(f"分析时间点: {fib_numbers}")
        print(f"布林带参数: MA周期={self.ma_period}, 标准差周期={self.std_period}, K值={self.k}")
        
        sensitivity_results = {}
        
        for t_exec in fib_numbers:
            try:
                # 评估策略在该时间点的表现
                mean_return, std_return = self.evaluate_return_T_p1(t_exec)
                sharpe_ratio = self.calculate_sharpe_ratio(t_exec=t_exec)
                
                # 计算信号统计
                returns = self.readouts[f"return_signal_{t_exec}"].dropna()
                total_signals = len(returns)
                positive_returns = sum(returns > 0)
                negative_returns = sum(returns < 0)
                win_rate = positive_returns / total_signals if total_signals > 0 else 0
                
                sensitivity_results[t_exec] = {
                    'mean_return': mean_return,
                    'std_return': std_return,
                    'sharpe_ratio': sharpe_ratio,
                    'total_signals': total_signals,
                    'win_rate': win_rate,
                    'positive_signals': positive_returns,
                    'negative_signals': negative_returns
                }
                
                print(f"\n时间点 {t_exec}:")
                print(f"  平均收益率: {mean_return:.6f}")
                print(f"  收益率标准差: {std_return:.6f}")
                print(f"  夏普比率: {sharpe_ratio:.6f}")
                print(f"  总信号数: {total_signals}")
                print(f"  胜率: {win_rate:.2%}")
                print(f"  正收益信号: {positive_returns}, 负收益信号: {negative_returns}")
                
            except Exception as e:
                print(f"时间点 {t_exec} 分析失败: {e}")
                sensitivity_results[t_exec] = None
        
        # 找出最佳和最差时间点
        valid_results = {k: v for k, v in sensitivity_results.items() if v is not None}
        if valid_results:
            best_time = max(valid_results.keys(), key=lambda x: valid_results[x]['sharpe_ratio'])
            worst_time = min(valid_results.keys(), key=lambda x: valid_results[x]['sharpe_ratio'])
            
            print(f"\n=== 时间敏感度分析结果 ===")
            print(f"最佳执行时间点: {best_time} (夏普比率: {valid_results[best_time]['sharpe_ratio']:.6f})")
            print(f"最差执行时间点: {worst_time} (夏普比率: {valid_results[worst_time]['sharpe_ratio']:.6f})")
            print(f"敏感度差异: {valid_results[best_time]['sharpe_ratio'] - valid_results[worst_time]['sharpe_ratio']:.6f}")
        
        return sensitivity_results
    
    def plot_sharpe_ratio_chart(self, sensitivity_results, save_path=None):
        """
        专门绘制夏普比率折线图
        
        参数:
        sensitivity_results: 敏感度分析结果字典
        save_path: 保存路径（可选）
        """
        valid_results = {k: v for k, v in sensitivity_results.items() if v is not None}
        if not valid_results:
            print("没有有效的敏感度分析结果")
            return
        
        # 提取数据
        times = list(valid_results.keys())
        sharpe_ratios = [valid_results[t]['sharpe_ratio'] for t in times]
        
        # 找出最高和最低夏普比率
        max_sharpe_idx = np.argmax(sharpe_ratios)
        min_sharpe_idx = np.argmin(sharpe_ratios)
        max_sharpe_time = times[max_sharpe_idx]
        max_sharpe_value = sharpe_ratios[max_sharpe_idx]
        min_sharpe_time = times[min_sharpe_idx]
        min_sharpe_value = sharpe_ratios[min_sharpe_idx]
        
        # 创建图表
        plt.figure(figsize=(14, 8))
        
        # 绘制折线图
        plt.plot(times, sharpe_ratios, marker='o', linewidth=3, markersize=8, 
                color='steelblue', markerfacecolor='steelblue', markeredgecolor='darkblue',
                label='夏普比率', alpha=0.8)
        
        # 特殊标注最高和最低夏普比率
        plt.scatter(max_sharpe_time, max_sharpe_value, color='red', s=150, zorder=5, 
                   marker='*', label=f'最高夏普比率: {max_sharpe_value:.4f}')
        plt.scatter(min_sharpe_time, min_sharpe_value, color='orange', s=150, zorder=5, 
                   marker='v', label=f'最低夏普比率: {min_sharpe_value:.4f}')
        
        # 添加水平参考线
        plt.axhline(y=np.mean(sharpe_ratios), color='green', linestyle='--', alpha=0.7, 
                   label=f'平均夏普比率: {np.mean(sharpe_ratios):.4f}')
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # 添加标注
        plt.annotate(f'最高\n{max_sharpe_value:.4f}', 
                    xy=(max_sharpe_time, max_sharpe_value),
                    xytext=(max_sharpe_time, max_sharpe_value + 0.1),
                    ha='center', va='bottom', fontweight='bold', color='red', fontsize=12,
                    arrowprops=dict(arrowstyle='->', color='red', lw=2))
        
        plt.annotate(f'最低\n{min_sharpe_value:.4f}', 
                    xy=(min_sharpe_time, min_sharpe_value),
                    xytext=(min_sharpe_time, min_sharpe_value - 0.1),
                    ha='center', va='top', fontweight='bold', color='orange', fontsize=12,
                    arrowprops=dict(arrowstyle='->', color='orange', lw=2))
        
        # 设置图表属性
        plt.xlabel('执行时间点', fontsize=12, fontweight='bold')
        plt.ylabel('夏普比率', fontsize=12, fontweight='bold')
        plt.title(f'布林带策略夏普比率趋势分析\n(MA周期={self.ma_period}, 标准差周期={self.std_period}, K值={self.k})', 
                 fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best', framealpha=0.8)
        
        # 设置x轴刻度
        plt.xticks(times, [f'{t}' for t in times])
        
        # 添加填充区域显示波动范围
        plt.fill_between(times, sharpe_ratios, alpha=0.2, color='steelblue', 
                        label=f'波动范围: [{min(sharpe_ratios):.4f}, {max(sharpe_ratios):.4f}]')
        
        # 添加统计信息文本框
        stats_text = f"""统计信息:
最高夏普比率: {max_sharpe_value:.4f} (时间点: {max_sharpe_time})
最低夏普比率: {min_sharpe_value:.4f} (时间点: {min_sharpe_time})
平均夏普比率: {np.mean(sharpe_ratios):.4f}
标准差: {np.std(sharpe_ratios):.4f}
夏普比率范围: [{min(sharpe_ratios):.4f}, {max(sharpe_ratios):.4f}]"""
        
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', 
                facecolor='lightblue', alpha=0.8), fontsize=10)
        
        plt.tight_layout()
        
        # 保存图表（如果指定了路径）
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"夏普比率折线图已保存到: {save_path}")
        
        plt.show()
        
        # 打印总结信息
        print(f"\n=== 夏普比率趋势分析总结 ===")
        print(f"最高夏普比率: {max_sharpe_value:.4f} (时间点: {max_sharpe_time})")
        print(f"最低夏普比率: {min_sharpe_value:.4f} (时间点: {min_sharpe_time})")
        print(f"策略对执行时间敏感度: {np.std(sharpe_ratios):.4f}")
        print(f"夏普比率范围: [{min(sharpe_ratios):.4f}, {max(sharpe_ratios):.4f}]")
        
        # 分析趋势
        if len(times) > 2:
            # 计算简单趋势
            early_avg = np.mean(sharpe_ratios[:len(sharpe_ratios)//2])
            late_avg = np.mean(sharpe_ratios[len(sharpe_ratios)//2:])
            trend = "上升" if late_avg > early_avg else "下降"
            print(f"整体趋势: {trend} (前期平均: {early_avg:.4f}, 后期平均: {late_avg:.4f})")
    
    def plot_fibonacci_sensitivity(self, sensitivity_results):
        """
        绘制斐波那契时间敏感度分析图表
        
        参数:
        sensitivity_results: 敏感度分析结果字典
        """
        valid_results = {k: v for k, v in sensitivity_results.items() if v is not None}
        if not valid_results:
            print("没有有效的敏感度分析结果")
            return
        
        # 提取数据
        times = list(valid_results.keys())
        sharpe_ratios = [valid_results[t]['sharpe_ratio'] for t in times]
        mean_returns = [valid_results[t]['mean_return'] for t in times]
        win_rates = [valid_results[t]['win_rate'] for t in times]
        total_signals = [valid_results[t]['total_signals'] for t in times]
        
        # 找出最高夏普比率
        max_sharpe_idx = np.argmax(sharpe_ratios)
        max_sharpe_time = times[max_sharpe_idx]
        max_sharpe_value = sharpe_ratios[max_sharpe_idx]
        
        # 创建图表
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'布林带策略斐波那契时间敏感度分析\n(MA周期={self.ma_period}, STD周期={self.std_period}, K值={self.k})', 
                    fontsize=16, fontweight='bold')
        
        # 1. 夏普比率
        bars1 = ax1.bar(times, sharpe_ratios, alpha=0.7, color='steelblue', edgecolor='black')
        bars1[max_sharpe_idx].set_color('red')  # 最高夏普比率用红色标注
        ax1.axhline(y=max_sharpe_value, color='red', linestyle='--', alpha=0.7, 
                   label=f'最高夏普比率: {max_sharpe_value:.4f}')
        ax1.set_xlabel('执行时间点')
        ax1.set_ylabel('夏普比率')
        ax1.set_title('夏普比率 vs 执行时间点')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 在最高夏普比率的柱子上添加标注
        ax1.annotate(f'最高\n{max_sharpe_value:.4f}', 
                    xy=(max_sharpe_time, max_sharpe_value),
                    xytext=(max_sharpe_time, max_sharpe_value + 0.1),
                    ha='center', va='bottom', fontweight='bold', color='red',
                    arrowprops=dict(arrowstyle='->', color='red'))
        
        # 2. 平均收益率
        ax2.bar(times, mean_returns, alpha=0.7, color='green', edgecolor='black')
        ax2.set_xlabel('执行时间点')
        ax2.set_ylabel('平均收益率')
        ax2.set_title('平均收益率 vs 执行时间点')
        ax2.grid(True, alpha=0.3)
        
        # 3. 胜率
        ax3.bar(times, [w*100 for w in win_rates], alpha=0.7, color='orange', edgecolor='black')
        ax3.set_xlabel('执行时间点')
        ax3.set_ylabel('胜率 (%)')
        ax3.set_title('胜率 vs 执行时间点')
        ax3.grid(True, alpha=0.3)
        
        # 4. 信号数量
        ax4.bar(times, total_signals, alpha=0.7, color='purple', edgecolor='black')
        ax4.set_xlabel('执行时间点')
        ax4.set_ylabel('总信号数')
        ax4.set_title('信号数量 vs 执行时间点')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # 打印总结信息
        print(f"\n=== 斐波那契时间敏感度图表总结 ===")
        print(f"最高夏普比率: {max_sharpe_value:.4f} (时间点: {max_sharpe_time})")
        print(f"策略对执行时间敏感度: {np.std(sharpe_ratios):.4f}")
        print(f"夏普比率范围: [{min(sharpe_ratios):.4f}, {max(sharpe_ratios):.4f}]")
    
    def run_fibonacci_analysis(self):
        """
        运行斐波那契数列时间敏感度分析（不生成图表）
        """
        print("=== 布林带策略斐波那契时间敏感度分析 ===")
        print(f"参数: MA周期={self.ma_period}, 标准差周期={self.std_period}, K值={self.k}")
        
        # 运行斐波那契时间敏感度分析
        fib_results = self.fibonacci_time_sensitivity_analysis()
        
        print("\n斐波那契时间敏感度分析完成！")
        return fib_results

def test_z_bolinger_fibonacci():
    """
    测试布林带策略的斐波那契时间敏感度分析
    """
    print("开始测试布林带策略的斐波那契时间敏感度分析...")
    
    # 创建策略实例
    strategy = ZBolinger(f"src/MinutesIdx.h5", "IC")
    
    # 运行斐波那契时间敏感度分析（不生成图表）
    fib_results = strategy.run_fibonacci_analysis()
    
    print("\n测试完成！")

def show_sharpe_ratio_chart():
    """
    专门展示夏普比率图表
    """
    print("开始生成布林带策略的夏普比率图表...")
    
    # 创建策略实例
    strategy = ZBolinger(f"src/MinutesIdx.h5", "IC")
    
    # 运行斐波那契时间敏感度分析
    fib_results = strategy.fibonacci_time_sensitivity_analysis()
    
    # 生成专门的夏普比率图表
    strategy.plot_sharpe_ratio_chart(fib_results, save_path="fibonacci_sharpe_ratio.png")
    
    print("\n夏普比率图表生成完成！")

def show_comprehensive_fibonacci_analysis():
    """
    展示完整的斐波那契时间敏感度分析（包含夏普比率图表）
    """
    print("开始生成完整的斐波那契时间敏感度分析...")
    
    # 创建策略实例
    strategy = ZBolinger(f"src/MinutesIdx.h5", "IC")
    
    # 运行斐波那契时间敏感度分析
    fib_results = strategy.fibonacci_time_sensitivity_analysis()
    
    # 生成专门的夏普比率图表
    strategy.plot_sharpe_ratio_chart(fib_results, save_path="fibonacci_sharpe_ratio.png")
    
    # 生成完整的四合一分析图表
    strategy.plot_fibonacci_sensitivity(fib_results)

def run_comprehensive_zbolinger_analysis():
    """
    运行布林带策略的完整综合分析，包括累计收益率分析
    """
    print("开始运行布林带策略的完整综合分析...")
    
    # 创建策略实例
    strategy = ZBolinger(f"src/MinutesIdx.h5", "IC")
    
    # 运行综合分析
    analysis_results = strategy.run_comprehensive_analysis()
    
    # 绘制累计收益率图表
    strategy.plot_cumulative_return_chart(save_path="zbolinger_cumulative_return.png")
    
    print("\n布林带策略综合分析完成！")
    
    return analysis_results

def show_cumulative_return_analysis():
    """
    专门展示累计收益率分析
    """
    print("开始布林带策略的累计收益率分析...")
    
    # 创建策略实例
    strategy = ZBolinger(f"src/MinutesIdx.h5", "IC")
    
    # 计算最高累计收益率
    max_cumulative_return = strategy.calculate_max_cumulative_return()
    print(f"最高累计收益率: {max_cumulative_return:.4f}")
    
    # 计算收益率标准差
    return_std = strategy.calculate_return_std()
    print(f"收益率标准差: {return_std:.4f}")
    
    # 绘制累计收益率图表
    strategy.plot_cumulative_return_chart(save_path="zbolinger_cumulative_return.png")
    
    print("\n累计收益率分析完成！")
    
    print("\n完整的斐波那契时间敏感度分析完成！")

if __name__ == "__main__":
    # 提供多种运行选项
    print("请选择运行模式:")
    print("1. 运行斐波那契时间敏感度分析（仅数值结果）")
    print("2. 生成夏普比率图表")
    print("3. 生成完整的斐波那契时间敏感度分析（包含图表）")
    
    choice = input("请输入选择 (1/2/3): ").strip()
    
    if choice == "1":
        # 运行斐波那契时间敏感度分析（仅数值结果）
        test_z_bolinger_fibonacci()
    elif choice == "2":
        # 生成夏普比率图表
        show_sharpe_ratio_chart()
    elif choice == "3":
        # 生成完整的斐波那契时间敏感度分析
        show_comprehensive_fibonacci_analysis()
        show_cumulative_return_analysis()
    else:
        print("无效选择，默认运行模式1")
        test_z_bolinger_fibonacci()