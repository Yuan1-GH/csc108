#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可视化模块
使用Plotly创建期货价格走势图表
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime, timedelta
import numpy as np

class FuturesVisualizer:
    """期货数据可视化器"""
    
    def __init__(self):
        """初始化可视化器"""
        self.colors = {
            'IF': '#1f77b4',  # 蓝色
            'IH': '#ff7f0e',  # 橙色
            'IC': '#2ca02c',  # 绿色
            'IM': '#d62728'   # 红色
        }
        
        self.product_names = {
            'IF': 'IF - 沪深300股指期货',
            'IH': 'IH - 上证50股指期货',
            'IC': 'IC - 中证500股指期货',
            'IM': 'IM - 中证1000股指期货'
        }
    
    def filter_data_by_time_range(self, data: pd.DataFrame, time_range: str) -> pd.DataFrame:
        """
        根据时间范围过滤数据
        
        Args:
            data (pd.DataFrame): 原始数据
            time_range (str): 时间范围 ('all', '1Y', '6M', '3M', '1M')
            
        Returns:
            pd.DataFrame: 过滤后的数据
        """
        if time_range == 'all':
            return data
        
        # 获取最新日期
        latest_date = data['DateIndex'].max()
        
        # 计算起始日期
        if time_range == '1Y':
            start_date = latest_date - timedelta(days=365)
        elif time_range == '6M':
            start_date = latest_date - timedelta(days=180)
        elif time_range == '3M':
            start_date = latest_date - timedelta(days=90)
        elif time_range == '1M':
            start_date = latest_date - timedelta(days=30)
        else:
            return data
        
        return data[data['DateIndex'] >= start_date]
    
    def calculate_moving_average(self, data: pd.DataFrame, window: int = 20) -> pd.Series:
        """
        计算移动平均线
        
        Args:
            data (pd.DataFrame): 数据
            window (int): 窗口大小
            
        Returns:
            pd.Series: 移动平均值
        """
        return data['Close'].rolling(window=window).mean()
    
    def calculate_bollinger_bands(self, data: pd.DataFrame, window: int = 20, num_std: float = 2):
        """
        计算布林带
        
        Args:
            data (pd.DataFrame): 数据
            window (int): 窗口大小
            num_std (float): 标准差倍数
            
        Returns:
            tuple: (上轨, 中轨, 下轨)
        """
        ma = self.calculate_moving_average(data, window)
        std = data['Close'].rolling(window=window).std()
        
        upper_band = ma + (std * num_std)
        lower_band = ma - (std * num_std)
        
        return upper_band, ma, lower_band
    
    def create_candlestick_chart(self, data: pd.DataFrame, product: str, options: dict):
        """
        创建K线图
        
        Args:
            data (pd.DataFrame): 期货数据
            product (str): 产品代码
            options (dict): 可视化选项
            
        Returns:
            plotly.graph_objects.Figure: 图表对象
        """
        # 过滤时间范围
        filtered_data = self.filter_data_by_time_range(data, options.get('time_range', 'all'))
        
        # 为了提高性能，对于大数据集进行采样
        if len(filtered_data) > 10000:
            # 按日期分组，每天取一个代表性数据点
            daily_data = filtered_data.groupby(filtered_data['DateIndex'].dt.date).agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum',
                'DateTime': 'first'
            }).reset_index()
            daily_data['DateIndex'] = pd.to_datetime(daily_data['DateIndex'])
            plot_data = daily_data
            freq_text = "日线"
        else:
            plot_data = filtered_data
            freq_text = "分钟线"
        
        # 创建子图
        has_volume = 'volume' in options.get('indicators', [])
        if has_volume:
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.1,
                subplot_titles=(f'{self.product_names[product]} - 价格走势', '成交量'),
                row_width=[0.7, 0.3]
            )
        else:
            fig = make_subplots(rows=1, cols=1)
        
        # 添加K线图
        fig.add_trace(
            go.Candlestick(
                x=plot_data['DateIndex'],
                open=plot_data['Open'],
                high=plot_data['High'],
                low=plot_data['Low'],
                close=plot_data['Close'],
                name=f'{product} K线',
                increasing_line_color='#26a69a',
                decreasing_line_color='#ef5350'
            ),
            row=1, col=1
        )
        
        # 添加技术指标
        indicators = options.get('indicators', [])
        
        # 移动平均线
        if 'ma' in indicators:
            ma5 = self.calculate_moving_average(plot_data, 5)
            ma20 = self.calculate_moving_average(plot_data, 20)
            
            fig.add_trace(
                go.Scatter(
                    x=plot_data['DateIndex'],
                    y=ma5,
                    mode='lines',
                    name='MA5',
                    line=dict(color='orange', width=1)
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=plot_data['DateIndex'],
                    y=ma20,
                    mode='lines',
                    name='MA20',
                    line=dict(color='blue', width=1)
                ),
                row=1, col=1
            )
        
        # 布林带
        if 'bollinger' in indicators:
            upper, middle, lower = self.calculate_bollinger_bands(plot_data)
            
            fig.add_trace(
                go.Scatter(
                    x=plot_data['DateIndex'],
                    y=upper,
                    mode='lines',
                    name='布林上轨',
                    line=dict(color='gray', width=1, dash='dash'),
                    showlegend=False
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=plot_data['DateIndex'],
                    y=lower,
                    mode='lines',
                    name='布林下轨',
                    line=dict(color='gray', width=1, dash='dash'),
                    fill='tonexty',
                    fillcolor='rgba(128,128,128,0.1)',
                    showlegend=False
                ),
                row=1, col=1
            )
        
        # 成交量
        if has_volume:
            colors = ['red' if close < open else 'green' 
                     for close, open in zip(plot_data['Close'], plot_data['Open'])]
            
            fig.add_trace(
                go.Bar(
                    x=plot_data['DateIndex'],
                    y=plot_data['Volume'],
                    name='成交量',
                    marker_color=colors,
                    opacity=0.7
                ),
                row=2, col=1
            )
        
        # 更新布局
        title = f'{self.product_names[product]} - {freq_text}走势图'
        if options.get('time_range') != 'all':
            time_range_names = {
                '1Y': '最近1年',
                '6M': '最近6个月', 
                '3M': '最近3个月',
                '1M': '最近1个月'
            }
            title += f' ({time_range_names.get(options["time_range"], "")})'
        
        fig.update_layout(
            title=title,
            xaxis_title='时间',
            yaxis_title='价格',
            template='plotly_white',
            height=600 if not has_volume else 800,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # 更新x轴格式
        fig.update_xaxes(
            rangeslider_visible=False,
            type='date'
        )
        
        return fig
    
    def create_line_chart(self, data: pd.DataFrame, product: str, options: dict):
        """
        创建线形图
        
        Args:
            data (pd.DataFrame): 期货数据
            product (str): 产品代码
            options (dict): 可视化选项
            
        Returns:
            plotly.graph_objects.Figure: 图表对象
        """
        # 过滤时间范围
        filtered_data = self.filter_data_by_time_range(data, options.get('time_range', 'all'))
        
        # 为了提高性能，对大数据集进行采样
        if len(filtered_data) > 5000:
            step = len(filtered_data) // 5000
            plot_data = filtered_data.iloc[::step]
        else:
            plot_data = filtered_data
        
        fig = go.Figure()
        
        # 添加收盘价线
        fig.add_trace(
            go.Scatter(
                x=plot_data['DateTime'],
                y=plot_data['Close'],
                mode='lines',
                name=f'{product} 收盘价',
                line=dict(color=self.colors.get(product, '#1f77b4'), width=2)
            )
        )
        
        # 添加技术指标
        indicators = options.get('indicators', [])
        
        if 'ma' in indicators:
            ma5 = self.calculate_moving_average(plot_data, 5)
            ma20 = self.calculate_moving_average(plot_data, 20)
            
            fig.add_trace(
                go.Scatter(
                    x=plot_data['DateTime'],
                    y=ma5,
                    mode='lines',
                    name='MA5',
                    line=dict(color='orange', width=1)
                )
            )
            
            fig.add_trace(
                go.Scatter(
                    x=plot_data['DateTime'],
                    y=ma20,
                    mode='lines',
                    name='MA20',
                    line=dict(color='blue', width=1)
                )
            )
        
        # 更新布局
        title = f'{self.product_names[product]} - 价格走势线形图'
        if options.get('time_range') != 'all':
            time_range_names = {
                '1Y': '最近1年',
                '6M': '最近6个月',
                '3M': '最近3个月', 
                '1M': '最近1个月'
            }
            title += f' ({time_range_names.get(options["time_range"], "")})'
        
        fig.update_layout(
            title=title,
            xaxis_title='时间',
            yaxis_title='价格',
            template='plotly_white',
            height=600,
            showlegend=True
        )
        
        return fig
    
    def create_area_chart(self, data: pd.DataFrame, product: str, options: dict):
        """
        创建面积图
        
        Args:
            data (pd.DataFrame): 期货数据
            product (str): 产品代码
            options (dict): 可视化选项
            
        Returns:
            plotly.graph_objects.Figure: 图表对象
        """
        # 过滤时间范围
        filtered_data = self.filter_data_by_time_range(data, options.get('time_range', 'all'))
        
        # 为了提高性能，对大数据集进行采样
        if len(filtered_data) > 5000:
            step = len(filtered_data) // 5000
            plot_data = filtered_data.iloc[::step]
        else:
            plot_data = filtered_data
        
        fig = go.Figure()
        
        # 添加面积图
        fig.add_trace(
            go.Scatter(
                x=plot_data['DateTime'],
                y=plot_data['Close'],
                mode='lines',
                name=f'{product} 收盘价',
                line=dict(color=self.colors.get(product, '#1f77b4'), width=2),
                fill='tonexty',
                fillcolor=f'rgba{tuple(list(px.colors.hex_to_rgb(self.colors.get(product, "#1f77b4"))) + [0.3])}'
            )
        )
        
        # 更新布局
        title = f'{self.product_names[product]} - 价格走势面积图'
        if options.get('time_range') != 'all':
            time_range_names = {
                '1Y': '最近1年',
                '6M': '最近6个月',
                '3M': '最近3个月',
                '1M': '最近1个月'
            }
            title += f' ({time_range_names.get(options["time_range"], "")})'
        
        fig.update_layout(
            title=title,
            xaxis_title='时间',
            yaxis_title='价格',
            template='plotly_white',
            height=600,
            showlegend=True
        )
        
        return fig
    
    def create_chart(self, data: pd.DataFrame, product: str, options: dict):
        """
        根据选项创建图表
        
        Args:
            data (pd.DataFrame): 期货数据
            product (str): 产品代码
            options (dict): 可视化选项
            
        Returns:
            plotly.graph_objects.Figure: 图表对象
        """
        chart_type = options.get('chart_type', 'candlestick')
        
        if chart_type == 'line':
            return self.create_line_chart(data, product, options)
        elif chart_type == 'area':
            return self.create_area_chart(data, product, options)
        else:
            return self.create_candlestick_chart(data, product, options)
    
    def show_chart(self, fig):
        """
        显示图表
        
        Args:
            fig: Plotly图表对象
        """
        fig.show()
    
    def save_chart(self, fig, product: str, chart_type: str, time_range: str, technical_indicators: list = None):
        """
        保存图表到指定目录，使用新的命名格式
        
        Args:
            fig: Plotly图表对象
            product (str): 期货产品名称
            chart_type (str): 图表类型
            time_range (str): 时间范围
            technical_indicators (list): 技术指标列表
        """
        import os
        
        try:
            # 创建保存目录
            save_dir = os.path.join("..", "中信建投", "任务一")
            os.makedirs(save_dir, exist_ok=True)
            
            # 生成技术指标字符串
            if technical_indicators:
                tech_str = "-".join(technical_indicators)
            else:
                tech_str = chart_type
            
            # 生成文件名：{产品}-{技术指标}-{时间范围}.html
            filename = f"{product}-{tech_str}-{time_range}.html"
            filepath = os.path.join(save_dir, filename)
            
            # 保存文件
            fig.write_html(filepath)
            print(f"✅ 图表已保存为: {filepath}")
            return filepath
        except Exception as e:
            print(f"❌ 保存图表失败: {e}")
            return None

def test_visualization():
    """测试可视化功能"""
    print("=== 测试可视化模块 ===")
    
    # 创建测试数据
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    np.random.seed(42)
    
    test_data = pd.DataFrame({
        'DateTime': dates,
        'DateIndex': dates,
        'Open': 3000 + np.cumsum(np.random.randn(100) * 10),
        'High': 3000 + np.cumsum(np.random.randn(100) * 10) + 20,
        'Low': 3000 + np.cumsum(np.random.randn(100) * 10) - 20,
        'Close': 3000 + np.cumsum(np.random.randn(100) * 10),
        'Volume': np.random.randint(1000, 10000, 100)
    })
    
    # 创建可视化器
    visualizer = FuturesVisualizer()
    
    # 测试选项
    options = {
        'chart_type': 'candlestick',
        'time_range': 'all',
        'indicators': ['ma', 'volume']
    }
    
    # 创建图表
    fig = visualizer.create_chart(test_data, 'IF', options)
    print("✅ 测试图表创建成功")

if __name__ == "__main__":
    test_visualization()