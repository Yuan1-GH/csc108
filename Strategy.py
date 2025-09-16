#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
策略函数库
包含所有交易策略的实现，供参数优化器调用
"""

import numpy as np
import pandas as pd

class StrategyFunctions:
    """策略函数集合"""

    @staticmethod
    def simple_reversal_strategy(price_data, **kwargs):
        """
        简单反转策略

        Args:
            price_data: 价格数据
            **kwargs: 其他参数（该策略不需要参数）

        Returns:
            tuple: (收益率序列, 策略信息字典)
        """
        returns = price_data.pct_change().dropna()
        signals = -np.sign(returns.shift(1))
        strategy_returns = signals * returns

        strategy_info = {
            'strategy_name': 'simple_reversal',
            'parameters': {'method': 'baseline'},
            'description': '简单反转策略：昨天跌今天就买，昨天涨今天就卖'
        }

        return strategy_returns.dropna(), strategy_info

    @staticmethod
    def fixed_threshold_strategy(price_data, threshold=0.002, **kwargs):
        """
        固定阈值策略

        Args:
            price_data: 价格数据
            threshold: 阈值参数
            **kwargs: 其他参数

        Returns:
            tuple: (收益率序列, 策略信息字典)
        """
        returns = price_data.pct_change().dropna()
        rolling_mean = returns.rolling(window=20).mean()
        rolling_std = returns.rolling(window=20).std()
        z_score = (returns - rolling_mean) / rolling_std

        signals = np.where(z_score < -threshold, 1, np.where(z_score > threshold, -1, 0))
        strategy_returns = signals * returns

        strategy_info = {
            'strategy_name': 'fixed_threshold',
            'parameters': {'threshold': threshold},
            'description': f'固定阈值策略：z_score < -{threshold}买入，z_score > {threshold}卖出'
        }

        return strategy_returns.dropna(), strategy_info

    @staticmethod
    def adaptive_threshold_strategy(price_data, window=7, std_multiplier=1.5, **kwargs):
        """
        自适应阈值策略

        Args:
            price_data: 价格数据
            window: 窗口大小
            std_multiplier: 标准差倍数
            **kwargs: 其他参数

        Returns:
            tuple: (收益率序列, 策略信息字典)
        """
        returns = price_data.pct_change().dropna()

        # 计算自适应阈值
        rolling_mean = returns.rolling(window=window).mean()
        rolling_std = returns.rolling(window=window).std()
        upper_band = rolling_mean + std_multiplier * rolling_std
        lower_band = rolling_mean - std_multiplier * rolling_std

        # 生成信号：均值回归策略
        signals = np.zeros_like(returns.values)
        signals[returns.values < lower_band.values] = 1    # 低于下轨买入
        signals[returns.values > upper_band.values] = -1   # 高于上轨卖出

        strategy_returns = pd.Series(signals * returns.values, index=returns.index)

        strategy_info = {
            'strategy_name': 'adaptive_threshold',
            'parameters': {'window': window, 'std_multiplier': std_multiplier},
            'description': f'自适应阈值策略：window={window}, std_multiplier={std_multiplier}'
        }

        return strategy_returns.dropna(), strategy_info

    @staticmethod
    def adaptive_trend_strategy(price_data, window=7, std_multiplier=1.5, ma_length=20, **kwargs):
        """
        自适应阈值+趋势过滤策略

        Args:
            price_data: 价格数据
            window: 窗口大小
            std_multiplier: 标准差倍数
            ma_length: 移动平均线长度
            **kwargs: 其他参数

        Returns:
            tuple: (收益率序列, 策略信息字典)
        """
        returns = price_data.pct_change().dropna()

        # 趋势过滤
        ma = price_data.rolling(window=ma_length).mean()
        trend = price_data > ma

        # 自适应阈值
        rolling_mean = returns.rolling(window=window).mean()
        rolling_std = returns.rolling(window=window).std()
        upper_band = rolling_mean + std_multiplier * rolling_std
        lower_band = rolling_mean - std_multiplier * rolling_std

        # 生成信号（对齐索引）
        signals = pd.Series(0, index=returns.index)
        long_signals = (returns < lower_band) & trend
        short_signals = (returns > upper_band) & (~trend)

        signals[long_signals] = 1
        signals[short_signals] = -1
        strategy_returns = signals * returns

        strategy_info = {
            'strategy_name': 'adaptive_trend',
            'parameters': {'window': window, 'std_multiplier': std_multiplier, 'ma_length': ma_length},
            'description': f'自适应趋势策略：window={window}, std_multiplier={std_multiplier}, ma_length={ma_length}'
        }

        return strategy_returns.dropna(), strategy_info

    @staticmethod
    def get_strategy_function(strategy_name):
        """
        获取策略函数

        Args:
            strategy_name: 策略名称

        Returns:
            function: 策略函数
        """
        strategy_map = {
            'simple_reversal': StrategyFunctions.simple_reversal_strategy,
            'fixed_threshold': StrategyFunctions.fixed_threshold_strategy,
            'adaptive_threshold': StrategyFunctions.adaptive_threshold_strategy,
            'adaptive_trend': StrategyFunctions.adaptive_trend_strategy
        }

        if strategy_name not in strategy_map:
            raise ValueError(f"未知的策略: {strategy_name}. 可用策略: {list(strategy_map.keys())}")

        return strategy_map[strategy_name]

    @staticmethod
    def get_strategy_parameters(strategy_name):
        """
        获取策略的参数信息

        Args:
            strategy_name: 策略名称

        Returns:
            dict: 参数信息
        """
        parameter_info = {
            'simple_reversal': {
                'parameters': {},
                'description': '简单反转策略，无需参数',
                'optimization_method': 'baseline'  # 基准方法，无需优化
            },
            'fixed_threshold': {
                'parameters': {
                    'threshold': {
                        'type': 'float',
                        'range': [0.001, 0.01],
                        'default': 0.002,
                        'description': '阈值参数'
                    }
                },
                'description': '固定阈值策略',
                'optimization_method': 'grid_search'  # 网格搜索
            },
            'adaptive_threshold': {
                'parameters': {
                    'window': {
                        'type': 'int',
                        'range': [3, 20],
                        'default': 7,
                        'description': '窗口大小'
                    },
                    'std_multiplier': {
                        'type': 'float',
                        'range': [0.5, 3.0],
                        'default': 1.5,
                        'description': '标准差倍数'
                    }
                },
                'description': '自适应阈值策略',
                'optimization_method': 'ml_sampling'  # 机器学习采样
            },
            'adaptive_trend': {
                'parameters': {
                    'window': {
                        'type': 'int',
                        'range': [3, 20],
                        'default': 7,
                        'description': '窗口大小'
                    },
                    'std_multiplier': {
                        'type': 'float',
                        'range': [0.5, 3.0],
                        'default': 1.5,
                        'description': '标准差倍数'
                    },
                    'ma_length': {
                        'type': 'int',
                        'range': [5, 30],
                        'default': 20,
                        'description': '移动平均线长度'
                    }
                },
                'description': '自适应趋势策略',
                'optimization_method': 'ml_sampling'  # 机器学习采样
            }
        }

        return parameter_info.get(strategy_name, {})

    @staticmethod
    def validate_parameters(strategy_name, parameters):
        """
        验证策略参数

        Args:
            strategy_name: 策略名称
            parameters: 参数字典

        Returns:
            bool: 参数是否有效
        """
        param_info = StrategyFunctions.get_strategy_parameters(strategy_name)

        if not param_info:
            return False

        for param_name, param_value in parameters.items():
            if param_name not in param_info['parameters']:
                return False

            param_config = param_info['parameters'][param_name]
            min_val, max_val = param_config['range']

            if not (min_val <= param_value <= max_val):
                return False

        return True