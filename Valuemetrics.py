#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
性能指标计算库
包含夏普比率和其他效益衡量指数的计算函数
"""

import numpy as np
import pandas as pd
from scipy import stats

class PerformanceMetrics:
    """性能指标计算类"""

    @staticmethod
    def calculate_sharpe_ratio(returns, risk_free_rate=0.02, frequency='daily', annualization_method='standard'):
        """
        计算夏普比率

        Args:
            returns: 收益率序列
            risk_free_rate: 无风险利率（默认2%）
            frequency: 数据频率 ('daily', 'minute', 'hourly', 'weekly', 'monthly')
            annualization_method: 年化方法 ('standard', 'log')

        Returns:
            float: 夏普比率
        """
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0

        # 计算超额收益
        excess_returns = returns - risk_free_rate / 252  # 日化无风险利率

        # 根据频率选择年化因子
        if frequency == 'daily':
            annual_factor = np.sqrt(252)
        elif frequency == 'minute':
            annual_factor = np.sqrt(252 * 240)  # 假设每天240分钟
        elif frequency == 'hourly':
            annual_factor = np.sqrt(252 * 6.5)  # 假设每天6.5小时
        elif frequency == 'weekly':
            annual_factor = np.sqrt(52)
        elif frequency == 'monthly':
            annual_factor = np.sqrt(12)
        else:
            annual_factor = np.sqrt(252)

        # 计算夏普比率
        mean_excess_return = np.mean(excess_returns)
        std_excess_return = np.std(excess_returns)

        if std_excess_return == 0:
            return 0.0

        sharpe_ratio = mean_excess_return / std_excess_return * annual_factor

        # 限制极端值
        sharpe_ratio = np.clip(sharpe_ratio, -10, 10)

        return sharpe_ratio

    @staticmethod
    def calculate_sortino_ratio(returns, risk_free_rate=0.02, frequency='daily'):
        """
        计算索提诺比率（只考虑下行风险）

        Args:
            returns: 收益率序列
            risk_free_rate: 无风险利率
            frequency: 数据频率

        Returns:
            float: 索提诺比率
        """
        if len(returns) == 0:
            return 0.0

        excess_returns = returns - risk_free_rate / 252

        # 只考虑下行风险（负收益）
        downside_returns = excess_returns[excess_returns < 0]
        if len(downside_returns) == 0:
            return 0.0

        downside_std = np.std(downside_returns)

        if downside_std == 0:
            return 0.0

        # 年化因子
        if frequency == 'daily':
            annual_factor = np.sqrt(252)
        elif frequency == 'minute':
            annual_factor = np.sqrt(252 * 240)
        else:
            annual_factor = np.sqrt(252)

        mean_excess_return = np.mean(excess_returns)
        sortino_ratio = mean_excess_return / downside_std * annual_factor

        return np.clip(sortino_ratio, -10, 10)

    @staticmethod
    def calculate_calmar_ratio(returns, frequency='daily'):
        """
        计算卡玛比率（年化收益率 / 最大回撤）

        Args:
            returns: 收益率序列
            frequency: 数据频率

        Returns:
            float: 卡玛比率
        """
        if len(returns) == 0:
            return 0.0

        # 计算年化收益率
        if frequency == 'daily':
            annual_factor = 252
        elif frequency == 'minute':
            annual_factor = 252 * 240
        else:
            annual_factor = 252

        cumulative_returns = (1 + returns).cumprod()
        total_return = cumulative_returns.iloc[-1] - 1
        annual_return = (1 + total_return) ** (annual_factor / len(returns)) - 1

        # 计算最大回撤
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = abs(drawdown.min())

        if max_drawdown == 0:
            return 0.0

        calmar_ratio = annual_return / max_drawdown

        return np.clip(calmar_ratio, -10, 10)

    @staticmethod
    def calculate_information_ratio(returns, benchmark_returns, frequency='daily'):
        """
        计算信息比率（相对于基准的超额收益）

        Args:
            returns: 策略收益率序列
            benchmark_returns: 基准收益率序列
            frequency: 数据频率

        Returns:
            float: 信息比率
        """
        if len(returns) == 0 or len(benchmark_returns) == 0:
            return 0.0

        # 确保长度一致
        min_len = min(len(returns), len(benchmark_returns))
        returns = returns.iloc[:min_len]
        benchmark_returns = benchmark_returns.iloc[:min_len]

        excess_returns = returns - benchmark_returns

        if len(excess_returns) == 0 or np.std(excess_returns) == 0:
            return 0.0

        # 年化因子
        if frequency == 'daily':
            annual_factor = np.sqrt(252)
        elif frequency == 'minute':
            annual_factor = np.sqrt(252 * 240)
        else:
            annual_factor = np.sqrt(252)

        information_ratio = np.mean(excess_returns) / np.std(excess_returns) * annual_factor

        return np.clip(information_ratio, -10, 10)

    @staticmethod
    def calculate_win_rate(returns):
        """
        计算胜率

        Args:
            returns: 收益率序列

        Returns:
            float: 胜率（0-1之间）
        """
        if len(returns) == 0:
            return 0.0

        win_trades = returns[returns > 0]
        total_trades = len(returns[returns != 0])

        if total_trades == 0:
            return 0.0

        return len(win_trades) / total_trades

    @staticmethod
    def calculate_profit_factor(returns):
        """
        计算盈利因子（总盈利 / 总亏损）

        Args:
            returns: 收益率序列

        Returns:
            float: 盈利因子
        """
        if len(returns) == 0:
            return 0.0

        winning_trades = returns[returns > 0]
        losing_trades = returns[returns < 0]

        total_profit = winning_trades.sum() if len(winning_trades) > 0 else 0
        total_loss = abs(losing_trades.sum()) if len(losing_trades) > 0 else 0

        if total_loss == 0:
            return float('inf') if total_profit > 0 else 0.0

        return total_profit / total_loss

    @staticmethod
    def calculate_max_drawdown(returns):
        """
        计算最大回撤

        Args:
            returns: 收益率序列

        Returns:
            tuple: (最大回撤, 回撤持续时间)
        """
        if len(returns) == 0:
            return 0.0, 0

        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max

        max_drawdown = abs(drawdown.min())

        # 计算回撤持续时间
        is_drawdown = drawdown < 0
        drawdown_periods = []
        start_idx = None

        for i, is_dd in enumerate(is_drawdown):
            if is_dd and start_idx is None:
                start_idx = i
            elif not is_dd and start_idx is not None:
                drawdown_periods.append(i - start_idx)
                start_idx = None

        if start_idx is not None:
            drawdown_periods.append(len(is_drawdown) - start_idx)

        max_duration = max(drawdown_periods) if drawdown_periods else 0

        return max_drawdown, max_duration

    @staticmethod
    def calculate_all_metrics(returns, benchmark_returns=None, risk_free_rate=0.02, frequency='daily'):
        """
        计算所有性能指标

        Args:
            returns: 收益率序列
            benchmark_returns: 基准收益率序列（可选）
            risk_free_rate: 无风险利率
            frequency: 数据频率

        Returns:
            dict: 包含所有性能指标的字典
        """
        metrics = {}

        # 基础指标
        metrics['sharpe_ratio'] = PerformanceMetrics.calculate_sharpe_ratio(returns, risk_free_rate, frequency)
        metrics['sortino_ratio'] = PerformanceMetrics.calculate_sortino_ratio(returns, risk_free_rate, frequency)
        metrics['calmar_ratio'] = PerformanceMetrics.calculate_calmar_ratio(returns, frequency)
        metrics['win_rate'] = PerformanceMetrics.calculate_win_rate(returns)
        metrics['profit_factor'] = PerformanceMetrics.calculate_profit_factor(returns)

        # 回撤指标
        max_drawdown, max_duration = PerformanceMetrics.calculate_max_drawdown(returns)
        metrics['max_drawdown'] = max_drawdown
        metrics['max_drawdown_duration'] = max_duration

        # 基准相关指标
        if benchmark_returns is not None:
            metrics['information_ratio'] = PerformanceMetrics.calculate_information_ratio(returns, benchmark_returns, frequency)

            # 计算beta和alpha
            min_len = min(len(returns), len(benchmark_returns))
            returns_aligned = returns.iloc[:min_len]
            benchmark_aligned = benchmark_returns.iloc[:min_len]

            if len(returns_aligned) > 0 and np.std(benchmark_aligned) > 0:
                beta = np.cov(returns_aligned, benchmark_aligned)[0, 1] / np.var(benchmark_aligned)
                alpha = np.mean(returns_aligned) - beta * np.mean(benchmark_aligned)
                metrics['beta'] = beta
                metrics['alpha'] = alpha * 252  # 年化alpha

        # 统计指标
        metrics['total_return'] = (1 + returns).prod() - 1
        metrics['annual_return'] = (1 + metrics['total_return']) ** (252 / len(returns)) - 1
        metrics['volatility'] = np.std(returns) * np.sqrt(252)
        metrics['skewness'] = stats.skew(returns.dropna())
        metrics['kurtosis'] = stats.kurtosis(returns.dropna())

        return metrics

    @staticmethod
    def get_metric_description(metric_name):
        """
        获取指标的描述

        Args:
            metric_name: 指标名称

        Returns:
            dict: 指标描述
        """
        descriptions = {
            'sharpe_ratio': {
                'name': '夏普比率',
                'description': '风险调整后收益，越高越好',
                'range': '(-10, 10)',
                'preference': 'higher'
            },
            'sortino_ratio': {
                'name': '索提诺比率',
                'description': '下行风险调整后收益，越高越好',
                'range': '(-10, 10)',
                'preference': 'higher'
            },
            'calmar_ratio': {
                'name': '卡玛比率',
                'description': '年化收益率与最大回撤的比率，越高越好',
                'range': '(-10, 10)',
                'preference': 'higher'
            },
            'information_ratio': {
                'name': '信息比率',
                'description': '相对于基准的超额收益，越高越好',
                'range': '(-10, 10)',
                'preference': 'higher'
            },
            'win_rate': {
                'name': '胜率',
                'description': '盈利交易占比，越高越好',
                'range': '(0, 1)',
                'preference': 'higher'
            },
            'profit_factor': {
                'name': '盈利因子',
                'description': '总盈利与总亏损的比率，越高越好',
                'range': '(0, ∞)',
                'preference': 'higher'
            },
            'max_drawdown': {
                'name': '最大回撤',
                'description': '最大损失幅度，越低越好',
                'range': '(0, 1)',
                'preference': 'lower'
            },
            'total_return': {
                'name': '总收益率',
                'description': '策略总收益',
                'range': '(-∞, ∞)',
                'preference': 'higher'
            },
            'annual_return': {
                'name': '年化收益率',
                'description': '策略年化收益率',
                'range': '(-∞, ∞)',
                'preference': 'higher'
            },
            'volatility': {
                'name': '年化波动率',
                'description': '策略波动率，越低越好',
                'range': '(0, ∞)',
                'preference': 'lower'
            }
        }

        return descriptions.get(metric_name, {
            'name': metric_name,
            'description': '未知指标',
            'range': 'N/A',
            'preference': 'higher'
        })

    @staticmethod
    def composite_score(metrics, weights=None):
        """
        计算综合评分（多个指标的加权平均）

        Args:
            metrics: 指标字典
            weights: 权重字典（可选）

        Returns:
            float: 综合评分（0-100）
        """
        if weights is None:
            # 默认权重
            weights = {
                'sharpe_ratio': 0.3,
                'sortino_ratio': 0.2,
                'calmar_ratio': 0.2,
                'win_rate': 0.1,
                'profit_factor': 0.1,
                'max_drawdown': 0.1
            }

        total_score = 0
        total_weight = 0

        for metric_name, weight in weights.items():
            if metric_name in metrics:
                value = metrics[metric_name]
                metric_desc = PerformanceMetrics.get_metric_description(metric_name)

                # 归一化到0-1范围
                if metric_desc['preference'] == 'higher':
                    # 越高越好的指标
                    if metric_name in ['sharpe_ratio', 'sortino_ratio', 'calmar_ratio', 'information_ratio']:
                        normalized = np.clip((value + 10) / 20, 0, 1)  # (-10, 10) -> (0, 1)
                    elif metric_name in ['win_rate', 'profit_factor']:
                        normalized = np.clip(value / 2, 0, 1)  # 假设2为很好的值
                    else:
                        normalized = np.clip(value / 10, 0, 1)  # 其他指标
                else:
                    # 越低越好的指标
                    if metric_name == 'max_drawdown':
                        normalized = np.clip((1 - value) / 1, 0, 1)  # (0, 1) -> (1, 0)
                    else:
                        normalized = np.clip(1 / (1 + abs(value)), 0, 1)

                total_score += normalized * weight
                total_weight += weight

        if total_weight == 0:
            return 0.0

        return (total_score / total_weight) * 100