#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
期货数据读取模块
负责从CSV文件中读取和处理期货分钟级数据
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os

class FuturesDataLoader:
    """期货数据加载器"""
    
    def __init__(self, csv_path="dataVisual/csv/MinutesIdxClnd.csv"):
        """
        初始化数据加载器
        
        Args:
            csv_path (str): CSV文件路径
        """
        self.csv_path = csv_path
        self.data = None
        self.available_products = []
        
    def load_data(self):
        """
        加载CSV数据
        
        Returns:
            bool: 加载是否成功
        """
        try:
            if not os.path.exists(self.csv_path):
                print(f"错误: 文件 {self.csv_path} 不存在")
                return False
            
            print(f"正在加载数据文件: {self.csv_path}")
            self.data = pd.read_csv(self.csv_path)
            
            # 数据预处理
            self._preprocess_data()
            
            # 获取可用的期货产品
            self.available_products = sorted(self.data['Product'].unique().tolist())
            
            print(f"数据加载成功!")
            print(f"数据形状: {self.data.shape}")
            print(f"可用期货产品: {self.available_products}")
            print(f"时间范围: {self.data['DateIndex'].min()} 到 {self.data['DateIndex'].max()}")
            
            return True
            
        except Exception as e:
            print(f"加载数据时发生错误: {e}")
            return False
    
    def _preprocess_data(self):
        """数据预处理"""
        # 转换日期格式
        self.data['DateIndex'] = pd.to_datetime(self.data['DateIndex'])
        
        # 创建完整的时间戳（日期 + 时间）
        self.data['DateTime'] = pd.to_datetime(
            self.data['DateIndex'].dt.strftime('%Y-%m-%d') + ' ' + 
            self.data['Time'].astype(str).str.zfill(4).str[:2] + ':' + 
            self.data['Time'].astype(str).str.zfill(4).str[2:]
        )
        
        # 确保数值列为浮点型
        numeric_columns = ['Close', 'High', 'Low', 'Open', 'Amount', 'Volume']
        for col in numeric_columns:
            self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
        
        # 按产品和时间排序
        self.data = self.data.sort_values(['Product', 'DateTime']).reset_index(drop=True)
    
    def get_product_data(self, product_code):
        """
        获取指定期货产品的数据
        
        Args:
            product_code (str): 期货产品代码 (IF, IH, IC, IM)
            
        Returns:
            pd.DataFrame: 指定产品的数据，如果产品不存在则返回None
        """
        if self.data is None:
            print("错误: 请先加载数据")
            return None
        
        if product_code not in self.available_products:
            print(f"错误: 产品代码 '{product_code}' 不存在")
            print(f"可用产品: {self.available_products}")
            return None
        
        product_data = self.data[self.data['Product'] == product_code].copy()
        
        print(f"\n{product_code} 产品数据统计:")
        print(f"数据行数: {len(product_data):,}")
        print(f"时间范围: {product_data['DateIndex'].min()} 到 {product_data['DateIndex'].max()}")
        print(f"交易日数: {product_data['DateIndex'].dt.date.nunique()}")
        print(f"价格范围: {product_data['Low'].min():.2f} - {product_data['High'].max():.2f}")
        
        return product_data
    
    def get_available_products(self):
        """
        获取可用的期货产品列表
        
        Returns:
            list: 期货产品代码列表
        """
        return self.available_products
    
    def get_data_summary(self):
        """
        获取数据概览
        
        Returns:
            dict: 数据统计信息
        """
        if self.data is None:
            return None
        
        summary = {}
        for product in self.available_products:
            product_data = self.data[self.data['Product'] == product]
            summary[product] = {
                'rows': len(product_data),
                'start_date': product_data['DateIndex'].min(),
                'end_date': product_data['DateIndex'].max(),
                'trading_days': product_data['DateIndex'].dt.date.nunique(),
                'price_range': {
                    'min': product_data['Low'].min(),
                    'max': product_data['High'].max()
                },
                'volume_range': {
                    'min': product_data['Volume'].min(),
                    'max': product_data['Volume'].max()
                }
            }
        
        return summary

def test_data_loader():
    """测试数据加载器功能"""
    print("=== 测试期货数据加载器 ===")
    
    # 创建数据加载器
    loader = FuturesDataLoader()
    
    # 加载数据
    if not loader.load_data():
        return
    
    # 获取数据概览
    summary = loader.get_data_summary()
    print("\n=== 数据概览 ===")
    for product, info in summary.items():
        print(f"\n{product}:")
        print(f"  数据行数: {info['rows']:,}")
        print(f"  时间范围: {info['start_date']} 到 {info['end_date']}")
        print(f"  交易日数: {info['trading_days']}")
        print(f"  价格范围: {info['price_range']['min']:.2f} - {info['price_range']['max']:.2f}")
    
    # 测试获取单个产品数据
    print("\n=== 测试获取IF产品数据 ===")
    if_data = loader.get_product_data('IF')
    if if_data is not None:
        print(f"IF数据前5行:")
        print(if_data[['DateTime', 'Open', 'High', 'Low', 'Close', 'Volume']].head())

if __name__ == "__main__":
    test_data_loader()