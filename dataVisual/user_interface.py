#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用户界面模块
负责处理用户输入和选择期货产品
"""

import sys
from typing import List, Optional

class UserInterface:
    """用户界面类"""
    
    def __init__(self):
        """初始化用户界面"""
        self.available_products = ['IF', 'IH', 'IC', 'IM']
        self.product_names = {
            'IF': 'IF - 沪深300股指期货',
            'IH': 'IH - 上证50股指期货', 
            'IC': 'IC - 中证500股指期货',
            'IM': 'IM - 中证1000股指期货'
        }
    
    def show_welcome(self):
        """显示欢迎信息"""
        print("=" * 60)
        print("🚀 期货价格走势可视化系统")
        print("=" * 60)
        print("本系统支持四种股指期货产品的价格走势分析:")
        for i, product in enumerate(self.available_products, 1):
            print(f"  {i}. {self.product_names[product]}")
        print("=" * 60)
    
    def get_product_choice(self) -> Optional[str]:
        """
        获取用户选择的期货产品
        
        Returns:
            str: 选择的产品代码，如果用户选择退出则返回None
        """
        while True:
            print("\n请选择要分析的期货产品:")
            for i, product in enumerate(self.available_products, 1):
                print(f"  {i}. {self.product_names[product]}")
            print("  0. 退出程序")
            
            try:
                choice = input("\n请输入选项编号 (0-4): ").strip()
                
                if choice == '0':
                    print("感谢使用，再见! 👋")
                    return None
                
                choice_num = int(choice)
                if 1 <= choice_num <= 4:
                    selected_product = self.available_products[choice_num - 1]
                    print(f"\n✅ 您选择了: {self.product_names[selected_product]}")
                    return selected_product
                else:
                    print("❌ 无效选择，请输入 0-4 之间的数字")
                    
            except ValueError:
                print("❌ 请输入有效的数字")
            except KeyboardInterrupt:
                print("\n\n程序被用户中断，再见! 👋")
                return None
    
    def get_visualization_options(self) -> dict:
        """
        获取可视化选项
        
        Returns:
            dict: 可视化配置选项
        """
        options = {
            'chart_type': 'candlestick',  # 默认K线图
            'time_range': 'all',          # 默认全部时间
            'indicators': []              # 技术指标
        }
        
        print("\n📊 可视化选项配置:")
        
        # 选择图表类型
        print("\n1. 选择图表类型:")
        print("  1. K线图 (推荐)")
        print("  2. 线形图")
        print("  3. 面积图")
        
        try:
            chart_choice = input("请选择图表类型 (1-3, 默认1): ").strip()
            if chart_choice == '2':
                options['chart_type'] = 'line'
            elif chart_choice == '3':
                options['chart_type'] = 'area'
            else:
                options['chart_type'] = 'candlestick'
        except:
            pass
        
        # 选择时间范围
        print("\n2. 选择时间范围:")
        print("  1. 全部数据")
        print("  2. 最近1年")
        print("  3. 最近6个月")
        print("  4. 最近3个月")
        print("  5. 最近1个月")
        
        try:
            time_choice = input("请选择时间范围 (1-5, 默认1): ").strip()
            time_map = {
                '2': '1Y',
                '3': '6M', 
                '4': '3M',
                '5': '1M'
            }
            options['time_range'] = time_map.get(time_choice, 'all')
        except:
            pass
        
        # 选择技术指标
        print("\n3. 选择技术指标 (可多选):")
        print("  1. 移动平均线 (MA)")
        print("  2. 成交量")
        print("  3. 布林带")
        print("  4. 不添加指标")
        
        try:
            indicators_input = input("请输入指标编号，用逗号分隔 (如: 1,2 或直接回车跳过): ").strip()
            if indicators_input:
                indicator_choices = [x.strip() for x in indicators_input.split(',')]
                for choice in indicator_choices:
                    if choice == '1':
                        options['indicators'].append('ma')
                    elif choice == '2':
                        options['indicators'].append('volume')
                    elif choice == '3':
                        options['indicators'].append('bollinger')
        except:
            pass
        
        return options
    
    def confirm_analysis(self, product: str, options: dict) -> bool:
        """
        确认分析配置
        
        Args:
            product (str): 选择的产品
            options (dict): 可视化选项
            
        Returns:
            bool: 是否确认继续
        """
        print("\n" + "=" * 50)
        print("📋 分析配置确认:")
        print(f"  期货产品: {self.product_names[product]}")
        
        chart_type_names = {
            'candlestick': 'K线图',
            'line': '线形图', 
            'area': '面积图'
        }
        print(f"  图表类型: {chart_type_names.get(options['chart_type'], 'K线图')}")
        
        time_range_names = {
            'all': '全部数据',
            '1Y': '最近1年',
            '6M': '最近6个月',
            '3M': '最近3个月', 
            '1M': '最近1个月'
        }
        print(f"  时间范围: {time_range_names.get(options['time_range'], '全部数据')}")
        
        if options['indicators']:
            indicator_names = {
                'ma': '移动平均线',
                'volume': '成交量',
                'bollinger': '布林带'
            }
            indicators_str = ', '.join([indicator_names.get(ind, ind) for ind in options['indicators']])
            print(f"  技术指标: {indicators_str}")
        else:
            print("  技术指标: 无")
        
        print("=" * 50)
        
        try:
            confirm = input("\n确认开始分析? (y/n, 默认y): ").strip().lower()
            return confirm != 'n'
        except:
            return True
    
    def show_progress(self, message: str):
        """
        显示进度信息
        
        Args:
            message (str): 进度消息
        """
        print(f"⏳ {message}")
    
    def show_success(self, message: str):
        """
        显示成功信息
        
        Args:
            message (str): 成功消息
        """
        print(f"✅ {message}")
    
    def show_error(self, message: str):
        """
        显示错误信息
        
        Args:
            message (str): 错误消息
        """
        print(f"❌ {message}")
    
    def ask_continue(self) -> bool:
        """
        询问是否继续分析其他产品
        
        Returns:
            bool: 是否继续
        """
        try:
            choice = input("\n是否分析其他期货产品? (y/n, 默认n): ").strip().lower()
            return choice == 'y'
        except:
            return False

def test_user_interface():
    """测试用户界面功能"""
    print("=== 测试用户界面 ===")
    
    ui = UserInterface()
    ui.display_welcome()
    
    # 模拟用户选择
    print("\n模拟用户选择 IF 产品...")
    
    # 模拟获取可视化选项
    options = {
        'chart_type': 'candlestick',
        'time_range': '3M',
        'indicators': ['ma', 'volume']
    }
    
    # 模拟确认
    print("\n模拟确认配置...")
    ui.confirm_analysis('IF', options)

if __name__ == "__main__":
    test_user_interface()