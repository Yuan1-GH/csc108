#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
期货价格走势可视化项目主程序
整合数据读取、用户交互和可视化功能
"""

import os
import sys
from datetime import datetime
import traceback

# 导入自定义模块
from data_loader import FuturesDataLoader
from user_interface import UserInterface
from visualization import FuturesVisualizer

class FuturesAnalysisApp:
    """期货分析应用主类"""
    
    def __init__(self):
        """初始化应用"""
        self.data_loader = FuturesDataLoader()
        self.ui = UserInterface()
        self.visualizer = FuturesVisualizer()
        self.data = None
        
    def initialize_data(self):
        """初始化数据"""
        try:
            self.ui.show_progress("正在加载期货数据...")
            
            # 加载数据
            success = self.data_loader.load_data()
            
            if not success:
                self.ui.show_error("数据加载失败")
                return False
            
            # 获取加载的数据
            self.data = self.data_loader.data
            
            if self.data is None or self.data.empty:
                self.ui.show_error("数据为空")
                return False
            
            # 显示数据基本信息
            total_records = len(self.data)
            products = self.data['Product'].unique()
            date_range = f"{self.data['DateIndex'].min().strftime('%Y-%m-%d')} 至 {self.data['DateIndex'].max().strftime('%Y-%m-%d')}"
            
            print(f"\n📊 数据加载成功!")
            print(f"   总记录数: {total_records:,}")
            print(f"   期货产品: {', '.join(products)}")
            print(f"   时间范围: {date_range}")
            
            return True
            
        except Exception as e:
            self.ui.show_error(f"数据初始化失败: {e}")
            return False
    
    def get_user_choices(self):
        """获取用户选择"""
        try:
            # 获取产品选择
            product = self.ui.get_product_choice()
            if not product:
                return None, None
            
            # 检查产品数据是否存在
            product_data = self.data_loader.get_product_data(product)
            if product_data.empty:
                self.ui.show_error(f"未找到产品 {product} 的数据")
                return None, None
            
            # 显示产品数据信息
            records_count = len(product_data)
            date_range = f"{product_data['DateIndex'].min().strftime('%Y-%m-%d')} 至 {product_data['DateIndex'].max().strftime('%Y-%m-%d')}"
            print(f"\n📈 {product} 数据信息:")
            print(f"   记录数: {records_count:,}")
            print(f"   时间范围: {date_range}")
            
            # 获取可视化选项
            options = self.ui.get_visualization_options()
            
            return product, options
            
        except Exception as e:
            self.ui.show_error(f"获取用户选择失败: {e}")
            return None, None
    
    def create_visualization(self, product: str, options: dict):
        """创建可视化图表"""
        try:
            self.ui.show_progress(f"正在生成 {product} 的价格走势图...")
            
            # 获取产品数据
            product_data = self.data_loader.get_product_data(product)
            
            if product_data.empty:
                self.ui.show_error(f"产品 {product} 没有可用数据")
                return False
            
            # 创建图表
            fig = self.visualizer.create_chart(product_data, product, options)
            
            # 显示图表
            self.visualizer.show_chart(fig)
            
            # 询问是否保存图表
            try:
                save_choice = input("\n是否保存图表到HTML文件? (y/n): ").strip().lower()
                if save_choice in ['y', 'yes', '是']:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"futures_chart_{product}_{timestamp}.html"
                    self.visualizer.save_chart(fig, filename)
            except EOFError:
                # 处理输入流结束的情况
                print("\n⚠️ 输入流结束，跳过保存图表")
            except KeyboardInterrupt:
                print("\n⚠️ 用户中断，跳过保存图表")
            
            self.ui.show_success(f"{product} 价格走势图生成完成!")
            return True
            
        except Exception as e:
            self.ui.show_error(f"创建可视化失败: {e}")
            print(f"详细错误信息: {traceback.format_exc()}")
            return False
    
    def run(self):
        """运行主程序"""
        try:
            # 显示欢迎信息
            self.ui.show_welcome()
            
            # 初始化数据
            if not self.initialize_data():
                return
            
            # 主循环
            while True:
                print("\n" + "="*60)
                
                # 获取用户选择
                product, options = self.get_user_choices()
                
                if not product or not options:
                    print("❌ 获取用户选择失败")
                    if not self.ui.ask_continue():
                        break
                    continue
                
                # 确认分析
                if not self.ui.confirm_analysis(product, options):
                    if not self.ui.ask_continue():
                        break
                    continue
                
                # 创建可视化
                success = self.create_visualization(product, options)
                
                if not success:
                    print("❌ 可视化创建失败")
                
                # 询问是否继续
                if not self.ui.ask_continue():
                    break
            
            print("\n👋 感谢使用期货价格走势分析系统!")
            
        except KeyboardInterrupt:
            print("\n\n👋 程序被用户中断，感谢使用!")
        except Exception as e:
            print(f"\n❌ 程序运行出错: {e}")
            print(f"详细错误信息: {traceback.format_exc()}")

def check_dependencies():
    """检查依赖包"""
    required_packages = ['pandas', 'plotly', 'numpy']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ 缺少以下依赖包:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n请运行以下命令安装:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True

def check_data_file():
    """检查数据文件是否存在"""
    data_file = "dataVisual/csv/MinutesIdxClnd.csv"
    
    if not os.path.exists(data_file):
        print(f"❌ 数据文件不存在: {data_file}")
        print("请确保数据文件位于正确的路径")
        return False
    
    return True

def main():
    """主函数"""
    print("🚀 启动期货价格走势分析系统...")
    
    # 检查依赖
    if not check_dependencies():
        return
    
    # 检查数据文件
    if not check_data_file():
        return
    
    # 创建并运行应用
    app = FuturesAnalysisApp()
    app.run()

if __name__ == "__main__":
    main()