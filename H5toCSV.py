"""
H5转CSV处理器
功能：将H5文件中的四个期货产品(IF、IH、IC、IM)合并转换为CSV格式
只生成两个文件：清洗前和清洗后的合并CSV
"""

import pandas as pd
import numpy as np
import os
import time
import json
from typing import Optional, Tuple, Dict, Any, List
import warnings
import logging
from pathlib import Path

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class H5toCSVConverter:
    """H5转CSV转换器 - 合并所有期货产品到一个CSV"""
    
    def __init__(self):
        """初始化转换器"""
        self.h5_data = None
        self.file_path = None
        self.products = ['IF', 'IH', 'IC', 'IM']
        self.combined_raw_data = None
        self.combined_cleaned_data = None
        self.results = {}
        
    def load_h5_file(self, file_path: str) -> bool:
        """
        加载H5文件
        
        Args:
            file_path: H5文件路径
            
        Returns:
            bool: 是否成功加载
        """
        try:
            logger.info(f"正在加载H5文件: {file_path}")
            
            if not os.path.exists(file_path):
                logger.error(f"文件不存在: {file_path}")
                return False
                
            # 读取H5文件
            self.h5_data = pd.read_hdf(file_path)
            self.file_path = file_path
            
            logger.info(f"成功加载H5文件，数据形状: {self.h5_data.shape}")
            logger.info(f"列名: {list(self.h5_data.columns)}")
            
            return True
            
        except Exception as e:
            logger.error(f"加载H5文件失败: {e}")
            return False
    
    def extract_all_products_data(self) -> bool:
        """
        提取所有产品数据并合并
        
        Returns:
            bool: 是否成功提取
        """
        try:
            logger.info("开始提取所有产品数据...")
            
            all_product_data = []
            
            for product_code in self.products:
                logger.info(f"正在提取{product_code}产品数据...")
                
                if product_code not in self.h5_data.columns:
                    logger.warning(f"未找到{product_code}产品数据，跳过")
                    continue
                
                product_column = self.h5_data[product_code]
                
                # 处理所有产品：每个元素都是DataFrame
                for idx, df_value in product_column.items():
                    if isinstance(df_value, pd.DataFrame) and not df_value.empty:
                        # 添加产品标识和日期索引
                        df_copy = df_value.copy()
                        df_copy['Product'] = product_code
                        df_copy['DateIndex'] = str(idx)
                        all_product_data.append(df_copy)
            
            if not all_product_data:
                logger.error("未找到任何有效的产品数据")
                return False
            
            # 合并所有产品数据
            self.combined_raw_data = pd.concat(all_product_data, ignore_index=True)
            logger.info(f"成功合并所有产品数据，总形状: {self.combined_raw_data.shape}")
            
            return True
            
        except Exception as e:
            logger.error(f"提取产品数据失败: {e}")
            return False
    
    def clean_data(self) -> bool:
        """
        清洗合并后的数据
        
        Returns:
            bool: 是否成功清洗
        """
        try:
            logger.info("开始清洗合并数据...")
            
            if self.combined_raw_data is None:
                logger.error("原始数据为空，无法清洗")
                return False
            
            # 复制原始数据
            self.combined_cleaned_data = self.combined_raw_data.copy()
            
            # 数据清洗步骤
            original_rows = len(self.combined_cleaned_data)
            
            # 1. 删除重复行
            self.combined_cleaned_data = self.combined_cleaned_data.drop_duplicates()
            after_dedup = len(self.combined_cleaned_data)
            logger.info(f"删除重复行: {original_rows - after_dedup} 行")
            
            # 2. 处理缺失值
            # 对于数值列，删除包含NaN的行
            numeric_columns = self.combined_cleaned_data.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) > 0:
                before_na = len(self.combined_cleaned_data)
                self.combined_cleaned_data = self.combined_cleaned_data.dropna(subset=numeric_columns)
                after_na = len(self.combined_cleaned_data)
                logger.info(f"删除包含缺失值的行: {before_na - after_na} 行")
            
            # 3. 数据类型优化
            for col in self.combined_cleaned_data.columns:
                if self.combined_cleaned_data[col].dtype == 'float64':
                    # 尝试转换为float32以节省内存
                    try:
                        self.combined_cleaned_data[col] = pd.to_numeric(
                            self.combined_cleaned_data[col], downcast='float'
                        )
                        logger.info(f"优化列 {col} 数据类型为 {self.combined_cleaned_data[col].dtype}")
                    except:
                        pass
            
            final_rows = len(self.combined_cleaned_data)
            retention_rate = (final_rows / original_rows) * 100 if original_rows > 0 else 0
            
            logger.info(f"数据清洗完成:")
            logger.info(f"  - 原始行数: {original_rows:,}")
            logger.info(f"  - 清洗后行数: {final_rows:,}")
            logger.info(f"  - 数据保留率: {retention_rate:.2f}%")
            
            return True
            
        except Exception as e:
            logger.error(f"数据清洗失败: {e}")
            return False
    
    def save_to_csv(self, output_dir: str) -> Tuple[bool, str, str]:
        """
        保存合并的CSV文件
        
        Args:
            output_dir: 输出目录
            
        Returns:
            Tuple[bool, str, str]: (是否成功, 原始CSV路径, 清洗后CSV路径)
        """
        try:
            # 创建输出目录
            os.makedirs(output_dir, exist_ok=True)
            
            # 获取基础文件名
            base_name = Path(self.file_path).stem
            
            # 定义文件路径
            original_csv_path = os.path.join(output_dir, f"{base_name}Orign.csv")
            cleaned_csv_path = os.path.join(output_dir, f"{base_name}Clnd.csv")
            
            # 保存原始数据
            if self.combined_raw_data is not None:
                logger.info(f"保存合并原始CSV文件: {original_csv_path}")
                self.combined_raw_data.to_csv(original_csv_path, index=False, encoding='utf-8-sig')
                
                # 获取文件大小
                original_size = os.path.getsize(original_csv_path) / (1024 * 1024)  # MB
                logger.info(f"合并原始CSV: {original_size:.2f} MB, 形状: {self.combined_raw_data.shape}")
            
            # 保存清洗后数据
            if self.combined_cleaned_data is not None:
                logger.info(f"保存合并清洗后CSV文件: {cleaned_csv_path}")
                self.combined_cleaned_data.to_csv(cleaned_csv_path, index=False, encoding='utf-8-sig')
                
                # 获取文件大小
                cleaned_size = os.path.getsize(cleaned_csv_path) / (1024 * 1024)  # MB
                logger.info(f"合并清洗后CSV: {cleaned_size:.2f} MB, 形状: {self.combined_cleaned_data.shape}")
            
            return True, original_csv_path, cleaned_csv_path
            
        except Exception as e:
            logger.error(f"保存CSV文件失败: {e}")
            return False, "", ""

    
    def print_summary(self) -> None:
        """打印处理总结"""
        print("\n" + "="*60)
        print("H5转CSV处理完成！")
        print("="*60)
        print(f"源文件: {self.file_path}")
        print()
        
        if self.combined_raw_data is not None and self.combined_cleaned_data is not None:
            original_rows = len(self.combined_raw_data)
            cleaned_rows = len(self.combined_cleaned_data)
            retention_rate = (cleaned_rows / original_rows * 100) if original_rows > 0 else 0
            
            print("✅ 合并数据处理成功")
            print(f"  - 原始数据: {original_rows:,} 行 × {len(self.combined_raw_data.columns)} 列")
            print(f"  - 清洗后数据: {cleaned_rows:,} 行 × {len(self.combined_cleaned_data.columns)} 列")
            print(f"  - 数据保留率: {retention_rate:.2f}%")
        else:
            print("❌ 数据处理失败")
        
        print("="*60)

def main():
    """主函数"""
    print("H5转CSV处理器 - 合并所有期货产品")
    print("="*60)
    
    # 配置参数
    h5_file_path = "src/MinutesIdx.h5"
    output_dir = "dataVisual/csv"
    
    # 是否使用小型测试数据集（仅处理少量数据，用于测试）
    use_test_data = False  # 设置为True使用小型测试数据集，False使用完整数据集
    
    # 检查文件是否存在
    if not os.path.exists(h5_file_path):
        logger.error(f"H5文件不存在: {h5_file_path}")
        print(f"❌ 错误：找不到文件 {h5_file_path}")
        return
    
    # 创建转换器
    converter = H5toCSVConverter()
    
    # 加载H5文件
    if not converter.load_h5_file(h5_file_path):
        logger.error("H5文件加载失败")
        return
    
    # 如果使用测试数据集，取包含IM数据的部分（从索引1158开始取100行）
    if use_test_data:
        print("⚠️  使用小型测试数据集（取包含IM数据的100行数据）")
        if isinstance(converter.h5_data, pd.DataFrame):
            # 从索引1158开始取100行，确保包含IM数据
            start_idx = 1158
            end_idx = start_idx + 100
            converter.h5_data = converter.h5_data.iloc[start_idx:end_idx].copy()
            print(f"使用数据范围: 索引 {start_idx} 到 {end_idx-1}")
    
    # 提取所有产品数据
    if not converter.extract_all_products_data():
        logger.error("产品数据提取失败")
        return
    
    # 清洗数据
    if not converter.clean_data():
        logger.error("数据清洗失败")
        return
    
    # 保存CSV文件
    success, original_csv, cleaned_csv = converter.save_to_csv(output_dir)
    if not success:
        logger.error("CSV文件保存失败")
        return
    

    
    # 打印总结
    converter.print_summary()
    
    print(f"\n生成的文件:")
    print(f"  - 原始数据CSV: {original_csv}")
    print(f"  - 清洗后CSV: {cleaned_csv}")

if __name__ == "__main__":
    main()