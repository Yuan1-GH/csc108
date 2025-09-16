"""
H5转CSV处理器
功能：将H5文件转换为CSV格式，生成原始版本和清洗后版本两个文件
"""

import pandas as pd
import numpy as np
import os
from typing import Optional, Tuple, Dict, Any
import warnings
import logging
from pathlib import Path

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class H5ToCSVProcessor:
    """H5转CSV处理器，生成原始和清洗后的CSV文件"""
    
    def __init__(self):
        self.raw_data = None
        self.cleaned_data = None
        self.file_path = None
        
    def load_h5_data(self, file_path: str) -> bool:
        """
        加载H5文件数据
        
        Args:
            file_path: H5文件路径
            
        Returns:
            bool: 是否成功加载
        """
        try:
            if not os.path.exists(file_path):
                logger.error(f"文件不存在: {file_path}")
                return False
            
            logger.info(f"正在加载H5文件: {file_path}")
            
            # 尝试读取H5文件
            self.raw_data = pd.read_hdf(file_path)
            self.file_path = file_path
            
            logger.info(f"成功加载H5数据，形状: {self.raw_data.shape}")
            logger.info(f"列名: {list(self.raw_data.columns)}")
            
            return True
            
        except Exception as e:
            logger.error(f"加载H5文件失败: {e}")
            return False
    
    def analyze_data_quality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        分析数据质量
        
        Args:
            data: 要分析的数据
            
        Returns:
            Dict: 数据质量报告
        """
        report = {
            "基本信息": {
                "数据形状": data.shape,
                "列数": len(data.columns),
                "行数": len(data),
                "内存使用": f"{data.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB"
            },
            "缺失值统计": {},
            "数据类型": {},
            "异常值统计": {}
        }
        
        # 缺失值统计
        missing_stats = data.isnull().sum()
        report["缺失值统计"] = {
            col: {
                "缺失数量": int(missing_stats[col]),
                "缺失比例": f"{missing_stats[col] / len(data) * 100:.2f}%"
            }
            for col in data.columns if missing_stats[col] > 0
        }
        
        # 数据类型
        report["数据类型"] = {col: str(dtype) for col, dtype in data.dtypes.items()}
        
        # 数值列的异常值统计
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if len(data[col].dropna()) > 0:
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = ((data[col] < lower_bound) | (data[col] > upper_bound)).sum()
                if outliers > 0:
                    report["异常值统计"][col] = {
                        "异常值数量": int(outliers),
                        "异常值比例": f"{outliers / len(data) * 100:.2f}%",
                        "下界": lower_bound,
                        "上界": upper_bound
                    }
        
        return report
    
    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        数据清洗
        
        Args:
            data: 原始数据
            
        Returns:
            pd.DataFrame: 清洗后的数据
        """
        logger.info("开始数据清洗...")
        
        cleaned_data = data.copy()
        cleaning_log = []
        
        # 1. 处理缺失值
        numeric_columns = cleaned_data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            missing_count = cleaned_data[col].isnull().sum()
            if missing_count > 0:
                # 使用前向填充和后向填充
                cleaned_data[col] = cleaned_data[col].fillna(method='ffill').fillna(method='bfill')
                
                # 如果还有缺失值，用均值填充
                if cleaned_data[col].isnull().sum() > 0:
                    mean_value = cleaned_data[col].mean()
                    cleaned_data[col] = cleaned_data[col].fillna(mean_value)
                
                cleaning_log.append(f"处理列 {col} 的 {missing_count} 个缺失值")
        
        # 处理非数值列的缺失值
        non_numeric_columns = cleaned_data.select_dtypes(exclude=[np.number]).columns
        for col in non_numeric_columns:
            missing_count = cleaned_data[col].isnull().sum()
            if missing_count > 0:
                # 用前向填充处理
                cleaned_data[col] = cleaned_data[col].fillna(method='ffill').fillna('Unknown')
                cleaning_log.append(f"处理列 {col} 的 {missing_count} 个缺失值")
        
        # 2. 处理异常值（使用IQR方法）
        for col in numeric_columns:
            if len(cleaned_data[col].dropna()) > 0:
                Q1 = cleaned_data[col].quantile(0.25)
                Q3 = cleaned_data[col].quantile(0.75)
                IQR = Q3 - Q1
                
                if IQR > 0:  # 避免除零错误
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    # 统计异常值数量
                    outliers_mask = (cleaned_data[col] < lower_bound) | (cleaned_data[col] > upper_bound)
                    outliers_count = outliers_mask.sum()
                    
                    if outliers_count > 0:
                        # 用边界值替换异常值
                        cleaned_data.loc[cleaned_data[col] < lower_bound, col] = lower_bound
                        cleaned_data.loc[cleaned_data[col] > upper_bound, col] = upper_bound
                        cleaning_log.append(f"处理列 {col} 的 {outliers_count} 个异常值")
        
        # 3. 去除重复行
        duplicate_count = cleaned_data.duplicated().sum()
        if duplicate_count > 0:
            cleaned_data = cleaned_data.drop_duplicates()
            cleaning_log.append(f"删除 {duplicate_count} 个重复行")
        
        # 4. 数据类型优化
        for col in numeric_columns:
            if cleaned_data[col].dtype == 'float64':
                # 尝试转换为float32以节省内存
                if cleaned_data[col].min() >= np.finfo(np.float32).min and cleaned_data[col].max() <= np.finfo(np.float32).max:
                    cleaned_data[col] = cleaned_data[col].astype('float32')
                    cleaning_log.append(f"优化列 {col} 数据类型为 float32")
        
        logger.info("数据清洗完成")
        for log in cleaning_log:
            logger.info(f"  - {log}")
        
        return cleaned_data
    
    def save_to_csv(self, data: pd.DataFrame, filename: str, description: str = "") -> bool:
        """
        保存数据为CSV文件
        
        Args:
            data: 要保存的数据
            filename: 文件名
            description: 文件描述
            
        Returns:
            bool: 是否成功保存
        """
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
            
            # 保存CSV文件
            data.to_csv(filename, index=False, encoding='utf-8-sig')
            
            file_size = os.path.getsize(filename) / 1024 / 1024  # MB
            logger.info(f"成功保存{description}: {filename}")
            logger.info(f"  - 文件大小: {file_size:.2f} MB")
            logger.info(f"  - 数据形状: {data.shape}")
            
            return True
            
        except Exception as e:
            logger.error(f"保存{description}失败: {e}")
            return False
    
    def generate_summary_report(self, original_data: pd.DataFrame, cleaned_data: pd.DataFrame) -> Dict[str, Any]:
        """
        生成处理总结报告
        
        Args:
            original_data: 原始数据
            cleaned_data: 清洗后数据
            
        Returns:
            Dict: 总结报告
        """
        report = {
            "处理时间": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "源文件": self.file_path,
            "原始数据": {
                "形状": original_data.shape,
                "缺失值总数": int(original_data.isnull().sum().sum()),
                "重复行数": int(original_data.duplicated().sum())
            },
            "清洗后数据": {
                "形状": cleaned_data.shape,
                "缺失值总数": int(cleaned_data.isnull().sum().sum()),
                "重复行数": int(cleaned_data.duplicated().sum())
            },
            "清洗效果": {
                "删除的缺失值": int(original_data.isnull().sum().sum() - cleaned_data.isnull().sum().sum()),
                "删除的重复行": int(original_data.duplicated().sum() - cleaned_data.duplicated().sum()),
                "数据保留率": f"{len(cleaned_data) / len(original_data) * 100:.2f}%"
            }
        }
        
        return report
    
    def process_h5_to_csv(self, h5_file_path: str, output_dir: str = "output") -> bool:
        """
        完整的H5转CSV处理流程
        
        Args:
            h5_file_path: H5文件路径
            output_dir: 输出目录
            
        Returns:
            bool: 是否成功处理
        """
        try:
            # 1. 加载H5数据
            if not self.load_h5_data(h5_file_path):
                return False
            
            # 2. 生成输出文件名
            base_name = Path(h5_file_path).stem
            os.makedirs(output_dir, exist_ok=True)
            
            original_csv = os.path.join(output_dir, f"{base_name}_original.csv")
            cleaned_csv = os.path.join(output_dir, f"{base_name}_cleaned.csv")
            
            # 3. 分析原始数据质量
            logger.info("分析原始数据质量...")
            original_quality = self.analyze_data_quality(self.raw_data)
            
            # 4. 保存原始CSV
            logger.info("保存原始CSV文件...")
            if not self.save_to_csv(self.raw_data, original_csv, "原始数据CSV"):
                return False
            
            # 5. 数据清洗
            logger.info("执行数据清洗...")
            self.cleaned_data = self.clean_data(self.raw_data)
            
            # 6. 保存清洗后的CSV
            logger.info("保存清洗后CSV文件...")
            if not self.save_to_csv(self.cleaned_data, cleaned_csv, "清洗后数据CSV"):
                return False
            
            # 7. 分析清洗后数据质量
            logger.info("分析清洗后数据质量...")
            cleaned_quality = self.analyze_data_quality(self.cleaned_data)
            
            # 8. 生成总结报告
            summary_report = self.generate_summary_report(self.raw_data, self.cleaned_data)
            
            # 9. 保存报告
            report_file = os.path.join(output_dir, f"{base_name}_processing_report.json")
            full_report = {
                "summary": summary_report,
                "original_data_quality": original_quality,
                "cleaned_data_quality": cleaned_quality
            }
            
            import json
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(full_report, f, ensure_ascii=False, indent=2, default=str)
            
            logger.info(f"处理报告已保存: {report_file}")
            
            # 10. 打印总结
            print("\n" + "="*60)
            print("H5转CSV处理完成！")
            print("="*60)
            print(f"源文件: {h5_file_path}")
            print(f"输出目录: {output_dir}")
            print(f"\n生成的文件:")
            print(f"  📄 原始数据CSV: {original_csv}")
            print(f"  🧹 清洗后CSV: {cleaned_csv}")
            print(f"  📊 处理报告: {report_file}")
            
            print(f"\n数据统计:")
            print(f"  原始数据: {self.raw_data.shape[0]:,} 行 × {self.raw_data.shape[1]} 列")
            print(f"  清洗后数据: {self.cleaned_data.shape[0]:,} 行 × {self.cleaned_data.shape[1]} 列")
            print(f"  数据保留率: {len(self.cleaned_data) / len(self.raw_data) * 100:.2f}%")
            
            if summary_report["清洗效果"]["删除的缺失值"] > 0:
                print(f"  处理缺失值: {summary_report['清洗效果']['删除的缺失值']:,} 个")
            if summary_report["清洗效果"]["删除的重复行"] > 0:
                print(f"  删除重复行: {summary_report['清洗效果']['删除的重复行']:,} 行")
            
            print("="*60)
            
            return True
            
        except Exception as e:
            logger.error(f"处理过程中出现错误: {e}")
            return False

def main():
    """主函数 - 演示使用"""
    print("H5转CSV处理器")
    print("="*50)
    
    # 创建处理器实例
    processor = H5ToCSVProcessor()
    
    # 示例：处理H5文件
    h5_files = [
        "中信建投/任务一/MinutesIdx.h5",  # 如果存在的话
        # 可以添加更多H5文件路径
    ]
    
    success_count = 0
    
    for h5_file in h5_files:
        if os.path.exists(h5_file):
            print(f"\n正在处理: {h5_file}")
            if processor.process_h5_to_csv(h5_file, "converted_data"):
                success_count += 1
                print(f"✅ {h5_file} 处理成功")
            else:
                print(f"❌ {h5_file} 处理失败")
        else:
            print(f"⚠️  文件不存在: {h5_file}")
    
    if success_count == 0:
        print("\n没有找到可处理的H5文件")
        
       
if __name__ == "__main__":
    main()