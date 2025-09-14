import h5py
import numpy as np
import pandas as pd

def check_hdf5_structure():
    """检查HDF5文件的数据结构"""
    file_path = "F:/HelloWorld/Code/中信建投/任务一/MinutesIdx.h5"
    
    with h5py.File(file_path, 'r') as f:
        print("=== HDF5文件结构分析 ===")
        print("\n1. 顶层结构:")
        for key in f.keys():
            item = f[key]
            if hasattr(item, 'shape'):
                print(f"  {key}: 数组, 形状={item.shape}, 类型={item.dtype}")
            else:
                print(f"  {key}: 组")
        
        print("\n2. 'd'组内部结构:")
        if 'd' in f:
            for key in f['d'].keys():
                item = f['d'][key]
                if hasattr(item, 'shape'):
                    print(f"  d/{key}: 数组, 形状={item.shape}, 类型={item.dtype}")
                else:
                    print(f"  d/{key}: 组")
        
        print("\n3. 关键数据集分析:")
        
        # 重新分析axis0和axis1
        print("\n  重新分析数据轴:")
        if 'd/axis0' in f:
            axis0_data = f['d/axis0'][:]
            print(f"  axis0数据: {axis0_data}")
            print(f"  axis0可能是: {'期货品种' if any(b'I' in x for x in axis0_data) else '其他'}")
        
        if 'd/axis1' in f:
            axis1_data = f['d/axis1'][:]
            print(f"  axis1数量: {len(axis1_data)}")
            print(f"  axis1前10个: {axis1_data[:10]}")
            print(f"  axis1后10个: {axis1_data[-10:]}")
            
            # 检查是否包含日期格式
            date_like = [x for x in axis1_data[:20] if b'20' in x]
            print(f"  axis1中类似日期的项: {date_like[:5]}")
            
            # 检查是否包含期货代码
            ic_like = [i for i, x in enumerate(axis1_data) if b'IC' in x]
            print(f"  axis1中IC相关项索引: {ic_like[:5]}")
            if ic_like:
                for idx in ic_like[:3]:
                    print(f"    索引{idx}: {axis1_data[idx]}")
        
        # 检查价格数据
        if 'd/block0_values' in f:
            raw_data = f['d/block0_values']
            print(f"\n4. 价格数据详细分析:")
            print(f"  数据形状: {raw_data.shape}")
            print(f"  数据类型: {raw_data.dtype}")
            
            # 由于是object类型，需要特殊处理
            actual_data = raw_data[0]  # 获取实际的数据数组
            print(f"  实际数据形状: {actual_data.shape}")
            print(f"  实际数据类型: {actual_data.dtype}")
            print(f"  数据统计:")
            print(f"    最小值: {actual_data.min()}")
            print(f"    最大值: {actual_data.max()}")
            print(f"    平均值: {actual_data.mean():.2f}")
            print(f"    前20个值: {actual_data[:20]}")
            
            # 根据正确的数据结构分析IC期货
            # axis0: [IF, IH, IC, IM] - 4个品种
            # axis1: 日期序列 - 3518个交易日
            # 数据应该是 (日期数 × 品种数) 的一维数组
            
            n_symbols = 4  # IF, IH, IC, IM
            n_dates = 3518
            ic_index = 2   # IC在axis0中的索引
            
            print(f"\n5. IC期货数据提取 (品种索引: {ic_index}):")
            
            if len(actual_data) >= n_dates * n_symbols:
                # 重塑数据: (日期, 品种)
                reshaped_data = actual_data[:n_dates * n_symbols].reshape(n_dates, n_symbols)
                ic_raw = reshaped_data[:, ic_index]
                
                print(f"  IC原始数据统计:")
                print(f"    数据点数: {len(ic_raw)}")
                print(f"    最小值: {ic_raw.min()}")
                print(f"    最大值: {ic_raw.max()}")
                print(f"    平均值: {ic_raw.mean():.2f}")
                print(f"    标准差: {ic_raw.std():.2f}")
                print(f"    前10个值: {ic_raw[:10]}")
                print(f"    后10个值: {ic_raw[-10:]}")
                
                # 检查数据分布
                unique_values = np.unique(ic_raw)
                print(f"    唯一值数量: {len(unique_values)}")
                if len(unique_values) <= 50:
                    print(f"    唯一值分布: {np.bincount(ic_raw.astype(int))}")
                
                # 分析数据是否需要特殊转换
                print(f"\n6. 数据转换分析:")
                print(f"  原始数据范围: 0-255 (uint8)")
                print(f"  是否需要价格转换: 是")
                
                # 尝试不同的转换方法
                print(f"\n  转换方法测试:")
                
                # 方法1: 简单线性映射到合理价格区间
                base_price = 4500
                method1 = base_price + (ic_raw / 255.0 - 0.5) * (base_price * 0.4)
                print(f"  方法1 (线性映射): 范围 {method1.min():.2f} - {method1.max():.2f}")
                
                # 方法2: 将uint8值作为价格变化的编码
                method2 = base_price * (1 + (ic_raw - 128) / 1280.0)  # ±10%变化
                print(f"  方法2 (变化编码): 范围 {method2.min():.2f} - {method2.max():.2f}")
                
                # 检查连续性
                diff1 = np.diff(method1)
                diff2 = np.diff(method2)
                print(f"  方法1日变化统计: 均值={diff1.mean():.4f}, 标准差={diff1.std():.4f}")
                print(f"  方法2日变化统计: 均值={diff2.mean():.4f}, 标准差={diff2.std():.4f}")
            else:
                print(f"  数据长度不足，期望 {n_dates * n_symbols}，实际 {len(actual_data)}")

if __name__ == "__main__":
    check_hdf5_structure()