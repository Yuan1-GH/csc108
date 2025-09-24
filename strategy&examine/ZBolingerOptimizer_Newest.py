import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import warnings
import os
import json
from datetime import datetime
warnings.filterwarnings('ignore')

# 导入原始的ZBolinger策略
from ZBolinger import ZBolinger

class ZBolingerOptimizer:
    """1
    使用机器学习梯度下降方法优化ZBolinger策略参数
    
    目标：最大化策略的平均夏普比率
    特征：MA长度、STD周期、k值
    方法：随机森林 + 梯度下降优化
    """
    
    def __init__(self, filepath, product, n_samples=500):
        """
        初始化优化器
        
        参数:
        filepath: H5文件路径
        product: 产品名称
        n_samples: 采样数量，用于生成训练数据
        """
        self.filepath = filepath
        self.product = product
        self.n_samples = n_samples
        
        # 参数范围
        self.param_ranges = {
            'ma_period': (0, 100),      # MA长度范围
            'std_period': (5, 50),       # STD周期范围
            'k': (1.0, 4.0)              # k值范围
        }
        
        # 机器学习模型
        self.model = None
        self.scaler = StandardScaler()
        self.training_data = None
        self.best_params = None
        self.best_sharpe = None
        
        # 新增：拟合的多项式系数
        self.poly_coeffs = None
        self.poly_degree = 2  # 默认二次多项式
        
    def generate_training_data(self):
        """
        生成训练数据：随机采样参数组合并计算对应的夏普比率
        
        返回:
        X: 特征矩阵 (ma_period, std_period, k)
        y: 目标值 (平均夏普比率)
        """
        print("=== 生成训练数据 ===")
        
        X = []
        y = []
        
        np.random.seed(42)  # 设置随机种子保证可重复性
        
        for i in range(self.n_samples):
            # 随机生成参数组合
            ma_period = np.random.randint(self.param_ranges['ma_period'][0], 
                                        self.param_ranges['ma_period'][1] + 1)
            std_period = np.random.randint(self.param_ranges['std_period'][0], 
                                         self.param_ranges['std_period'][1] + 1)
            k = np.random.uniform(self.param_ranges['k'][0], 
                                self.param_ranges['k'][1])
            
            # 确保std_period <= ma_period
            if std_period > ma_period:
                std_period = ma_period
            
            print(f"采样 {i+1}/{self.n_samples}: MA={ma_period}, STD={std_period}, k={k:.2f}")
            
            try:
                # 创建策略实例并设置参数
                strategy = ZBolinger(self.filepath, self.product)
                strategy.ma_period = ma_period
                strategy.std_period = std_period
                strategy.k = k
                
                # 重新计算布林带
                strategy.calculate_bollinger_bands()
                
                # 计算多个时间点的夏普比率并取平均
                fib_numbers = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89]
                sharpe_ratios = []
                
                for t_exec in fib_numbers:
                    try:
                        sharpe = strategy.calculate_sharpe_ratio(t_exec=t_exec)
                        if not np.isnan(sharpe) and not np.isinf(sharpe):
                            sharpe_ratios.append(sharpe)
                    except Exception as e:
                        continue
                
                if len(sharpe_ratios) > 0:
                    avg_sharpe = np.mean(sharpe_ratios)
                    
                    X.append([ma_period, std_period, k])
                    y.append(avg_sharpe)
                    
                    print(f"  平均夏普比率: {avg_sharpe:.4f}")
                else:
                    print(f"  无法计算有效的夏普比率")
                    
            except Exception as e:
                print(f"  采样失败: {e}")
                continue
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"\n成功生成 {len(X)} 个有效样本")
        print(f"夏普比率范围: [{np.min(y):.4f}, {np.max(y):.4f}]")
        
        return X, y
    
    def train_model(self, X, y):
        """
        训练机器学习模型
        
        参数:
        X: 特征矩阵
        y: 目标值
        """
        print("\n=== 训练机器学习模型 ===")
        
        # 标准化特征
        X_scaled = self.scaler.fit_transform(X)
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        # 训练随机森林模型
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X_train, y_train)
        
        # 评估模型性能
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)
        
        train_mse = mean_squared_error(y_train, y_pred_train)
        test_mse = mean_squared_error(y_test, y_pred_test)
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        
        print(f"训练集 - MSE: {train_mse:.4f}, R²: {train_r2:.4f}")
        print(f"测试集 - MSE: {test_mse:.4f}, R²: {test_r2:.4f}")
        
        # 拟合多项式函数
        self.fit_polynomial_model(X, y)
        
        # 保存训练数据
        self.training_data = {
            'X': X,
            'y': y,
            'X_scaled': X_scaled,
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'y_pred_train': y_pred_train,
            'y_pred_test': y_pred_test,
            'train_r2': train_r2,
            'test_r2': test_r2
        }
        
        return train_r2, test_r2
    
    def fit_polynomial_model(self, X, y):
        """
        拟合多项式模型，用于生成多项式形式的表达式
        
        参数:
        X: 特征矩阵
        y: 目标值
        """
        print("\n=== 拟合多项式模型 ===")
        
        # 使用numpy的多项式拟合
        # 对每个特征分别拟合多项式
        poly_coeffs = {}
        feature_names = ['ma_period', 'std_period', 'k']
        
        for i, feature_name in enumerate(feature_names):
            # 提取单个特征
            x_feature = X[:, i]
            # 拟合多项式 (degree=2 表示二次多项式)
            coeffs = np.polyfit(x_feature, y, self.poly_degree)
            poly_coeffs[feature_name] = coeffs
            
            # 计算拟合优度
            y_pred = np.polyval(coeffs, x_feature)
            r2 = r2_score(y, y_pred)
            print(f"{feature_name} 多项式拟合 R²: {r2:.4f}")
        
        # 保存多项式系数
        self.poly_coeffs = poly_coeffs
        
        # 输出多项式表达式
        self.print_polynomial_expressions()
    
    def print_polynomial_expressions(self):
        """
        在控制台输出多项式表达式
        """
        print("\n=== 多项式拟合结果 ===")
        print("夏普比率与各参数的多项式关系:")
        
        feature_names_chinese = ['MA周期', 'STD周期', 'k值']
        
        for i, (feature_key, feature_name) in enumerate(zip(['ma_period', 'std_period', 'k'], feature_names_chinese)):
            coeffs = self.poly_coeffs[feature_key]
            
            # 构建多项式表达式字符串
            expr_parts = []
            for power, coeff in enumerate(reversed(coeffs)):
                if abs(coeff) > 1e-10:  # 忽略非常小的系数
                    if power == 0:
                        expr_parts.append(f"{coeff:.6f}")
                    elif power == 1:
                        expr_parts.append(f"{coeff:.6f}*{feature_name}")
                    else:
                        expr_parts.append(f"{coeff:.6f}*{feature_name}^{power}")
            
            polynomial_expr = " + ".join(expr_parts)
            print(f"{feature_name}: 夏普比率 = {polynomial_expr}")
        
        print("\n")
    
    def save_parameters_to_file(self, filename=None):
        """
        保存参数到文件
        
        参数:
        filename: 文件名，如果为None则自动生成
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"zbolinger_params_{self.product}_{timestamp}.json"
        
        # 构建参数字典
        params_dict = {
            'product': self.product,
            'timestamp': datetime.now().isoformat(),
            'best_params': {
                'ma_period': float(self.best_params[0]) if self.best_params is not None else None,
                'std_period': float(self.best_params[1]) if self.best_params is not None else None,
                'k': float(self.best_params[2]) if self.best_params is not None else None
            },
            'best_sharpe': float(self.best_sharpe) if self.best_sharpe is not None else None,
            'poly_coeffs': {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in (self.poly_coeffs or {}).items()},
            'poly_degree': self.poly_degree,
            'param_ranges': self.param_ranges,
            'model_performance': {
                'train_r2': float(self.training_data['train_r2']) if self.training_data and 'train_r2' in self.training_data else None,
                'test_r2': float(self.training_data['test_r2']) if self.training_data and 'test_r2' in self.training_data else None
            } if self.training_data else None
        }
        
        # 保存到文件
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(params_dict, f, indent=2, ensure_ascii=False)
            print(f"参数已保存到文件: {filename}")
            return filename
        except Exception as e:
            print(f"保存参数文件失败: {e}")
            return None
    
    def load_parameters_from_file(self, filename):
        """
        从文件加载参数
        
        参数:
        filename: 参数文件名
        
        返回:
        是否成功加载
        """
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                params_dict = json.load(f)
            
            # 加载最佳参数
            if params_dict['best_params'] and all(v is not None for v in params_dict['best_params'].values()):
                self.best_params = [
                    params_dict['best_params']['ma_period'],
                    params_dict['best_params']['std_period'],
                    params_dict['best_params']['k']
                ]
                self.best_sharpe = params_dict['best_sharpe']
                print(f"从文件加载最佳参数:")
                print(f"  MA周期: {self.best_params[0]:.0f}")
                print(f"  STD周期: {self.best_params[1]:.0f}")
                print(f"  k值: {self.best_params[2]:.3f}")
                print(f"  夏普比率: {self.best_sharpe:.4f}")
            
            # 加载多项式系数
            if 'poly_coeffs' in params_dict and params_dict['poly_coeffs']:
                self.poly_coeffs = {k: np.array(v) if isinstance(v, list) else v for k, v in params_dict['poly_coeffs'].items()}
                self.poly_degree = params_dict.get('poly_degree', 2)
                print("多项式系数已加载")
                self.print_polynomial_expressions()
            
            return True
            
        except Exception as e:
            print(f"加载参数文件失败: {e}")
            return False
    
    def find_parameter_files(self, directory="."):
        """
        在指定目录中查找参数文件
        
        参数:
        directory: 要搜索的目录
        
        返回:
        参数文件列表
        """
        param_files = []
        try:
            for file in os.listdir(directory):
                if file.endswith('.json') and ('zbolinger' in file.lower() or 'params' in file.lower()):
                    filepath = os.path.join(directory, file)
                    # 验证文件内容是否为有效的参数文件
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            if 'best_params' in data and 'product' in data:
                                param_files.append({
                                    'filename': file,
                                    'filepath': filepath,
                                    'product': data['product'],
                                    'timestamp': data.get('timestamp', '未知'),
                                    'best_sharpe': data.get('best_sharpe', '未知')
                                })
                    except:
                        continue
        except Exception as e:
            print(f"查找参数文件时出错: {e}")
        
        return param_files
    
    def interactive_parameter_selection(self):
        """
        交互式参数选择
        
        返回:
        选择的模式 ('file' 或 'ml') 和相关参数
        """
        print("\n=== 参数获取方式选择 ===")
        print("请选择获取最优参数的方式:")
        print("1. 从已有文件获取参数")
        print("2. 机器学习得到")
        
        while True:
            choice = input("请输入选择 (1 或 2): ").strip()
            
            if choice == '1':
                # 从文件获取参数
                print("\n正在查找参数文件...")
                param_files = self.find_parameter_files()
                
                if not param_files:
                    print("未找到有效的参数文件，将使用机器学习方法")
                    return 'ml', None
                
                print(f"找到 {len(param_files)} 个参数文件:")
                for i, file_info in enumerate(param_files):
                    print(f"{i+1}. {file_info['filename']}")
                    print(f"   产品: {file_info['product']}")
                    print(f"   夏普比率: {file_info['best_sharpe']}")
                    print(f"   时间: {file_info['timestamp']}")
                    print()
                
                while True:
                    file_choice = input("请选择文件编号 (或输入 'back' 返回上级选择): ").strip()
                    if file_choice.lower() == 'back':
                        break
                    
                    try:
                        file_idx = int(file_choice) - 1
                        if 0 <= file_idx < len(param_files):
                            selected_file = param_files[file_idx]['filepath']
                            print(f"已选择文件: {param_files[file_idx]['filename']}")
                            return 'file', selected_file
                        else:
                            print("无效的文件编号")
                    except ValueError:
                        print("请输入有效的数字")
                
            elif choice == '2':
                return 'ml', None
            else:
                print("无效的选择，请输入 1 或 2")
    
    def run_optimization_with_interaction(self):
        """
        带交互的优化流程
        
        返回:
        优化结果字典
        """
        print("=== 开始ZBolinger策略参数优化 (交互模式) ===")
        
        # 交互式选择参数获取方式
        mode, param_file = self.interactive_parameter_selection()
        
        if mode == 'file':
            # 从文件加载参数
            if self.load_parameters_from_file(param_file):
                print("\n=== 参数加载成功 ===")
                # 验证参数
                validation_result = self.validate_optimization()
                
                # 保存当前参数（可选）
                save_choice = input("是否保存当前参数到新文件? (y/n): ").strip().lower()
                if save_choice == 'y':
                    self.save_parameters_to_file()
                
                return {
                    'mode': 'file',
                    'best_params': self.best_params,
                    'best_sharpe': self.best_sharpe,
                    'validation': validation_result,
                    'source_file': param_file
                }
            else:
                print("文件加载失败，将使用机器学习方法")
                mode = 'ml'
        
        if mode == 'ml':
            # 使用机器学习方法
            print("\n使用机器学习方法获取参数...")
            
            # 1. 生成训练数据
            X, y = self.generate_training_data()
            
            # 2. 训练模型
            train_r2, test_r2 = self.train_model(X, y)
            
            # 3. 参数优化
            best_params, best_sharpe = self.optimize_parameters()
            
            # 4. 验证结果
            validation_result = self.validate_optimization()
            
            # 5. 保存参数
            save_choice = input("是否保存优化结果到文件? (y/n): ").strip().lower()
            if save_choice == 'y' or save_choice == '':
                saved_file = self.save_parameters_to_file()
            
            # 6. 汇总结果
            optimization_result = {
                'mode': 'ml',
                'best_params': best_params,
                'predicted_sharpe': best_sharpe,
                'model_performance': {
                    'train_r2': train_r2,
                    'test_r2': test_r2
                },
                'validation': validation_result,
                'training_samples': len(X)
            }
            
            print("\n=== 优化完成 ===")
            self.print_optimization_summary(optimization_result)
            
            return optimization_result
    
    def predict_sharpe(self, params):
        """
        使用模型预测给定参数的夏普比率
        
        参数:
        params: 参数数组 [ma_period, std_period, k]
        
        返回:
        预测的夏普比率
        """
        if self.model is None:
            raise ValueError("模型尚未训练")
        
        # 确保参数在合理范围内
        ma_period, std_period, k = params
        ma_period = int(np.clip(ma_period, *self.param_ranges['ma_period']))
        std_period = int(np.clip(std_period, *self.param_ranges['std_period']))
        k = np.clip(k, *self.param_ranges['k'])
        
        # 确保std_period <= ma_period
        if std_period > ma_period:
            std_period = ma_period
        
        # 标准化并预测
        params_scaled = self.scaler.transform([[ma_period, std_period, k]])
        predicted_sharpe = self.model.predict(params_scaled)[0]
        
        return predicted_sharpe
    
    def objective_function(self, params):
        """
        目标函数：返回负的夏普比率（因为我们要最大化）
        
        参数:
        params: 参数数组 [ma_period, std_period, k]
        
        返回:
        负的预测夏普比率
        """
        try:
            predicted_sharpe = self.predict_sharpe(params)
            return -predicted_sharpe  # 负值因为minimize函数寻找最小值
        except Exception as e:
            print(f"目标函数计算失败: {e}")
            return 1e10  # 返回一个大值表示失败
    
    def optimize_parameters(self):
        """
        使用梯度下降优化参数
        
        返回:
        最优参数和对应的夏普比率
        """
        print("\n=== 参数优化 ===")
        
        # 初始参数（使用训练数据中的最佳参数作为起点）
        if self.training_data is not None:
            best_idx = np.argmax(self.training_data['y'])
            initial_params = self.training_data['X'][best_idx]
        else:
            initial_params = [50, 20, 2.0]  # 默认初始参数
        
        print(f"初始参数: MA={initial_params[0]}, STD={initial_params[1]}, k={initial_params[2]:.2f}")
        
        # 定义参数边界
        bounds = [
            self.param_ranges['ma_period'],
            self.param_ranges['std_period'], 
            self.param_ranges['k']
        ]
        
        # 使用多种优化方法
        best_result = None
        best_sharpe = -np.inf
        
        # 尝试不同的优化方法
        methods = ['L-BFGS-B', 'TNC', 'SLSQP']
        
        for method in methods:
            try:
                print(f"\n使用优化方法: {method}")
                result = minimize(
                    self.objective_function,
                    initial_params,
                    method=method,
                    bounds=bounds,
                    options={'maxiter': 100, 'disp': False}
                )
                
                if result.success and -result.fun > best_sharpe:
                    best_result = result
                    best_sharpe = -result.fun
                    
                print(f"{method} - 最优夏普比率: {-result.fun:.4f}")
                print(f"参数: MA={result.x[0]:.0f}, STD={result.x[1]:.0f}, k={result.x[2]:.3f}")
                
            except Exception as e:
                print(f"{method} 优化失败: {e}")
                continue
        
        if best_result is not None:
            self.best_params = best_result.x
            self.best_sharpe = -best_result.fun
            
            print(f"\n=== 最优参数 ===")
            print(f"MA周期: {self.best_params[0]:.0f}")
            print(f"STD周期: {self.best_params[1]:.0f}")
            print(f"k值: {self.best_params[2]:.3f}")
            print(f"预测夏普比率: {self.best_sharpe:.4f}")
            
            return self.best_params, self.best_sharpe
        else:
            print("优化失败，使用训练数据中的最佳参数")
            best_idx = np.argmax(self.training_data['y'])
            self.best_params = self.training_data['X'][best_idx]
            self.best_sharpe = self.training_data['y'][best_idx]
            
            return self.best_params, self.best_sharpe
    
    def validate_optimization(self):
        """
        验证优化结果
        
        返回:
        验证结果字典
        """
        print("\n=== 验证优化结果 ===")
        
        if self.best_params is None:
            print("尚未进行优化")
            return None
        
        # 使用最优参数创建策略实例
        strategy = ZBolinger(self.filepath, self.product)
        strategy.ma_period = int(self.best_params[0])
        strategy.std_period = int(self.best_params[1])
        strategy.k = self.best_params[2]
        
        # 重新计算布林带
        strategy.calculate_bollinger_bands()
        
        # 在多个时间点验证夏普比率
        fib_numbers = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89]
        actual_sharpe_ratios = []
        
        print("验证各时间点的实际夏普比率:")
        for t_exec in fib_numbers:
            try:
                sharpe = strategy.calculate_sharpe_ratio(t_exec=t_exec)
                if not np.isnan(sharpe) and not np.isinf(sharpe):
                    actual_sharpe_ratios.append(sharpe)
                    print(f"  时间点 {t_exec}: {sharpe:.4f}")
            except Exception as e:
                print(f"  时间点 {t_exec}: 计算失败 - {e}")
                continue
        
        if len(actual_sharpe_ratios) > 0:
            actual_avg_sharpe = np.mean(actual_sharpe_ratios)
            
            # 如果有训练模型，使用模型预测；否则使用文件中的夏普比率
            if self.model is not None:
                predicted_sharpe = self.predict_sharpe(self.best_params)
                print(f"\n验证结果:")
                print(f"预测平均夏普比率: {predicted_sharpe:.4f}")
                print(f"实际平均夏普比率: {actual_avg_sharpe:.4f}")
                print(f"预测误差: {abs(predicted_sharpe - actual_avg_sharpe):.4f}")
                
                validation_result = {
                    'predicted_sharpe': predicted_sharpe,
                    'actual_sharpe': actual_avg_sharpe,
                    'prediction_error': abs(predicted_sharpe - actual_avg_sharpe),
                    'individual_sharpe_ratios': actual_sharpe_ratios
                }
            else:
                print(f"\n验证结果 (从文件加载):")
                print(f"文件中的夏普比率: {self.best_sharpe:.4f}")
                print(f"实际平均夏普比率: {actual_avg_sharpe:.4f}")
                print(f"差异: {abs(self.best_sharpe - actual_avg_sharpe):.4f}")
                
                validation_result = {
                    'file_sharpe': self.best_sharpe,
                    'actual_sharpe': actual_avg_sharpe,
                    'difference': abs(self.best_sharpe - actual_avg_sharpe),
                    'individual_sharpe_ratios': actual_sharpe_ratios
                }
            
            return validation_result
        else:
            print("无法计算有效的验证结果")
            return None
    
    def plot_optimization_results(self):
        """
        绘制优化结果图表
        """
        if self.training_data is None:
            print("尚未生成训练数据")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('ZBolinger策略参数优化结果', fontsize=16, fontweight='bold')
        
        X = self.training_data['X']
        y = self.training_data['y']
        
        # 1. 参数空间分布
        ax = axes[0, 0]
        scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', alpha=0.6, s=50)
        ax.set_xlabel('MA周期')
        ax.set_ylabel('STD周期')
        ax.set_title('参数空间分布 (颜色=夏普比率)')
        plt.colorbar(scatter, ax=ax)
        
        # 标记最优参数
        if self.best_params is not None:
            ax.scatter(self.best_params[0], self.best_params[1], 
                      c='red', s=200, marker='*', label='最优参数')
            ax.legend()
        
        # 2. k值 vs 夏普比率
        ax = axes[0, 1]
        ax.scatter(X[:, 2], y, alpha=0.6, s=50)
        ax.set_xlabel('k值')
        ax.set_ylabel('夏普比率')
        ax.set_title('k值 vs 夏普比率')
        ax.grid(True, alpha=0.3)
        
        if self.best_params is not None:
            predicted_best = self.predict_sharpe(self.best_params)
            ax.scatter(self.best_params[2], predicted_best, 
                      c='red', s=200, marker='*', label='最优k值')
            ax.legend()
        
        # 3. 模型预测 vs 实际值
        ax = axes[1, 0]
        y_test = self.training_data['y_test']
        y_pred_test = self.training_data['y_pred_test']
        ax.scatter(y_test, y_pred_test, alpha=0.6, s=50)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        ax.set_xlabel('实际夏普比率')
        ax.set_ylabel('预测夏普比率')
        ax.set_title('模型预测 vs 实际值')
        ax.grid(True, alpha=0.3)
        
        # 4. 特征重要性
        ax = axes[1, 1]
        if hasattr(self.model, 'feature_importances_'):
            feature_names = ['MA周期', 'STD周期', 'k值']
            importances = self.model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            ax.bar(range(len(importances)), importances[indices])
            ax.set_xlabel('特征')
            ax.set_ylabel('重要性')
            ax.set_title('特征重要性')
            ax.set_xticks(range(len(importances)))
            ax.set_xticklabels([feature_names[i] for i in indices], rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def run_optimization(self):
        """
        运行完整的优化流程
        
        返回:
        优化结果字典
        """
        print("=== 开始ZBolinger策略参数优化 ===")
        
        # 1. 生成训练数据
        X, y = self.generate_training_data()
        
        # 2. 训练模型
        train_r2, test_r2 = self.train_model(X, y)
        
        # 3. 参数优化
        best_params, best_sharpe = self.optimize_parameters()
        
        # 4. 验证结果
        validation_result = self.validate_optimization()
        
        # 5. 汇总结果
        optimization_result = {
            'best_params': best_params,
            'predicted_sharpe': best_sharpe,
            'model_performance': {
                'train_r2': train_r2,
                'test_r2': test_r2
            },
            'validation': validation_result,
            'training_samples': len(X)
        }
        
        print("\n=== 优化完成 ===")
        self.print_optimization_summary(optimization_result)
        
        return optimization_result
    
    def print_optimization_summary(self, result):
        """
        打印优化结果摘要
        
        参数:
        result: 优化结果字典
        """
        print("\n" + "="*50)
        print("ZBolinger策略参数优化结果摘要")
        print("="*50)
        
        if result['best_params'] is not None:
            print(f"最优参数:")
            print(f"  MA周期: {result['best_params'][0]:.0f}")
            print(f"  STD周期: {result['best_params'][1]:.0f}")
            print(f"  k值: {result['best_params'][2]:.3f}")
            print(f"预测夏普比率: {result['predicted_sharpe']:.4f}")
        
        print(f"\n模型性能:")
        print(f"  训练集R²: {result['model_performance']['train_r2']:.4f}")
        print(f"  测试集R²: {result['model_performance']['test_r2']:.4f}")
        
        if result['validation'] is not None:
            print(f"\n验证结果:")
            print(f"  预测夏普比率: {result['validation']['predicted_sharpe']:.4f}")
            print(f"  实际夏普比率: {result['validation']['actual_sharpe']:.4f}")
            print(f"  预测误差: {result['validation']['prediction_error']:.4f}")
        
        print(f"\n训练样本数: {result['training_samples']}")
        print("="*50)


def test_optimizer():
    """
    测试优化器
    """
    print("开始测试ZBolinger策略参数优化器...")
    
    # 创建优化器实例
    optimizer = ZBolingerOptimizer(
        filepath="src/MinutesIdx.h5",
        product="IC",
        n_samples=5000  # 减少样本数以加快测试速度
    )
    
    # 运行带交互的优化
    result = optimizer.run_optimization_with_interaction()
    
    # 绘制结果（仅在机器学习模式下）
    if result['mode'] == 'ml':
        optimizer.plot_optimization_results()
    
    print("\n优化器测试完成！")
    return result


if __name__ == "__main__":
    # 运行优化器测试
    result = test_optimizer()