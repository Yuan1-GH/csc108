import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# 导入原始的ZBolinger策略
from ZBolinger import ZBolinger

class ZBolingerOptimizer:
    """
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
            'ma_period': (10, 100),      # MA长度范围
            'std_period': (5, 50),       # STD周期范围
            'k': (1.0, 4.0)              # k值范围
        }
        
        # 机器学习模型
        self.model = None
        self.scaler = StandardScaler()
        self.training_data = None
        self.best_params = None
        self.best_sharpe = None
        
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
            'y_pred_test': y_pred_test
        }
        
        return train_r2, test_r2
    
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
    
    # 运行优化
    result = optimizer.run_optimization()
    
    # 绘制结果
    optimizer.plot_optimization_results()
    
    print("\n优化器测试完成！")
    return result


if __name__ == "__main__":
    # 运行优化器测试
    result = test_optimizer()