# 量化交易策略对比分析平台

## 📁 文件结构
```
webapp/
├── app.py                 # Flask后端应用
├── start_web.py          # 启动脚本
├── templates/
│   └── index.html        # 前端页面
├── scripts/
│   ├── test_api.py       # API测试脚本
│   └── debug_strategies.py # 策略调试脚本
└── static/               # 静态文件目录
```

## 🚀 快速启动

### 方法1：使用启动脚本（推荐）
```bash
cd webapp
python start_web.py
```

### 方法2：直接启动
```bash
cd webapp
python app.py
```

然后在浏览器中访问：http://localhost:5000

## 📊 功能特点

- **6种交易策略实时对比**
  - 基础策略
  - 固定阈值策略
  - 自适应阈值策略
  - 自适应阈值+趋势过滤
  - 动态仓位控制
  - 多因子策略

- **多维度性能分析图表**
  - 累计收益率对比曲线
  - 夏普比率柱状图
  - 最大回撤对比
  - 风险收益散点图

- **交互式功能**
  - 策略选择器（可勾选显示/隐藏）
  - 时间范围筛选
  - 数据导出功能

## 🔧 技术栈

- **后端**: Flask + Python + pandas
- **前端**: HTML + JavaScript + Chart.js
- **数据**: HDF5格式的高频数据

## 📈 使用说明

1. **启动服务器**后等待2秒自动打开浏览器
2. **选择策略**：勾选想要对比的策略
3. **查看分析**：观察不同维度的表现对比
4. **导出数据**：点击"导出数据"按钮下载JSON格式结果

## 🧪 测试和调试

### 测试API功能
```bash
cd webapp
python scripts/test_api.py
```

### 调试策略计算
```bash
cd webapp
python scripts/debug_strategies.py
```

## 📝 注意事项

- 确保已安装所需依赖：`pip install flask flask-cors pandas numpy matplotlib`
- 数据文件路径已配置为相对路径，指向`../中信建投/任务一/MinutesIdx.h5`
- 如使用其他数据文件，请修改`app.py`中的`filepath`参数

## 🎯 策略说明

所有策略都已修复前视偏差问题，确保回测结果的真实性和可靠性。动态仓位策略经过修正后，夏普比率从异常值回归到正常水平。