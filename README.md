#### 中信建投h5文件下载
https://drive.google.com/file/d/1OEcVhOUiidUZrR9mmZXa9qLyCCnzzDJS/view?usp=share_link
# 📊 量化交易策略对比分析平台

## 🎯 项目概述

我已经为你创建了一个完整的量化交易策略对比分析网页，所有文件已整理到 `webapp/` 文件夹中。

## 📁 文件结构

```
HelloWorld/Code/
├── webapp/                     # 网页应用文件夹
│   ├── app.py                  # Flask后端主程序
│   ├── start_web.py           # webapp内启动脚本
│   ├── README.md              # 详细说明文档
│   ├── templates/
│   │   └── index.html         # 前端页面
│   ├── scripts/
│   │   ├── test_api.py        # API测试脚本
│   │   └── debug_strategies.py # 策略调试脚本
│   └── static/                # 静态资源目录
├── launch_webapp.py           # 根目录启动脚本 ⭐推荐使用
└── WEBAPP使用说明.md           # 本文件
```

## 🚀 启动方式

### 方式1：根目录启动（推荐）
```bash
python launch_webapp.py
```

### 方式2：webapp目录启动
```bash
cd webapp
python start_web.py
```

### 方式3：直接启动
```bash
cd webapp
python app.py
```

启动后自动打开浏览器访问：http://localhost:5000

## ✨ 主要功能

### 🔧 后端功能
- **6种交易策略实现**：
  - 基础策略（简单反转）
  - 固定阈值策略
  - 自适应阈值策略
  - 自适应阈值+趋势过滤
  - 动态仓位控制
  - 多因子策略

- **RESTful API接口**：
  - `/api/strategies` - 获取所有策略数据
  - `/api/compare` - 获取策略对比数据
  - `/api/strategy/<name>` - 获取单个策略数据

### 🎨 前端功能
- **4个交互式图表**：
  - 累计收益率对比曲线
  - 夏普比率柱状图
  - 最大回撤对比
  - 风险收益散点图

- **交互功能**：
  - 策略选择器（可勾选显示/隐藏）
  - 实时数据刷新
  - JSON格式数据导出

## 📊 策略表现（最新数据）

根据API测试结果，策略排名如下：

1. **动态仓位策略** - 夏普比率: 0.3724
2. **多因子策略** - 夏普比率: 0.0805
3. **自适应+趋势策略** - 夏普比率: -0.0754
4. **固定阈值策略** - 夏普比率: -0.4583
5. **基础策略** - 夏普比率: -0.5425
6. **自适应阈值策略** - 夏普比率: -0.6254

## 🔧 技术栈

- **后端**: Flask + Python + pandas + numpy
- **前端**: HTML5 + JavaScript + Chart.js
- **数据源**: HDF5格式高频数据
- **可视化**: Chart.js图表库

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

### 查看详细日志
启动时会显示数据加载状态和策略计算过程。

## ⚠️ 重要说明

1. **数据来源**: 使用 `中信建投/任务一/MinutesIdx.h5` 文件
2. **前视偏差修复**: 所有策略已修复前视偏差问题，确保回测结果真实可靠
3. **动态仓位策略**: 已修复异常夏普比率问题，现在数据真实可信
4. **路径配置**: 已配置相对路径，确保文件移动后仍能正常工作

## 🎯 使用建议

1. **首次使用**: 建议使用 `python launch_webapp.py` 从根目录启动
2. **策略分析**: 观察不同策略在各个维度的表现
3. **数据导出**: 点击"导出数据"按钮获取详细分析结果
4. **参数调优**: 可以修改 `app.py` 中的策略参数进行优化

## 🔄 更新日志

- v1.0: 初始版本，包含6种策略对比分析
- v1.1: 修复动态仓位策略前视偏差问题
- v1.2: 整理文件结构，优化启动方式

---

**网页已成功部署并运行在 http://localhost:5000** 🎉