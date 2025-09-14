# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a comprehensive quantitative trading research repository focused on developing and testing algorithmic trading strategies. The project has evolved into two main components:

1. **中信建投/任务一/** - IC futures trading strategy backtesting system (Jupyter notebook based)
2. **webapp/** - Flask-based web application for real-time strategy visualization and analysis
3. **uni/Orcale/** - Oracle database management tools

## Key Dependencies

The project relies on standard data science and trading libraries:
- numpy, pandas, matplotlib, seaborn
- yfinance for market data
- sklearn for machine learning strategies
- oracledb for Oracle database operations
- flask, flask-cors for web application

**No formal dependency management** - install packages manually as needed:
```bash
pip install numpy pandas matplotlib seaborn yfinance scikit-learn oracledb flask flask-cors
```

## Architecture Overview

### Trading Strategy System (Dual Implementation)

#### Jupyter Notebook Implementation (中信建投/任务一/)
- **MinutesIdx.h5** - HDF5 file containing minute-level IC futures data (202MB)
- **new_cala.ipynb** - Complete strategy development workflow with 10 strategies
- **Core Strategy Class**: `stock_info` with comprehensive strategy methods

#### Web Application Implementation (webapp/)
- **app.py** - Flask application with `StrategyAnalyzer` class
- **Real-time strategy comparison and visualization**
- **Interactive web interface for strategy analysis**
- **HTML5 report generation**

### Strategy Implementation Details

The project implements **10 different trading strategies**:

1. **Fixed Threshold** - Simple mean reversion with fixed z-score threshold
2. **Improved Threshold** - Dual threshold bandwidth with state machine
3. **Adaptive Threshold** - Dynamic threshold based on rolling volatility
4. **Adaptive + Trend** - Trend-filtered adaptive strategy
5. **Dynamic Position** - Smooth position sizing using tanh function
6. **Volatility Weighted** - Inverse volatility scaling for position sizing
7. **Vol Weighted + Dynamic Threshold** - Combined volatility and threshold adaptation
8. **Multi Timeframe** - 5/20/60 minute multi-period signal combination
9. **ML Enhanced** - Random forest classifier with feature engineering
10. **Risk Parity** - Target volatility control with dynamic leverage

### Data Structure

**HDF5 File Structure**:
- Complex nested data structures requiring special handling
- `StrategyAnalyzer` class automatically extracts and processes data
- Supports various formats: DataFrame, Series, nested structures
- Converts to consistent price series for analysis

## Development Workflow

### Running Trading Strategies

#### Option 1: Jupyter Notebook (Traditional)
```bash
# Launch Jupyter for strategy development
jupyter notebook 中信建投/任务一/new_cala.ipynb
```

#### Option 2: Web Application (Recommended)
```bash
# Quick launch from project root
python launch_webapp.py

# Or manual launch
cd webapp && python app.py
# Access at http://localhost:5000
```

### Web Application Features
- **6 trading strategies** with real-time comparison
- **Interactive strategy selection**
- **Multi-dimensional performance analysis charts**
- **Data export functionality**
- **Responsive design for mobile and desktop**

### Common Development Commands
```bash
# Launch Jupyter for strategy development
jupyter notebook

# Run individual Python scripts
python uni/Orcale/编辑表.py

# Debug specific strategies
python webapp/scripts/debug_strategies.py

# Test API endpoints
python webapp/scripts/test_api.py

# Quick webapp launch
python launch_webapp.py
```

## Key Files and Data

### Primary Data Files
- **中信建投/任务一/MinutesIdx.h5** - Main dataset (202MB), minute-level futures data
- **中信建投/任务一/new_cala.ipynb** - Complete strategy implementations
- **.gitignore** - Excludes the large HDF5 data file

### Web Application Files
- **webapp/app.py** - Main Flask application with StrategyAnalyzer class
- **webapp/templates/** - HTML templates for web interface
- **webapp/static/** - CSS, JavaScript, and static assets
- **webapp/scripts/** - Debug and testing utilities
- **launch_webapp.py** - Quick launcher from project root

### Strategy Class Architecture

Both implementations share similar core functionality:

```python
class StrategyAnalyzer / stock_info:
    def __init__(self, filepath, target_col)
    def load_data()  # Handles complex HDF5 structures
    def calculate_sharpe(self, pnl)
    # 10 strategy methods including:
    # - strategy_fixed_threshold()
    # - strategy_ml_enhanced()
    # - strategy_risk_parity()
    # - generate_html_report()  # Web app only
```

## Advanced Features

### HTML5 Report Generation
- Professional interactive reports with strategy comparisons
- Performance metrics and visualizations
- Responsive design with mobile support
- Real-time data updates

### Machine Learning Integration
- Random forest classification for price direction prediction
- Feature engineering with technical indicators
- Probability-weighted position sizing

### Risk Management
- Volatility-based position sizing
- Maximum leverage constraints
- Dynamic threshold adaptation
- Risk parity allocation

## Testing and Debugging

### Strategy Testing
- Jupyter notebook execution and validation
- Web application real-time testing
- Manual verification of Sharpe ratios and returns
- Visual inspection of cumulative return plots

### Debug Tools
- **webapp/scripts/debug_strategies.py** - Individual strategy analysis
- **webapp/scripts/test_api.py** - API endpoint testing
- Comprehensive error handling for data structure issues

## Performance Optimization

### Data Handling
- Efficient HDF5 data extraction
- Rolling window calculations with pandas
- Memory optimization for large datasets
- Parallel processing for multiple strategies

### Web Application
- CORS enabled for cross-origin requests
- Non-interactive matplotlib backend for server use
- Responsive design principles
- Real-time data updates without page refresh