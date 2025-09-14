# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a quantitative trading research repository focused on developing and testing algorithmic trading strategies. The main work is concentrated in two areas:

1. **中信建投/任务一/** - IC futures trading strategy backtesting system
2. **uni/Orcale/** - Oracle database management tools

## Key Dependencies

The project relies on standard data science and trading libraries:
- numpy, pandas, matplotlib, seaborn
- yfinance for market data
- sklearn for machine learning strategies
- oracledb for Oracle database operations

**No formal dependency management** - install packages manually as needed:
```bash
pip install numpy pandas matplotlib seaborn yfinance scikit-learn oracledb
```

## Main Project Structure

### Trading Strategy Development (中信建投/任务一/)
- **MinutesIdx.h5** - HDF5 file containing minute-level IC futures data
- **new_cala.ipynb** - Jupyter notebook with complete strategy development workflow
  - 10 different trading strategies implemented
  - Strategy comparison and Sharpe ratio analysis
  - Parameter optimization and sensitivity analysis
  - HTML report generation functionality

**Core Strategy Class:**
```python
class stock_info:
    def __init__(self, filepath, target_col)
    def calculate_sharpe(self, pnl)
    # 10 strategy methods including:
    # - strategy_fixed_threshold()
    # - strategy_ml_enhanced()
    # - strategy_risk_parity()
    # etc.
```

### Database Tools (uni/Orcale/)
- **编辑表.py** - Interactive Oracle table management tool
- **谁的表.py** - Oracle table inspection utility

## Development Workflow

### Running Trading Strategies
1. Execute the Jupyter notebook: `jupyter notebook 中信建投/任务一/new_cala.ipynb`
2. Strategies automatically load from `MinutesIdx.h5`
3. Results include Sharpe ratios, cumulative returns, and HTML reports

### Common Development Commands
```bash
# Launch Jupyter for strategy development
jupyter notebook

# Run individual Python scripts
python uni/Orcale/编辑表.py

# Generate HTML strategy reports
# (Run from within the Jupyter notebook)
```

## Key Files and Data

- **中信建投/任务一/MinutesIdx.h5** - Primary dataset (202MB), contains minute-level futures data
- **中信建投/任务一/new_cala.ipynb** - Main development notebook with complete strategy implementations
- **.gitignore** - Excludes the large HDF5 data file

## Data Structure Notes

The HDF5 file contains complex nested data structures. The `stock_info` class handles data extraction automatically, converting various formats (DataFrame, Series, nested structures) to consistent price series for analysis.

## Testing

No formal test suite exists. Testing is done through:
- Jupyter notebook execution and strategy validation
- Manual verification of Sharpe ratios and returns
- Visual inspection of cumulative return plots