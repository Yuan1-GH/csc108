# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **quantitative trading research repository** focused on developing and testing algorithmic trading strategies for Chinese futures markets (IF, IH, IC, IM contracts). The project implements a complete parameter optimization system using both traditional grid search and machine learning approaches with random sampling.

## Architecture Overview

### Core Components

The system consists of **4 main Python modules**:

1. **`MLParameterOpt.py`** (17KB) - ML-based parameter optimizer using random sampling
2. **`ProductAnal.py`** (20KB) - Product visualization and analysis
3. **`Strategy.py`** (9KB) - Core trading strategy implementations
4. **`Valuemetrics.py`** (15KB) - Financial metrics and value calculations

### Data Pipeline

```
HDF5 Data → Random Sampling → Strategy Calculation → Performance Metrics → Parameter Optimization
```

**Data Source**: `中信建投/任务一/MinutesIdx.h5` (202MB, excluded from git) containing minute-level futures data

## Key Dependencies

**Missing Dependencies**: The project references modules that don't exist:
- `strategy_functions.py` (imported in MLParameterOpt.py)
- `performance_metrics.py` (imported in MLParameterOpt.py and ProductAnal.py)

**Actual Dependencies Used**:
```python
# Core data science
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# Standard library
import json, time, os, warnings
```

**No Formal Dependency Management**: No requirements.txt, setup.py, or pyproject.toml files exist.

## Development Commands

### Primary Entry Points
```bash
# Main operations
python MLParameterOpt.py    # ML parameter optimization with random sampling
python ProductAnal.py       # Product analysis and visualization
python Strategy.py          # Strategy implementations
python Valuemetrics.py      # Financial metrics calculations

# Interactive development
jupyter notebook 中信建投/任务一/new_cala.ipynb
```

### No Build System
The project lacks traditional build/test commands. Files are executed directly without formal build processes.

## Strategy Implementation

### Core Strategies
1. **Simple Reversal Strategy** - Basic mean reversion (baseline)
2. **Fixed Threshold Strategy** - Static z-score based trading
3. **Adaptive Threshold Strategy** - Dynamic threshold adjustment using rolling statistics
4. **ML Enhanced Strategy** - Random sampling based parameter optimization

### Key Classes
- `SampleBasedMLOptimizer` - ML parameter optimization with configurable sampling
- `StrategyFunctions` - Collection of trading strategy implementations
- Performance calculation utilities for Sharpe ratios and returns

## Technical Architecture

### Random Sampling Method
The ML optimizer uses random sampling to reduce computational complexity:
```python
n_samples = 30      # 30 random samples
sample_days = 15   # 15 days per sample
total_data = 450 days (vs 3500+ days original)
```

**Benefits**: 87% reduction in computation while maintaining statistical representativeness.

### Parameter Search Space
```python
# Fixed threshold strategy
thresholds = [0.001, 0.002, 0.005, 0.01]

# Adaptive threshold strategy
windows = [3, 5, 7, 10, 15]
std_multipliers = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
# Total: 30 parameter combinations
```

## Project Structure

```
F:\HelloWorld\Code\
├── Core Python Scripts:
│   ├── Strategy.py (9KB) - Strategy function implementations
│   ├── MLParameterOpt.py (17KB) - ML-based parameter optimizer
│   ├── ProductAnal.py (20KB) - Product visualization analyzer
│   └── Valuemetrics.py (15KB) - Value metrics calculations
├── 中信建投/任务一/:
│   ├── MinutesIdx.h5 (202MB) - Main dataset (gitignored)
│   ├── new_cala.ipynb (683KB) - Jupyter notebook
│   └── Generated PNG files - Analysis charts
├── Configuration:
│   ├── .gitignore - Excludes HDF5 data file
│   ├── .vscode/settings.json - VS Code Python configuration
│   └── .claude/settings.local.json - Claude Code permissions
└── Documentation:
    ├── README.md - Project documentation (Chinese)
    └── CLAUDE.md - This file
```

## Current Issues

### Missing Components
1. **Missing Module Dependencies**: Code imports non-existent modules (`strategy_functions`, `performance_metrics`)
2. **No Web Application**: Despite documentation mentioning webapp components, they don't exist
3. **No Dependency Management**: No formal package management system
4. **No Tests**: No testing framework or test files
5. **No CI/CD**: No continuous integration configuration

### Documentation Inconsistencies
The existing CLAUDE.md file describes features (webapp, 10 strategies) that don't exist in the current codebase, suggesting documentation drift from actual implementation.

## Working with This Codebase

### Before Starting
1. **Check Missing Dependencies**: Verify that `strategy_functions.py` and `performance_metrics.py` exist or create them
2. **Data File**: Ensure `中信建投/任务一/MinutesIdx.h5` is available (202MB, not in git)
3. **Environment Setup**: Install required packages manually (no requirements.txt)

### Common Development Patterns
- **Strategy Development**: Add new strategies to `Strategy.py` following the existing pattern
- **Parameter Optimization**: Extend the sampling methods in `MLParameterOpt.py`
- **Analysis**: Use `ProductAnal.py` for visualization and `Valuemetrics.py` for metrics
- **Interactive Work**: Use the Jupyter notebook for experimentation

### File Naming Conventions
- Main modules use CamelCase.py (`MLParameterOpt.py`, `ProductAnal.py`)
- Classes use CamelCase (`SampleBasedMLOptimizer`, `StrategyFunctions`)
- Methods use snake_case (`simple_reversal_strategy`, `calculate_sharpe`)

## Output Files

### Parameter Configuration Files
- `latest_optimal_params.json` - Traditional optimization results
- `latest_sampling_ml_params.json` - ML optimization results

### Analysis Outputs
- PNG charts in `中信建投/任务一/` directory
- JSON parameter files with optimal configurations
- Console output with performance metrics and recommendations