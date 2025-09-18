# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **quantitative trading research repository** focused on developing and testing algorithmic trading strategies for Chinese futures markets (IF, IH, IC, IM contracts). The project implements a complete parameter optimization system using both traditional grid search and machine learning approaches with random sampling. The system has been migrated from HDF5 data storage to CSV format for improved compatibility and easier data inspection.

## Architecture Overview

### Core Components

The system consists of **6 main Python modules**:

1. **`MLParameterOpt.py`** (411 lines) - ML-based parameter optimizer using random sampling
2. **`ProductAnal.py`** (732 lines) - Product visualization and analysis
3. **`Strategy.py`** (272 lines) - Core trading strategy implementations
4. **`Valuemetrics.py`** (452 lines) - Financial metrics and value calculations
5. **`CSVloader.py`** (222 lines) - Unified CSV data loader
6. **`H5toCSV.py`** (378 lines) - HDF5 to CSV data migration utility

### Data Pipeline

```
HDF5 Data → CSV Conversion → Random Sampling → Strategy Calculation → Performance Metrics → Parameter Optimization
```

**Data Source**: `中信建投/任务一/MinutesIdx.h5` (203MB, excluded from git) containing minute-level futures data for IF, IH, IC, IM contracts, converted to `csv/MinutesIdx_original.csv` (12MB)

## Development Commands

### Primary Entry Points
```bash
# Main operations
python MLParameterOpt.py    # ML parameter optimization with random sampling
python ProductAnal.py       # Product analysis and visualization
python Strategy.py          # Strategy implementations
python Valuemetrics.py      # Financial metrics calculations
python H5toCSV.py           # Convert HDF5 data to CSV format

# Interactive development
jupyter notebook 中信建投/任务一/origin.ipynb
```

### Data Processing
```bash
# Convert HDF5 to CSV (one-time migration)
python H5toCSV.py

# The system now uses CSV data via CSVloader.py instead of direct HDF5 access
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
- `PerformanceMetrics` - Financial performance calculations
- `CSVDataLoader` - Unified data access layer
- `DataAnalyzer` - Product visualization and analysis

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

### Data Migration Architecture
The system has been migrated from HDF5 to CSV format:
- **Original**: Direct HDF5 access via `pd.read_hdf()`
- **Current**: CSV-based access via `CSVDataLoader` class
- **Benefits**: Improved compatibility, easier data inspection, better version control

## Project Structure

```
F:\HelloWorld\csc108\
├── Core Python Scripts:
│   ├── Strategy.py (272 lines) - Strategy function implementations
│   ├── MLParameterOpt.py (417 lines) - ML-based parameter optimizer
│   ├── ProductAnal.py (544 lines) - Product visualization analyzer
│   ├── Valuemetrics.py (452 lines) - Value metrics calculations
│   ├── CSVloader.py (69 lines) - Unified CSV data loader
│   └── H5toCSV.py (378 lines) - HDF5 to CSV migration utility
├── Data:
│   ├── csv/MinutesIdx_original.csv (13MB) - Main dataset
│   └── 中信建投/任务一/MinutesIdx.h5 (202MB, gitignored) - Original data
├── Notebooks:
│   └── 中信建投/任务一/origin.ipynb - Interactive analysis
├── Configuration:
│   ├── .gitignore - Excludes HDF5 data file
│   └── .vscode/settings.json - VS Code Python configuration
└── Documentation:
    ├── README.md - Project documentation (Chinese)
    └── CLAUDE.md - This file
```

## Current Issues

### Missing Components
1. **No Dependency Management**: No requirements.txt, setup.py, or pyproject.toml files exist
2. **No Tests**: No testing framework or test files
3. **No CI/CD**: No continuous integration configuration

### Data Migration Status
- **Complete**: HDF5 to CSV migration utility exists (`H5toCSV.py`)
- **Usage**: All main modules now use `CSVDataLoader` instead of direct HDF5 access
- **Current**: Migration completed, system fully operates on CSV data
- **Note**: Original HDF5 file preserved for backup purposes

## Dependencies

### Required Packages
```python
# Core data science
numpy >= 1.24.4
pandas >= 1.5.3
matplotlib.pyplot
scipy
seaborn

# Standard library
json, time, os, warnings, logging
```

### Environment Setup
The project references a specific conda environment:
```bash
# Environment: data_env with numpy==1.24.4, pandas==1.5.3
conda activate data_env
```

## Working with This Codebase

### Before Starting
1. **Data Migration**: Run `python H5toCSV.py` to convert HDF5 to CSV format
2. **Data File**: Ensure `csv/MinutesIdx_original.csv` exists (13MB)
3. **Environment Setup**: Install required packages manually (no requirements.txt)

### Common Development Patterns
- **Strategy Development**: Add new strategies to `Strategy.py` following the existing pattern
- **Parameter Optimization**: Extend the sampling methods in `MLParameterOpt.py`
- **Analysis**: Use `ProductAnal.py` for visualization and `Valuemetrics.py` for metrics
- **Interactive Work**: Use the Jupyter notebook for experimentation
- **Data Access**: Use `CSVDataLoader` class for all data operations

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

### Data Files
- `csv/MinutesIdx_original.csv` - Main working dataset
- `csv/MinutesIdx_cleaned.csv` - Cleaned version (if generated)