"""
Time Series Train/Test Splitting Guide
Comprehensive examples of proper time series splitting techniques
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')


class TimeSeriesSplitter:
    """Comprehensive time series splitting utilities"""
    
    @staticmethod
    def simple_holdout_split(data: pd.Series, test_size: float = 0.2) -> Tuple[pd.Series, pd.Series]:
        """
        Simple hold-out split for time series
        
        Args:
            data: Time series data
            test_size: Proportion of data for testing (0.0 to 1.0)
            
        Returns:
            Tuple of (train_data, test_data)
        """
        split_point = int(len(data) * (1 - test_size))
        train = data.iloc[:split_point]
        test = data.iloc[split_point:]
        
        return train, test
    
    @staticmethod
    def date_based_split(data: pd.Series, split_date: str) -> Tuple[pd.Series, pd.Series]:
        """
        Split based on specific date
        
        Args:
            data: Time series data with datetime index
            split_date: Date string to split on (e.g., '2023-01-01')
            
        Returns:
            Tuple of (train_data, test_data)
        """
        split_date = pd.to_datetime(split_date)
        train = data[data.index < split_date]
        test = data[data.index >= split_date]
        
        return train, test
    
    @staticmethod
    def expanding_window_split(data: pd.Series, 
                              min_train_size: int = 100, 
                              test_size: int = 30,
                              step_size: int = 30) -> List[Tuple[pd.Series, pd.Series]]:
        """
        Expanding window (walk-forward) cross-validation
        Train set grows, test set slides forward
        
        Args:
            data: Time series data
            min_train_size: Minimum size of training set
            test_size: Size of each test set
            step_size: Step size between splits
            
        Returns:
            List of (train_data, test_data) tuples
        """
        splits = []
        
        start_test = min_train_size
        while start_test + test_size <= len(data):
            train = data.iloc[:start_test]
            test = data.iloc[start_test:start_test + test_size]
            splits.append((train, test))
            start_test += step_size
        
        return splits
    
    @staticmethod
    def rolling_window_split(data: pd.Series,
                           train_size: int = 252,  # 1 year of trading days
                           test_size: int = 30,
                           step_size: int = 30) -> List[Tuple[pd.Series, pd.Series]]:
        """
        Rolling window cross-validation
        Both train and test sets slide forward
        
        Args:
            data: Time series data
            train_size: Fixed size of training set
            test_size: Size of each test set
            step_size: Step size between splits
            
        Returns:
            List of (train_data, test_data) tuples
        """
        splits = []
        
        start_train = 0
        while start_train + train_size + test_size <= len(data):
            train = data.iloc[start_train:start_train + train_size]
            test = data.iloc[start_train + train_size:start_train + train_size + test_size]
            splits.append((train, test))
            start_train += step_size
        
        return splits
    
    @staticmethod
    def purged_split(data: pd.Series, 
                    test_size: float = 0.2, 
                    gap: int = 5) -> Tuple[pd.Series, pd.Series]:
        """
        Split with a gap to prevent information leakage
        Useful when predictions might affect future values
        
        Args:
            data: Time series data
            test_size: Proportion for testing
            gap: Number of periods to exclude between train and test
            
        Returns:
            Tuple of (train_data, test_data)
        """
        split_point = int(len(data) * (1 - test_size))
        train = data.iloc[:split_point - gap]
        test = data.iloc[split_point:]
        
        return train, test
    
    @staticmethod
    def combinatorial_purged_split(data: pd.Series,
                                  n_splits: int = 5,
                                  test_size: int = 50,
                                  gap: int = 10) -> List[Tuple[pd.Series, pd.Series]]:
        """
        Combinatorial purged cross-validation
        Non-overlapping test sets with gaps
        
        Args:
            data: Time series data
            n_splits: Number of splits
            test_size: Size of each test set
            gap: Gap between consecutive test sets
            
        Returns:
            List of (train_data, test_data) tuples
        """
        splits = []
        total_test_size = n_splits * test_size + (n_splits - 1) * gap
        
        if total_test_size > len(data) // 2:
            raise ValueError("Test sets too large for available data")
        
        # Calculate test set positions
        start_pos = len(data) - total_test_size
        
        for i in range(n_splits):
            test_start = start_pos + i * (test_size + gap)
            test_end = test_start + test_size
            
            # Create train set (everything before test set)
            train = data.iloc[:test_start - gap] if gap > 0 else data.iloc[:test_start]
            test = data.iloc[test_start:test_end]
            
            splits.append((train, test))
        
        return splits


def demonstrate_splitting_methods():
    """Demonstrate different splitting methods with visualizations"""
    
    # Create sample time series data
    dates = pd.date_range('2020-01-01', '2024-01-01', freq='D')
    trend = np.linspace(100, 200, len(dates))
    seasonal = 10 * np.sin(2 * np.pi * np.arange(len(dates)) / 365.25)
    noise = np.random.normal(0, 5, len(dates))
    ts = pd.Series(trend + seasonal + noise, index=dates)
    
    print("=== Time Series Splitting Methods ===")
    print(f"Data: {len(ts)} points from {ts.index[0].date()} to {ts.index[-1].date()}")
    
    # 1. Simple Hold-out Split
    print("\n1. Simple Hold-out Split (80/20)")
    train, test = TimeSeriesSplitter.simple_holdout_split(ts, test_size=0.2)
    print(f"   Train: {len(train)} points ({train.index[0].date()} to {train.index[-1].date()})")
    print(f"   Test:  {len(test)} points ({test.index[0].date()} to {test.index[-1].date()})")
    
    # 2. Date-based Split
    print("\n2. Date-based Split (split at 2023-01-01)")
    train, test = TimeSeriesSplitter.date_based_split(ts, '2023-01-01')
    print(f"   Train: {len(train)} points ({train.index[0].date()} to {train.index[-1].date()})")
    print(f"   Test:  {len(test)} points ({test.index[0].date()} to {test.index[-1].date()})")
    
    # 3. Expanding Window
    print("\n3. Expanding Window Cross-Validation")
    splits = TimeSeriesSplitter.expanding_window_split(ts, min_train_size=500, test_size=30, step_size=90)
    print(f"   Number of splits: {len(splits)}")
    for i, (train, test) in enumerate(splits[:3]):  # Show first 3
        print(f"   Split {i+1}: Train={len(train)} points, Test={len(test)} points")
        print(f"            Train: {train.index[-1].date()}, Test: {test.index[0].date()} to {test.index[-1].date()}")
    
    # 4. Rolling Window
    print("\n4. Rolling Window Cross-Validation")
    splits = TimeSeriesSplitter.rolling_window_split(ts, train_size=365, test_size=30, step_size=90)
    print(f"   Number of splits: {len(splits)}")
    for i, (train, test) in enumerate(splits[:3]):  # Show first 3
        print(f"   Split {i+1}: Train={len(train)} points, Test={len(test)} points")
        print(f"            Train: {train.index[0].date()} to {train.index[-1].date()}")
        print(f"            Test: {test.index[0].date()} to {test.index[-1].date()}")
    
    # 5. Purged Split
    print("\n5. Purged Split (with 10-day gap)")
    train, test = TimeSeriesSplitter.purged_split(ts, test_size=0.2, gap=10)
    print(f"   Train: {len(train)} points ({train.index[0].date()} to {train.index[-1].date()})")
    print(f"   Test:  {len(test)} points ({test.index[0].date()} to {test.index[-1].date()})")
    print(f"   Gap:   {(test.index[0] - train.index[-1]).days} days")
    
    return ts


def visualize_splits(ts: pd.Series):
    """Create visualizations of different splitting methods"""
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle('Time Series Splitting Methods Comparison', fontsize=16, fontweight='bold')
    
    # 1. Simple Hold-out
    train, test = TimeSeriesSplitter.simple_holdout_split(ts, test_size=0.2)
    axes[0, 0].plot(train.index, train.values, label='Train', color='blue', alpha=0.7)
    axes[0, 0].plot(test.index, test.values, label='Test', color='red', alpha=0.7)
    axes[0, 0].set_title('Simple Hold-out Split (80/20)')
    axes[0, 0].legend()
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # 2. Date-based Split
    train, test = TimeSeriesSplitter.date_based_split(ts, '2023-01-01')
    axes[0, 1].plot(train.index, train.values, label='Train', color='blue', alpha=0.7)
    axes[0, 1].plot(test.index, test.values, label='Test', color='red', alpha=0.7)
    axes[0, 1].axvline(x=pd.to_datetime('2023-01-01'), color='green', linestyle='--', alpha=0.7, label='Split Date')
    axes[0, 1].set_title('Date-based Split (2023-01-01)')
    axes[0, 1].legend()
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # 3. Expanding Window (show multiple splits)
    splits = TimeSeriesSplitter.expanding_window_split(ts, min_train_size=500, test_size=60, step_size=200)
    axes[1, 0].plot(ts.index, ts.values, color='lightgray', alpha=0.5, label='Full Data')
    colors = ['red', 'orange', 'green', 'purple', 'brown']
    for i, (train, test) in enumerate(splits[:5]):
        axes[1, 0].plot(test.index, test.values, color=colors[i % len(colors)], 
                       linewidth=2, label=f'Test {i+1}')
    axes[1, 0].set_title('Expanding Window Cross-Validation')
    axes[1, 0].legend()
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # 4. Rolling Window (show multiple splits)
    splits = TimeSeriesSplitter.rolling_window_split(ts, train_size=365, test_size=60, step_size=200)
    axes[1, 1].plot(ts.index, ts.values, color='lightgray', alpha=0.5, label='Full Data')
    for i, (train, test) in enumerate(splits[:5]):
        axes[1, 1].plot(train.index, train.values, color='blue', alpha=0.3)
        axes[1, 1].plot(test.index, test.values, color=colors[i % len(colors)], 
                       linewidth=2, label=f'Test {i+1}')
    axes[1, 1].set_title('Rolling Window Cross-Validation')
    axes[1, 1].legend()
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    # 5. Purged Split
    train, test = TimeSeriesSplitter.purged_split(ts, test_size=0.2, gap=10)
    gap_start = train.index[-1]
    gap_end = test.index[0]
    axes[2, 0].plot(train.index, train.values, label='Train', color='blue', alpha=0.7)
    axes[2, 0].plot(test.index, test.values, label='Test', color='red', alpha=0.7)
    axes[2, 0].axvspan(gap_start, gap_end, alpha=0.3, color='yellow', label='Purge Gap')
    axes[2, 0].set_title('Purged Split (10-day gap)')
    axes[2, 0].legend()
    axes[2, 0].tick_params(axis='x', rotation=45)
    
    # 6. Combinatorial Purged
    splits = TimeSeriesSplitter.combinatorial_purged_split(ts, n_splits=4, test_size=50, gap=20)
    axes[2, 1].plot(ts.index, ts.values, color='lightgray', alpha=0.5, label='Full Data')
    for i, (train, test) in enumerate(splits):
        axes[2, 1].plot(test.index, test.values, color=colors[i % len(colors)], 
                       linewidth=3, label=f'Test {i+1}')
    axes[2, 1].set_title('Combinatorial Purged CV')
    axes[2, 1].legend()
    axes[2, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()


def best_practices_guide():
    """Print best practices for time series splitting"""
    
    print("\n" + "="*60)
    print("🎯 TIME SERIES SPLITTING BEST PRACTICES")
    print("="*60)
    
    practices = [
        ("✅ Always Respect Time Order", 
         "Never use future data to predict past values. Always split chronologically."),
        
        ("🔄 Use Cross-Validation", 
         "Single train/test split can be misleading. Use multiple temporal splits."),
        
        ("📏 Choose Appropriate Test Size", 
         "Test size should match your forecasting horizon. For 30-day forecasts, use 30-day test sets."),
        
        ("🚧 Consider Purging", 
         "Add gaps between train/test if predictions might affect future values (e.g., trading)."),
        
        ("📊 Match Business Reality", 
         "Split based on when you'd actually retrain models in production."),
        
        ("⚖️ Balance Train Size", 
         "Ensure sufficient training data but don't make test sets too small."),
        
        ("🔍 Validate Assumptions", 
         "Check that your test period is representative of deployment conditions."),
        
        ("📈 Consider Regime Changes", 
         "Market crashes, policy changes, etc. can make old data less relevant."),
        
        ("🎯 Use Multiple Metrics", 
         "Don't rely on single metric. Use MAE, RMSE, MAPE, and domain-specific metrics."),
        
        ("🔄 Implement Walk-Forward", 
         "For production, use expanding window that mimics real deployment.")
    ]
    
    for i, (title, description) in enumerate(practices, 1):
        print(f"\n{i:2d}. {title}")
        print(f"    {description}")


def splitting_recommendations_by_use_case():
    """Provide specific recommendations based on use case"""
    
    print("\n" + "="*60)
    print("📋 SPLITTING RECOMMENDATIONS BY USE CASE")
    print("="*60)
    
    use_cases = {
        "Stock Price Forecasting": {
            "method": "Expanding Window with Purging",
            "reasoning": "Markets evolve, need recent data. Purging prevents look-ahead bias.",
            "parameters": "min_train_size=252, test_size=21, gap=5, step_size=21"
        },
        
        "Sales Forecasting": {
            "method": "Rolling Window",
            "reasoning": "Seasonal patterns important. Fixed window captures recent trends.",
            "parameters": "train_size=104 weeks, test_size=4 weeks, step_size=4 weeks"
        },
        
        "Demand Forecasting": {
            "method": "Expanding Window",
            "reasoning": "More historical data helps capture long-term patterns.",
            "parameters": "min_train_size=365, test_size=30, step_size=30"
        },
        
        "Energy Consumption": {
            "method": "Seasonal Splits",
            "reasoning": "Strong seasonal patterns. Test on full seasons.",
            "parameters": "Split by seasons, test on recent year"
        },
        
        "Economic Indicators": {
            "method": "Simple Hold-out with Large Gap",
            "reasoning": "Regime changes common. Large gap prevents contamination.",
            "parameters": "test_size=0.2, gap=90 days"
        }
    }
    
    for use_case, details in use_cases.items():
        print(f"\n📊 {use_case}")
        print(f"   Method: {details['method']}")
        print(f"   Why: {details['reasoning']}")
        print(f"   Params: {details['parameters']}")


def common_mistakes():
    """Highlight common mistakes in time series splitting"""
    
    print("\n" + "="*60)
    print("❌ COMMON MISTAKES TO AVOID")
    print("="*60)
    
    mistakes = [
        ("Random Splitting", 
         "Using train_test_split() from sklearn on time series data.",
         "Always use temporal splits!"),
        
        ("Too Small Test Sets", 
         "Using test sets smaller than forecast horizon.",
         "Test set should match your prediction horizon."),
        
        ("Data Leakage", 
         "Including future information in features.",
         "Be careful with lag features and rolling statistics."),
        
        ("Ignoring Seasonality", 
         "Splitting without considering seasonal patterns.",
         "Ensure test sets cover full seasonal cycles."),
        
        ("Fixed Cutoff Date", 
         "Always splitting at the same date regardless of data.",
         "Use percentage-based or rolling splits."),
        
        ("No Cross-Validation", 
         "Relying on single train/test split.",
         "Use multiple temporal splits for robust evaluation."),
        
        ("Overfitting to Test Set", 
         "Repeatedly evaluating and tuning on same test set.",
         "Use separate validation set or proper CV."),
        
        ("Ignoring Gaps", 
         "Not accounting for weekends, holidays, or missing data.",
         "Handle gaps appropriately in your splitting strategy.")
    ]
    
    for i, (mistake, description, solution) in enumerate(mistakes, 1):
        print(f"\n{i}. ❌ {mistake}")
        print(f"   Problem: {description}")
        print(f"   Solution: {solution}")


if __name__ == "__main__":
    # Set random seed for reproducible results
    np.random.seed(42)
    
    # Demonstrate all splitting methods
    print("Time Series Splitting Comprehensive Guide")
    print("="*50)
    
    # Create sample data and demonstrate splits
    ts = demonstrate_splitting_methods()
    
    # Show visualizations
    visualize_splits(ts)
    
    # Print best practices
    best_practices_guide()
    
    # Use case recommendations
    splitting_recommendations_by_use_case()
    
    # Common mistakes
    common_mistakes()
    
    print(f"\n{'='*60}")
    print("🎯 Key Takeaway: NEVER use random splits for time series!")
    print("Always respect the temporal order of your data.")
    print(f"{'='*60}") 