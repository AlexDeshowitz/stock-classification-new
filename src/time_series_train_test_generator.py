"""
Time Series Training and Test Set Generator
Creates multiple train/test splits with specified window sizes
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import warnings
warnings.filterwarnings('ignore')


def create_time_series_splits(
    data: pd.Series,
    train_window: int = 150,
    forecast_horizon: int = 30,
    step_size: int = 1,
    min_periods: Optional[int] = None
) -> List[Tuple[pd.Series, pd.Series]]:
    """
    Create multiple train/test splits for time series forecasting
    
    Args:
        data: Time series data with datetime index
        train_window: Number of days for training (default: 150)
        forecast_horizon: Number of days to forecast ahead (default: 30)
        step_size: Step size between consecutive splits (default: 1)
        min_periods: Minimum periods required (default: train_window)
        
    Returns:
        List of (train_data, test_data) tuples
        
    Example:
        >>> ts = pd.Series([...], index=pd.date_range('2020-01-01', periods=500))
        >>> splits = create_time_series_splits(ts, train_window=150, forecast_horizon=30)
        >>> print(f"Created {len(splits)} train/test splits")
    """
    
    if min_periods is None:
        min_periods = train_window
    
    if len(data) < min_periods + forecast_horizon:
        raise ValueError(f"Data length ({len(data)}) is too short for train_window ({train_window}) + forecast_horizon ({forecast_horizon})")
    
    splits = []
    
    # Start from the minimum required position
    start_pos = 0
    
    while start_pos + train_window + forecast_horizon <= len(data):
        # Define train and test windows
        train_end = start_pos + train_window
        test_start = train_end
        test_end = test_start + forecast_horizon
        
        # Extract train and test data
        train_data = data.iloc[start_pos:train_end]
        test_data = data.iloc[test_start:test_end]
        
        # Only add if we have enough data
        if len(train_data) >= min_periods and len(test_data) == forecast_horizon:
            splits.append((train_data, test_data))
        
        # Move to next position
        start_pos += step_size
    
    return splits


def create_time_series_splits_with_features(
    data: pd.DataFrame,
    target_column: str,
    feature_columns: Optional[List[str]] = None,
    train_window: int = 150,
    forecast_horizon: int = 30,
    step_size: int = 1,
    min_periods: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Create train/test splits with both target and feature variables
    
    Args:
        data: DataFrame with datetime index
        target_column: Name of target column to forecast
        feature_columns: List of feature column names (optional)
        train_window: Number of days for training
        forecast_horizon: Number of days to forecast ahead
        step_size: Step size between splits
        min_periods: Minimum periods required
        
    Returns:
        List of dictionaries with keys: 'X_train', 'y_train', 'X_test', 'y_test'
    """
    
    if target_column not in data.columns:
        raise ValueError(f"Target column '{target_column}' not found in data")
    
    if feature_columns is None:
        feature_columns = [col for col in data.columns if col != target_column]
    
    if min_periods is None:
        min_periods = train_window
    
    splits = []
    start_pos = 0
    
    while start_pos + train_window + forecast_horizon <= len(data):
        # Define windows
        train_end = start_pos + train_window
        test_start = train_end
        test_end = test_start + forecast_horizon
        
        # Extract data
        train_data = data.iloc[start_pos:train_end]
        test_data = data.iloc[test_start:test_end]
        
        if len(train_data) >= min_periods and len(test_data) == forecast_horizon:
            split_dict = {
                'X_train': train_data[feature_columns],
                'y_train': train_data[target_column],
                'X_test': test_data[feature_columns],
                'y_test': test_data[target_column],
                'train_dates': train_data.index,
                'test_dates': test_data.index
            }
            splits.append(split_dict)
        
        start_pos += step_size
    
    return splits


def create_stock_forecasting_splits(
    stock_data: pd.DataFrame,
    ticker: str,
    target_column: str = 'close',
    train_window: int = 150,
    forecast_horizon: int = 30,
    step_size: int = 30,
    include_features: bool = True
) -> List[Dict[str, Any]]:
    """
    Create train/test splits specifically for stock forecasting
    
    Args:
        stock_data: DataFrame with stock data
        ticker: Stock ticker symbol
        target_column: Column to forecast (default: 'close')
        train_window: Training window size (default: 150 days)
        forecast_horizon: Forecast horizon (default: 30 days)
        step_size: Step between splits (default: 30 days)
        include_features: Whether to include feature columns
        
    Returns:
        List of split dictionaries
    """
    
    # Filter for specific ticker
    if 'ticker' in stock_data.columns:
        ticker_data = stock_data[stock_data['ticker'] == ticker].copy()
    else:
        ticker_data = stock_data.copy()
    
    # Sort by date
    if 'date' in ticker_data.columns:
        ticker_data = ticker_data.sort_values('date')
        ticker_data.set_index('date', inplace=True)
    
    # Remove ticker column if it exists (since we've filtered)
    if 'ticker' in ticker_data.columns:
        ticker_data = ticker_data.drop('ticker', axis=1)
    
    if target_column not in ticker_data.columns:
        raise ValueError(f"Target column '{target_column}' not found in data")
    
    if include_features:
        # Use all columns except target as features
        feature_columns = [col for col in ticker_data.columns if col != target_column]
        
        splits = create_time_series_splits_with_features(
            ticker_data,
            target_column=target_column,
            feature_columns=feature_columns,
            train_window=train_window,
            forecast_horizon=forecast_horizon,
            step_size=step_size
        )
    else:
        # Use only target column
        target_series = ticker_data[target_column]
        splits = create_time_series_splits(
            target_series,
            train_window=train_window,
            forecast_horizon=forecast_horizon,
            step_size=step_size
        )
        
        # Convert to dictionary format for consistency
        splits = [
            {
                'y_train': split[0],
                'y_test': split[1],
                'train_dates': split[0].index,
                'test_dates': split[1].index
            }
            for split in splits
        ]
    
    return splits


def generate_multiple_splits_summary(splits: List[Any]) -> pd.DataFrame:
    """
    Generate a summary of the created splits
    
    Args:
        splits: List of split dictionaries or tuples
        
    Returns:
        DataFrame with split information
    """
    
    summary_data = []
    
    for i, split in enumerate(splits):
        if isinstance(split, dict):
            # Dictionary format (with features)
            if 'y_train' in split:
                train_start = split['train_dates'][0]
                train_end = split['train_dates'][-1]
                test_start = split['test_dates'][0]
                test_end = split['test_dates'][-1]
                train_size = len(split['y_train'])
                test_size = len(split['y_test'])
            else:
                # Fallback for other dictionary formats
                continue
        else:
            # Tuple format (train, test)
            train_data, test_data = split
            train_start = train_data.index[0]
            train_end = train_data.index[-1]
            test_start = test_data.index[0] 
            test_end = test_data.index[-1]
            train_size = len(train_data)
            test_size = len(test_data)
        
        summary_data.append({
            'Split': i + 1,
            'Train_Start': train_start.strftime('%Y-%m-%d'),
            'Train_End': train_end.strftime('%Y-%m-%d'),
            'Test_Start': test_start.strftime('%Y-%m-%d'),
            'Test_End': test_end.strftime('%Y-%m-%d'),
            'Train_Size': train_size,
            'Test_Size': test_size,
            'Gap_Days': (test_start - train_end).days
        })
    
    return pd.DataFrame(summary_data)


def demonstrate_split_creation():
    """Demonstrate the split creation functions"""
    
    print("=== Time Series Train/Test Split Generation ===")
    
    # Create sample data
    dates = pd.date_range('2020-01-01', '2024-01-01', freq='D')
    np.random.seed(42)
    
    # Sample stock data
    stock_data = pd.DataFrame({
        'date': dates,
        'close': 100 + np.cumsum(np.random.randn(len(dates)) * 0.5),
        'volume': np.random.randint(1000000, 10000000, len(dates)),
        'high': 100 + np.cumsum(np.random.randn(len(dates)) * 0.5) + 2,
        'low': 100 + np.cumsum(np.random.randn(len(dates)) * 0.5) - 2,
        'ticker': 'DEMO'
    })
    
    print(f"Sample data: {len(stock_data)} days from {dates[0].date()} to {dates[-1].date()}")
    
    # Method 1: Simple target-only splits
    print("\n1. Creating simple target-only splits...")
    target_series = pd.Series(stock_data['close'].values, index=stock_data['date'])
    simple_splits = create_time_series_splits(
        target_series,
        train_window=150,
        forecast_horizon=30,
        step_size=60  # Create splits every 60 days
    )
    
    print(f"   Created {len(simple_splits)} splits")
    print(f"   First split: Train {len(simple_splits[0][0])} days, Test {len(simple_splits[0][1])} days")
    
    # Method 2: Stock-specific splits with features
    print("\n2. Creating stock forecasting splits with features...")
    stock_splits = create_stock_forecasting_splits(
        stock_data,
        ticker='DEMO',
        target_column='close',
        train_window=150,
        forecast_horizon=30,
        step_size=30,
        include_features=True
    )
    
    print(f"   Created {len(stock_splits)} splits")
    if stock_splits:
        print(f"   Features in each split: {stock_splits[0]['X_train'].columns.tolist()}")
        print(f"   First split train shape: {stock_splits[0]['X_train'].shape}")
        print(f"   First split test shape: {stock_splits[0]['X_test'].shape}")
    
    # Method 3: Generate summary
    print("\n3. Split summary:")
    summary = generate_multiple_splits_summary(stock_splits)
    print(summary.head().to_string(index=False))
    
    return stock_splits, summary


# Main function for your specific use case
def create_150_30_splits(
    data: pd.Series,
    step_size: int = 1
) -> List[Tuple[pd.Series, pd.Series]]:
    """
    Create train/test splits with exactly 150 days training and 30 days forecasting
    This is the main function you requested!
    
    Args:
        data: Time series data with datetime index
        step_size: Step size between splits (default: 1 for maximum overlap)
        
    Returns:
        List of (train_150_days, test_30_days) tuples
        
    Example:
        >>> # Load your stock data
        >>> stock_ts = pd.Series(stock_data['close'].values, index=stock_data['date'])
        >>> splits = create_150_30_splits(stock_ts, step_size=30)
        >>> print(f"Created {len(splits)} train/test splits")
        >>> 
        >>> # Use first split
        >>> train, test = splits[0]
        >>> print(f"Train: {len(train)} days, Test: {len(test)} days")
    """
    
    return create_time_series_splits(
        data=data,
        train_window=150,
        forecast_horizon=30,
        step_size=step_size,
        min_periods=150
    )


if __name__ == "__main__":
    # Demonstrate the functions
    print("Time Series Train/Test Split Generator")
    print("=" * 50)
    
    # Run demonstrations
    splits, summary = demonstrate_split_creation()
    
    print(f"\n{'='*50}")
    print("🎯 Main Function: create_150_30_splits()")
    print("   - 150 days training")
    print("   - 30 days forecasting")
    print("   - Sliding window with replacement")
    print("   - Use step_size=1 for maximum overlap")
    print("   - Use step_size=30 for non-overlapping splits")
    print(f"{'='*50}")
    
    # Example usage
    print("\n📖 Example Usage:")
    print("""
    # For your stock data:
    stock_ts = pd.Series(df['close'].values, index=df['date'])
    splits = create_150_30_splits(stock_ts, step_size=30)
    
    # Use splits for training/testing
    for i, (train, test) in enumerate(splits):
        print(f"Split {i+1}: {train.index[0]} to {train.index[-1]} → {test.index[0]} to {test.index[-1]}")
        # Your model training and testing code here
    """) 