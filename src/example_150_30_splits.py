"""
Example: Using create_150_30_splits() with your stock data
"""

import pandas as pd
import numpy as np
from time_series_train_test_generator import create_150_30_splits, generate_multiple_splits_summary

def example_with_your_data():
    """Example using your actual stock data format"""
    
    # Load your data (replace with your actual path)
    df = pd.read_csv('../data/03_primary/combined_modeling_input.csv')
    df['date'] = pd.to_datetime(df['date'])
    
    # Select a specific ticker
    ticker = 'AAPL'
    stock_data = df[df['ticker'] == ticker].copy()
    stock_data = stock_data.sort_values('date')
    
    # Create time series for the close price
    stock_ts = pd.Series(
        stock_data['close'].values, 
        index=stock_data['date']
    )
    
    print(f"Stock data for {ticker}:")
    print(f"  Total days: {len(stock_ts)}")
    print(f"  Date range: {stock_ts.index[0].date()} to {stock_ts.index[-1].date()}")
    
    # Create 150/30 splits
    splits = create_150_30_splits(
        stock_ts, 
        step_size=30  # New split every 30 days
    )
    
    print(f"\n✅ Created {len(splits)} train/test splits")
    print("Each split: 150 days training → 30 days forecasting")
    
    # Show first few splits
    print("\nFirst 5 splits:")
    for i, (train, test) in enumerate(splits[:5]):
        print(f"  Split {i+1}: {train.index[0].date()} to {train.index[-1].date()} → {test.index[0].date()} to {test.index[-1].date()}")
    
    # Generate summary table
    summary = generate_multiple_splits_summary(splits)
    print(f"\nSummary table (first 10 splits):")
    print(summary.head(10).to_string(index=False))
    
    return splits

def use_splits_for_training():
    """Example of how to use the splits for training models"""
    
    print("\n" + "="*50)
    print("📚 How to use splits for training:")
    print("="*50)
    
    # Example code structure
    example_code = '''
    # 1. Create your splits
    splits = create_150_30_splits(stock_ts, step_size=30)
    
    # 2. Loop through each split for training/testing
    results = []
    
    for i, (train, test) in enumerate(splits):
        print(f"Processing split {i+1}/{len(splits)}")
        
        # Your model training
        # model = YourModel()
        # model.fit(train)
        
        # Your predictions
        # predictions = model.predict(steps=30)
        
        # Evaluate performance
        # mae = mean_absolute_error(test, predictions)
        # results.append({'split': i+1, 'mae': mae})
    
    # 3. Analyze results across all splits
    # performance_df = pd.DataFrame(results)
    # print(f"Average MAE: {performance_df['mae'].mean():.4f}")
    '''
    
    print(example_code)

if __name__ == "__main__":
    print("Example: 150 Days Training + 30 Days Forecasting")
    print("=" * 50)
    
    try:
        # Run example with your data
        splits = example_with_your_data()
        
        # Show usage example
        use_splits_for_training()
        
    except FileNotFoundError:
        print("⚠️  Could not find your data file. Here's how to use it:")
        print("""
        # With your data:
        df = pd.read_csv('path/to/your/data.csv')
        df['date'] = pd.to_datetime(df['date'])
        
        # Select ticker and create time series
        ticker_data = df[df['ticker'] == 'AAPL']
        stock_ts = pd.Series(ticker_data['close'].values, index=ticker_data['date'])
        
        # Create splits
        splits = create_150_30_splits(stock_ts, step_size=30)
        
        # Use splits for training
        for train, test in splits:
            # Your model training code here
            pass
        """)
        
    print("\n🎯 Key Points:")
    print("• step_size=1: Maximum overlap (sliding window)")
    print("• step_size=30: Non-overlapping splits")
    print("• step_size=7: Weekly intervals")
    print("• Each split guarantees exactly 150 train + 30 test days") 