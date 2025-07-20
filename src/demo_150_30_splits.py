"""
Demo: 150 Days Training + 30 Days Forecasting Splits
Complete working example with sample data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from time_series_train_test_generator import create_150_30_splits, generate_multiple_splits_summary

def create_sample_stock_data():
    """Create sample stock data that mimics your actual data structure"""
    
    # Create 2 years of daily data
    dates = pd.date_range('2022-01-01', '2024-01-01', freq='D')
    np.random.seed(42)
    
    # Generate realistic stock price data
    initial_price = 100
    returns = np.random.normal(0.0005, 0.02, len(dates))  # Daily returns
    prices = [initial_price]
    
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    # Create DataFrame similar to your data
    sample_data = pd.DataFrame({
        'date': dates,
        'close': prices,
        'ticker': 'SAMPLE'
    })
    
    return sample_data

def demo_create_150_30_splits():
    """Demonstrate the 150/30 split function"""
    
    print("🎯 Demo: create_150_30_splits() Function")
    print("=" * 50)
    
    # Create sample data
    sample_data = create_sample_stock_data()
    print(f"📊 Sample data: {len(sample_data)} days")
    print(f"   Date range: {sample_data['date'].min().date()} to {sample_data['date'].max().date()}")
    
    # Create time series
    stock_ts = pd.Series(
        sample_data['close'].values,
        index=sample_data['date']
    )
    
    # Test different step sizes
    step_sizes = [1, 7, 30]
    
    for step_size in step_sizes:
        print(f"\n🔄 Testing step_size={step_size}")
        
        # Create splits
        splits = create_150_30_splits(stock_ts, step_size=step_size)
        
        print(f"   ✅ Created {len(splits)} splits")
        
        if len(splits) > 0:
            # Show first split details
            train, test = splits[0]
            print(f"   📅 First split:")
            print(f"      Train: {train.index[0].date()} to {train.index[-1].date()} ({len(train)} days)")
            print(f"      Test:  {test.index[0].date()} to {test.index[-1].date()} ({len(test)} days)")
            
            # Show split summary
            if len(splits) <= 10:
                summary = generate_multiple_splits_summary(splits)
                print(f"   📋 All splits:")
                print(summary.to_string(index=False))
            else:
                print(f"   📋 First 5 splits:")
                summary = generate_multiple_splits_summary(splits[:5])
                print(summary.to_string(index=False))
    
    # Return the most useful configuration
    optimal_splits = create_150_30_splits(stock_ts, step_size=30)
    return optimal_splits, stock_ts

def visualize_splits(splits, stock_ts):
    """Visualize the train/test splits"""
    
    print(f"\n📈 Visualizing splits...")
    
    # Create visualization
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    
    # Plot 1: Full time series with split boundaries
    axes[0].plot(stock_ts.index, stock_ts.values, color='blue', alpha=0.7, linewidth=1)
    axes[0].set_title('Full Time Series with Split Boundaries')
    axes[0].set_ylabel('Price')
    
    # Add vertical lines for split boundaries
    colors = ['red', 'green', 'orange', 'purple', 'brown']
    for i, (train, test) in enumerate(splits[:5]):  # Show first 5 splits
        color = colors[i % len(colors)]
        axes[0].axvline(x=train.index[0], color=color, alpha=0.5, linestyle='--', 
                       label=f'Split {i+1} start')
        axes[0].axvline(x=test.index[-1], color=color, alpha=0.5, linestyle='-', 
                       label=f'Split {i+1} end')
    
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Detailed view of first split
    if len(splits) > 0:
        train, test = splits[0]
        
        # Plot training data
        axes[1].plot(train.index, train.values, color='blue', linewidth=2, 
                    label=f'Training Data ({len(train)} days)')
        
        # Plot test data
        axes[1].plot(test.index, test.values, color='red', linewidth=2, 
                    label=f'Test Data ({len(test)} days)')
        
        # Add divider
        axes[1].axvline(x=train.index[-1], color='black', linestyle=':', alpha=0.7, 
                       label='Train/Test Split')
        
        axes[1].set_title('Detailed View: First Split (150 days train + 30 days test)')
        axes[1].set_xlabel('Date')
        axes[1].set_ylabel('Price')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def practical_usage_example(splits):
    """Show how to use the splits in practice"""
    
    print(f"\n💡 Practical Usage Example")
    print("=" * 30)
    
    print("Here's how you'd use these splits for model training:")
    print()
    
    # Example workflow
    results = []
    
    for i, (train, test) in enumerate(splits[:3]):  # Show first 3 splits
        print(f"📊 Split {i+1}:")
        print(f"   Train period: {train.index[0].date()} to {train.index[-1].date()}")
        print(f"   Test period:  {test.index[0].date()} to {test.index[-1].date()}")
        
        # Simulate model training and evaluation
        # In practice, you'd use your actual model here
        
        # Simple example: predict mean of training data
        train_mean = train.mean()
        predictions = np.full(len(test), train_mean)
        
        # Calculate error
        mae = np.mean(np.abs(test.values - predictions))
        results.append({
            'split': i+1,
            'train_mean': train_mean,
            'test_mean': test.mean(),
            'mae': mae
        })
        
        print(f"   Train mean: ${train_mean:.2f}")
        print(f"   Test mean:  ${test.mean():.2f}")
        print(f"   MAE:        ${mae:.2f}")
        print()
    
    # Summary
    results_df = pd.DataFrame(results)
    print(f"📋 Summary across splits:")
    print(f"   Average MAE: ${results_df['mae'].mean():.2f}")
    print(f"   MAE Std:     ${results_df['mae'].std():.2f}")
    
    return results_df

def integration_with_your_data():
    """Show how to integrate with your actual data"""
    
    print(f"\n🔗 Integration with Your Data")
    print("=" * 40)
    
    code_example = '''
# Step 1: Load your data
df = pd.read_csv('data/03_primary/combined_modeling_input.csv')
df['date'] = pd.to_datetime(df['date'])

# Step 2: Select ticker and create time series
ticker = 'AAPL'  # or any ticker you want
stock_data = df[df['ticker'] == ticker].copy()
stock_data = stock_data.sort_values('date')

# Step 3: Create time series
stock_ts = pd.Series(stock_data['close'].values, index=stock_data['date'])

# Step 4: Create 150/30 splits
splits = create_150_30_splits(stock_ts, step_size=30)

# Step 5: Use splits for training
for i, (train, test) in enumerate(splits):
    print(f"Split {i+1}: {len(train)} train days, {len(test)} test days")
    
    # Your model training code here
    # model = YourModel()
    # model.fit(train)
    # predictions = model.predict(steps=30)
    # mae = mean_absolute_error(test, predictions)
    
# Step 6: Collect results and analyze performance
'''
    
    print(code_example)

if __name__ == "__main__":
    print("Complete Demo: 150 Days Training + 30 Days Forecasting")
    print("=" * 60)
    
    # Run the demo
    splits, stock_ts = demo_create_150_30_splits()
    
    # Visualize (optional - comment out if no display)
    try:
        visualize_splits(splits, stock_ts)
    except Exception as e:
        print(f"⚠️  Visualization skipped: {e}")
    
    # Show practical usage
    results = practical_usage_example(splits)
    
    # Show integration guide
    integration_with_your_data()
    
    print(f"\n🎯 Summary:")
    print(f"   ✅ Function created: create_150_30_splits()")
    print(f"   ✅ Tested with sample data")
    print(f"   ✅ Generated {len(splits)} splits")
    print(f"   ✅ Each split: 150 train + 30 test days")
    print(f"   ✅ Ready to use with your stock data!") 