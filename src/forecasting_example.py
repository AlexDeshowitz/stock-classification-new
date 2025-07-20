"""
Simple example of using the Time Series Forecasting Library
"""

import pandas as pd
import numpy as np
from time_series_forecaster import TimeSeriesForecaster, quick_forecast, NaiveModel, LinearTrendModel

def create_sample_data():
    """Create sample time series data for demonstration"""
    dates = pd.date_range('2020-01-01', '2024-01-01', freq='D')
    
    # Create synthetic stock price data with trend and seasonality
    trend = np.linspace(100, 200, len(dates))
    seasonal = 10 * np.sin(2 * np.pi * np.arange(len(dates)) / 365.25)
    weekly = 5 * np.sin(2 * np.pi * np.arange(len(dates)) / 7)
    noise = np.random.normal(0, 5, len(dates))
    
    prices = trend + seasonal + weekly + noise
    
    return pd.Series(prices, index=dates)

def example_quick_forecast():
    """Example using the quick_forecast function"""
    print("=== Quick Forecast Example ===")
    
    # Create sample data
    ts = create_sample_data()
    print(f"Created sample time series with {len(ts)} points")
    
    # Quick forecast
    results = quick_forecast(ts, steps=30)
    
    print(f"Best model: {results.get('best_model', 'N/A')}")
    
    if 'cv_results' in results:
        print("\nCross-validation results:")
        for model, scores in results['cv_results'].items():
            print(f"  {model}: MAE={scores['mae_mean']:.2f}, RMSE={scores['rmse_mean']:.2f}")
    
    return results

def example_comprehensive_forecast():
    """Example using the comprehensive TimeSeriesForecaster"""
    print("\n=== Comprehensive Forecast Example ===")
    
    # Create sample data
    ts = create_sample_data()
    
    # Initialize forecaster
    forecaster = TimeSeriesForecaster()
    
    # Add models manually
    forecaster.add_model(NaiveModel(method='last'))
    forecaster.add_model(NaiveModel(method='seasonal', seasonal_period=7))
    forecaster.add_model(LinearTrendModel(include_seasonal=True, seasonal_period=7))
    
    print(f"Added {len(forecaster.models)} models")
    
    # Run comprehensive forecast
    results = forecaster.forecast(ts, steps=30, run_cv=True)
    
    print(f"Best model: {results.get('best_model', 'N/A')}")
    
    # Display results summary
    summary = forecaster.get_results_summary()
    print("\nModel Performance Summary:")
    print(summary.to_string(index=False))
    
    return results, forecaster

def forecast_stock_data(ticker='AAPL', data_path='../data/03_primary/combined_modeling_input.csv'):
    """Example forecasting real stock data"""
    print(f"\n=== Real Stock Data Forecast: {ticker} ===")
    
    try:
        # Load data
        df = pd.read_csv(data_path)
        df['date'] = pd.to_datetime(df['date'])
        
        # Filter for specific ticker
        stock_data = df[df['ticker'] == ticker].copy()
        stock_data = stock_data.sort_values('date')
        
        # Create time series
        ts = pd.Series(
            stock_data['close'].values,
            index=stock_data['date']
        )
        
        print(f"Loaded {len(ts)} data points for {ticker}")
        print(f"Date range: {ts.index.min()} to {ts.index.max()}")
        
        # Quick forecast
        results = quick_forecast(ts, steps=30)
        
        print(f"Best model: {results.get('best_model', 'N/A')}")
        
        if 'forecast' in results:
            current_price = ts.iloc[-1]
            forecast_end = results['forecast'].iloc[-1]
            change_pct = ((forecast_end / current_price) - 1) * 100
            
            print(f"Current price: ${current_price:.2f}")
            print(f"30-day forecast: ${forecast_end:.2f}")
            print(f"Expected change: {change_pct:+.2f}%")
        
        return results
        
    except FileNotFoundError:
        print(f"Data file not found: {data_path}")
        return None
    except Exception as e:
        print(f"Error processing {ticker}: {e}")
        return None

if __name__ == "__main__":
    # Set random seed for reproducible results
    np.random.seed(42)
    
    # Run examples
    print("Time Series Forecasting Examples")
    print("=" * 50)
    
    # Example 1: Quick forecast with synthetic data
    quick_results = example_quick_forecast()
    
    # Example 2: Comprehensive forecast with synthetic data
    comp_results, forecaster = example_comprehensive_forecast()
    
    # Example 3: Real stock data (if available)
    stock_results = forecast_stock_data()
    
    print("\n" + "=" * 50)
    print("Examples completed!")
    
    # Basic usage instructions
    print("\nBasic Usage:")
    print("1. Import: from time_series_forecaster import quick_forecast")
    print("2. Use: results = quick_forecast(your_time_series, steps=30)")
    print("3. Access: best_forecast = results['forecast']") 