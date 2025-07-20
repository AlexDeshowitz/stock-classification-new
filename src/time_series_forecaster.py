"""
Time Series Forecasting Library
Comprehensive forecasting toolkit with proper cross-validation and multiple models
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Optional imports for advanced models
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.exponential_smoothing.ets import ETSModel
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False


class TimeSeriesCrossValidator:
    """Time series cross-validation with proper temporal splits"""
    
    def __init__(self, n_splits: int = 5, test_size: int = 30, gap: int = 0):
        """
        Initialize time series cross validator
        
        Args:
            n_splits: Number of cross-validation splits
            test_size: Size of test set in each split
            gap: Gap between train and test sets
        """
        self.n_splits = n_splits
        self.test_size = test_size
        self.gap = gap
    
    def split(self, X: pd.DataFrame) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate time series splits
        
        Args:
            X: Input dataframe with time index
            
        Returns:
            List of (train_idx, test_idx) tuples
        """
        n_samples = len(X)
        splits = []
        
        # Calculate split points
        total_test_size = self.test_size * self.n_splits + self.gap * (self.n_splits - 1)
        min_train_size = n_samples - total_test_size
        
        for i in range(self.n_splits):
            # Calculate indices for this split
            test_end = n_samples - i * (self.test_size + self.gap)
            test_start = test_end - self.test_size
            train_end = test_start - self.gap
            train_start = 0
            
            if train_end <= train_start:
                break
                
            train_idx = np.arange(train_start, train_end)
            test_idx = np.arange(test_start, test_end)
            
            splits.append((train_idx, test_idx))
        
        return splits[::-1]  # Return in chronological order


class BaseTimeSeriesModel:
    """Base class for time series models"""
    
    def __init__(self, name: str):
        self.name = name
        self.model = None
        self.is_fitted = False
    
    def fit(self, y: pd.Series, X: Optional[pd.DataFrame] = None):
        """Fit the model"""
        raise NotImplementedError
    
    def predict(self, steps: int, X: Optional[pd.DataFrame] = None) -> np.ndarray:
        """Make predictions"""
        raise NotImplementedError
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters"""
        return {}


class NaiveModel(BaseTimeSeriesModel):
    """Naive forecasting model (last value or seasonal naive)"""
    
    def __init__(self, method: str = 'last', seasonal_period: int = 1):
        super().__init__('Naive')
        self.method = method
        self.seasonal_period = seasonal_period
        self.last_values = None
    
    def fit(self, y: pd.Series, X: Optional[pd.DataFrame] = None):
        self.last_values = y.values
        self.is_fitted = True
        return self
    
    def predict(self, steps: int, X: Optional[pd.DataFrame] = None) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        if self.method == 'last':
            return np.full(steps, self.last_values[-1])
        elif self.method == 'seasonal':
            # Seasonal naive
            period = min(self.seasonal_period, len(self.last_values))
            seasonal_values = self.last_values[-period:]
            predictions = np.tile(seasonal_values, (steps // period) + 1)[:steps]
            return predictions
        else:
            return np.full(steps, np.mean(self.last_values))


class LinearTrendModel(BaseTimeSeriesModel):
    """Linear regression with time trend"""
    
    def __init__(self, include_seasonal: bool = False, seasonal_period: int = 12):
        super().__init__('LinearTrend')
        self.include_seasonal = include_seasonal
        self.seasonal_period = seasonal_period
        self.model = LinearRegression()
        self.scaler = StandardScaler()
        self.trend_end = None
    
    def _create_features(self, length: int, start_idx: int = 0) -> pd.DataFrame:
        """Create time-based features"""
        features = pd.DataFrame({
            'trend': np.arange(start_idx, start_idx + length)
        })
        
        if self.include_seasonal:
            for i in range(1, self.seasonal_period):
                features[f'seasonal_{i}'] = np.sin(2 * np.pi * i * features['trend'] / self.seasonal_period)
                features[f'seasonal_cos_{i}'] = np.cos(2 * np.pi * i * features['trend'] / self.seasonal_period)
        
        return features
    
    def fit(self, y: pd.Series, X: Optional[pd.DataFrame] = None):
        features = self._create_features(len(y))
        features_scaled = self.scaler.fit_transform(features)
        self.model.fit(features_scaled, y.values)
        self.trend_end = len(y)
        self.is_fitted = True
        return self
    
    def predict(self, steps: int, X: Optional[pd.DataFrame] = None) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        future_features = self._create_features(steps, self.trend_end)
        future_features_scaled = self.scaler.transform(future_features)
        predictions = self.model.predict(future_features_scaled)
        return predictions


class ARIMAModel(BaseTimeSeriesModel):
    """ARIMA model wrapper"""
    
    def __init__(self, order: Tuple[int, int, int] = (1, 1, 1), seasonal_order: Optional[Tuple[int, int, int, int]] = None):
        super().__init__('ARIMA')
        self.order = order
        self.seasonal_order = seasonal_order
        self.fitted_values = None
        
        if not STATSMODELS_AVAILABLE:
            raise ImportError("statsmodels is required for ARIMA model")
    
    def fit(self, y: pd.Series, X: Optional[pd.DataFrame] = None):
        try:
            self.model = ARIMA(y, order=self.order, seasonal_order=self.seasonal_order)
            self.fitted_model = self.model.fit()
            self.is_fitted = True
        except Exception as e:
            print(f"ARIMA fitting failed: {e}")
            # Fallback to simpler model
            self.model = ARIMA(y, order=(1, 1, 1))
            self.fitted_model = self.model.fit()
            self.is_fitted = True
        return self
    
    def predict(self, steps: int, X: Optional[pd.DataFrame] = None) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        forecast = self.fitted_model.forecast(steps=steps)
        return forecast.values if hasattr(forecast, 'values') else forecast


class ProphetModel(BaseTimeSeriesModel):
    """Facebook Prophet model wrapper"""
    
    def __init__(self, yearly_seasonality: bool = True, weekly_seasonality: bool = True, daily_seasonality: bool = False):
        super().__init__('Prophet')
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.last_date = None
        
        if not PROPHET_AVAILABLE:
            raise ImportError("prophet is required for Prophet model")
    
    def fit(self, y: pd.Series, X: Optional[pd.DataFrame] = None):
        # Prepare data for Prophet
        df = pd.DataFrame({
            'ds': y.index,
            'y': y.values
        })
        
        self.model = Prophet(
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality
        )
        self.model.fit(df)
        self.last_date = y.index[-1]
        self.is_fitted = True
        return self
    
    def predict(self, steps: int, X: Optional[pd.DataFrame] = None) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Create future dates
        future_dates = pd.date_range(
            start=self.last_date + pd.Timedelta(days=1),
            periods=steps,
            freq='D'
        )
        
        future_df = pd.DataFrame({'ds': future_dates})
        forecast = self.model.predict(future_df)
        return forecast['yhat'].values


class RandomForestTimeSeriesModel(BaseTimeSeriesModel):
    """Random Forest for time series with lag features"""
    
    def __init__(self, n_lags: int = 5, n_estimators: int = 100):
        super().__init__('RandomForest')
        self.n_lags = n_lags
        self.model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
        self.last_values = None
    
    def _create_lag_features(self, y: pd.Series) -> pd.DataFrame:
        """Create lagged features"""
        features = pd.DataFrame()
        for lag in range(1, self.n_lags + 1):
            features[f'lag_{lag}'] = y.shift(lag)
        return features.dropna()
    
    def fit(self, y: pd.Series, X: Optional[pd.DataFrame] = None):
        features = self._create_lag_features(y)
        target = y[features.index]
        
        self.model.fit(features, target)
        self.last_values = y.tail(self.n_lags).values
        self.is_fitted = True
        return self
    
    def predict(self, steps: int, X: Optional[pd.DataFrame] = None) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        predictions = []
        current_values = self.last_values.copy()
        
        for _ in range(steps):
            # Create features from current values
            features = current_values[-self.n_lags:].reshape(1, -1)
            pred = self.model.predict(features)[0]
            predictions.append(pred)
            
            # Update current values for next prediction
            current_values = np.append(current_values[1:], pred)
        
        return np.array(predictions)


class TimeSeriesForecaster:
    """Main forecasting class that orchestrates multiple models"""
    
    def __init__(self):
        self.models = {}
        self.results = {}
        self.best_model = None
        self.cv_results = {}
    
    def add_model(self, model: BaseTimeSeriesModel):
        """Add a model to the forecasting pipeline"""
        self.models[model.name] = model
    
    def add_default_models(self):
        """Add a default set of models"""
        self.add_model(NaiveModel(method='last'))
        self.add_model(NaiveModel(method='seasonal', seasonal_period=7))
        self.add_model(LinearTrendModel(include_seasonal=True, seasonal_period=7))
        self.add_model(RandomForestTimeSeriesModel(n_lags=7))
        
        if STATSMODELS_AVAILABLE:
            self.add_model(ARIMAModel(order=(1, 1, 1)))
        
        if PROPHET_AVAILABLE:
            self.add_model(ProphetModel())
    
    def cross_validate(self, y: pd.Series, n_splits: int = 5, test_size: int = 30) -> Dict[str, Dict[str, float]]:
        """Perform time series cross-validation"""
        cv = TimeSeriesCrossValidator(n_splits=n_splits, test_size=test_size)
        splits = cv.split(pd.DataFrame(index=y.index))
        
        cv_results = {}
        
        for model_name, model in self.models.items():
            print(f"Cross-validating {model_name}...")
            model_scores = {'mae': [], 'rmse': [], 'mape': []}
            
            for train_idx, test_idx in splits:
                try:
                    # Split data
                    y_train = y.iloc[train_idx]
                    y_test = y.iloc[test_idx]
                    
                    # Fit and predict
                    model_copy = self._copy_model(model)
                    model_copy.fit(y_train)
                    predictions = model_copy.predict(len(y_test))
                    
                    # Calculate metrics
                    mae = mean_absolute_error(y_test, predictions)
                    rmse = np.sqrt(mean_squared_error(y_test, predictions))
                    mape = mean_absolute_percentage_error(y_test, predictions) * 100
                    
                    model_scores['mae'].append(mae)
                    model_scores['rmse'].append(rmse)
                    model_scores['mape'].append(mape)
                
                except Exception as e:
                    print(f"Error in {model_name}: {e}")
                    continue
            
            # Calculate mean scores
            if model_scores['mae']:
                cv_results[model_name] = {
                    'mae_mean': np.mean(model_scores['mae']),
                    'mae_std': np.std(model_scores['mae']),
                    'rmse_mean': np.mean(model_scores['rmse']),
                    'rmse_std': np.std(model_scores['rmse']),
                    'mape_mean': np.mean(model_scores['mape']),
                    'mape_std': np.std(model_scores['mape'])
                }
        
        self.cv_results = cv_results
        return cv_results
    
    def fit_all_models(self, y: pd.Series, X: Optional[pd.DataFrame] = None):
        """Fit all models"""
        for model_name, model in self.models.items():
            try:
                print(f"Fitting {model_name}...")
                model.fit(y, X)
                print(f"✓ {model_name} fitted successfully")
            except Exception as e:
                print(f"✗ Error fitting {model_name}: {e}")
    
    def predict_all_models(self, steps: int, X: Optional[pd.DataFrame] = None) -> Dict[str, np.ndarray]:
        """Generate predictions from all fitted models"""
        predictions = {}
        
        for model_name, model in self.models.items():
            if model.is_fitted:
                try:
                    pred = model.predict(steps, X)
                    predictions[model_name] = pred
                    print(f"✓ {model_name} predictions generated")
                except Exception as e:
                    print(f"✗ Error predicting with {model_name}: {e}")
        
        return predictions
    
    def select_best_model(self, metric: str = 'mae_mean') -> str:
        """Select best model based on cross-validation results"""
        if not self.cv_results:
            raise ValueError("No cross-validation results available. Run cross_validate() first.")
        
        best_score = float('inf')
        best_model_name = None
        
        for model_name, scores in self.cv_results.items():
            if metric in scores and scores[metric] < best_score:
                best_score = scores[metric]
                best_model_name = model_name
        
        self.best_model = best_model_name
        return best_model_name
    
    def forecast(self, y: pd.Series, steps: int, use_best_model: bool = True, 
                run_cv: bool = True, X: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Complete forecasting pipeline
        
        Args:
            y: Time series data
            steps: Number of steps to forecast
            use_best_model: Whether to return only best model results
            run_cv: Whether to run cross-validation
            X: External features (optional)
            
        Returns:
            Dictionary with predictions and model performance
        """
        results = {}
        
        # Add default models if none exist
        if not self.models:
            self.add_default_models()
        
        # Cross-validation
        if run_cv:
            print("Running cross-validation...")
            cv_results = self.cross_validate(y)
            results['cv_results'] = cv_results
            
            # Select best model
            best_model_name = self.select_best_model()
            results['best_model'] = best_model_name
            print(f"Best model: {best_model_name}")
        
        # Fit models on full data
        print("\nFitting models on full dataset...")
        self.fit_all_models(y, X)
        
        # Generate predictions
        print("\nGenerating predictions...")
        predictions = self.predict_all_models(steps, X)
        results['predictions'] = predictions
        
        # Create forecast dates
        if hasattr(y.index, 'freq') and y.index.freq:
            freq = y.index.freq
        else:
            freq = pd.infer_freq(y.index) or 'D'
        
        forecast_index = pd.date_range(
            start=y.index[-1] + pd.Timedelta(days=1),
            periods=steps,
            freq=freq
        )
        results['forecast_dates'] = forecast_index
        
        # Return best model results if requested
        if use_best_model and self.best_model and self.best_model in predictions:
            results['forecast'] = pd.Series(
                predictions[self.best_model], 
                index=forecast_index
            )
        
        return results
    
    def _copy_model(self, model: BaseTimeSeriesModel) -> BaseTimeSeriesModel:
        """Create a copy of a model for cross-validation"""
        model_class = model.__class__
        
        if isinstance(model, NaiveModel):
            return NaiveModel(model.method, model.seasonal_period)
        elif isinstance(model, LinearTrendModel):
            return LinearTrendModel(model.include_seasonal, model.seasonal_period)
        elif isinstance(model, RandomForestTimeSeriesModel):
            return RandomForestTimeSeriesModel(model.n_lags, model.model.n_estimators)
        elif STATSMODELS_AVAILABLE and isinstance(model, ARIMAModel):
            return ARIMAModel(model.order, model.seasonal_order)
        elif PROPHET_AVAILABLE and isinstance(model, ProphetModel):
            return ProphetModel(model.yearly_seasonality, model.weekly_seasonality, model.daily_seasonality)
        else:
            return model_class()
    
    def get_results_summary(self) -> pd.DataFrame:
        """Get a summary of cross-validation results"""
        if not self.cv_results:
            return pd.DataFrame()
        
        summary_data = []
        for model_name, scores in self.cv_results.items():
            summary_data.append({
                'Model': model_name,
                'MAE (mean ± std)': f"{scores['mae_mean']:.4f} ± {scores['mae_std']:.4f}",
                'RMSE (mean ± std)': f"{scores['rmse_mean']:.4f} ± {scores['rmse_std']:.4f}",
                'MAPE (mean ± std)': f"{scores['mape_mean']:.2f}% ± {scores['mape_std']:.2f}%"
            })
        
        return pd.DataFrame(summary_data).sort_values('Model')


# Convenience function for quick forecasting
def quick_forecast(y: pd.Series, steps: int = 30, models: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Quick forecasting function with sensible defaults
    
    Args:
        y: Time series data
        steps: Number of steps to forecast (default: 30)
        models: List of model names to use (default: all available)
        
    Returns:
        Dictionary with forecast results
    """
    forecaster = TimeSeriesForecaster()
    
    if models is None:
        forecaster.add_default_models()
    else:
        # Add specific models based on names
        model_map = {
            'naive': NaiveModel(),
            'seasonal_naive': NaiveModel(method='seasonal', seasonal_period=7),
            'linear_trend': LinearTrendModel(include_seasonal=True),
            'random_forest': RandomForestTimeSeriesModel(),
        }
        
        if STATSMODELS_AVAILABLE:
            model_map['arima'] = ARIMAModel()
        
        if PROPHET_AVAILABLE:
            model_map['prophet'] = ProphetModel()
        
        for model_name in models:
            if model_name.lower() in model_map:
                forecaster.add_model(model_map[model_name.lower()])
    
    return forecaster.forecast(y, steps)


if __name__ == "__main__":
    # Example usage
    print("Time Series Forecasting Library")
    print("Available models:", ['Naive', 'LinearTrend', 'RandomForest'])
    if STATSMODELS_AVAILABLE:
        print("✓ ARIMA available")
    if PROPHET_AVAILABLE:
        print("✓ Prophet available") 