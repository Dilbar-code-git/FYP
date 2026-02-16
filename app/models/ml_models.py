import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
import warnings

warnings.filterwarnings('ignore')

class ModelTrainer:
    """Class to train and evaluate multiple ML models"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.results = {}
    
    def train_linear_regression(self, X_train, y_train):
        """Train Linear Regression model"""
        try:
            model = LinearRegression()
            model.fit(X_train, y_train)
            self.models['Linear Regression'] = model
            return model
        except Exception as e:
            raise Exception(f"Error training Linear Regression: {str(e)}")
    
    def train_random_forest(self, X_train, y_train, n_estimators=100):
        """Train Random Forest model"""
        try:
            model = RandomForestRegressor(
                n_estimators=n_estimators,
                random_state=self.random_state,
                n_jobs=-1
            )
            model.fit(X_train, y_train)
            self.models['Random Forest'] = model
            return model
        except Exception as e:
            raise Exception(f"Error training Random Forest: {str(e)}")
    
    def train_xgboost(self, X_train, y_train):
        """Train XGBoost model"""
        try:
            model = XGBRegressor(
                random_state=self.random_state,
                n_jobs=-1,
                verbosity=0
            )
            model.fit(X_train, y_train)
            self.models['XGBoost'] = model
            return model
        except Exception as e:
            raise Exception(f"Error training XGBoost: {str(e)}")
    
    def train_arima(self, series, order=(5, 1, 2)):
        """Train ARIMA model for time series"""
        try:
            from statsmodels.tsa.arima.model import ARIMA
            model = ARIMA(series, order=order)
            fitted_model = model.fit()
            self.models['ARIMA'] = fitted_model
            return fitted_model
        except Exception as e:
            raise Exception(f"Error training ARIMA: {str(e)}")
    
    def evaluate_model(self, model_name, X_test, y_test):
        """Evaluate a trained model"""
        try:
            if model_name not in self.models:
                raise ValueError(f"Model {model_name} not trained")
            
            model = self.models[model_name]
            y_pred = model.predict(X_test)
            
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Safe MAPE calculation (avoid division by zero)
            mask = y_test != 0
            if mask.sum() > 0:
                mape = np.mean(np.abs((y_test[mask] - y_pred[mask]) / y_test[mask])) * 100
            else:
                mape = 0.0
            
            # Handle any remaining NaN or Inf values
            mape = 0.0 if np.isnan(mape) or np.isinf(mape) else mape
            
            # Calculate Accuracy (% of predictions within 10% error margin)
            percent_error = np.abs((y_test - y_pred) / (np.abs(y_test) + 1e-10)) * 100
            accuracy = float(round(np.mean(percent_error <= 10) * 100, 2))
            
            # Calculate Correlation coefficient
            correlation = np.corrcoef(y_test, y_pred)[0, 1]
            correlation = 0.0 if np.isnan(correlation) else float(round(correlation, 4))
            
            # Calculate Adjusted R-squared
            n = len(y_test)
            k = X_test.shape[1]
            adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - k - 1) if n > k + 1 else r2
            adjusted_r2 = float(round(adjusted_r2, 4))
            
            # Calculate RMSE percentage
            rmse_percentage = (rmse / (np.mean(np.abs(y_test)) + 1e-10)) * 100
            rmse_percentage = float(round(rmse_percentage, 2))
            
            metrics = {
                'model': model_name,
                'performance': {
                    'accuracy': accuracy,  # % of predictions within 10% error
                    'r2_score': float(round(r2, 4)),
                    'adjusted_r2': adjusted_r2,
                    'correlation': correlation,
                    'rmse_percentage': rmse_percentage
                },
                'errors': {
                    'mse': float(round(mse, 4)),
                    'rmse': float(round(rmse, 4)),
                    'mae': float(round(mae, 4)),
                    'mape': float(round(mape, 4))
                },
                'predictions': y_pred.tolist() if hasattr(y_pred, 'tolist') else y_pred
            }
            
            self.results[model_name] = metrics
            return metrics
        except Exception as e:
            raise Exception(f"Error evaluating {model_name}: {str(e)}")
    
    def train_all_models(self, X_train, y_train):
        """Train all available models"""
        results = {}
        
        try:
            self.train_linear_regression(X_train, y_train)
            results['Linear Regression'] = 'Trained'
        except Exception as e:
            results['Linear Regression'] = f'Failed: {str(e)}'
        
        try:
            self.train_random_forest(X_train, y_train)
            results['Random Forest'] = 'Trained'
        except Exception as e:
            results['Random Forest'] = f'Failed: {str(e)}'
        
        try:
            self.train_xgboost(X_train, y_train)
            results['XGBoost'] = 'Trained'
        except Exception as e:
            results['XGBoost'] = f'Failed: {str(e)}'
        
        return results
    
    def evaluate_all_models(self, X_test, y_test):
        """Evaluate all trained models"""
        evaluation_results = {}
        
        for model_name in self.models.keys():
            if model_name != 'ARIMA':
                evaluation_results[model_name] = self.evaluate_model(model_name, X_test, y_test)
        
        return evaluation_results
    
    def get_best_model(self):
        """Get best performing model based on RMSE"""
        if not self.results:
            return None
        
        best_model = min(self.results.items(), key=lambda x: x[1]['errors']['rmse'])
        return {
            'model_name': best_model[0],
            'metrics': best_model[1]
        }
    
    def predict(self, model_name, X):
        """Make predictions using a trained model"""
        try:
            if model_name not in self.models:
                raise ValueError(f"Model {model_name} not trained")
            
            model = self.models[model_name]
            predictions = model.predict(X)
            return predictions
        except Exception as e:
            raise Exception(f"Error making predictions: {str(e)}")
    
    def get_feature_importance(self, model_name, feature_names=None):
        """Get feature importance from tree-based models"""
        try:
            if model_name not in self.models:
                raise ValueError(f"Model {model_name} not trained")
            
            model = self.models[model_name]
            
            if not hasattr(model, 'feature_importances_'):
                raise ValueError(f"{model_name} does not have feature importance")
            
            importances = model.feature_importances_
            
            if feature_names is None:
                feature_names = [f"Feature {i}" for i in range(len(importances))]
            
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            return importance_df.to_dict('records')
        except Exception as e:
            raise Exception(f"Error getting feature importance: {str(e)}")
