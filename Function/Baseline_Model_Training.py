# -*- coding: utf-8 -*-
"""
Steam Turbine - Baseline Model Training Script

Trains GLM models for reference values and standard deviations
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.externals import joblib

def train_reference_model(variable_name, X, y, degree=2):
    """
    Train polynomial regression model for reference values
    
    Parameters:
    -----------
    variable_name : str
        Name of the variable (e.g., 'hp_efficiency')
    X : DataFrame
        Features (Power, Temperature, etc.)
    y : Series
        Target variable values
    degree : int
        Polynomial degree (default: 2)
    
    Returns:
    --------
    model : Pipeline
        Trained model
    metrics : dict
        Performance metrics
    """
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Create pipeline
    model = Pipeline([
        ('poly', PolynomialFeatures(degree=degree)),
        ('linear', LinearRegression())
    ])
    
    # Train
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    metrics = {
        'train_r2': r2_score(y_train, y_pred_train),
        'test_r2': r2_score(y_test, y_pred_test),
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
        'train_mae': np.mean(np.abs(y_train - y_pred_train)),
        'test_mae': np.mean(np.abs(y_test - y_pred_test))
    }
    
    # Save model
    joblib.dump(model, f'model/ST_GLM_ref_{variable_name}.pkl')
    
    print(f"\n{variable_name} Reference Model:")
    print(f"  Train R²: {metrics['train_r2']:.4f}")
    print(f"  Test R²: {metrics['test_r2']:.4f}")
    print(f"  Test RMSE: {metrics['test_rmse']:.4f}")
    
    return model, metrics


def train_std_model(variable_name, X, reference_data, actual_data, degree=1):
    """
    Train model for standard deviation prediction
    
    The std model predicts how much variation to expect
    at different operating conditions
    """
    
    # Calculate residuals grouped by operating conditions
    residuals = actual_data - reference_data
    
    # Group by power and temperature bins
    X_binned = X.copy()
    X_binned['power_bin'] = pd.cut(X['Power'], bins=20)
    X_binned['temp_bin'] = pd.cut(X['T'], bins=10)
    
    # Calculate std for each bin
    std_data = residuals.groupby(
        [X_binned['power_bin'], X_binned['temp_bin']]
    ).std().reset_index()
    
    # Get mean power and temp for each bin
    X_std = X.groupby(
        [X_binned['power_bin'], X_binned['temp_bin']]
    ).mean().reset_index()
    
    y_std = std_data[0]  # Standard deviation values
    
    # Train model (usually linear is sufficient)
    model = Pipeline([
        ('poly', PolynomialFeatures(degree=degree)),
        ('linear', LinearRegression())
    ])
    
    model.fit(X_std[['Power', 'T']], y_std)
    
    # Save model
    joblib.dump(model, f'model/ST_GLM_std_{variable_name}.pkl')
    
    print(f"\n{variable_name} Std Model trained and saved.")
    
    return model


# Main training script
if __name__ == '__main__':
    
    print("="*60)
    print("Steam Turbine Baseline Model Training")
    print("="*60)
    
    # Load historical data (6-12 months of healthy operation)
    data = pd.read_csv('data/st_historical_data.csv')
    
    # Features (boundary conditions)
    X = data[['Power', 'T', 'H', 'P']]
    X_basic = data[['Power', 'T']]  # Most models use just Power and Temp
    
    # Variables to model
    variables = [
        'hp_p_in', 'hp_t_in', 'hp_p_out', 'hp_t_out', 
        'hp_efficiency', 'hp_pressure_ratio',
        'ip_p_in', 'ip_t_in', 'ip_p_out', 'ip_t_out',
        'ip_efficiency', 'ip_pressure_ratio',
        'lp_p_in', 'lp_t_in', 'lp_t_out', 'lp_vacuum',
        'lp_efficiency', 'lp_pressure_ratio',
        'overall_efficiency', 'heat_rate', 'condenser_perf'
    ]
    
    # Train reference models
    print("\n" + "="*60)
    print("Training Reference Models...")
    print("="*60)
    
    reference_models = {}
    reference_predictions = {}
    
    for var in variables:
        if var in data.columns:
            model, metrics = train_reference_model(var, X_basic, data[var])
            reference_models[var] = model
            reference_predictions[var] = model.predict(X_basic)
        else:
            print(f"Warning: {var} not found in data. Skipping...")
    
    # Train std models
    print("\n" + "="*60)
    print("Training Standard Deviation Models...")
    print("="*60)
    
    std_models = {}
    
    for var in variables:
        if var in data.columns and var in reference_predictions:
            model = train_std_model(
                var, X_basic, 
                reference_predictions[var], 
                data[var]
            )
            std_models[var] = model
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Total models trained: {len(reference_models) + len(std_models)}")
    print(f"Models saved to: model/ST_GLM_*.pkl")
