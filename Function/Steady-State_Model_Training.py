
"""
Steam Turbine - Steady State Detection Model Training
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.mixture import GaussianMixture
from sklearn.externals import joblib

def divide_steady_unsteady(power, interval=20, alpha=0.05):
    """
    Divide data into steady and unsteady periods
    
    Uses interval estimation of mean power change
    """
    
    delta_power = power.diff().dropna()
    
    steady_indices = []
    unsteady_indices = []
    steady_delta = []
    unsteady_delta = []
    
    # Critical value for confidence interval
    z_critical = stats.norm.ppf(1 - alpha/2)
    
    # Estimate std from stable periods
    # (you may need to adjust this based on your data)
    sigma = delta_power[abs(delta_power) < 1].std()
    
    for i in range(0, len(delta_power) - interval, interval):
        window = delta_power.iloc[i:i+interval]
        mu = window.mean()
        
        # Confidence interval for mean
        margin = z_critical * sigma / np.sqrt(interval)
        ci_lower = mu - margin
        ci_upper = mu + margin
        
        # If 0 is in confidence interval → steady
        if ci_lower <= 0 <= ci_upper:
            steady_indices.extend(range(i, i+interval))
            steady_delta.extend(window.values)
        else:
            unsteady_indices.extend(range(i, i+interval))
            unsteady_delta.extend(window.values)
    
    steady_ratio = len(steady_indices) / (len(steady_indices) + len(unsteady_indices))
    
    print(f"Steady data ratio: {steady_ratio:.2%}")
    print(f"Steady samples: {len(steady_indices)}")
    print(f"Unsteady samples: {len(unsteady_indices)}")
    
    return (steady_indices, unsteady_indices), (steady_delta, unsteady_delta), steady_ratio


def train_steady_state_models(steady_delta, unsteady_delta, n_components=5):
    """
    Train Gaussian and GMM models for steady-state detection
    """
    
    # Train Gaussian model for steady data
    mu_steady = np.mean(steady_delta)
    sigma_steady = np.std(steady_delta, ddof=1)
    
    print(f"\nSteady Model (Gaussian):")
    print(f"  μ = {mu_steady:.4f}")
    print(f"  σ = {sigma_steady:.4f}")
    
    # Train GMM for unsteady data
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type='full',
        random_state=42
    )
    
    unsteady_array = np.array(unsteady_delta).reshape(-1, 1)
    gmm.fit(unsteady_array)
    
    print(f"\nUnsteady Model (GMM with {n_components} components):")
    for i in range(n_components):
        print(f"  Component {i+1}: μ={gmm.means_[i][0]:.4f}, "
              f"σ={np.sqrt(gmm.covariances_[i][0][0]):.4f}, "
              f"weight={gmm.weights_[i]:.3f}")
    
    # Save models
    joblib.dump((mu_steady, sigma_steady), 'model/ST_steady_model.pkl')
    joblib.dump(gmm, 'model/ST_unsteady_model.pkl')
    
    return (mu_steady, sigma_steady), gmm


# Main training script
if __name__ == '__main__':
    
    print("="*60)
    print("Steam Turbine Steady-State Model Training")
    print("="*60)
    
    # Load historical power data
    data = pd.read_csv('data/st_historical_data.csv')
    power = data['Power']
    
    # Divide into steady and unsteady periods
    indices, delta_power, steady_ratio = divide_steady_unsteady(
        power, 
        interval=20, 
        alpha=0.05
    )
    
    steady_indices, unsteady_indices = indices
    steady_delta, unsteady_delta = delta_power
    
    # Train models
    steady_model, unsteady_model = train_steady_state_models(
        steady_delta, 
        unsteady_delta, 
        n_components=5
    )
    
    # Save steady ratio
    joblib.dump(steady_ratio, 'model/ST_steady_ratio.pkl')
    
    # Test the models
    print("\n" + "="*60)
    print("Model Testing")
    print("="*60)
    
    # Test on sample data
    test_deltas = [-0.1, 0.05, 5.0, -10.0, 0.2]
    test_labels = ['Steady', 'Steady', 'Unsteady', 'Unsteady', 'Steady']
    
    mu_steady, sigma_steady = steady_model
    
    for delta, label in zip(test_deltas, test_labels):
        # Calculate likelihoods
        prob_steady = stats.norm.pdf(delta, mu_steady, sigma_steady)
        prob_unsteady = np.exp(unsteady_model.score_samples([[delta]])[0])
        
        # Likelihood ratio
        lr = (prob_steady * steady_ratio) / (prob_unsteady * (1 - steady_ratio))
        
        prediction = "Steady" if lr > 1 else "Unsteady"
        status = "✓" if prediction == label else "✗"
        
        print(f"ΔP={delta:6.1f} MW: {prediction:8s} (expected: {label:8s}) {status}")
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print("Models saved:")
    print("  - model/ST_steady_model.pkl")
    print("  - model/ST_unsteady_model.pkl")
    print("  - model/ST_steady_ratio.pkl")
