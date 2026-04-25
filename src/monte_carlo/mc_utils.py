from optparse import Option
import numpy as np
from typing import Tuple, Optional, Callable
from scipy.stats import norm
from config.mc_config import MonteCarloConfig

def generate_correlated_brownian(n_paths:int, n_steps:int, correlation: float, seed: Optional[int]=None):
    if seed is not None:
        np.random.seed(seed)
    
    # Generate independent std normal variables
    Z1 = np.random.standard_normal((n_paths, n_steps))
    Z2 = np.random.standard_normal((n_paths, n_steps))

    # Create correlated Brownian increments using Cholesky decomposition
    dW1 = Z1
    dW2 = correlation * Z1 + np.sqrt(1-correlation**2) * Z2

    return dW1, dW2

def generate_antithetic_normals(n_paths:int, n_steps:int, seed: Optional[int] = None):
    """Used for variance reduction of Monte Carlo estimates without increasing n_simulations"""

    if seed is not None:
        np.random.seed(seed)
    
    # Generate pairs
    n_half = n_paths // 2
    Z_half = np.random.standard_normal((n_half, n_steps))

    if n_paths%2 == 0:
        Z = np.vstack([Z_half, -Z_half])
    else:
        Z_extra = np.random.standard_normal((1, n_steps))
        Z = np.vstack([Z_half, -Z_half, Z_extra])
    
    return Z

def compute_path_stats(paths: np.ndarray):
    final_values = paths[:, -1]
    return {
        'mean': np.mean(final_values),
        'std': np.std(final_values),
        'min': np.min(final_values),
        'max': np.max(final_values),
        'median': np.median(final_values),
        'percentile_5': np.percentile(final_values, 5),
        'percentile_25': np.percentile(final_values, 25),
        'percentile_75': np.percentile(final_values, 75),
        'percentile_95': np.percentile(final_values, 95)
    }

def price_european(paths: np.ndarray, strike: float, r: float, T: float, option_type: str = 'call'):
    S_T = paths[:, -1]

    # Find payoffs based on option type
    if option_type == 'call':
        payoffs = np.maximum(S_T - strike, 0)
    elif option_type == 'put':
        payoffs = np.maximum(strike - S_T, 0)
    else:
        raise ValueError(f"Unknown option type: {option_type}")
    
    # Discount and average payoffs to get price
    pv_payoffs = np.exp(-r * T) * payoffs
    price = np.mean(pv_payoffs)
    std_error = np.std(pv_payoffs) / np.sqrt(len(pv_payoffs))

    return price, std_error

def convergence_test(simulation_func: Callable, test_sizes: list, true_value: Optional[float]=None):
    """Test convergence of Monte Carlo based on increasing number of simulations"""
    results = {
        'n_paths': [], 
        'prices': [],
        'std_errors': [],
        'errors': [] if true_value is not None else None
    }

    # Run simulations and store results
    for n_paths in test_sizes:
        price, std_error = simulation_func(n_paths)

        results['n_paths'].append(n_paths)
        results['prices'].append(price)
        results['std_errors'].append(std_error)

        if true_value is not None:
            results['errors'].append(abs(price - true_value))

    # Convert results to arrays
    results['n_paths'] = np.array(results['n_paths'])
    results['prices'] = np.array(results['prices'])
    results['std_errors'] = np.array(results['std_errors'])
    if results['errors'] is not None:
        results['errors'] = np.array(results['errors'])
    
    return results

def estimate_confidence_interval(mean: float, std_error: float, confidence_level: float = 0.95):
    z = norm.ppf((1+confidence_level)/2)
    margin = z * std_error
    lower = mean - margin
    upper = mean + margin

    return lower, upper

def truncate_variance(variance: np.ndarray, min_var: Optional[float] = None, max_var: Optional[float] = None):
    """Truncate variance to ensure variance is positive"""

    if min_var is None:
        min_var = MonteCarloConfig.MIN_VARIANCE
    if max_var is None:
        max_var = MonteCarloConfig.MAX_VARIANCE
    
    return np.clip(variance, min_var, max_var)