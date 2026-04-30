import numpy as np
from typing import Tuple, Optional, Dict
from config.mc_config import MonteCarloConfig
from src.monte_carlo.mc_utils import *

class HestonMonteCarlo:
    def __init__(self, r: float, q: float, kappa: float, theta: float, xi: float, rho: float, v0: float, config: Optional[Dict]=None):
        self.r = r
        self.q = q
        self.kappa = kappa
        self.theta = theta
        self.xi = xi
        self.rho = rho
        self.v0 = v0
        
        if config is None:
            config = MonteCarloConfig.get_config()
        
        self.n_simulations = config['n_simulations']
        self.n_time_steps = config['n_time_steps']
        self.random_seed = config['random_seed']
        self.use_antithetic = config['use_antithetic']
        self.min_variance = config['min_variance']
        self.max_variance = config['max_variance']

    def simulate_paths(self, S0: float, T: float, n_paths: Optional[int] = None):
        if n_paths is None:
            n_paths = self.n_simulations

        n_steps = self.n_time_steps
        dt = T/n_steps

        S = np.zeros((n_paths, n_steps + 1))
        v = np.zeros((n_paths, n_steps + 1))

        S[:, 0] = S0
        v[:, 0] = self.v0

        if self.use_antithetic:
            Z1 = generate_antithetic_normals(n_paths, n_steps, self.random_seed)
            Z2 = generate_antithetic_normals(n_paths, n_steps, self.random_seed + 1 if self.random_seed else None)

            dW_S = Z1
            dW_v = self.rho * Z1 + np.sqrt(1-self.rho**2) * Z2
        else:
            dW_S, dW_v = generate_correlated_brownian(n_paths, n_steps, self.rho, self.random_seed)

        # Scale by sqrt(dt)
        dW_S = dW_S * np.sqrt(dt)
        dW_v = dW_v * np.sqrt(dt)

        for t in range(n_steps):
            v_t = truncate_variance(v[:, t], self.min_variance, self.max_variance)
            st_dev_t = np.sqrt(v_t)

            v[:, t+1] = v_t + self.kappa * (self.theta - v_t) * dt + self.xi * st_dev_t * dW_v[:,t] + 0.25 * self.xi**2 * (dW_v[:, t]**2 - dt)
            v[:, t+1] = truncate_variance(v[:, t+1], self.min_variance, self.max_variance)

            S[:, t+1] = S[:, t] * np.exp((self.r - self.q - 0.5 * v_t) * dt + st_dev_t * dW_S[:, t])
        return S, v

    def price_european_option(self, S0: float, K: float, T: float, option_type:str = 'call', n_paths: Optional[int] = None):
        S_paths, v_paths = self.simulate_paths(S0, T, n_paths)
        price, std_error = price_european(S_paths, K, self.r, T, option_type)

        ci_lower, ci_upper = estimate_confidence_interval(price, std_error)

        path_stats = compute_path_stats(S_paths)
        variance_stats = compute_path_stats(v_paths)
        diagnostics = {
            'price': price,
            'std_error': std_error,
            'ci_95': (ci_lower, ci_upper),
            'n_paths': S_paths.shape[0],
            'n_steps': S_paths.shape[1] - 1,
            'asset_stats': path_stats,
            'variance_stats': variance_stats,
            'antithetic_used': self.use_antithetic
        }
        return price, std_error, diagnostics

    def price_multiple_options(self, S0: float, strikes: np.ndarray, maturities: np.ndarray, option_type: str = 'call',
                               n_paths: Optional[int] = None):
        if len(strikes) != len(maturities):
            raise ValueError("strikes and maturities must have same length")
        
        prices = np.zeros(len(strikes))
        std_errors = np.zeros(len(strikes))
        
        for i, (K, T) in enumerate(zip(strikes, maturities)):
            price, std_error, _ = self.price_european_option(
                S0, K, T, option_type, n_paths
            )
            prices[i] = price
            std_errors[i] = std_error
        
        return prices, std_errors