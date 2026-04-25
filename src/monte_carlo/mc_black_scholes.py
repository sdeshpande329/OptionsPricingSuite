import numpy as np
from typing import Optional, Dict

from config.mc_config import MonteCarloConfig
from src.monte_carlo.mc_utils import *

class BlackScholesMonteCarlo:
    """Monte Carlo pricer for European options under the Black-Scholes model."""
    def __init__(self, r: float, q: float, sigma: float, config: Optional[Dict] = None):
        """Initialize Black-Scholes Monte Carlo pricer."""
        self.r = r
        self.q = q
        self.sigma = sigma

        if self.sigma <= 0:
            raise ValueError("Sigma (volatility) must be positive")

        if config is None:
            config = MonteCarloConfig.get_config()

        self.n_simulations = config["n_simulations"]
        self.n_time_steps = config["n_time_steps"]
        self.random_seed = config["random_seed"]
        self.use_antithetic = config["use_antithetic"]

    def simulate_paths(self, S0: float, T: float, n_paths: Optional[int] = None) -> np.ndarray:
        """Simulate Black-Scholes stock-price paths using exact GBM discretization."""
        if S0 <= 0:
            raise ValueError("S0 must be positive")
        if T <= 0:
            raise ValueError("T must be positive")

        if n_paths is None:
            n_paths = self.n_simulations

        n_steps = self.n_time_steps
        dt = T / n_steps

        S = np.zeros((n_paths, n_steps + 1))
        S[:, 0] = S0

        # generating standard normal shocks:
        if self.use_antithetic:
            Z = generate_antithetic_normals(n_paths, n_steps, self.random_seed)
        else:
            if self.random_seed is not None:
                np.random.seed(self.random_seed)
            Z = np.random.standard_normal((n_paths, n_steps))

        drift = (self.r - self.q - 0.5 * self.sigma**2) * dt
        diffusion_scale = self.sigma * np.sqrt(dt)

        # exact geometric brownian motion update:
        for t in range(n_steps):
            S[:, t + 1] = S[:, t] * np.exp(drift + diffusion_scale * Z[:, t])

        return S

    def price_european_option(self, S0: float, K: float, T: float, option_type: str = "call", n_paths: Optional[int] = None):
        """Price a European option using Monte Carlo simulation."""
        if K <= 0:
            raise ValueError("K must be positive")

        S_paths = self.simulate_paths(S0, T, n_paths)

        # pricing from terminal payoffs:
        price, std_error = price_european(S_paths, K, self.r, T, option_type)

        ci_lower, ci_upper = estimate_confidence_interval(price, std_error)

        path_stats = compute_path_stats(S_paths)

        diagnostics = {
            "price": price,
            "std_error": std_error,
            "ci_95": (ci_lower, ci_upper),
            "n_paths": S_paths.shape[0],
            "n_steps": S_paths.shape[1] - 1,
            "asset_stats": path_stats,
            "antithetic_used": self.use_antithetic,
        }

        return price, std_error, diagnostics

    def price_multiple_options(self, S0: float, strikes: np.ndarray, maturities: np.ndarray, option_type: str = "call",
                               n_paths: Optional[int] = None):
        """Price multiple European options with different strikes and maturities."""
        if len(strikes) != len(maturities):
            raise ValueError("strikes and maturities must have same length")

        prices = np.zeros(len(strikes))
        std_errors = np.zeros(len(strikes))

        for i, (K, T) in enumerate(zip(strikes, maturities)):
            price, std_error, _ = self.price_european_option(S0=S0, K=K, T=T, option_type=option_type, n_paths=n_paths)
            prices[i] = price
            std_errors[i] = std_error

        return prices, std_errors