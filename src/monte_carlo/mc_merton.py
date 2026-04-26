import numpy as np
from typing import Optional, Dict
from config.mc_config import MonteCarloConfig
from src.monte_carlo.mc_utils import *


class MertonMonteCarlo:
    """Monte Carlo pricer for European options under the Merton jump-diffusion model."""

    def __init__(
        self,
        r: float,
        q: float,
        sigma: float,
        lambda_jump: float,
        jump_mean: float,
        jump_std: float,
        config: Optional[Dict] = None,
    ):
        self.r = r
        self.q = q
        self.sigma = sigma
        self.lambda_jump = lambda_jump
        self.jump_mean = jump_mean
        self.jump_std = jump_std

        if self.sigma <= 0:
            raise ValueError("Sigma (volatility) must be positive")
        if self.lambda_jump < 0:
            raise ValueError("Jump intensity must be non-negative")
        if self.jump_std < 0:
            raise ValueError("Jump standard deviation must be non-negative")

        if config is None:
            config = MonteCarloConfig.get_config()

        self.n_simulations = config["n_simulations"]
        self.n_time_steps = config["n_time_steps"]
        self.random_seed = config["random_seed"]
        self.use_antithetic = config["use_antithetic"]

    def simulate_paths(self, S0: float, T: float, n_paths: Optional[int] = None):
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

        if self.random_seed is not None:
            np.random.seed(self.random_seed)

        if self.use_antithetic:
            Z = generate_antithetic_normals(n_paths, n_steps, self.random_seed)
        else:
            Z = np.random.standard_normal((n_paths, n_steps))

        dW_S = Z * np.sqrt(dt)

        jump_counts = np.random.poisson(self.lambda_jump * dt, size=(n_paths, n_steps))
        jump_scales = np.sqrt(jump_counts) * self.jump_std
        jump_sizes = np.random.normal(
            loc=jump_counts * self.jump_mean,
            scale=jump_scales,
        )

        jump_compensator = np.exp(self.jump_mean + 0.5 * self.jump_std**2) - 1.0

        for t in range(n_steps):
            S[:, t + 1] = S[:, t] * np.exp(
                (self.r - self.q - self.lambda_jump * jump_compensator - 0.5 * self.sigma**2) * dt
                + self.sigma * dW_S[:, t]
                + jump_sizes[:, t]
            )

        return S

    def price_european_option(
        self,
        S0: float,
        K: float,
        T: float,
        option_type: str = "call",
        n_paths: Optional[int] = None,
    ):
        if K <= 0:
            raise ValueError("K must be positive")

        S_paths = self.simulate_paths(S0, T, n_paths)
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
            "jump_intensity": self.lambda_jump,
            "jump_mean": self.jump_mean,
            "jump_std": self.jump_std,
        }
        return price, std_error, diagnostics

    def price_multiple_options(
        self,
        S0: float,
        strikes: np.ndarray,
        maturities: np.ndarray,
        option_type: str = "call",
        n_paths: Optional[int] = None,
    ):
        if len(strikes) != len(maturities):
            raise ValueError("strikes and maturities must have same length")

        prices = np.zeros(len(strikes))
        std_errors = np.zeros(len(strikes))

        for i, (K, T) in enumerate(zip(strikes, maturities)):
            price, std_error, _ = self.price_european_option(
                S0=S0,
                K=K,
                T=T,
                option_type=option_type,
                n_paths=n_paths,
            )
            prices[i] = price
            std_errors[i] = std_error

        return prices, std_errors
