from dataclasses import dataclass
from typing import Dict
from math import exp, sqrt

import numpy as np

from src.numerical.finite_difference import FiniteDifferenceSolver

# store model parameters:
@dataclass
class BlackScholesParams:
    r: float
    sigma: float
    q: float = 0.0 # default no dividend yield

# store contract parameters:
@dataclass
class EuropeanOption:
    K: float
    T: float
    option_type: str = "call" # default option is a call option

    def __post_init__(self):
        valid_types = {"call", "put"}
        # checking validity of option type provided:
        if self.option_type not in valid_types:
            raise ValueError(f"option_type must be one of {valid_types}, got {self.option_type}")

# main model wrapper class:
class BlackScholesModel:
    """Black-Scholes pricing model wrapper for 1D finite-difference schemes."""

    def __init__(self, params: BlackScholesParams):
        self.params = params

    def payoff(self, S: np.ndarray, contract: EuropeanOption) -> np.ndarray:
        """Terminal payoff at maturity."""
        # the payoffs are standard well known closed form in finance:
        if contract.option_type == "call":
            return np.maximum(S - contract.K, 0.0)
        else:
            return np.maximum(contract.K - S, 0.0)

    def left_boundary(self, S_min: float, t: float, params: Dict[str, float]) -> float:
        """Left boundary condition at S = S_min (left edge of the stock grid)."""
        # note the dividend yield is irrelevant for this boundary.
        K = params["K"]
        r = params["r"]
        T = params["T"]
        option_type = params["option_type"]

        # well known boundary limits for call and put as S approaches 0:
        if option_type == "call":
            return 0.0
        else:
            return K * np.exp(-r * (T - t))

    def right_boundary(self, S_max: float, t: float, params: Dict[str, float]) -> float:
        """Right boundary condition at S = S_max (right edge of the stock grid)."""
        K = params["K"]
        r = params["r"]
        q = params.get("q", 0.0) # defensive coding cause q is not always used in finance
        T = params["T"]
        option_type = params["option_type"]

        # well known boundary limits for call and put as S approaches inf:
        if option_type == "call":
            return S_max * np.exp(-q * (T - t)) - K * np.exp(-r * (T - t))
        else:
            return 0.0

    def build_solver_params(self, contract: EuropeanOption) -> Dict[str, float]:
        """Build the parameter dictionary expected by FiniteDifferenceSolver."""
        return {
            "r": self.params.r,
            "sigma": self.params.sigma,
            "q": self.params.q,
            "K": contract.K,
            "T": contract.T,
            "option_type": contract.option_type,
            "boundary_conditions": {
                "left": self.left_boundary,
                "right": self.right_boundary,
            },
        }

    def price(self, contract: EuropeanOption, spot: float, scheme: str, S_min: float, S_max: float, N_S: int, N_t: int,
          theta_cn: float = 0.5) -> float:
        """
        Price a European option using the chosen finite-difference scheme.
        Returns the price at time 0 at the grid point closest to the current spot.
        """
        # applying the FiniteDifferenceSolver:
        solver = FiniteDifferenceSolver(scheme=scheme, theta_cn=theta_cn)
        solver.setup_grid(S_min=S_min, S_max=S_max, N_S=N_S, T=contract.T, N_t=N_t)

        params = self.build_solver_params(contract)

        # option prices solution grid:
        V = solver.solve(lambda S: self.payoff(S, contract), params)

        # price of the option today is at V(S_0, t), we get the value in V closest in terms of S_0:
        idx = np.argmin(np.abs(solver.S - spot))
        return V[0, idx]
    
    def compute_greeks(self, contract: EuropeanOption, spot: float, scheme: str = "crank_nicolson",
                       S_min: float = None, S_max: float = None, N_S: int = 100, N_t: int = 100,
                       theta_cn: float = 0.5) -> Dict[str, float]:
        """
        Compute Delta, Gamma, Vega, and Theta via finite differences on the PDE solution grid.
        """
        
        
        # Auto-compute grid bounds if not provided
        if S_min is None or S_max is None:
            sigma = self.params.sigma
            r = self.params.r
            q = self.params.q
            T = contract.T
            K = contract.K
            
            # Log-normal bounds
            drift = (r - q - 0.5 * sigma**2) * T
            spread = 4.0 * sigma * sqrt(T)
            S_min = max(1e-8, spot * exp(drift - spread))
            S_max = max(S_min + 1.0, spot * exp(drift + spread), 2.0 * K)
        
        # Solve PDE
        solver = FiniteDifferenceSolver(scheme=scheme, theta_cn=theta_cn)
        solver.setup_grid(S_min=S_min, S_max=S_max, N_S=N_S, T=contract.T, N_t=N_t)
        params = self.build_solver_params(contract)
        V = solver.solve(lambda S: self.payoff(S, contract), params)
        
        # Find index closest to spot
        idx = np.argmin(np.abs(solver.S - spot))
        S_at_spot = solver.S[idx]
        
        # Get price at spot
        price_at_spot = V[0, idx]
        
        # Delta and Gamma: dV/dS and d²V/dS²
        if idx > 0 and idx < N_S - 1:
            delta = (V[0, idx + 1] - V[0, idx - 1]) / (2 * solver.dS)
            gamma = (V[0, idx + 1] - 2 * price_at_spot + V[0, idx - 1]) / (solver.dS ** 2)
        else:
            delta = (V[0, idx + 1] - V[0, idx]) / solver.dS if idx < N_S - 1 else delta
            gamma = (V[0, idx + 1] - 2 * price_at_spot + V[0, idx - 1]) / (solver.dS ** 2)
        
        # Vega: dV/dσ 
        sigma_bump = self.params.sigma * 1.01
        params_vega = params.copy()
        params_vega["sigma"] = sigma_bump
        solver_vega = FiniteDifferenceSolver(scheme=scheme, theta_cn=theta_cn)
        solver_vega.setup_grid(S_min=S_min, S_max=S_max, N_S=N_S, T=contract.T, N_t=N_t)
        V_vega = solver_vega.solve(lambda S: self.payoff(S, contract), params_vega)
        vega = (V_vega[0, idx] - price_at_spot) / 0.01
        
        # Theta: dV/dt 
        if N_t > 1:
            theta = -(V[1, idx] - V[0, idx]) / solver.dt
        else:
            theta = float('nan')
        
        return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta}