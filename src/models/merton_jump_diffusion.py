from dataclasses import dataclass
from typing import Dict

import numpy as np

from src.numerical.imex_schemes import IMEXSolver

# store model parameters:
@dataclass
class MertonJumpParams:
    r: float
    sigma: float
    q: float = 0.0 # default no dividend yield
    lambda_jump: float = 0.0 # jump intensity
    jump_mean: float = 0.0 # mean of the log jump size
    jump_std: float = 0.0 # standard deviation of the log jump size

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
class MertonJumpDiffusionModel:
    """Merton Jump-Diffusion pricing model wrapper for 1D finite-difference schemes."""

    def __init__(self, params: MertonJumpParams):
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
        """Build the parameter dictionary expected by IMEXSolver."""
        return {
            "r": self.params.r,
            "sigma": self.params.sigma,
            "q": self.params.q,
            "lambda_jump": self.params.lambda_jump,
            "jump_mean": self.params.jump_mean,
            "jump_std": self.params.jump_std,
            "K": contract.K,
            "T": contract.T,
            "option_type": contract.option_type,
            "boundary_conditions": {
                "left": self.left_boundary,
                "right": self.right_boundary,
            },
        }

    def price(self, contract: EuropeanOption, spot: float, scheme: str, S_min: float, S_max: float, N_S: int, N_t: int) -> float:
        """
        Price a European option using the chosen IMEX jump-diffusion scheme.
        Returns the price at time 0 at the grid point closest to the current spot.
        """
        # applying the IMEX Solver for the jump-diffusion PIDE:
        solver = IMEXSolver(scheme=scheme)
        solver.setup_grid(S_min=S_min, S_max=S_max, N_S=N_S, T=contract.T, N_t=N_t)

        params = self.build_solver_params(contract)

        # option prices solution grid:
        V = solver.solve(lambda S: self.payoff(S, contract), params)

        # price of the option today is at V(S_0, t), we get the value in V closest in terms of S_0:
        idx = np.argmin(np.abs(solver.S - spot))
        return V[0, idx]
