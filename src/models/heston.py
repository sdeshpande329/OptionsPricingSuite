"""
Implements the 2D Heston PDE using ADI schemes from src.numerical.adi_schemes.

The model assumes:
    dS_t = rS_t dt + sqrt(v_t)S_t dW_t^S
    dv_t = kappa(theta - v_t)dt + xi*sqrt(v_t)dW_t^v
    dW_t^S dW_t^v = rho dt

"""

import numpy as np
from typing import Callable, Dict, Tuple, Optional
from scipy.optimize import brentq
from scipy.stats import norm
from src.numerical.adi_schemes import ADISolver


class HestonModel:
    """
    Attributes:
    -----------
    r : float
        Risk-free interest rate
    q : float
        Dividend yield
    kappa : float
        Mean reversion speed for variance
    theta : float
        Long-run variance level
    xi : float
        Volatility of volatility (vol of vol)
    rho : float
        Correlation between asset returns and variance
    v0 : float
        Initial variance
    """
    
    def __init__(self, r: float, q: float, kappa: float, theta: float, 
                 xi: float, rho: float, v0: float):
        """
        Initialize Heston model with parameters.
        """
        self.r = r
        self.q = q
        self.kappa = kappa
        self.theta = theta
        self.xi = xi
        self.rho = rho
        self.v0 = v0

        # Validate parameter constraints
        if not self.feller_condition_satisfied(kappa, theta, xi):
            print(f"Warning: Feller condition not satisfied: "
                  f"2*kappa*theta = {2*kappa*theta:.4f} <= xi^2 = {xi**2:.4f}")
        if rho < -1 or rho > 1:
            raise ValueError("Correlation must be between -1 and 1")
        if v0 <= 0:
            raise ValueError("Initial variance must be positive")
    
    def get_pde_coefficients(self) -> Dict[str, float]:
        """
        Return PDE coefficients for the Heston model as a dictionary.
        
        Returns:

        pde_coefficients : dict
            Dictionary containing the PDE coefficients: 'r', 'q', 'kappa', 'theta', 'xi', 'rho'
        """
        return {
            'r': self.r,
            'q': self.q,
            'kappa': self.kappa,
            'theta': self.theta,
            'xi': self.xi,
            'rho': self.rho
        }
    
    def price_european_option(self, S0: float, K: float, T: float, 
                             option_type: str = 'call',
                             scheme: str = 'craig-sneyd',
                             N_S: int = 100, N_v: int = 50, N_t: int = 100,
                             S_max: Optional[float] = None,
                             v_max: Optional[float] = None) -> float:
        """
        Price a European option using ADI finite difference.
        
        Parameters:
        
        S0 : float
            Current stock price
        K : float
            Strike price
        T : float
            Time to maturity (years)
        option_type : str
            'call' or 'put'
        scheme : str
            ADI scheme to use: 'douglas', 'craig-sneyd', 'mcs', 'hv'
        N_S : int
            Number of grid points in S direction
        N_v : int
            Number of grid points in v direction
        N_t : int
            Number of time steps
        S_max : float, optional
            Maximum S for grid (default: 3*S0)
        v_max : float, optional
            Maximum v for grid (default: 5*theta)
        
        Returns:
        
        price : float
            Option price
        """
        # Set default grid bounds
        if S_max is None:
            S_max = 3 * S0
        if v_max is None:
            v_max = 5 * self.theta
        
        # Initialize ADI solver
        solver = ADISolver(scheme=scheme, theta=0.5)
        
        # Setup grid
        solver.setup_grid(
            S_min=0.0, S_max=S_max, N_S=N_S,
            v_min=0.0, v_max=v_max, N_v=N_v,
            T=T, N_t=N_t
        )
        
        # Define payoff function
        def payoff(S_grid, v_grid):
            if option_type == 'call':
                return np.maximum(S_grid - K, 0)
            else:
                return np.maximum(K - S_grid, 0)
        
        # Get PDE coefficients
        params = self.get_pde_coefficients()
        
        # Solve PDE
        V = solver.solve(payoff, params)
        
        # Interpolate to get price at (S0, v0, t=0)
        price = self._interpolate_grid_value(
            V[0, :, :], solver.S, solver.v, S0, self.v0
        )
        
        return price
    
    def price_multiple_options(self, S0: float, strikes: np.ndarray, 
                               maturities: np.ndarray, 
                               option_type: str = 'call',
                               scheme: str = 'craig-sneyd',
                               **grid_params) -> np.ndarray:
        """
        Price multiple options with different strikes and maturities.
        
        Useful for calibration where we need prices for many market options.
        
        Parameters:
        
        S0 : float
            Current stock price
        strikes : np.ndarray
            Array of strike prices
        maturities : np.ndarray
            Array of maturities (same length as strikes)
        option_type : str
            'call' or 'put'
        scheme : str
            ADI scheme to use
        **grid_params
            Grid parameters (N_S, N_v, N_t, S_max, v_max)
        
        Returns:
        
        prices : np.ndarray
            Array of option prices
        """
        # Validate inputs
        if len(strikes) != len(maturities):
            raise ValueError("strikes and maturities must have same length")
        
        prices = np.zeros(len(strikes))
        for i, (K, T) in enumerate(zip(strikes, maturities)):
            prices[i] = self.price_european_option(
                S0=S0, K=K, T=T, 
                option_type=option_type, 
                scheme=scheme, 
                **grid_params
            )
        
        return prices
    
    def compute_greeks(self, S0: float, K: float, T: float,
                      option_type: str = 'call',
                      scheme: str = 'craig-sneyd',
                      N_S: int = 100, N_v: int = 50, N_t: int = 100,
                      S_max: Optional[float] = None,
                      v_max: Optional[float] = None) -> Dict[str, float]:
        """
        Compute option Greeks using finite differences on the PDE grid.
        
        Parameters:
        
        S0, K, T : float
            Option parameters
        option_type : str
            'call' or 'put'
        scheme : str
            ADI scheme to use
        N_S, N_v, N_t : int
            Grid parameters
        S_max, v_max : float, optional
            Grid bounds
        
        Returns:
        
        greeks : dict
            Greeks: 'delta', 'gamma', 'vega', 'theta'
        """
        # Set defaults
        if S_max is None:
            S_max = 3 * S0
        if v_max is None:
            v_max = 5 * self.theta
        
        # Setup solver
        solver = ADISolver(scheme=scheme, theta=0.5)
        solver.setup_grid(
            S_min=0.0, S_max=S_max, N_S=N_S,
            v_min=0.0, v_max=v_max, N_v=N_v,
            T=T, N_t=N_t
        )
        
        # Define payoff
        def payoff(S_grid, v_grid):
            if option_type == 'call':
                return np.maximum(S_grid - K, 0)
            else:
                return np.maximum(K - S_grid, 0)
        
        # Solve
        params = self.get_pde_coefficients()
        V = solver.solve(payoff, params)
        
        # Find grid indices closest to (S0, v0)
        i_S = np.argmin(np.abs(solver.S - S0))
        i_v = np.argmin(np.abs(solver.v - self.v0))
        
        dS = solver.dS
        dv = solver.dv
        dt = solver.dt
        
        # Compute Greeks using central differences
        greeks = {}
        
        # Delta: dV/dS
        if i_S > 0 and i_S < N_S - 1:
            greeks['delta'] = (V[0, i_v, i_S+1] - V[0, i_v, i_S-1]) / (2 * dS)
        else:
            greeks['delta'] = np.nan
        
        # Gamma: d²V/dS²
        if i_S > 0 and i_S < N_S - 1:
            greeks['gamma'] = (V[0, i_v, i_S+1] - 2*V[0, i_v, i_S] + V[0, i_v, i_S-1]) / (dS**2)
        else:
            greeks['gamma'] = np.nan
        
        # Vega: dV/dv
        if i_v > 0 and i_v < N_v - 1:
            greeks['vega'] = (V[0, i_v+1, i_S] - V[0, i_v-1, i_S]) / (2 * dv)
        else:
            greeks['vega'] = np.nan
        
        # Theta: dV/dt (use backward difference)
        if V.shape[0] > 1:
            greeks['theta'] = -(V[1, i_v, i_S] - V[0, i_v, i_S]) / dt
        else:
            greeks['theta'] = np.nan
        
        return greeks
    
    def _interpolate_grid_value(self, grid: np.ndarray, 
                                S_grid: np.ndarray, v_grid: np.ndarray,
                                S_target: float, v_target: float) -> float:
        """
        Interpolate grid value at arbitrary (S, v) point using bilinear interpolation.
        
        Parameters:
        
        grid : np.ndarray, shape (N_v, N_S)
            2D grid of values
        S_grid : np.ndarray
            S coordinate values
        v_grid : np.ndarray
            v coordinate values
        S_target : float
            Target S value
        v_target : float
            Target v value
        
        Returns:
        
        V_interp : float
            Interpolated value
        """
        # Find bracketing indices
        i_S = np.searchsorted(S_grid, S_target)
        i_v = np.searchsorted(v_grid, v_target)
        
        # Clamp to valid range
        i_S = np.clip(i_S, 1, len(S_grid) - 2)
        i_v = np.clip(i_v, 1, len(v_grid) - 2)
        
        # Get bracketing values
        S_low, S_high = S_grid[i_S-1], S_grid[i_S]
        v_low, v_high = v_grid[i_v-1], v_grid[i_v]
        
        # Bilinear interpolation weights
        w_S = (S_target - S_low) / (S_high - S_low)
        w_v = (v_target - v_low) / (v_high - v_low)
        
        # Get corner values
        V_00 = grid[i_v-1, i_S-1]
        V_10 = grid[i_v-1, i_S]
        V_01 = grid[i_v, i_S-1]
        V_11 = grid[i_v, i_S]
        
        # Bilinear interpolation
        V_interp = (1-w_S)*(1-w_v)*V_00 + w_S*(1-w_v)*V_10 + \
                   (1-w_S)*w_v*V_01 + w_S*w_v*V_11
        
        return V_interp
    
    @staticmethod
    def feller_condition_satisfied(kappa: float, theta: float, xi: float) -> bool:
        """
        Check if Feller condition is satisfied.
        
        The Feller condition 2*kappa*theta > xi² ensures that variance
        remains strictly positive.
        
        Parameters:
        
        kappa, theta, xi : float
            Heston parameters
        
        Returns:
        
        bool
            True if Feller condition satisfied
        """
        return 2*kappa*theta > xi**2
    
    def __repr__(self) -> str:
        """String representation of model parameters."""
        feller_status = "✓" if self.feller_condition_satisfied(self.kappa, self.theta, self.xi) else "✗"
        return (f"HestonModel(r={self.r:.4f}, q={self.q:.4f}, "
                f"kappa={self.kappa:.4f}, theta={self.theta:.4f}, "
                f"xi={self.xi:.4f}, rho={self.rho:.4f}, v0={self.v0:.4f}) "
                f"[Feller: {feller_status}]")