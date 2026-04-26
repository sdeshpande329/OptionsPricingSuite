from __future__ import annotations

import numpy as np
import scipy.integrate as integrate
from typing import Dict, Optional

from src.numerical.adi_schemes import ADISolver


_VALID_SCHEMES = frozenset(
    ["douglas", "craig_sneyd", "modified_craig_sneyd", "hundsdorfer_verwer"]
)


class HestonModel:
    """Prices European options under the Heston stochastic volatility model."""
    def __init__(self, r: float, q: float, kappa: float, theta: float, xi: float, rho: float, v0: float) -> None:
        if not -1.0 <= rho <= 1.0:
            raise ValueError(f"rho must be in [-1, 1], got {rho}")
        if v0 <= 0.0:
            raise ValueError(f"v0 must be positive, got {v0}")
        if kappa <= 0.0:
            raise ValueError(f"kappa must be positive, got {kappa}")
        if theta <= 0.0:
            raise ValueError(f"theta must be positive, got {theta}")
        if xi <= 0.0:
            raise ValueError(f"xi must be positive, got {xi}")

        self.r = r
        self.q = q
        self.kappa = kappa
        self.theta = theta
        self.xi = xi
        self.rho = rho
        self.v0 = v0

        if not self.feller_condition_satisfied(kappa, theta, xi):
            print(
                f"Warning: Feller condition not satisfied: "
                f"2*kappa*theta = {2 * kappa * theta:.4f} <= xi^2 = {xi ** 2:.4f}. "
                f"Variance may reach zero."
            )
    
    def get_pde_coefficients(self) -> Dict[str, float]:
        """Return the PDE coefficients required by ADISolver."""
        return {
            "r": self.r,
            "q": self.q,
            "kappa": self.kappa,
            "theta": self.theta,
            "xi": self.xi,
            "rho": self.rho,
        }

    def price_european_option(self,S0: float,K: float,T: float,option_type: str = "call",scheme: str = "craig_sneyd",N_S: int = 100,N_v: int = 50,N_t: int = 100,S_max: Optional[float] = None,
    v_max: Optional[float] = None) -> float:
        """Price a European option using an ADI finite-difference scheme."""
        if scheme not in _VALID_SCHEMES:
            raise ValueError(
                f"Unknown scheme '{scheme}'. Valid options: {sorted(_VALID_SCHEMES)}"
            )
        option_type = option_type.lower()
        if option_type not in ("call", "put"):
            raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")

        if S_max is None:
            S_max = 4.0 * S0
        if v_max is None:
            v_max = 5.0 * self.theta

        solver = ADISolver(scheme=scheme)
        solver.setup_grid(
            S_min=0.0, S_max=S_max, N_S=N_S,
            v_min=0.0, v_max=v_max, N_v=N_v,
            T=T, N_t=N_t,
        )

        def payoff(S_grid: np.ndarray, v_grid: np.ndarray) -> np.ndarray:
            if option_type == "call":
                return np.maximum(S_grid - K, 0.0)
            else:
                return np.maximum(K - S_grid, 0.0)

        V = solver.solve(payoff, self.get_pde_coefficients())
        price_surface = V[0]

        return self._interpolate_grid_value(
            price_surface, solver.S, solver.v, S0, self.v0
        )

    def price_multiple_options(self,S0: float,strikes: np.ndarray,maturities: np.ndarray,option_type: str = "call",scheme: str = "craig_sneyd",N_S: int = 100,N_v: int = 50,N_t: int = 100,S_max: Optional[float] = None,
    v_max: Optional[float] = None) -> np.ndarray:
        """Price multiple European options efficiently by grouping by maturity."""
        strikes    = np.asarray(strikes,    dtype=float)
        maturities = np.asarray(maturities, dtype=float)
        if strikes.shape != maturities.shape:
            raise ValueError("strikes and maturities must have the same shape.")
        if scheme not in _VALID_SCHEMES:
            raise ValueError(
                f"Unknown scheme '{scheme}'. Valid options: {sorted(_VALID_SCHEMES)}"
            )
        option_type = option_type.lower()

        if S_max is None:
            S_max = 4.0 * S0
        if v_max is None:
            v_max = 5.0 * self.theta

        prices = np.zeros(len(strikes))

        for T in np.unique(maturities):
            mask         = maturities == T
            strikes_at_T = strikes[mask]

            solver = ADISolver(scheme=scheme)
            solver.setup_grid(
                S_min=0.0, S_max=S_max, N_S=N_S,
                v_min=0.0, v_max=v_max, N_v=N_v,
                T=T, N_t=N_t,
            )

            for local_idx, K in enumerate(strikes_at_T):
                def payoff(
                    S_grid: np.ndarray,
                    v_grid: np.ndarray,
                    _K: float = K,
                ) -> np.ndarray:
                    if option_type == "call":
                        return np.maximum(S_grid - _K, 0.0)
                    else:
                        return np.maximum(_K - S_grid, 0.0)

                V             = solver.solve(payoff, self.get_pde_coefficients())
                price_surface = V[0]
                global_idx    = np.where(mask)[0][local_idx]
                prices[global_idx] = self._interpolate_grid_value(
                    price_surface, solver.S, solver.v, S0, self.v0
                )

        return prices

    def compute_greeks(self, S0: float, K: float,T: float, option_type: str = "call", scheme: str = "craig_sneyd", N_S: int = 100,N_v: int = 50,N_t: int = 100,S_max: Optional[float] = None,
    v_max: Optional[float] = None,) -> Dict[str, float]:
        """Compute Delta, Gamma, Vega, and Theta via finite differences on the PDE solution grid using bilinear interpolation at perturbed points."""
        if S_max is None:
            S_max = 4.0 * S0
        if v_max is None:
            v_max = 5.0 * self.theta

        solver = ADISolver(scheme=scheme)
        solver.setup_grid(
            S_min=0.0, S_max=S_max, N_S=N_S,
            v_min=0.0, v_max=v_max, N_v=N_v,
            T=T, N_t=N_t,
        )

        def payoff(S_grid: np.ndarray, v_grid: np.ndarray) -> np.ndarray:
            if option_type == "call":
                return np.maximum(S_grid - K, 0.0)
            else:
                return np.maximum(K - S_grid, 0.0)

        V = solver.solve(payoff, self.get_pde_coefficients())

        def interp(surface: np.ndarray, S_t: float, v_t: float) -> float:
            return self._interpolate_grid_value(surface, solver.S, solver.v, S_t, v_t)

        surface_t0 = V[0]
        dS_bump    = solver.dS
        dv_bump    = solver.dv

        V_Sup = interp(surface_t0, S0 + dS_bump, self.v0)
        V_Sdn = interp(surface_t0, S0 - dS_bump, self.v0)
        V_mid = interp(surface_t0, S0,            self.v0)

        delta = (V_Sup - V_Sdn) / (2.0 * dS_bump)
        gamma = (V_Sup - 2.0 * V_mid + V_Sdn) / dS_bump**2

        V_vup = interp(surface_t0, S0, self.v0 + dv_bump)
        V_vdn = interp(surface_t0, S0, self.v0 - dv_bump)
        vega  = (V_vup - V_vdn) / (2.0 * dv_bump)

        if V.shape[0] > 1:
            V_next      = interp(V[1], S0, self.v0)
            theta_greek = -(V_next - V_mid) / solver.dt
        else:
            theta_greek = float("nan")

        return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta_greek}

    @staticmethod
    def characteristic_function_price(S0: float,K: float,T: float,r: float,q: float,kappa: float,theta: float,xi: float,rho: float,v0: float,option_type: str = "call",integration_limit: float = 200.0,) -> float:
        """Semi-analytic Heston (1993) price via characteristic function inversion."""

        def _heston_cf(phi: complex) -> complex:
            i = 1j
            x = np.log(S0)
            d = np.sqrt(
                (rho * xi * i * phi - kappa) ** 2
                + xi**2 * (i * phi + phi**2)
            )
            g = (kappa - rho * xi * i * phi - d) / (kappa - rho * xi * i * phi + d)
            exp_dT    = np.exp(-d * T)
            denom_log = 1.0 - g * exp_dT
            if np.abs(denom_log) < 1e-14:
                denom_log = 1e-14
            C = (r - q) * i * phi * T + (kappa * theta / xi**2) * (
                (kappa - rho * xi * i * phi - d) * T
                - 2.0 * np.log(denom_log / (1.0 - g))
            )
            D = (
                (kappa - rho * xi * i * phi - d)
                / xi**2
                * (1.0 - exp_dT)
                / denom_log
            )
            return np.exp(C + D * v0 + i * phi * x)

        log_K = np.log(K)

        def _integrand_P1(phi: float) -> float:
            cf_num = _heston_cf(phi - 1j)
            cf_den = _heston_cf(-1j)
            return float(np.real(
                np.exp(-1j * phi * log_K) * cf_num / (1j * phi * cf_den)
            ))

        def _integrand_P2(phi: float) -> float:
            return float(np.real(
                np.exp(-1j * phi * log_K) * _heston_cf(phi) / (1j * phi)
            ))

        P1 = 0.5 + integrate.quad(_integrand_P1, 1e-6, integration_limit, limit=200)[0] / np.pi
        P2 = 0.5 + integrate.quad(_integrand_P2, 1e-6, integration_limit, limit=200)[0] / np.pi

        call_price = S0 * np.exp(-q * T) * P1 - K * np.exp(-r * T) * P2

        if option_type.lower() == "call":
            return float(call_price)
        return float(call_price - S0 * np.exp(-q * T) + K * np.exp(-r * T))

    @staticmethod
    def feller_condition_satisfied(kappa: float, theta: float, xi: float) -> bool:
        """Return True if 2*kappa*theta > xi^2 (Feller condition)."""
        return 2.0 * kappa * theta > xi**2

    @staticmethod
    def _interpolate_grid_value(grid: np.ndarray, S_grid: np.ndarray, v_grid: np.ndarray, S_target: float,v_target: float) -> float:
        """Bilinear interpolation of a (N_v, N_S) grid at an arbitrary (S, v) point."""
        i_S = int(np.clip(np.searchsorted(S_grid, S_target), 1, len(S_grid) - 1))
        i_v = int(np.clip(np.searchsorted(v_grid, v_target), 1, len(v_grid) - 1))

        S_lo, S_hi = S_grid[i_S - 1], S_grid[i_S]
        v_lo, v_hi = v_grid[i_v - 1], v_grid[i_v]
        w_S = (S_target - S_lo) / (S_hi - S_lo)
        w_v = (v_target - v_lo) / (v_hi - v_lo)

        V_00 = grid[i_v - 1, i_S - 1]  
        V_01 = grid[i_v - 1, i_S]        
        V_10 = grid[i_v,     i_S - 1]    
        V_11 = grid[i_v,     i_S]        

        return float(
            (1.0 - w_v) * ((1.0 - w_S) * V_00 + w_S * V_01)
            + w_v       * ((1.0 - w_S) * V_10 + w_S * V_11)
        )

    def __repr__(self) -> str:
        feller = "OK" if self.feller_condition_satisfied(self.kappa, self.theta, self.xi) else "VIOLATED"
        return (
            f"HestonModel(r={self.r}, q={self.q}, kappa={self.kappa}, "
            f"theta={self.theta}, xi={self.xi}, rho={self.rho}, v0={self.v0}) "
            f"[Feller: {feller}]"
        )