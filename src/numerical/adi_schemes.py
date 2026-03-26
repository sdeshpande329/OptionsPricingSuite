"""Alternating Direction Implicit (ADI) Schemes are used for solving multi-dimensional parabolic partial differential equations by breaking down the problem into a series of 1D problems."""

import numpy as np
from typing import Callable, Dict, Tuple
from numerical.linear_solvers import *

class ADISolver:
    def __init__(self, scheme: str, theta: float = 0.5):
        self.scheme = scheme
        self.theta = theta
        self.S = None
        self.v = None
        self.t = None
        self.dS = None
        self.dv = None
        self.dt = None

    def setup_grid(self, S_min: float, S_max: float, N_S: int,
                   v_min: float, v_max: float, N_v: int,
                   T: float, N_t: int):
        """Sets up the grid for the problem."""
        self.S = np.linspace(S_min, S_max, N_S)
        self.v = np.linspace(v_min, v_max, N_v)
        self.t = np.linspace(0, T, N_t)

        self.dS = (self.S[1] - self.S[0]) / (N_S - 1)
        self.dv = (self.v[1] - self.v[0]) / (N_v - 1)
        self.dt = T/ (N_t - 1)

    def solve(self, payoff: Callable, params: Dict[str, float]):
        """Solves the problem using the selected scheme."""

        V = np.zeros((self.N_t, self.N_v, self.N_S))
        S_grid, v_grid = np.meshgrid(self.S, self.v, indexing='ij')
        V[-1, :, :] = payoff(S_grid.T, v_grid.T)

        if self.scheme == 'douglas':
            step_method = self.douglas_scheme
        elif self.scheme == 'craig_sneyd':
            step_method = self.craig_sneyd_scheme
        elif self.scheme == 'modified_craig_sneyd':
            step_method = self.modified_craig_sneyd_scheme
        elif self.scheme == 'hundsdorfer_verwer':
            step_method = self.hundsdorfer_verwer_scheme

        for i in range(self.N_t - 1, 0, -1):
            V[i-1] = step_method(V[i], params)
            V[i-1] = self._apply_boundary_conditions(V[i-1], params)

        return V

    # Schemes

    def douglas_scheme(self, V_n: np.ndarray, params:Dict) -> np.ndarray:
        """
        The Douglas scheme is an ADI method that splits the 2D problem into two 1D problems solved sequentially. Each spatial direction is treated implicitly in alternating half-steps.
        First-order accurate in time O(Δt) and second-order accurate in space O(Δx², Δy²) with total accuracy O(Δt, Δx², Δy²). Unconditionally stable for diffusion terms.
        """
        N_v, N_S = V_n.shape
        V_star = np.zeros_like(V_n)
        V_next = np.zeros_like(V_n)

        # S-direction sweep
        for j in range(1, N_v - 1):
            v_j = self.v[j]
            lower, diag, upper = self._build_tridiagonal_S(j, v_j, params)
            rhs = self._build_rhs_douglas_step1(V_n, j, v_j, params)
            V_star[j, 1:-1] = solve_tridiagonal(lower, diag, upper, rhs)
        
        # Copy boundary conditions from V_n to V_star
        V_star[0, :] = V_n[0, :]
        V_star[-1, :] = V_n[-1, :]
        V_star[:, 0] = V_n[:, 0]
        V_star[:, -1] = V_n[:, -1]

        # Step 2: v-direction sweep (solve all columns)
        for i in range(1, N_S - 1):
            S_i = self.S[i]
            lower, diag, upper = self._build_tridiagonal_v(i, S_i, params)
            rhs = self._build_rhs_douglas_step2(V_star, i, S_i, params)
            V_next[1:-1, i] = solve_tridiagonal(lower, diag, upper, rhs)

        # Copy boundary conditions from V_star to V_next
        V_next[0, :] = V_star[0, :]
        V_next[-1, :] = V_star[-1, :]
        V_next[:, 0] = V_star[:, 0]
        V_next[:, -1] = V_star[:, -1]

        return V_next

    def craig_sneyd_scheme(self, V_n: np.ndarray, params:Dict) -> np.ndarray:
        """
        The Craig-Sneyd scheme improves upon Douglas by adding a correction step that better handles mixed derivative terms. 
        It uses a three-stage process: two half-step ADI sweeps followed by a correction for the mixed derivative. 
        Second-order accurate in time O(Δt²) and second-order accurate in space O(Δx², Δy²) with total accuracy O(Δt², Δx², Δy²). 
        Unconditionally stable for the full Heston PDE. Handles strong correlation (large |ρ|) much better than Douglas. Stable even when ρ → ±1. No spurious oscillations observed in practice.
        """
        N_v, N_S = V_n.shape
        V_bar = np.zeros_like(V_n)
        
        # S-direction sweep
        for j in range(1, N_v - 1):
            v_j = self.v[j]
            lower, diag, upper = self._build_tridiagonal_S(j, v_j, params)
            rhs = self._build_rhs_craig_sneyd_step_1(V_n, j, v_j, params)
            V_bar[j, 1:-1] = solve_tridiagonal(lower, diag, upper, rhs)

        # Copy boundary conditions from V_n to V_bar
        V_bar[0, :] = V_n[0, :]
        V_bar[-1, :] = V_n[-1, :]
        V_bar[:, 0] = V_n[:, 0]
        V_bar[:, -1] = V_n[:, -1]

        # v-direction sweep
        V_star = np.zeros_like(V_n)
        for i in range(1, N_S - 1):
            S_i = self.S[i]
            lower, diag, upper = self._build_tridiagonal_v(i, S_i, params)
            rhs = self._build_rhs_craig_sneyd_step_2(V_bar, i, S_i, params)
            V_star[1:-1, i] = solve_tridiagonal(lower, diag, upper, rhs)
            
        # Copy boundary conditions from V_bar to V_star
        V_star[0, :] = V_bar[0, :]
        V_star[-1, :] = V_bar[-1, :]
        V_star[:, 0] = V_bar[:, 0]
        V_star[:, -1] = V_bar[:, -1]

        # S-direction correction
        V_tilde = np.zeros_like(V_n)
        for j in range(1, N_v - 1):
            v_j = self.v[j]
            lower, diag, upper = self._build_tridiagonal_S(j, v_j, params)
            rhs = self._build_rhs_craig_sneyd_step_3(V_star, V_n, j, v_j, params)
            V_tilde[j, 1:-1] = solve_tridiagonal(lower, diag, upper, rhs)
            
        # Copy boundary conditions from V_star to V_tilde
        V_tilde[0, :] = V_star[0, :]
        V_tilde[-1, :] = V_star[-1, :]
        V_tilde[:, 0] = V_star[:, 0]
        V_tilde[:, -1] = V_star[:, -1]

        # Step 4: v-direction correction
        V_next = np.zeros_like(V_n)
        for i in range(1, N_S - 1):
            S_i = self.S[i]
            lower, diag, upper = self._build_tridiagonal_v(i, S_i, params)
            rhs = self._build_rhs_craig_sneyd_step_4(V_tilde, V_star, i, S_i, params)
            V_next[1:-1, i] = solve_tridiagonal(lower, diag, upper, rhs)
            
        # Copy boundary conditions from V_tilde to V_next
        V_next[0, :] = V_tilde[0, :]
        V_next[-1, :] = V_tilde[-1, :]
        V_next[:, 0] = V_tilde[:, 0]
        V_next[:, -1] = V_tilde[:, -1]

        return V_next

    def modified_craig_sneyd_scheme(self):
        """
        The MCS scheme modifies the standard Craig-Sneyd by adjusting how the mixed derivative correction is applied. 
        It uses a different splitting that improves stability properties while maintaining accuracy. Second-order accurate in time O(Δt²) and second-order accurate in space O(Δx², Δy²) with total accuracy O(Δt², Δx², Δy²). 
        Unconditionally stable. Enhanced stability compared to standard Craig-Sneyd. Better damping of high-frequency errors. Particularly stable for large time steps. Excellent performance with strong correlation.
        """
        pass


    def hundsdorfer_verwer_scheme(self):
        """
        The Hundsdorfer-Verwer scheme is a sophisticated ADI method that combines features of Craig-Sneyd with additional correction terms. It achieves excellent accuracy for mixed derivatives through a carefully constructed multi-stage process. 
        Second-order accurate in time O(Δt²) and second-order accurate in space O(Δx², Δy²) with total accuracy O(Δt², Δx², Δy²). 
        Best accuracy among ADI schemes for mixed derivative problems. Smallest error constants (lowest actual errors for given Δt, Δx, Δy). 
        Unconditionally stable. Excellent stability properties for all correlation values. Superior damping of oscillations. Most robust ADI scheme for difficult problems.
        """
        pass

    # Helper Functions

    def _apply_boundary_conditions(self, V, params):
        """Applies the boundary conditions to the solution."""
        return V

    def _build_tridiagonal_S(self, j:int, v_j:float, params: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Builds the tridiagonal system for the S direction."""
        r = params['r']
        q = params.get('q', 0.0)
        
        N = self.N_S - 2  # Interior points only
        lower = np.zeros(N - 1)
        diag = np.zeros(N)
        upper = np.zeros(N - 1)
        
        for idx in range(N):
            i = idx + 1  # Actual grid index (skip boundary at i=0)
            S_i = self.S[i]
            
            
            alpha = 0.5 * v_j * S_i**2 / self.dS**2
            beta = (r - q) * S_i / (2 * self.dS)
            
            if idx > 0:
                lower[idx - 1] = -self.theta * self.dt * (alpha - beta)
            
            diag[idx] = 1 + self.theta * self.dt * (2 * alpha + r)
            
            if idx < N - 1:
                upper[idx] = -self.theta * self.dt * (alpha + beta)
        
        return lower, diag, upper

    def _build_tridiagonal_v(self, i:int, S_i:float, params: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Builds the tridiagonal system for the v direction."""
        r = params['r']
        kappa = params['kappa']
        theta_v = params['theta']
        xi = params['xi']

        N = self.N_v - 2  # Interior points only
        lower = np.zeros(N - 1)
        diag = np.zeros(N)
        upper = np.zeros(N - 1)
        
        for idx in range(N):
            j = idx + 1  # Actual grid index (skip boundary at j=0)
            v_j = self.v[j]
            
            alpha = 0.5 * xi**2 * v_j / self.dv**2
            beta = kappa * (theta_v - v_j) / (2 * self.dv)

            if idx > 0:
                lower[idx - 1] = -self.theta * self.dt * (alpha - beta)

            diag[idx] = 1 + self.theta * self.dt * (2 * alpha + kappa)

            if idx < N - 1:
                upper[idx] = -self.theta * self.dt * (alpha + beta)

        return lower, diag, upper

    def _apply_mixed_derivative_correction(self, V:np.ndarray, j:int, i:int, params: Dict) -> float:
        """Applies the mixed derivative correction to the solution."""
        rho = params['rho']
        xi = params['xi']
        
        v_j = self.v[j]
        S_i = self.S[i]
        
        # Check bounds
        if j == 0 or j == self.N_v - 1 or i == 0 or i == self.N_S - 1:
            return 0.0
        
        # Mixed derivative (central difference)
        d2V_dSdv = (V[j+1, i+1] - V[j+1, i-1] - V[j-1, i+1] + V[j-1, i-1]) / (4 * self.dS * self.dv)
        
        L_Sv = rho * xi * v_j * S_i * d2V_dSdv
        
        return L_Sv
        
    def _build_rhs_douglass_step_1(self, V_n: np.ndarray, j:int, v_j:float, params: Dict) -> np.ndarray:
        """Builds the right-hand side for the first step of Douglas scheme."""
        r = params['r']
        q = params.get('q', 0.0)
        kappa = params['kappa']
        theta_v = params['theta']
        xi = params['xi']
        
        N = self.N_S - 2  # Interior points only
        rhs = np.zeros(N)
        
        for idx in range(N):
            i = idx + 1  # Actual grid index (skip boundary at i=0)
            S_i = self.S[i]

            alpha_S = 0.5 * v_j * S_i**2 / self.dS**2
            beta_S = (r - q) * S_i / (2 * self.dS)
            L_S_explicit = (1-self.theta) * self.dt * (
                alpha_S * (V_n[j, i+1] - 2 * V_n[j, i] + V_n[j, i-1]) + beta_S * (V_n[j, i+1] - V_n[j, i-1])
            )
            
            alpha_v = 0.5 * xi**2 * v_j / self.dv**2
            beta_v = kappa * (theta_v - v_j) / (2 * self.dv)
            L_v_explicit = self.dt * (
                alpha_v * (V_n[j+1, i] - 2*V_n[j,i] + V_n[j-1, i]) + beta_v * (V_n[j+1, i] - V_n[j-1, i])
            )

            L_Sv = self.dt * self._apply_mixed_derivative_correction(V_n, j, i, params)

            reaction_term = -self.dt * r * V_n[j,i]

            rhs[idx] = V_n[j, i] + L_S_explicit + L_v_explicit + L_Sv + reaction_term

        return rhs

    def _build_rhs_douglass_step_2(self, V_star: np.ndarray, i: int, S_i: float, params:Dict) -> np.ndarray:
        """Builds the right-hand side for the second step of Douglas scheme."""
        r = params['r']
        q = params.get('q', 0.0)
        
        N = self.N_v - 2  # Interior points only
        rhs = np.zeros(N)
        
        for idx in range(N):
            j = idx + 1 
            v_j = self.v[j]

            alpha_S = 0.5 * v_j * S_i**2 / self.dS**2
            beta_S = (r-q) *S_i/(2*self.dS)
            L_S_explicit = (1-self.theta) * self.dt * (
                alpha_S * (V_star[j, i+1] - 2 * V_star[j, i] + V_star[j, i-1]) + beta_S * (V_star[j, i+1] - V_star[j, i-1])
            )

            rhs[idx] = V_star[j, i] + L_S_explicit

        return rhs

    def _build_rhs_craig_sneyd_step_1(self, V_n: np.ndarray, j:int, v_j:float, params: Dict) -> np.ndarray:
        """Builds the right-hand side for the first step of Craig-Sneyd scheme."""
        return self._build_rhs_douglass_step_1(V_n, j, v_j, params)

    def _build_rhs_craig_sneyd_step_2(self, V_bar: np.ndarray, i: int, S_i: float, params:Dict) -> np.ndarray:
        """Builds the right-hand side for the second step of Craig-Sneyd scheme."""
        return self._build_rhs_douglass_step_2(V_bar, i, S_i, params)

    def _build_rhs_craig_sneyd_step_3(self, V_star: np.ndarray, V_n: np.ndarray, j:int, v_j:float, params: Dict) -> np.ndarray:
        """Builds the right-hand side for the third step of Craig-Sneyd scheme."""
        N = self.N_v - 2  # Interior points only
        rhs = np.zeros(N)
        
        for idx in range(N):
            i = idx + 1 

            L_Sv_star = self._apply_mixed_derivative_correction(V_star, j, i, params)
            L_Sv_n = self._apply_mixed_derivative_correction(V_n, j, i, params)

            correction_term = self.dt * (L_Sv_star - L_Sv_n)

            rhs[idx] = V_star[j, i] + correction_term

        return rhs

    def _build_rhs_craig_sneyd_step_4(self, V_tilde:np.ndarray, V_star: np.ndarray, i:int, S_i:float, params:Dict) -> np.ndarray:
        """Builds the right-hand side for the fourth step of Craig-Sneyd scheme."""
        N = self.N_v - 2  # Interior points only
        rhs = np.zeros(N)
        
        for idx in range(N):
            j = idx + 1 
            rhs[idx] = V_tilde[j, i]

        return rhs