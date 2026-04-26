from __future__ import annotations
import numpy as np
from typing import Callable, Dict, Tuple
from .linear_solvers import solve_tridiagonal

# Theta values taken from literature
_DEFAULT_THETA: Dict[str, float] = {
    "douglas":               0.5,
    "craig_sneyd":           0.5,
    "modified_craig_sneyd":  1.0 / 3.0,
    "hundsdorfer_verwer":    0.5 + np.sqrt(3.0) / 6.0,
}

class ADISolver:
    """Solves the semi-discretised Heston PDE by an ADI splitting scheme."""

    def __init__(self, scheme: str) -> None:
        self.scheme = scheme
        self.theta = _DEFAULT_THETA[scheme]

        self.S: np.ndarray | None = None
        self.v: np.ndarray | None = None
        self.t: np.ndarray | None = None
        self.dS: float | None = None
        self.dv: float | None = None
        self.dt: float | None = None
        self.N_S: int | None = None
        self.N_v: int | None = None
        self.N_t: int | None = None

    
    def setup_grid(self, S_min: float, S_max: float, N_S: int, v_min: float, v_max: float, N_v: int, T: float, N_t: int) -> None:
        """Create uniform spatial and temporal grids."""
        self.S = np.linspace(S_min, S_max, N_S)
        self.v = np.linspace(v_min, v_max, N_v)
        self.t = np.linspace(0.0, T, N_t)

        self.dS = (S_max - S_min) / (N_S - 1)
        self.dv = (v_max - v_min) / (N_v - 1)
        self.dt = T / (N_t - 1)

        self.N_S = N_S
        self.N_v = N_v
        self.N_t = N_t

    
    def solve(self, payoff: Callable, params: Dict[str, float]) -> np.ndarray:
        """Solve the Heston PDE backward in time."""
        V = np.zeros((self.N_t, self.N_v, self.N_S))

        S_grid, v_grid = np.meshgrid(self.S, self.v, indexing='ij')
        V[-1] = payoff(S_grid.T, v_grid.T)

        dispatch = {
            'douglas':             self._douglas_step,
            'craig_sneyd':         self._craig_sneyd_step,
            'modified_craig_sneyd': self._modified_craig_sneyd_step,
            'hundsdorfer_verwer':  self._hundsdorfer_verwer_step,
        }
        if self.scheme not in dispatch:
            raise ValueError(f"Unknown scheme '{self.scheme}'. "
                             f"Choose from {sorted(dispatch)}")
        step = dispatch[self.scheme]
        
        # Rannacher damping: Replace first time step with two backward Euler half-steps to damp oscillations from the non-smooth payoff kink at S=K.
        dt_half = self.dt / 2.0
        V[-2] = self._backward_euler_step(V[-1], params, dt_half)
        V[-2] = self._backward_euler_step(V[-2], params, dt_half)
        V[-2] = self._apply_boundary_conditions(V[-2], params)

        for n in range(self.N_t - 2, 0, -1):
            V[n - 1] = step(V[n], params)
            V[n - 1] = self._apply_boundary_conditions(V[n - 1], params)

        return V
    
    def _backward_euler_step(self, U: np.ndarray, params: Dict, dt_override: float) -> np.ndarray:
        """One step of fully implicit (backward Euler) in both spatial directions. Used for Rannacher damping at t=0."""
        dt_orig  = self.dt
        th_orig  = self.theta
        self.dt    = dt_override
        self.theta = 1.0

        FU = self._apply_F(U, params)
        Y0 = U + dt_override * FU

        Y1 = Y0.copy()
        for j in range(1, self.N_v - 1):
            lo, diag, up = self._build_tridiagonal_S(j, self.v[j], params)
            rhs = self._rhs_implicit_S_correction(Y0, U, j, params, th=1.0)
            Y1[j, 1:-1] = solve_tridiagonal(lo, diag, up, rhs)
        self._copy_boundaries(Y0, Y1)

        Y2 = Y1.copy()
        for i in range(1, self.N_S - 1):
            lo, diag, up = self._build_tridiagonal_v(i, self.S[i], params)
            rhs = self._rhs_implicit_v_correction(Y1, U, i, params, th=1.0)
            Y2[1:-1, i] = solve_tridiagonal(lo, diag, up, rhs)
        self._copy_boundaries(Y1, Y2)

        self.dt    = dt_orig
        self.theta = th_orig

        return Y2
    
    def _douglas_step(self, U: np.ndarray, params: Dict) -> np.ndarray:
        """
        Douglas (Do) scheme:

            Y0 = U + dt * F(U)
            Y1 = Y0 + theta*dt * (F1(Y1) - F1(U))       [implicit in S]
            Y2 = Y1 + theta*dt * (F2(Y2) - F2(U))       [implicit in v]
            U_new = Y2

        F0 (mixed derivative) appears only in Y0 and is NOT corrected.
        """
        dt = self.dt
        th = self.theta

        Y0 = U + dt * self._apply_F(U, params)

        # Y1: implicit S-direction correction
        Y1 = Y0.copy()
        for j in range(1, self.N_v - 1):
            lo, diag, up = self._build_tridiagonal_S(j, self.v[j], params)
            rhs = self._rhs_implicit_S_correction(Y0, U, j, params, th)
            Y1[j, 1:-1] = solve_tridiagonal(lo, diag, up, rhs)
        self._copy_boundaries(Y0, Y1)

        # Y2: implicit v-direction correction
        Y2 = Y1.copy()
        for i in range(1, self.N_S - 1):
            lo, diag, up = self._build_tridiagonal_v(i, self.S[i], params)
            rhs = self._rhs_implicit_v_correction(Y1, U, i, params, th)
            Y2[1:-1, i] = solve_tridiagonal(lo, diag, up, rhs)
        self._copy_boundaries(Y1, Y2)

        return Y2

    def _craig_sneyd_step(self, U: np.ndarray, params: Dict) -> np.ndarray:
        """
        Craig-Sneyd (CS) scheme, equation (2.18):

            Y0  = U + dt*F(U)
            Y1  = Y0 + theta*dt*(F1(Y1) - F1(U))
            Y2  = Y1 + theta*dt*(F2(Y2) - F2(U))
            ~Y0 = Y0 + (1/2)*dt*(F0(Y2) - F0(U))        [mixed-deriv correction]
            ~Y1 = ~Y0 + theta*dt*(F1(~Y1) - F1(U))
            ~Y2 = ~Y1 + theta*dt*(F2(~Y2) - F2(U))
            U_new = ~Y2

        The corrector re-uses the SAME tridiagonal systems (I - theta*dt*A1/A2) as the predictor, only the RHS changes.
        """
        dt = self.dt
        th = self.theta

        # Predictor (identical to Douglas)
        Y0 = U + dt * self._apply_F(U, params)

        Y1 = Y0.copy()
        for j in range(1, self.N_v - 1):
            lo, diag, up = self._build_tridiagonal_S(j, self.v[j], params)
            rhs = self._rhs_implicit_S_correction(Y0, U, j, params, th)
            Y1[j, 1:-1] = solve_tridiagonal(lo, diag, up, rhs)
        self._copy_boundaries(Y0, Y1)

        Y2 = Y1.copy()
        for i in range(1, self.N_S - 1):
            lo, diag, up = self._build_tridiagonal_v(i, self.S[i], params)
            rhs = self._rhs_implicit_v_correction(Y1, U, i, params, th)
            Y2[1:-1, i] = solve_tridiagonal(lo, diag, up, rhs)
        self._copy_boundaries(Y1, Y2)

        # Corrector: ~Y0 = Y0 + (1/2)*dt*(F0(Y2) - F0(U))
        F0_Y2 = self._apply_F0(Y2, params)
        F0_U  = self._apply_F0(U,  params)
        tY0 = Y0 + 0.5 * dt * (F0_Y2 - F0_U)
        self._copy_boundaries(Y0, tY0)

        tY1 = tY0.copy()
        for j in range(1, self.N_v - 1):
            lo, diag, up = self._build_tridiagonal_S(j, self.v[j], params)
            rhs = self._rhs_implicit_S_correction(tY0, U, j, params, th)
            tY1[j, 1:-1] = solve_tridiagonal(lo, diag, up, rhs)
        self._copy_boundaries(tY0, tY1)

        tY2 = tY1.copy()
        for i in range(1, self.N_S - 1):
            lo, diag, up = self._build_tridiagonal_v(i, self.S[i], params)
            rhs = self._rhs_implicit_v_correction(tY1, U, i, params, th)
            tY2[1:-1, i] = solve_tridiagonal(lo, diag, up, rhs)
        self._copy_boundaries(tY1, tY2)

        return tY2

    def _modified_craig_sneyd_step(self, U: np.ndarray, params: Dict) -> np.ndarray:
        """
        Modified Craig-Sneyd (MCS) scheme, equation (2.19):

            Y0  = U + dt*F(U)
            Y1  = Y0 + theta*dt*(F1(Y1) - F1(U))
            Y2  = Y1 + theta*dt*(F2(Y2) - F2(U))
            ^Y0 = Y0 + theta*dt*(F0(Y2) - F0(U))
            ~Y0 = ^Y0 + (1/2 - theta)*dt*(F(Y2) - F(U))
            ~Y1 = ~Y0 + theta*dt*(F1(~Y1) - F1(U))
            ~Y2 = ~Y1 + theta*dt*(F2(~Y2) - F2(U))
            U_new = ~Y2

        Note the corrector sweeps subtract F1(U) / F2(U), i.e. the OLD level, unlike HV which subtracts F1(Y2) / F2(Y2).
        """
        dt = self.dt
        th = self.theta

        # Predictor
        FU  = self._apply_F(U, params)
        Y0 = U + dt * FU

        Y1 = Y0.copy()
        for j in range(1, self.N_v - 1):
            lo, diag, up = self._build_tridiagonal_S(j, self.v[j], params)
            rhs = self._rhs_implicit_S_correction(Y0, U, j, params, th)
            Y1[j, 1:-1] = solve_tridiagonal(lo, diag, up, rhs)
        self._copy_boundaries(Y0, Y1)

        Y2 = Y1.copy()
        for i in range(1, self.N_S - 1):
            lo, diag, up = self._build_tridiagonal_v(i, self.S[i], params)
            rhs = self._rhs_implicit_v_correction(Y1, U, i, params, th)
            Y2[1:-1, i] = solve_tridiagonal(lo, diag, up, rhs)
        self._copy_boundaries(Y1, Y2)

        # ^Y0 = Y0 + theta*dt*(F0(Y2) - F0(U))
        F0_Y2 = self._apply_F0(Y2, params)
        F0_U  = self._apply_F0(U,  params)
        hY0 = Y0 + th * dt * (F0_Y2 - F0_U)
        self._copy_boundaries(Y0, hY0)

        # ~Y0 = ^Y0 + (1/2 - theta)*dt*(F(Y2) - F(U))
        FY2 = self._apply_F(Y2, params)
        tY0 = hY0 + (0.5 - th) * dt * (FY2 - FU)
        self._copy_boundaries(hY0, tY0)

        # Corrector sweeps subtract F1(U) / F2(U) -- same as predictor reference
        tY1 = tY0.copy()
        for j in range(1, self.N_v - 1):
            lo, diag, up = self._build_tridiagonal_S(j, self.v[j], params)
            rhs = self._rhs_implicit_S_correction(tY0, U, j, params, th)
            tY1[j, 1:-1] = solve_tridiagonal(lo, diag, up, rhs)
        self._copy_boundaries(tY0, tY1)

        tY2 = tY1.copy()
        for i in range(1, self.N_S - 1):
            lo, diag, up = self._build_tridiagonal_v(i, self.S[i], params)
            rhs = self._rhs_implicit_v_correction(tY1, U, i, params, th)
            tY2[1:-1, i] = solve_tridiagonal(lo, diag, up, rhs)
        self._copy_boundaries(tY1, tY2)

        return tY2

    def _hundsdorfer_verwer_step(self, U: np.ndarray, params: Dict) -> np.ndarray:
        """
        Hundsdorfer-Verwer (HV) scheme, equation (2.20):

            Y0  = U + dt*F(U)
            Y1  = Y0 + theta*dt*(F1(Y1) - F1(U))
            Y2  = Y1 + theta*dt*(F2(Y2) - F2(U))
            ~Y0 = Y0 + (1/2)*dt*(F(Y2)  - F(U))         [full F, not just F0]
            ~Y1 = ~Y0 + theta*dt*(F1(~Y1) - F1(Y2))     [subtracts F1(Y2), not F1(U)]
            ~Y2 = ~Y1 + theta*dt*(F2(~Y2) - F2(Y2))     [subtracts F2(Y2), not F2(U)]
            U_new = ~Y2

        Key difference from MCS: corrector sweeps reference Y2, not U.
        """
        dt = self.dt
        th = self.theta

        # Predictor
        FU = self._apply_F(U, params)
        Y0 = U + dt * FU

        Y1 = Y0.copy()
        for j in range(1, self.N_v - 1):
            lo, diag, up = self._build_tridiagonal_S(j, self.v[j], params)
            rhs = self._rhs_implicit_S_correction(Y0, U, j, params, th)
            Y1[j, 1:-1] = solve_tridiagonal(lo, diag, up, rhs)
        self._copy_boundaries(Y0, Y1)

        Y2 = Y1.copy()
        for i in range(1, self.N_S - 1):
            lo, diag, up = self._build_tridiagonal_v(i, self.S[i], params)
            rhs = self._rhs_implicit_v_correction(Y1, U, i, params, th)
            Y2[1:-1, i] = solve_tridiagonal(lo, diag, up, rhs)
        self._copy_boundaries(Y1, Y2)

        # ~Y0 = Y0 + (1/2)*dt*(F(Y2) - F(U))   -- uses FULL F, not just F0
        FY2 = self._apply_F(Y2, params)
        tY0 = Y0 + 0.5 * dt * (FY2 - FU)
        self._copy_boundaries(Y0, tY0)

        # Corrector sweeps reference Y2 (not U) -- this is the HV distinction
        tY1 = tY0.copy()
        for j in range(1, self.N_v - 1):
            lo, diag, up = self._build_tridiagonal_S(j, self.v[j], params)
            rhs = self._rhs_implicit_S_correction(tY0, Y2, j, params, th)
            tY1[j, 1:-1] = solve_tridiagonal(lo, diag, up, rhs)
        self._copy_boundaries(tY0, tY1)

        tY2 = tY1.copy()
        for i in range(1, self.N_S - 1):
            lo, diag, up = self._build_tridiagonal_v(i, self.S[i], params)
            rhs = self._rhs_implicit_v_correction(tY1, Y2, i, params, th)
            tY2[1:-1, i] = solve_tridiagonal(lo, diag, up, rhs)
        self._copy_boundaries(tY1, tY2)

        return tY2


    def _apply_F(self, V: np.ndarray, params: Dict) -> np.ndarray:
        """Apply the full operator F = F0 + F1 + F2 explicitly."""
        return self._apply_F0(V, params) + self._apply_F1(V, params) + self._apply_F2(V, params)

    def _apply_F0(self, V: np.ndarray, params: Dict) -> np.ndarray:
        """
        Apply the mixed derivative operator F0 explicitly.

        F0[j,i] = rho * xi * v_j * S_i * d2V/dSdv  (central differences)
        """
        rho = params['rho']
        xi  = params['xi']

        result = np.zeros_like(V)
        for j in range(1, self.N_v - 1):
            for i in range(1, self.N_S - 1):
                v_j = self.v[j]
                S_i = self.S[i]
                d2V = (V[j+1, i+1] - V[j+1, i-1]
                       - V[j-1, i+1] + V[j-1, i-1]) / (4.0 * self.dS * self.dv)
                result[j, i] = rho * xi * v_j * S_i * d2V
        return result

    def _apply_F1(self, V: np.ndarray, params: Dict) -> np.ndarray:
        """
        Apply the S-direction operator F1 explicitly.

        F1[j,i] = (1/2)*v_j*S_i^2 * d2V/dS^2
                + (r-q)*S_i * dV/dS
                - (r/2) * V
        """
        r  = params['r']
        q  = params.get('q', 0.0)
        dS = self.dS

        result = np.zeros_like(V)
        for j in range(1, self.N_v - 1):
            v_j = self.v[j]
            for i in range(1, self.N_S - 1):
                S_i = self.S[i]
                d2V_dS2 = (V[j, i+1] - 2.0*V[j, i] + V[j, i-1]) / dS**2
                dV_dS   = (V[j, i+1] - V[j, i-1]) / (2.0 * dS)
                result[j, i] = (0.5 * v_j * S_i**2 * d2V_dS2
                                + (r - q) * S_i * dV_dS
                                - 0.5 * r * V[j, i])
        return result

    def _apply_F2(self, V: np.ndarray, params: Dict) -> np.ndarray:
        """
        Apply the v-direction operator F2 explicitly.

        F2[j,i] = (1/2)*xi^2*v_j * d2V/dv^2
                + kappa*(theta-v_j) * dV/dv
                - (r/2) * V
        """
        r       = params['r']
        kappa   = params['kappa']
        theta_v = params['theta']
        xi      = params['xi']
        dv      = self.dv

        result = np.zeros_like(V)
        for j in range(1, self.N_v - 1):
            v_j = self.v[j]
            for i in range(1, self.N_S - 1):
                d2V_dv2 = (V[j+1, i] - 2.0*V[j, i] + V[j-1, i]) / dv**2
                dV_dv   = (V[j+1, i] - V[j-1, i]) / (2.0 * dv)
                result[j, i] = (0.5 * xi**2 * v_j * d2V_dv2
                                + kappa * (theta_v - v_j) * dV_dv
                                - 0.5 * r * V[j, i])
        return result

    def _build_tridiagonal_S(self, j: int, v_j: float, params: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build (I - theta*dt*A1) tridiagonal system for the S-direction."""
        r  = params['r']
        q  = params.get('q', 0.0)
        th = self.theta
        dt = self.dt
        dS = self.dS

        N = self.N_S - 2
        lower = np.zeros(N - 1)
        diag  = np.zeros(N)
        upper = np.zeros(N - 1)

        for idx in range(N):
            i   = idx + 1
            S_i = self.S[i]
            alpha = 0.5 * v_j * S_i**2 / dS**2
            beta  = (r - q) * S_i / (2.0 * dS)

            if idx > 0:
                lower[idx - 1] = -th * dt * (alpha - beta)
            diag[idx] = 1.0 + th * dt * (2.0 * alpha + 0.5 * r)
            if idx < N - 1:
                upper[idx] = -th * dt * (alpha + beta)

        return lower, diag, upper

    def _build_tridiagonal_v(self, i: int, S_i: float, params: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build (I - theta*dt*A2) tridiagonal system for the v-direction."""
        r       = params['r']
        kappa   = params['kappa']
        theta_v = params['theta']
        xi      = params['xi']
        th      = self.theta
        dt      = self.dt
        dv      = self.dv

        N = self.N_v - 2
        lower = np.zeros(N - 1)
        diag  = np.zeros(N)
        upper = np.zeros(N - 1)

        for idx in range(N):
            j   = idx + 1
            v_j = self.v[j]
            alpha = 0.5 * xi**2 * v_j / dv**2
            beta  = kappa * (theta_v - v_j) / (2.0 * dv)

            if idx > 0:
                lower[idx - 1] = -th * dt * (alpha - beta)
            diag[idx] = 1.0 + th * dt * (2.0 * alpha + 0.5 * r)
            if idx < N - 1:
                upper[idx] = -th * dt * (alpha + beta)

        return lower, diag, upper


    def _rhs_implicit_S_correction(self,Y_prev: np.ndarray, ref: np.ndarray, j: int, params: Dict, th: float) -> np.ndarray:
        """RHS for the implicit S-direction correction step."""
        N   = self.N_S - 2
        rhs = np.zeros(N)
        th_dt = th * self.dt

        for idx in range(N):
            i   = idx + 1
            S_i = self.S[i]
            v_j = self.v[j]
            r   = params['r']
            q   = params.get('q', 0.0)
            dS  = self.dS

            alpha = 0.5 * v_j * S_i**2 / dS**2
            beta  = (r - q) * S_i / (2.0 * dS)

            F1_ref = (alpha * (ref[j, i+1] - 2.0*ref[j, i] + ref[j, i-1])
                      + beta * (ref[j, i+1] - ref[j, i-1])
                      - 0.5 * r * ref[j, i])

            rhs[idx] = Y_prev[j, i] - th_dt * F1_ref

        return rhs

    def _rhs_implicit_v_correction(self, Y_prev: np.ndarray, ref: np.ndarray, i: int, params: Dict, th: float) -> np.ndarray:
        """RHS for the implicit v-direction correction step."""
        N   = self.N_v - 2
        rhs = np.zeros(N)
        th_dt = th * self.dt

        r       = params['r']
        kappa   = params['kappa']
        theta_v = params['theta']
        xi      = params['xi']
        dv      = self.dv

        for idx in range(N):
            j   = idx + 1
            v_j = self.v[j]

            alpha = 0.5 * xi**2 * v_j / dv**2
            beta  = kappa * (theta_v - v_j) / (2.0 * dv)

            F2_ref = (alpha * (ref[j+1, i] - 2.0*ref[j, i] + ref[j-1, i])
                      + beta * (ref[j+1, i] - ref[j-1, i])
                      - 0.5 * r * ref[j, i])

            rhs[idx] = Y_prev[j, i] - th_dt * F2_ref

        return rhs

    def _apply_boundary_conditions(self, V: np.ndarray, params: Dict) -> np.ndarray:
        """Apply Heston PDE boundary conditions (equations 2.3-2.5)."""
        return V

    @staticmethod
    def _copy_boundaries(src: np.ndarray, dst: np.ndarray) -> None:
        """Copy boundary rows/columns from src into dst (in-place)."""
        dst[0,  :] = src[0,  :]
        dst[-1, :] = src[-1, :]
        dst[:,  0] = src[:,  0]
        dst[:, -1] = src[:, -1]