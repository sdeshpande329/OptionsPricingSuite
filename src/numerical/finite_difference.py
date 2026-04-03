import numpy as np
from typing import Callable, Dict, Tuple

from numerical.linear_solvers import solve_tridiagonal


class FiniteDifferenceSolver:
    """
    1D finite-difference solver for the Black-Scholes PDE.

    Supported schemes: explicit, implicit, crank_nicolson
    """

    def __init__(self, scheme: str):
        valid_schemes = {"explicit", "implicit", "crank_nicolson"}
        if scheme not in valid_schemes:
            raise ValueError(f"scheme must be one of {valid_schemes}, got {scheme}")

        self.scheme = scheme
        
        # Crank-Nicolson theta weight:
        self.theta_cn = 0.5 # default theta = 0.5; theta = 1 is implicit scheme

        # placeholders for stock grid and time grid:
        self.S = None
        self.t = None

        # placeholders for number of stock grid points and time grid points:
        self.N_S = None
        self.N_t = None

        # placeholders for grid spacings in stock and time:
        self.dS = None
        self.dt = None

    def setup_grid(self, S_min: float, S_max: float, N_S: int, T: float, N_t: int) -> None:
        """ Set up the stock-price and time grids."""
        # creation of the stock and time grids in an evenly spaced way (grid is 0-indexed):
        self.S = np.linspace(S_min, S_max, N_S)
        self.t = np.linspace(0.0, T, N_t)

        # storing the number of grid points in stock and time:
        self.N_S = N_S
        self.N_t = N_t

        # computing and storing the stepsizes:
        self.dS = self.S[1] - self.S[0]
        self.dt = self.t[1] - self.t[0]

    def solve(self, payoff: Callable[[np.ndarray], np.ndarray], params: Dict[str, float]) -> np.ndarray:
        """
        Solve the PDE backward in time on the current grid.

        Parameters:
            payoff: terminal payoff function evaluated on self.S
            params: model and boundary-condition parameters

        Returns:
            Solution array V of shape (N_t, N_S)
        """
        # raise an error if the grdi does not exist prior to use:
        self._check_grid_initialized()

        V = np.zeros((self.N_t, self.N_S))

        # terminal condition at maturity (payoff at each end stock price in grid):
        V[-1, :] = payoff(self.S)

        # backward time stepping (from final time level to time = 0 [present]):
        for n in range(self.N_t - 1, 0, -1):
            # current time level:
            tau = self.t[n - 1]

            # scheme implementation on time-level:
            if self.scheme == "explicit":
                V[n - 1, :] = self.explicit_step(V[n, :], params, tau)
            elif self.scheme == "implicit":
                V[n - 1, :] = self.implicit_step(V[n, :], params, tau)
            elif self.scheme == "crank_nicolson":
                V[n - 1, :] = self.crank_nicolson_step(V[n, :], params, tau)
        return V

    def explicit_step(self, V_n: np.ndarray, params: Dict[str, float], tau: float) -> np.ndarray:
        """Single backward time step using the explicit scheme."""
        r = params["r"] # interest rate
        sigma = params["sigma"] # volatility

        V_next = np.zeros_like(V_n)

        # boundary values (in stock space) at V_next time-step:
        V_next = self._apply_boundary_conditions(V_next, params, tau)

        # interior update (excluding boundaries):
        for i in range(1, self.N_S - 1):
            S_i = self.S[i]

            # diffusion term of Black-Scholes PDE (uses standard centered difference formula):
            diffusion = 0.5 * sigma**2 * S_i**2 * (V_n[i + 1] - 2.0 * V_n[i] + V_n[i - 1]) / (self.dS**2)

            # convection term of Black-Scholes PDE (uses standard centered difference formula):
            convection = r * S_i * (V_n[i + 1] - V_n[i - 1]) / (2.0 * self.dS)

            # reaction term of the Black-Scholes PDE:
            reaction = -r * V_n[i]

            # explicit update formula:
            V_next[i] = V_n[i] + self.dt * (diffusion + convection + reaction)

        return V_next
    
    def implicit_step(self, V_n: np.ndarray, params: Dict[str, float], tau: float) -> np.ndarray:
        """Single backward time step using the implicit scheme (uses a linear system of equations)."""
        V_next = np.zeros_like(V_n)

        lower, diag, upper = self._build_tridiagonal_implicit(params)
        rhs = self._build_rhs_implicit(V_n, params, tau)

        # solving for interior nodes (excluding boundaries):
        V_next[1:-1] = solve_tridiagonal(lower, diag, upper, rhs)

        # applying boundary conditions at the new time level:
        V_next = self._apply_boundary_conditions(V_next, params, tau)

        return V_next

    def crank_nicolson_step(self, V_n: np.ndarray, params: Dict[str, float], tau: float) -> np.ndarray:
        """Single backward time step using the Crank-Nicolson scheme."""
        V_next = np.zeros_like(V_n)

        lower, diag, upper = self._build_tridiagonal_crank_nicolson(params)
        rhs = self._build_rhs_crank_nicolson(V_n, params, tau)
        
        # solving for interior nodes (excluding boundaries):
        V_next[1:-1] = solve_tridiagonal(lower, diag, upper, rhs)

        # applying boundary conditions at the new time level:
        V_next = self._apply_boundary_conditions(V_next, params, tau)

        return V_next

    def _apply_boundary_conditions(self, V: np.ndarray, params: Dict[str, float], tau: float) -> np.ndarray:
        """
        Apply left and right boundary conditions to a solution row.

        Expected:
            params["boundary_conditions"] = {"left": callable, "right": callable}
        """
        bc = params.get("boundary_conditions", None) # retrieves boundary condition dict

        # raise an error if the boundary dict was not provided in the parameters:
        if bc is None:
            raise ValueError("Missing 'boundary_conditions' in params.")

        # extracts the two boundary condition functions from the dict:
        left_bc = bc.get("left", None)
        right_bc = bc.get("right", None)

        # raise an error if one of the functions wasn't provided:
        if left_bc is None or right_bc is None:
            raise ValueError(
                "params['boundary_conditions'] must contain both 'left' and 'right'."
            )

        # computes and updates boundaries:
        V[0] = left_bc(self.S[0], tau, params)
        V[-1] = right_bc(self.S[-1], tau, params)

        return V

    def _build_tridiagonal_implicit(self, params: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Build the tridiagonal matrix for the implicit scheme.

        Returns arrays: lower, diag, upper corresponding to the interior nodes only.
        """
        r = params["r"] # interest rate
        sigma = params["sigma"] # volatility
        q = params.get("q", 0.0) # dividend yield (default to 0)

        N = self.N_S - 2 # number of interior stock price modes (excludes boundaries)
        lower = np.zeros(N - 1)
        diag = np.zeros(N)
        upper = np.zeros(N - 1)

        # looping through interior unknowns (excluding boundaries):
        for idx in range(N):
            # mapping system index to grid index (idx in system -> i in full stock grid):
            i = idx + 1
            S_i = self.S[i] # stock price at node i

            # finite-difference coefficients of Black-Scholes PDE (alpha:diffusion | beta: convection):
            alpha = 0.5 * sigma**2 * S_i**2 / (self.dS**2)
            beta = (r - q) * S_i / (2.0 * self.dS)

            # coefficients that go into the implicit matrix:
            a_i = self.dt * (alpha - beta) # left neighbor
            b_i = self.dt * (2.0 * alpha + r) # center node
            c_i = self.dt * (alpha + beta) # right neighbor

            # filling the diagonals (first row has no lower diag entry, last row has no upper diag entry):
            if idx > 0:
                lower[idx - 1] = -a_i

            diag[idx] = 1.0 + b_i

            if idx < N - 1:
                upper[idx] = -c_i

        return lower, diag, upper
    
    def _build_rhs_implicit(self, V_n: np.ndarray, params: Dict[str, float], tau: float) -> np.ndarray:
        """Build the right-hand side for the implicit scheme."""
        r = params["r"] # interest rate
        sigma = params["sigma"] # volatility
        q = params.get("q", 0.0) # dividend yield (default to 0)

        N = self.N_S - 2 # number of interior stock price modes (excludes boundaries)
        rhs = np.zeros(N)

        # boundaries at the new time level: 
        V_bc = np.zeros_like(V_n)
        V_bc = self._apply_boundary_conditions(V_bc, params, tau)

        # loops through interior system rows (excluding boundaries):
        for idx in range(N):
            # mapping system index to grid index (idx in system -> i in full stock grid): 
            i = idx + 1
            S_i = self.S[i] # stock price at node i

            # finite-difference coefficients of Black-Scholes PDE (alpha:diffusion | beta: convection):
            alpha = 0.5 * sigma**2 * S_i**2 / (self.dS**2)
            beta = (r - q) * S_i / (2.0 * self.dS)

            # coefficients that go into the implicit matrix:
            a_i = self.dt * (alpha - beta)
            c_i = self.dt * (alpha + beta)
            # no b_i here (RHS for implicit method is just known old row plus boundary corrections)

            # RHS entry starts as the known old value at node i:
            rhs[idx] = V_n[i]

            # left boundary contribution:
            if i == 1:
                rhs[idx] += a_i * V_bc[0]

            # right boundary contribution:
            if i == self.N_S - 2:
                rhs[idx] += c_i * V_bc[-1]

        return rhs
    
    def _build_tridiagonal_crank_nicolson(self, params: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build the tridiagonal matrix for Crank-Nicolson scheme."""
        r = params["r"] # interest rate
        sigma = params["sigma"] # volatility
        q = params.get("q", 0.0) # dividend yield (default to 0)

        N = self.N_S - 2 # number of interior stock price modes (excludes boundaries)
        lower = np.zeros(N - 1)
        diag = np.zeros(N)
        upper = np.zeros(N - 1)

        # looping through interior unknowns (excluding boundaries):
        for idx in range(N):
            # mapping system index to grid index (idx in system -> i in full stock grid):
            i = idx + 1 
            S_i = self.S[i] # stock price at node i
            
            # finite-difference coefficients of Black-Scholes PDE (alpha:diffusion | beta: convection):
            alpha = 0.5 * sigma**2 * S_i**2 / (self.dS**2)
            beta = (r - q) * S_i / (2.0 * self.dS)

            # coefficients that go into the Crank-Nicolson matrix (weight is 0.5):
            a_i = self.theta_cn * self.dt * (alpha - beta)
            b_i = self.theta_cn * self.dt * (2.0 * alpha + r)
            c_i = self.theta_cn * self.dt * (alpha + beta)

            # filling the diagonals (first row has no lower diag entry, last row has no upper diag entry):
            if idx > 0:
                lower[idx - 1] = -a_i

            diag[idx] = 1.0 + b_i

            if idx < N - 1:
                upper[idx] = -c_i

        return lower, diag, upper
    
    def _build_rhs_crank_nicolson(self, V_n: np.ndarray, params: Dict[str, float],
                              tau: float) -> np.ndarray:
        """Build the right-hand side for Crank-Nicolson scheme."""
        r = params["r"] # interest rate
        sigma = params["sigma"] # volatility
        q = params.get("q", 0.0) # dividend yield (default to 0)

        N = self.N_S - 2 # number of interior stock price modes (excludes boundaries)
        rhs = np.zeros(N)

        # boundaries at the new time level:
        V_bc = np.zeros_like(V_n)
        V_bc = self._apply_boundary_conditions(V_bc, params, tau)

        # loops through interior system rows (excluding boundaries):
        for idx in range(N):
            # mapping system index to grid index (idx in system -> i in full stock grid): 
            i = idx + 1
            S_i = self.S[i] # stock price at node i

            # finite-difference coefficients of Black-Scholes PDE (alpha:diffusion | beta: convection):
            alpha = 0.5 * sigma**2 * S_i**2 / (self.dS**2)
            beta = (r - q) * S_i / (2.0 * self.dS)

            # old-time (RHS) theta-method coefficients for interior nodes:
            a_i = (1.0 - self.theta_cn) * self.dt * (alpha - beta)
            b_i = (1.0 - self.theta_cn) * self.dt * (2.0 * alpha + r)
            c_i = (1.0 - self.theta_cn) * self.dt * (alpha + beta)

            # boundary coefficients from the new-time (implicit/LHS) side, moved to the RHS
            a_bc = self.theta_cn * self.dt * (alpha - beta)
            c_bc = self.theta_cn * self.dt * (alpha + beta)

            # RHS entry from the old-time level (explicit side of the theta-method / Crank-Nicolson update):
            rhs[idx] = (a_i * V_n[i - 1] + (1.0 - b_i) * V_n[i] + c_i * V_n[i + 1])

            # left boundary contribution:
            if i == 1:
                rhs[idx] += a_bc * V_bc[0]

            # right boundary contribution:
            if i == self.N_S - 2:
                rhs[idx] += c_bc * V_bc[-1]

        return rhs

    def _check_grid_initialized(self) -> None:
        """
        Ensure setup_grid has been called before solve.
        """
        if self.S is None or self.t is None:
            raise ValueError("Grid not initialized. Call setup_grid(...) first.")