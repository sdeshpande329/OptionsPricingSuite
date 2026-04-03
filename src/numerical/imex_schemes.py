import numpy as np
from typing import Callable, Dict, Tuple

from numerical.linear_solvers import solve_tridiagonal


class IMEXSolver:
    """
    1D IMEX finite-difference solver for the Merton jump-diffusion PIDE.

    The local diffusion/convection/reaction operator is treated implicitly.
    The nonlocal jump integral is treated explicitly.
    """

    def __init__(self, scheme: str = "imex_euler"):
        valid_schemes = {"imex_euler"}
        if scheme not in valid_schemes:
            raise ValueError(f"scheme must be one of {valid_schemes}, got {scheme}")

        self.scheme = scheme

        self.S = None
        self.t = None

        self.N_S = None
        self.N_t = None

        self.dS = None
        self.dt = None

    def setup_grid(self, S_min: float, S_max: float, N_S: int, T: float, N_t: int) -> None:
        """Set up the stock-price and time grids."""
        self.S = np.linspace(S_min, S_max, N_S)
        self.t = np.linspace(0.0, T, N_t)

        self.N_S = N_S
        self.N_t = N_t

        self.dS = self.S[1] - self.S[0]
        self.dt = self.t[1] - self.t[0]

    def solve(self, payoff: Callable[[np.ndarray], np.ndarray], params: Dict[str, float]) -> np.ndarray:
        """
        Solve the jump-diffusion PIDE backward in time on the current grid.

        Required params:
            r, sigma, lambda_jump, jump_mean, jump_std
            boundary_conditions = {"left": callable, "right": callable}

        Optional params:
            q
            quadrature_points
            jump_std_width
        """
        self._check_grid_initialized()
        self._validate_jump_params(params)

        V = np.zeros((self.N_t, self.N_S))
        V[-1, :] = payoff(self.S)

        for n in range(self.N_t - 1, 0, -1):
            tau = self.t[n - 1]

            if self.scheme == "imex_euler":
                V[n - 1, :] = self.imex_euler_step(V[n, :], params, tau)

        return V

    def imex_euler_step(self, V_n: np.ndarray, params: Dict[str, float], tau: float) -> np.ndarray:
        """Single backward time step using first-order IMEX Euler."""
        V_next = np.zeros_like(V_n)

        lower, diag, upper = self._build_tridiagonal_implicit(params)
        rhs = self._build_rhs_imex(V_n, params, tau)

        V_next[1:-1] = solve_tridiagonal(lower, diag, upper, rhs)
        V_next = self._apply_boundary_conditions(V_next, params, tau)

        return V_next

    def _apply_boundary_conditions(self, V: np.ndarray, params: Dict[str, float], tau: float) -> np.ndarray:
        """Apply left and right boundary conditions to a solution row."""
        bc = params.get("boundary_conditions")
        if bc is None:
            raise ValueError("Missing 'boundary_conditions' in params.")

        left_bc = bc.get("left")
        right_bc = bc.get("right")
        if left_bc is None or right_bc is None:
            raise ValueError("params['boundary_conditions'] must contain both 'left' and 'right'.")

        V[0] = left_bc(self.S[0], tau, params)
        V[-1] = right_bc(self.S[-1], tau, params)

        return V

    def _build_tridiagonal_implicit(self, params: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Build the tridiagonal matrix for the implicit local operator.

        The implicit part is the local Merton operator:
            0.5 * sigma^2 * S^2 * V_SS
            + (r - q - lambda * kappa) * S * V_S
            - (r + lambda) * V
        """
        r = params["r"]
        sigma = params["sigma"]
        q = params.get("q", 0.0)
        lambda_jump = params["lambda_jump"]
        kappa = self._jump_compensator(params)

        drift = r - q - lambda_jump * kappa

        N = self.N_S - 2
        lower = np.zeros(N - 1)
        diag = np.zeros(N)
        upper = np.zeros(N - 1)

        for idx in range(N):
            i = idx + 1
            S_i = self.S[i]

            alpha = 0.5 * sigma**2 * S_i**2 / (self.dS**2)
            beta = drift * S_i / (2.0 * self.dS)

            a_i = self.dt * (alpha - beta)
            b_i = self.dt * (2.0 * alpha + r + lambda_jump)
            c_i = self.dt * (alpha + beta)

            if idx > 0:
                lower[idx - 1] = -a_i

            diag[idx] = 1.0 + b_i

            if idx < N - 1:
                upper[idx] = -c_i

        return lower, diag, upper

    def _build_rhs_imex(self, V_n: np.ndarray, params: Dict[str, float], tau: float) -> np.ndarray:
        """Build the RHS for the IMEX Euler update."""
        lambda_jump = params["lambda_jump"]

        N = self.N_S - 2
        rhs = np.zeros(N)

        V_bc = np.zeros_like(V_n)
        V_bc = self._apply_boundary_conditions(V_bc, params, tau)

        jump_integral = self._compute_jump_integral(V_n, params)

        r = params["r"]
        sigma = params["sigma"]
        q = params.get("q", 0.0)
        kappa = self._jump_compensator(params)
        drift = r - q - lambda_jump * kappa

        for idx in range(N):
            i = idx + 1
            S_i = self.S[i]

            alpha = 0.5 * sigma**2 * S_i**2 / (self.dS**2)
            beta = drift * S_i / (2.0 * self.dS)

            a_i = self.dt * (alpha - beta)
            c_i = self.dt * (alpha + beta)

            rhs[idx] = V_n[i] + self.dt * lambda_jump * jump_integral[i]

            if i == 1:
                rhs[idx] += a_i * V_bc[0]

            if i == self.N_S - 2:
                rhs[idx] += c_i * V_bc[-1]

        return rhs

    def _compute_jump_integral(self, V_n: np.ndarray, params: Dict[str, float]) -> np.ndarray:
        """
        Approximate I[V](S) = integral V(S * exp(z)) f_Z(z) dz
        for Merton lognormal jumps with Z ~ N(jump_mean, jump_std^2).
        """
        z_nodes, weights = self._build_jump_quadrature(params)

        jump_integral = np.zeros_like(V_n)
        left_value = V_n[0]
        right_value = V_n[-1]

        for i, S_i in enumerate(self.S):
            shifted_spots = S_i * np.exp(z_nodes)
            sampled_values = np.interp(
                shifted_spots,
                self.S,
                V_n,
                left=left_value,
                right=right_value,
            )
            jump_integral[i] = np.sum(weights * sampled_values)

        return jump_integral

    def _build_jump_quadrature(self, params: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray]:
        """Build a truncated trapezoidal rule for the log-jump density."""
        mu_j = params["jump_mean"]
        sigma_j = params["jump_std"]
        quadrature_points = int(params.get("quadrature_points", 81))
        jump_std_width = float(params.get("jump_std_width", 5.0))

        if quadrature_points < 3:
            raise ValueError("quadrature_points must be at least 3.")
        if sigma_j <= 0.0:
            raise ValueError("jump_std must be positive.")

        z_min = mu_j - jump_std_width * sigma_j
        z_max = mu_j + jump_std_width * sigma_j
        z_nodes = np.linspace(z_min, z_max, quadrature_points)
        dz = z_nodes[1] - z_nodes[0]

        density = (
            np.exp(-0.5 * ((z_nodes - mu_j) / sigma_j) ** 2)
            / (sigma_j * np.sqrt(2.0 * np.pi))
        )

        weights = np.full(quadrature_points, dz)
        weights[0] *= 0.5
        weights[-1] *= 0.5
        weights *= density

        mass = np.sum(weights)
        if mass <= 0.0:
            raise ValueError("Jump quadrature produced non-positive total mass.")

        weights /= mass
        return z_nodes, weights

    def _jump_compensator(self, params: Dict[str, float]) -> float:
        """Return kappa = E[e^Z - 1] for Z ~ N(jump_mean, jump_std^2)."""
        mu_j = params["jump_mean"]
        sigma_j = params["jump_std"]
        return np.exp(mu_j + 0.5 * sigma_j**2) - 1.0

    def _validate_jump_params(self, params: Dict[str, float]) -> None:
        """Check required jump-diffusion parameters before solve."""
        required = {"r", "sigma", "lambda_jump", "jump_mean", "jump_std", "boundary_conditions"}
        missing = required.difference(params)
        if missing:
            raise ValueError(f"Missing required params: {sorted(missing)}")

    def _check_grid_initialized(self) -> None:
        """Ensure setup_grid has been called before solve."""
        if self.S is None or self.t is None:
            raise ValueError("Grid not initialized. Call setup_grid(...) first.")
