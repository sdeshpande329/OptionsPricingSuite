import numpy as np

def solve_tridiagonal(lower: np.ndarray, diag: np.ndarray,
                      upper: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Solves tridiagonal system using Thomas algorithm, which is a specialized Gaussian elimination for tridiagonal systems.
    
    Algorithm:
        1. Forward elimination: Eliminate lower diagonal
        2. Back substitution: Solve for x from bottom to top
    """
    n = len(diag)
    
    if len(lower) != n - 1:
        raise ValueError(f"lower must have length {n-1}, got {len(lower)}")
    if len(upper) != n - 1:
        raise ValueError(f"upper must have length {n-1}, got {len(upper)}")
    if len(rhs) != n:
        raise ValueError(f"rhs must have length {n}, got {len(rhs)}")
    
    # Create copies to avoid modifying inputs
    c = np.zeros(n - 1)  # Modified upper diagonal
    d = np.zeros(n)      # Modified rhs
    
    # Forward elimination
    c[0] = upper[0] / diag[0]
    d[0] = rhs[0] / diag[0]
    
    for i in range(1, n - 1):
        denominator = diag[i] - lower[i-1] * c[i-1]
        c[i] = upper[i] / denominator
        d[i] = (rhs[i] - lower[i-1] * d[i-1]) / denominator
    
    # Last row (no upper diagonal element)
    d[n-1] = (rhs[n-1] - lower[n-2] * d[n-2]) / (diag[n-1] - lower[n-2] * c[n-2])
    
    # Back substitution
    x = np.zeros(n)
    x[n-1] = d[n-1]
    
    for i in range(n - 2, -1, -1):
        x[i] = d[i] - c[i] * x[i+1]
    
    return x