from __future__ import annotations

from pathlib import Path
import sys
from typing import Optional

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.heston import HestonModel

DATA_PATH = REPO_ROOT / "data" / "options_metrics_processed" / "clean_options_data.csv"
RESULTS_DIR = REPO_ROOT / "data" / "results"


def infer_option_type(cp_flag: str) -> str:
    """Convert OptionMetrics cp_flag to 'call' or 'put'."""
    cp = str(cp_flag).strip().upper()
    if cp == "C":
        return "call"
    if cp == "P":
        return "put"
    raise ValueError(f"Unrecognized cp_flag: {cp_flag}")


def choose_grid_bounds(
    spot: float,
    strike: float,
    v0: float,
    theta: float,
    n_std_S: float = 4.0,
    n_std_v: float = 3.0
) -> tuple[float, float, float, float]:
    """
    Choose grid bounds for Heston model.
    
    Parameters:

    spot : float
        Current stock price
    strike : float
        Strike price
    v0 : float
        Initial variance
    theta : float
        Long-run variance
    n_std_S : float
        Number of std deviations for S domain
    n_std_v : float
        Number of std deviations for v domain
    
    Returns:
    
    S_min, S_max, v_min, v_max : float
    """
    # S domain: use spot and strike as reference
    S_min = max(1e-8, 0.5 * min(spot, strike))
    S_max = max(2.0 * max(spot, strike), S_min + 1.0)
    
    # v domain: based on initial and long-run variance
    v_min = 0.0
    v_max = max(n_std_v * max(v0, theta), 0.5)
    
    return S_min, S_max, v_min, v_max


def price_row(
    row: pd.Series,
    heston_params: dict,
    scheme: str,
    n_s: int,
    n_v: int,
    n_t: int
) -> dict:
    """
    Price one option contract using Heston model.
    
    Parameters:
    -----------
    row : pd.Series
        Row from OptionMetrics data
    heston_params : dict
        Heston model parameters {kappa, theta, xi, rho, v0}
    scheme : str
        ADI scheme: 'douglas', 'craig-sneyd', 'mcs', 'hv'
    n_s, n_v, n_t : int
        Grid dimensions
    
    Returns:
    --------
    dict
        Pricing results and errors
    """
    # Extract contract parameters
    spot = float(row["spot_price"])
    strike = float(row["strike_price"])
    tau = float(row["tau (time to maturity)"])
    market_price = float(row["mid_price"])
    option_type = infer_option_type(row["cp_flag"])
    r = float(row["rate"]) / 100.0
    
    # Create Heston model
    model = HestonModel(
        r=r,
        q=0.0,  # Assume no dividends
        kappa=heston_params['kappa'],
        theta=heston_params['theta'],
        xi=heston_params['xi'],
        rho=heston_params['rho'],
        v0=heston_params['v0']
    )
    
    # Choose grid bounds
    S_min, S_max, v_min, v_max = choose_grid_bounds(
        spot=spot,
        strike=strike,
        v0=heston_params['v0'],
        theta=heston_params['theta']
    )
    
    # Price option
    model_price = model.price_european_option(
        S0=spot,
        K=strike,
        T=tau,
        option_type=option_type,
        scheme=scheme,
        N_S=n_s,
        N_v=n_v,
        N_t=n_t,
        S_max=S_max,
        v_max=v_max
    )
    
    # Compute errors
    abs_error = abs(model_price - market_price)
    sq_error = (model_price - market_price) ** 2
    rel_error = abs_error / market_price if market_price > 0 else np.nan
    
    return {
        "date": row["date"],
        "exdate": row["exdate"],
        "cp_flag": row["cp_flag"],
        "spot_price": spot,
        "strike_price": strike,
        "tau (time to maturity)": tau,
        "market_mid_price": market_price,
        "model_price": model_price,
        "abs_error": abs_error,
        "sq_error": sq_error,
        "rel_error": rel_error,
        "scheme": scheme,
        "N_S": n_s,
        "N_v": n_v,
        "N_t": n_t,
    }


def main(
    scheme: str = "craig-sneyd",
    n_s: int = 100,
    n_v: int = 50,
    n_t: int = 100,
    max_rows: Optional[int] = 25,
    heston_params: Optional[dict] = None
) -> None:
    """
    Run Heston pricing on OptionMetrics data and save results.
    
    Parameters:
    
    scheme : str
        ADI scheme to use
    n_s, n_v, n_t : int
        Grid dimensions
    max_rows : int, optional
        Maximum number of rows to process (None for all)
    heston_params : dict, optional
        Heston parameters (use defaults if None)
    """
    # Check data exists
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Could not find processed data file: {DATA_PATH}")
    
    # Load data
    df = pd.read_csv(DATA_PATH)
    
    if max_rows is not None:
        df = df.head(max_rows).copy()
    
    # Default Heston parameters (typical values from literature)
    if heston_params is None:
        heston_params = {
            'kappa': 2.0,    # Mean reversion speed
            'theta': 0.04,   # Long-run variance (20% vol)
            'xi': 0.3,       # Vol of vol
            'rho': -0.7,     # Correlation (leverage effect)
            'v0': 0.04       # Initial variance (20% vol)
        }
    
    print(f"Using Heston parameters:")
    print(f"  kappa = {heston_params['kappa']:.4f}")
    print(f"  theta = {heston_params['theta']:.4f}")
    print(f"  xi    = {heston_params['xi']:.4f}")
    print(f"  rho   = {heston_params['rho']:.4f}")
    print(f"  v0    = {heston_params['v0']:.4f}")
    print()
    
    results = []
    
    # Price each option
    for idx, row in df.iterrows():
        priced = price_row(
            row=row,
            heston_params=heston_params,
            scheme=scheme,
            n_s=n_s,
            n_v=n_v,
            n_t=n_t
        )
        results.append(priced)
        # try:
            # priced = price_row(
            #     row=row,
            #     heston_params=heston_params,
            #     scheme=scheme,
            #     n_s=n_s,
            #     n_v=n_v,
            #     n_t=n_t
            # )
            # results.append(priced)
                
        # except Exception as exc:
        #     # Log error but continue
        #     results.append({
        #         "date": row["date"],
        #         "exdate": row["exdate"],
        #         "cp_flag": row["cp_flag"],
        #         "spot_price": row["spot_price"],
        #         "strike_price": row["strike_price"],
        #         "tau (time to maturity)": row["tau (time to maturity)"],
        #         "market_mid_price": row["mid_price"],
        #         "scheme": scheme,
        #         "N_S": n_s,
        #         "N_v": n_v,
        #         "N_t": n_t,
        #         "error_message": str(exc),
        #     })
        #     print(f"Error on row {idx}: {exc}")
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = RESULTS_DIR / f"heston_pricing_results_{scheme}.csv"
    results_df.to_csv(results_path, index=False)
    
    # Compute statistics
    successful = results_df["model_price"].notna().sum() if "model_price" in results_df.columns else 0

    
    if successful > 0:
        mae = results_df['abs_error'].dropna().mean()
        mse = results_df['sq_error'].dropna().mean()
        rmse = np.sqrt(mse)
        mape = results_df['rel_error'].dropna().mean() * 100
        
        print(f"\nPricing Error Statistics:")
        print(f"  Mean Absolute Error (MAE):  ${mae:.6f}")
        print(f"  Root Mean Squared Error (RMSE): ${rmse:.6f}")
        print(f"  Mean Absolute Percentage Error (MAPE): {mape:.2f}%")


if __name__ == "__main__":
    for scheme in ["douglas", "craig_sneyd", "modified_craig_sneyd", "hundsdorfer_verwer"]:
        print(f"Running scheme: {scheme}")
        
        main(
            scheme=scheme,
            n_s=80,      # Coarser grid for speed
            n_v=40,
            n_t=40,
            max_rows=25  # Limit for testing
        )