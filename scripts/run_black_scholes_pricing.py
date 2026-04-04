from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.models.black_scholes import BlackScholesModel, BlackScholesParams, EuropeanOption

REPO_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = REPO_ROOT / "data" / "options_metrics_processed" / "clean_options_data.csv"
RESULTS_DIR = REPO_ROOT / "data" / "results"

# retrieving the option type and change to format used in Black-Scholes PDE code:
def infer_option_type(cp_flag: str) -> str:
    """Convert OptionMetrics cp_flag to 'call' or 'put'."""
    cp = str(cp_flag).strip().upper()
    if cp == "C":
        return "call"
    if cp == "P":
        return "put"
    raise ValueError(f"Unrecognized cp_flag: {cp_flag}")

# setting the min and max of the stock price for the grid boundaries:
def choose_grid_bounds(spot: float, strike: float, r: float, q: float, sigma: float, tau: float,
    n_std: float = 4.0) -> tuple[float, float]:
    """Choose a stock-price domain using a lognormal heuristic (+/- n_std standard deviations)."""
    # drift scaled by tau (time-to-maturity):
    drift = (r - q - 0.5 * sigma**2) * tau

    # using standard deviation (volatility) and tau (time-to-maturity) to create a confident spread:
    spread = n_std * sigma * np.sqrt(tau)

    # applying the lognormal heuristic to determine confident S_min and S_max:
    s_min = spot * np.exp(drift - spread)
    s_max = spot * np.exp(drift + spread)

    # keep bounds numerically safe and valid:
    s_min = max(1e-8, s_min)
    s_max = max(s_max, 2.0 * strike, s_min + 1.0)

    return s_min, s_max

def price_row(row: pd.Series, scheme: str, theta_cn: float, n_s: int, n_t: int, rate_col: str) -> dict:
    """Price one contract (data row) with the Black-Scholes model wrapper."""
    # needed contract inputs/parameters:
    spot = float(row["spot_price"])
    strike = float(row["strike_price"])
    tau = float(row["tau (time to maturity)"])
    market_price = float(row["mid_price"])
    option_type = infer_option_type(row["cp_flag"])
    r = float(row["rate"]) / 100.0
    sigma = float(row["impl_volatility"])

    # building model parameters, option contract, and model wrapper:
    bs_params = BlackScholesParams(r=r, sigma=sigma, q=0.0)
    contract = EuropeanOption(K=strike, T=tau, option_type=option_type)
    model = BlackScholesModel(bs_params)

    # PDE stock-price domain:
    s_min, s_max = choose_grid_bounds(spot=spot, strike=strike, r=r, q=0.0, sigma=sigma, tau=tau, n_std=4.0)

    # run the Black-Scholes numerical pricing:
    model_price = model.price(
        contract=contract,
        spot=spot,
        scheme=scheme,
        S_min=s_min,
        S_max=s_max,
        N_S=n_s,
        N_t=n_t,
        theta_cn=theta_cn,
    )

    # computing the pricing errors:
    abs_error = abs(model_price - market_price)
    sq_error = (model_price - market_price) ** 2

    # returning a dictionary with the results of the numerical pricing and actual price comparison:
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
    }

def check_common_grid_explicit_stability(spot: float, strike: float, r: float, q: float, sigma: float, tau: float, n_s: int,
    n_t: int, n_std: float = 4.0) -> None:
    """Check chosen grid is stable for explicit scheme. All schemes are evaluated under the same parameters for consistency."""
    s_min, s_max = choose_grid_bounds(spot=spot, strike=strike, r=r, q=q, sigma=sigma, tau=tau, n_std=n_std)

    dS = (s_max - s_min) / (n_s - 1)
    dt = tau / (n_t - 1)

    # stability formula (derived from von Neumann and Eurler's formula):
    stability_value = dt * sigma**2 * s_max**2 / (dS**2)

    if stability_value > 1.0:
        raise ValueError(
            "Chosen common grid is not stable for the explicit scheme. "
            f"Computed stability value = {stability_value:.6f}, which exceeds 1.0. "
            "Choose a finer time grid (increase n_t) or a coarser stock grid (decrease n_s)."
        )

def main(scheme: str = "crank_nicolson", theta_cn: float = 0.5, n_s: int = 200, n_t: int = 2500,
    max_rows: Optional[int] = 25) -> None:
    """Run Black-Scholes pricing on cleaned OptionMetrics data and save results. """
    # checks that the cleaned data CSV exists before attempting to read:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Could not find processed data file: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    if max_rows is not None:
        df = df.head(max_rows).copy()

    # ensure the chosen common grid is also stable for explicit
    if len(df) > 0:
        first_row = df.iloc[0]

        spot = float(first_row["spot_price"])
        strike = float(first_row["strike_price"])
        tau = float(first_row["tau (time to maturity)"])
        r = float(first_row["rate"]) / 100.0
        sigma = float(first_row["impl_volatility"])

        check_common_grid_explicit_stability(spot=spot, strike=strike, r=r, q=0.0, sigma=sigma, tau=tau, n_s=n_s, n_t=n_t, n_std=4.0)

    results = []

    # looping through each row (option contract):
    for _, row in df.iterrows():
        try:
            priced = price_row(
                row=row,
                scheme=scheme,
                theta_cn=theta_cn,
                n_s=n_s,
                n_t=n_t,
            )
            results.append(priced)
        # if row (contract) fails then save error instead of crashing (numerical error still possible):
        except Exception as exc:
            results.append(
                {
                    "date": row["date"],
                    "exdate": row["exdate"],
                    "cp_flag": row["cp_flag"],
                    "spot_price": row["spot_price"],
                    "strike_price": row["strike_price"],
                    "tau (time to maturity)": row["tau (time to maturity)"],
                    "market_mid_price": row["mid_price"],
                    "scheme": scheme,
                    "theta_cn": theta_cn,
                    "N_S": n_s,
                    "N_t": n_t,
                    "error_message": str(exc),
                }
            )

    results_df = pd.DataFrame(results)

    # making sure the results directory exists and saves the CSV:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = RESULTS_DIR / f"black_scholes_pricing_results_{scheme}.csv"
    results_df.to_csv(results_path, index=False)

    # how many rows were priced successfully:
    successful = results_df["model_price"].notna().sum() if "model_price" in results_df.columns else 0

    # print basic summary:
    print(f"Saved results to: {results_path}")
    print(f"Rows processed: {len(results_df)}")
    print(f"Successful prices: {successful}")

    # MAE and MSE stats:
    if successful > 0:
        print(f"Mean absolute error: {results_df['abs_error'].dropna().mean():.6f}")
        print(f"Mean squared error: {results_df['sq_error'].dropna().mean():.6f}")


if __name__ == "__main__":
    for scheme in ["explicit", "implicit", "crank_nicolson"]:
        print(f"\nRunning scheme: {scheme}")
        main(scheme=scheme, max_rows=None)