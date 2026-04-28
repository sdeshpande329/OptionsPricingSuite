"""
Run Greeks computation and validation for all three models.

Compares computed Greeks against market-observed values from OptionMetrics.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Optional

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.black_scholes import BlackScholesModel, BlackScholesParams, EuropeanOption
from src.models.heston import HestonModel
from src.models.merton_jump_diffusion import MertonJumpDiffusionModel, MertonJumpParams

DATA_PATH = REPO_ROOT / "data" / "options_metrics_processed" / "clean_options_data.csv"
RESULTS_DIR = REPO_ROOT / "data" / "results"
HESTON_PARAMS_PATH = RESULTS_DIR / "heston_calibrated_parameters.csv"
MERTON_PARAMS_PATH = RESULTS_DIR / "merton_jump_calibration_results.csv"


def infer_option_type(cp_flag: str) -> str:
    """Convert OptionMetrics cp_flag to 'call' or 'put'."""
    cp = str(cp_flag).strip().upper()
    if cp == "C":
        return "call"
    if cp == "P":
        return "put"
    raise ValueError(f"Unrecognized cp_flag: {cp_flag}")


def load_heston_params() -> dict:
    """Load calibrated Heston parameters."""
    if not HESTON_PARAMS_PATH.exists():
        raise FileNotFoundError(f"Could not find Heston parameters: {HESTON_PARAMS_PATH}")
    params_df = pd.read_csv(HESTON_PARAMS_PATH)
    first_row = params_df.iloc[0]
    return {
        "kappa": float(first_row["kappa"]),
        "theta": float(first_row["theta"]),
        "xi": float(first_row["xi"]),
        "rho": float(first_row["rho"]),
        "v0": float(first_row["v0"]),
    }


def load_merton_params() -> dict:
    """Load calibrated Merton jump parameters."""
    if not MERTON_PARAMS_PATH.exists():
        raise FileNotFoundError(f"Could not find Merton parameters: {MERTON_PARAMS_PATH}")
    params_df = pd.read_csv(MERTON_PARAMS_PATH)
    first_row = params_df.iloc[0]
    return {
        "lambda_jump": float(first_row["lambda_jump"]),
        "jump_mean": float(first_row["jump_mean"]),
        "jump_std": float(first_row["jump_std"]),
    }


def compute_black_scholes_greeks(row: pd.Series, n_s: int = 100, n_t: int = 100) -> dict:
    """Compute Greeks using Black-Scholes PDE."""
    spot = float(row["spot_price"])
    strike = float(row["strike_price"])
    tau = float(row["tau (time to maturity)"])
    r = float(row["rate"]) / 100.0
    sigma = float(row["impl_volatility"])
    option_type = infer_option_type(row["cp_flag"])

    params = BlackScholesParams(r=r, sigma=sigma, q=0.0)
    model = BlackScholesModel(params)
    contract = EuropeanOption(K=strike, T=tau, option_type=option_type)

    greeks = model.compute_greeks(
        contract=contract,
        spot=spot,
        scheme="crank_nicolson",
        N_S=n_s,
        N_t=n_t
    )
    return greeks


def compute_heston_greeks(row: pd.Series, heston_params: dict, n_s: int = 80, n_v: int = 40, n_t: int = 40) -> dict:
    """Compute Greeks using Heston PDE."""
    spot = float(row["spot_price"])
    strike = float(row["strike_price"])
    tau = float(row["tau (time to maturity)"])
    r = float(row["rate"]) / 100.0
    option_type = infer_option_type(row["cp_flag"])

    model = HestonModel(
        r=r,
        q=0.0,
        kappa=heston_params["kappa"],
        theta=heston_params["theta"],
        xi=heston_params["xi"],
        rho=heston_params["rho"],
        v0=heston_params["v0"]
    )

    greeks = model.compute_greeks(
        S0=spot,
        K=strike,
        T=tau,
        option_type=option_type,
        scheme="craig_sneyd",
        N_S=n_s,
        N_v=n_v,
        N_t=n_t
    )
    return greeks


def compute_merton_greeks(row: pd.Series, merton_params: dict, n_s: int = 100, n_t: int = 100) -> dict:
    """Compute Greeks using Merton Jump-Diffusion PIDE."""
    spot = float(row["spot_price"])
    strike = float(row["strike_price"])
    tau = float(row["tau (time to maturity)"])
    r = float(row["rate"]) / 100.0
    sigma = float(row["impl_volatility"])
    option_type = infer_option_type(row["cp_flag"])

    params = MertonJumpParams(
        r=r,
        sigma=sigma,
        q=0.0,
        lambda_jump=merton_params["lambda_jump"],
        jump_mean=merton_params["jump_mean"],
        jump_std=merton_params["jump_std"]
    )
    model = MertonJumpDiffusionModel(params)
    contract = EuropeanOption(K=strike, T=tau, option_type=option_type)

    greeks = model.compute_greeks(
        contract=contract,
        spot=spot,
        scheme="imex_euler",
        N_S=n_s,
        N_t=n_t
    )
    return greeks


def main(max_rows: Optional[int] = None) -> None:
    """Run Greeks computation and validation."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Could not find data file: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    if max_rows is not None:
        df = df.head(max_rows).copy()


    # Load calibrated parameters
    heston_params = load_heston_params()
    merton_params = load_merton_params()

    print("Heston parameters:", heston_params)
    print("Merton parameters:", merton_params)
    print()

    results = []

    for idx, row in df.iterrows():

        spot = float(row["spot_price"])
        strike = float(row["strike_price"])
        tau = float(row["tau (time to maturity)"])
        option_type = infer_option_type(row["cp_flag"])

        # Market Greeks
        market_delta = float(row["delta"])
        market_gamma = float(row["gamma"])
        market_theta = float(row["theta"])
        market_vega = float(row["vega"])

        # Compute model Greeks
        try:
            bs_greeks = compute_black_scholes_greeks(row)
        except Exception as e:
            print(f"  BS error: {e}")
            bs_greeks = {"delta": np.nan, "gamma": np.nan, "vega": np.nan, "theta": np.nan}

        try:
            heston_greeks = compute_heston_greeks(row, heston_params)
        except Exception as e:
            print(f"  Heston error: {e}")
            heston_greeks = {"delta": np.nan, "gamma": np.nan, "vega": np.nan, "theta": np.nan}

        try:
            merton_greeks = compute_merton_greeks(row, merton_params)
        except Exception as e:
            print(f"  Merton error: {e}")
            merton_greeks = {"delta": np.nan, "gamma": np.nan, "vega": np.nan, "theta": np.nan}

        results.append({
            "date": row["date"],
            "strike_price": strike,
            "tau": tau,
            "option_type": option_type,
            "spot": spot,
            # Market
            "market_delta": market_delta,
            "market_gamma": market_gamma,
            "market_theta": market_theta,
            "market_vega": market_vega,
            # Black-Scholes
            "bs_delta": bs_greeks["delta"],
            "bs_gamma": bs_greeks["gamma"],
            "bs_vega": bs_greeks["vega"],
            "bs_theta": bs_greeks["theta"],
            # Heston
            "heston_delta": heston_greeks["delta"],
            "heston_gamma": heston_greeks["gamma"],
            "heston_vega": heston_greeks["vega"],
            "heston_theta": heston_greeks["theta"],
            # Merton
            "merton_delta": merton_greeks["delta"],
            "merton_gamma": merton_greeks["gamma"],
            "merton_vega": merton_greeks["vega"],
            "merton_theta": merton_greeks["theta"],
        })

    # Save results
    results_df = pd.DataFrame(results)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = RESULTS_DIR / "greeks_comparison_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved to: {results_path}")

    # Compute error statistics
    print("Greeks Comparison Statistics:")
    print("-" * 60)

    summary_data = []
    for model_name, prefix in [("Black-Scholes", "bs_"), ("Heston", "heston_"), ("Merton", "merton_")]:
        print(f"\n{model_name}:")
        for greek in ["delta", "gamma", "vega", "theta"]:
            model_col = f"{prefix}{greek}"
            market_col = f"market_{greek}"
            
            valid = results_df[model_col].notna() & results_df[market_col].notna()
            if valid.sum() > 0:
                errors = results_df.loc[valid, model_col] - results_df.loc[valid, market_col]
                mae = errors.abs().mean()
                rmse = np.sqrt((errors ** 2).mean())
                print(f"  {greek.capitalize():8s}: MAE = {mae:.6f}, RMSE = {rmse:.6f}")
                summary_data.append({
                    "model": model_name,
                    "greek": greek,
                    "mae": mae,
                    "rmse": rmse,
                    "n_samples": valid.sum()
                })

    # Save summary statistics
    summary_df = pd.DataFrame(summary_data)
    summary_path = RESULTS_DIR / "greeks_summary_statistics.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary statistics saved to: {summary_path}")


if __name__ == "__main__":
    main()