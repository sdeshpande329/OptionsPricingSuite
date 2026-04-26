import argparse
from pathlib import Path
import sys
from typing import Optional

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.monte_carlo.mc_black_scholes import BlackScholesMonteCarlo
from src.monte_carlo.mc_heston import HestonMonteCarlo
from src.monte_carlo.mc_merton import MertonMonteCarlo

DATA_PATH = REPO_ROOT / "data" / "options_metrics_processed" / "clean_options_data.csv"
RESULTS_DIR = REPO_ROOT / "data" / "results"
HESTON_PARAMS_PATH = RESULTS_DIR / "heston_calibrated_parameters.csv"
MERTON_PARAMS_PATH = RESULTS_DIR / "merton_jump_calibration_results.csv"

# retrieving the option type and change to format used in Monte Carlo code:
def infer_option_type(cp_flag: str) -> str:
    """Convert OptionMetrics cp_flag to 'call' or 'put'."""
    cp = str(cp_flag).strip().upper()
    if cp == "C":
        return "call"
    if cp == "P":
        return "put"
    raise ValueError(f"Unrecognized cp_flag: {cp_flag}")

# loading the heston parameters determined after calibration:
def load_heston_params() -> dict:
    """Load calibrated Heston parameters from the saved results file."""
    if not HESTON_PARAMS_PATH.exists():
        raise FileNotFoundError(f"Could not find Heston calibrated parameters file: {HESTON_PARAMS_PATH}")

    params_df = pd.read_csv(HESTON_PARAMS_PATH)
    first_row = params_df.iloc[0]

    return {
        "kappa": float(first_row["kappa"]),
        "theta": float(first_row["theta"]),
        "xi": float(first_row["xi"]),
        "rho": float(first_row["rho"]),
        "v0": float(first_row["v0"]),
    }

# loading the merton parameters determined after calibration:
def load_merton_params() -> dict:
    """Load calibrated Merton jump parameters from the saved results file."""
    if not MERTON_PARAMS_PATH.exists():
        raise FileNotFoundError(f"Could not find Merton calibration results file: {MERTON_PARAMS_PATH}")

    params_df = pd.read_csv(MERTON_PARAMS_PATH)
    first_row = params_df.iloc[0]

    return {
        "lambda_jump": float(first_row["lambda_jump"]),
        "jump_mean": float(first_row["jump_mean"]),
        "jump_std": float(first_row["jump_std"]),
    }

# Monte Carlo using Black-Scholes:
def price_row_black_scholes(row: pd.Series, n_paths: Optional[int] = None) -> dict:
    """Price one contract row with Black-Scholes Monte Carlo."""
    spot = float(row["spot_price"])
    strike = float(row["strike_price"])
    tau = float(row["tau (time to maturity)"])
    market_price = float(row["mid_price"])
    option_type = infer_option_type(row["cp_flag"])
    r = float(row["rate"]) / 100.0
    sigma = float(row["impl_volatility"])

    model = BlackScholesMonteCarlo(r=r, q=0.0, sigma=sigma)

    price, std_error, diagnostics = model.price_european_option(
        S0=spot, 
        K=strike, 
        T=tau, 
        option_type=option_type, 
        n_paths=n_paths)

    ci_lower, ci_upper = diagnostics["ci_95"]
    abs_error = abs(price - market_price)
    sq_error = (price - market_price) ** 2

    return {
        "date": row["date"],
        "exdate": row["exdate"],
        "cp_flag": row["cp_flag"],
        "spot_price": spot,
        "strike_price": strike,
        "tau (time to maturity)": tau,
        "market_mid_price": market_price,
        "model_price": price,
        "std_error": std_error,
        "ci_95_lower": ci_lower,
        "ci_95_upper": ci_upper,
        "n_paths": diagnostics["n_paths"],
        "n_steps": diagnostics["n_steps"],
        "antithetic_used": diagnostics["antithetic_used"],
        "abs_error": abs_error,
        "sq_error": sq_error,
        "model_name": "black_scholes",
    }

# Monte Carlo using Heston:
def price_row_heston(row: pd.Series, heston_params: dict, n_paths: Optional[int] = None) -> dict:
    """Price one contract row with Heston Monte Carlo."""
    spot = float(row["spot_price"])
    strike = float(row["strike_price"])
    tau = float(row["tau (time to maturity)"])
    market_price = float(row["mid_price"])
    option_type = infer_option_type(row["cp_flag"])
    r = float(row["rate"]) / 100.0

    model = HestonMonteCarlo(
        r=r,
        q=0.0,
        kappa=heston_params["kappa"],
        theta=heston_params["theta"],
        xi=heston_params["xi"],
        rho=heston_params["rho"],
        v0=heston_params["v0"],
    )

    price, std_error, diagnostics = model.price_european_option(
        S0=spot,
        K=strike,
        T=tau,
        option_type=option_type,
        n_paths=n_paths,
    )

    ci_lower, ci_upper = diagnostics["ci_95"]
    abs_error = abs(price - market_price)
    sq_error = (price - market_price) ** 2

    return {
        "date": row["date"],
        "exdate": row["exdate"],
        "cp_flag": row["cp_flag"],
        "spot_price": spot,
        "strike_price": strike,
        "tau (time to maturity)": tau,
        "market_mid_price": market_price,
        "model_price": price,
        "std_error": std_error,
        "ci_95_lower": ci_lower,
        "ci_95_upper": ci_upper,
        "n_paths": diagnostics["n_paths"],
        "n_steps": diagnostics["n_steps"],
        "antithetic_used": diagnostics["antithetic_used"],
        "abs_error": abs_error,
        "sq_error": sq_error,
        "model_name": "heston",
    }

def price_row_merton(row: pd.Series, merton_params: dict, n_paths: Optional[int] = None) -> dict:
    """Price one contract row with Merton Monte Carlo."""
    spot = float(row["spot_price"])
    strike = float(row["strike_price"])
    tau = float(row["tau (time to maturity)"])
    market_price = float(row["mid_price"])
    option_type = infer_option_type(row["cp_flag"])
    r = float(row["rate"]) / 100.0
    sigma = float(row["impl_volatility"])

    model = MertonMonteCarlo(
        r=r,
        q=0.0,
        sigma=sigma,
        lambda_jump=merton_params["lambda_jump"],
        jump_mean=merton_params["jump_mean"],
        jump_std=merton_params["jump_std"],
    )

    price, std_error, diagnostics = model.price_european_option(
        S0=spot,
        K=strike,
        T=tau,
        option_type=option_type,
        n_paths=n_paths,
    )

    ci_lower, ci_upper = diagnostics["ci_95"]
    abs_error = abs(price - market_price)
    sq_error = (price - market_price) ** 2

    return {
        "date": row["date"],
        "exdate": row["exdate"],
        "cp_flag": row["cp_flag"],
        "spot_price": spot,
        "strike_price": strike,
        "tau (time to maturity)": tau,
        "market_mid_price": market_price,
        "model_price": price,
        "std_error": std_error,
        "ci_95_lower": ci_lower,
        "ci_95_upper": ci_upper,
        "n_paths": diagnostics["n_paths"],
        "n_steps": diagnostics["n_steps"],
        "antithetic_used": diagnostics["antithetic_used"],
        "abs_error": abs_error,
        "sq_error": sq_error,
        "model_name": "merton",
    }


def main(model_name: str = "black_scholes", max_rows: Optional[int] = 25, n_paths: Optional[int] = None) -> None:
    """Run Monte Carlo pricing on cleaned OptionMetrics data and save results."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Could not find processed data file: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    if max_rows is not None:
        df = df.head(max_rows).copy()

    results = []

    if model_name == "heston":
        heston_params = load_heston_params()
    else:
        heston_params = None

    if model_name == "merton":
        merton_params = load_merton_params()
    else:
        merton_params = None

    for _, row in df.iterrows():
        try:
            if model_name == "black_scholes":
                priced = price_row_black_scholes(row=row, n_paths=n_paths)
            elif model_name == "heston":
                priced = price_row_heston(row=row, heston_params=heston_params, n_paths=n_paths)
            elif model_name == "merton":
                priced = price_row_merton(row=row, merton_params=merton_params, n_paths=n_paths)
            else:
                raise ValueError(
                    "model_name must be one of {'black_scholes', 'heston', 'merton'}"
                )

            results.append(priced)

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
                    "model_name": model_name,
                    "error_message": str(exc),
                }
            )

    results_df = pd.DataFrame(results)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = RESULTS_DIR / f"{model_name}_monte_carlo_results.csv"
    results_df.to_csv(results_path, index=False)

    successful = results_df["model_price"].notna().sum() if "model_price" in results_df.columns else 0

    print(f"Saved results to: {results_path}")
    print(f"Rows processed: {len(results_df)}")
    print(f"Successful prices: {successful}")

    if successful > 0:
        print(f"Mean absolute error: {results_df['abs_error'].dropna().mean():.6f}")
        print(f"Mean squared error: {results_df['sq_error'].dropna().mean():.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Monte Carlo pricing for a selected model.")
    parser.add_argument(
        "--model",
        type=str,
        default="black_scholes",
        choices=["black_scholes", "heston", "merton"],
        help="Model to run: black_scholes, heston, or merton",
    )
    parser.add_argument(
        "--max_rows",
        type=int,
        default=25,
        help="Maximum number of rows to process (use -1 for all rows)",
    )
    parser.add_argument(
        "--n_paths",
        type=int,
        default=None,
        help="Optional override for number of Monte Carlo simulation paths",
    )

    args = parser.parse_args()

    max_rows = None if args.max_rows == -1 else args.max_rows

    print(f"\nRunning Monte Carlo for model: {args.model}")
    main(model_name=args.model, max_rows=max_rows, n_paths=args.n_paths)
