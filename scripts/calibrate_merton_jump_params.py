from __future__ import annotations

from itertools import product
from pathlib import Path
import sys
from typing import Optional

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.merton_jump_diffusion import EuropeanOption, MertonJumpDiffusionModel, MertonJumpParams
from scripts.run_merton_pide_pricing import IMEX_SCHEME, choose_grid_bounds, infer_option_type

DATA_PATH = REPO_ROOT / "data" / "options_metrics_processed" / "clean_options_data.csv"
RESULTS_DIR = REPO_ROOT / "data" / "results"

DEFAULT_CP_FLAG = None
DEFAULT_MAX_ROWS = 60
DEFAULT_TAU_MIN = 0.0
DEFAULT_TAU_MAX = 1.0

LAMBDA_GRID = [0.0, 0.0001, 0.005, 0.01]
JUMP_MEAN_GRID = [-0.02, -0.01, -0.005, 0.0]
JUMP_STD_GRID = [0.01,0.02, 0.03, 0.05]


def load_calibration_data(
    max_rows: Optional[int] = DEFAULT_MAX_ROWS,
    cp_flag: Optional[str] = DEFAULT_CP_FLAG,
    tau_min: float = DEFAULT_TAU_MIN,
    tau_max: float = DEFAULT_TAU_MAX,
) -> pd.DataFrame:
    """Load and filter a tractable calibration sample."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Could not find processed data file: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    if cp_flag is not None:
        df = df[df["cp_flag"].astype(str).str.upper() == cp_flag.upper()].copy()

    df = df[
        (df["tau (time to maturity)"] >= tau_min)
        & (df["tau (time to maturity)"] <= tau_max)
    ].copy()

    df = df.sort_values(["date", "exdate", "strike_price"]).reset_index(drop=True)

    if max_rows is not None and len(df) > max_rows:
        if cp_flag is None:
            sampled_parts = []
            total_rows = len(df)
            for _, group in df.groupby(df["cp_flag"].astype(str).str.upper(), sort=True):
                share = len(group) / total_rows
                take = max(1, round(max_rows * share))
                sampled_parts.append(group.head(min(take, len(group))))

            df = pd.concat(sampled_parts, ignore_index=True)
            if len(df) > max_rows:
                df = df.head(max_rows).copy()
        else:
            df = df.head(max_rows).copy()

    if df.empty:
        raise ValueError("Calibration dataset is empty after filtering.")

    return df


def summarize_calibration_data(df: pd.DataFrame) -> dict:
    """Build a compact summary of the calibration sample."""
    cp_counts = df["cp_flag"].astype(str).str.upper().value_counts().to_dict()
    tau_series = df["tau (time to maturity)"]

    return {
        "rows": len(df),
        "call_count": int(cp_counts.get("C", 0)),
        "put_count": int(cp_counts.get("P", 0)),
        "tau_min": float(tau_series.min()),
        "tau_max": float(tau_series.max()),
    }


def price_contract_with_params(
    row: pd.Series,
    lambda_jump: float,
    jump_mean: float,
    jump_std: float,
    scheme: str = IMEX_SCHEME,
    n_s: int = 120,
    n_t: int = 400,
) -> float:
    """Price one contract under a candidate Merton parameter set."""
    spot = float(row["spot_price"])
    strike = float(row["strike_price"])
    tau = float(row["tau (time to maturity)"])
    option_type = infer_option_type(row["cp_flag"])
    r = float(row["rate"]) / 100.0
    sigma = float(row["impl_volatility"])

    params = MertonJumpParams(
        r=r,
        sigma=sigma,
        q=0.0,
        lambda_jump=lambda_jump,
        jump_mean=jump_mean,
        jump_std=jump_std,
    )
    contract = EuropeanOption(K=strike, T=tau, option_type=option_type)
    model = MertonJumpDiffusionModel(params)

    s_min, s_max = choose_grid_bounds(
        spot=spot,
        strike=strike,
        r=r,
        q=0.0,
        sigma=sigma,
        tau=tau,
        n_std=4.0,
    )

    return model.price(
        contract=contract,
        spot=spot,
        scheme=scheme,
        S_min=s_min,
        S_max=s_max,
        N_S=n_s,
        N_t=n_t,
    )


def evaluate_parameter_set(
    df: pd.DataFrame,
    lambda_jump: float,
    jump_mean: float,
    jump_std: float,
    scheme: str = IMEX_SCHEME,
    n_s: int = 120,
    n_t: int = 400,
) -> dict:
    """Compute summary error metrics for one candidate parameter set."""
    total_abs_error = 0.0
    total_sq_error = 0.0
    successful = 0
    failed = 0

    for _, row in df.iterrows():
        market_price = float(row["mid_price"])

        try:
            model_price = price_contract_with_params(
                row=row,
                lambda_jump=lambda_jump,
                jump_mean=jump_mean,
                jump_std=jump_std,
                scheme=scheme,
                n_s=n_s,
                n_t=n_t,
            )
            total_abs_error += abs(model_price - market_price)
            total_sq_error += (model_price - market_price) ** 2
            successful += 1
        except Exception:
            failed += 1

    mae = total_abs_error / successful if successful > 0 else float("inf")
    mse = total_sq_error / successful if successful > 0 else float("inf")
    rmse = mse ** 0.5 if mse != float("inf") else float("inf")

    return {
        "lambda_jump": lambda_jump,
        "jump_mean": jump_mean,
        "jump_std": jump_std,
        "scheme": scheme,
        "N_S": n_s,
        "N_t": n_t,
        "rows_used": successful,
        "rows_failed": failed,
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
    }


def run_grid_search(
    df: pd.DataFrame,
    scheme: str = IMEX_SCHEME,
    n_s: int = 120,
    n_t: int = 400,
) -> pd.DataFrame:
    """Evaluate every combination in the calibration grid."""
    results = []

    # For the no-jump baseline, lambda=0 shuts off the jump term entirely.
    # We still pass a positive jump_std because the IMEX solver validates it.
    parameter_sets = [(0.0, 0.0, JUMP_STD_GRID[0])]
    for lambda_jump, jump_mean, jump_std in product(LAMBDA_GRID, JUMP_MEAN_GRID, JUMP_STD_GRID):
        if lambda_jump == 0.0:
            continue
        parameter_sets.append((lambda_jump, jump_mean, jump_std))

    for lambda_jump, jump_mean, jump_std in parameter_sets:
        result = evaluate_parameter_set(
            df=df,
            lambda_jump=lambda_jump,
            jump_mean=jump_mean,
            jump_std=jump_std,
            scheme=scheme,
            n_s=n_s,
            n_t=n_t,
        )
        results.append(result)
        print(
            "tested "
            f"lambda={lambda_jump:.4f}, mean={jump_mean:.4f}, std={jump_std:.4f} "
            f"-> mae={result['mae']:.6f}, mse={result['mse']:.6f}, rmse={result['rmse']:.6f}"
        )

    return pd.DataFrame(results).sort_values(["mse", "mae", "rmse"]).reset_index(drop=True)


def main(
    max_rows: Optional[int] = DEFAULT_MAX_ROWS,
    cp_flag: Optional[str] = DEFAULT_CP_FLAG,
    tau_min: float = DEFAULT_TAU_MIN,
    tau_max: float = DEFAULT_TAU_MAX,
    n_s: int = 120,
    n_t: int = 400,
) -> None:
    """Run a simple grid-search calibration for the Merton jump parameters."""
    calibration_df = load_calibration_data(
        max_rows=max_rows,
        cp_flag=cp_flag,
        tau_min=tau_min,
        tau_max=tau_max,
    )
    sample_summary = summarize_calibration_data(calibration_df)
    results_df = run_grid_search(
        calibration_df,
        scheme=IMEX_SCHEME,
        n_s=n_s,
        n_t=n_t,
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = RESULTS_DIR / "merton_jump_calibration_results.csv"
    results_df.to_csv(results_path, index=False)

    best = results_df.iloc[0]

    print("\nCalibration sample summary:")
    print(f"Rows used: {sample_summary['rows']}")
    print(f"Option type filter: {cp_flag if cp_flag is not None else 'all'}")
    print(f"Calls: {sample_summary['call_count']}")
    print(f"Puts: {sample_summary['put_count']}")
    print(f"Requested tau range: [{tau_min}, {tau_max}]")
    print(f"Actual tau range in sample: [{sample_summary['tau_min']:.6f}, {sample_summary['tau_max']:.6f}]")

    print("\nBest parameter set by MSE:")
    print(
        f"lambda_jump={best['lambda_jump']}, "
        f"jump_mean={best['jump_mean']}, "
        f"jump_std={best['jump_std']}"
    )
    print(f"MAE={best['mae']:.6f}")
    print(f"MSE={best['mse']:.6f}")
    print(f"RMSE={best['rmse']:.6f}")
    print(f"Saved calibration results to: {results_path}")


if __name__ == "__main__":
    main()
