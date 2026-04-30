from __future__ import annotations

from pathlib import Path
import sys
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.heston import HestonModel

DATA_PATH = REPO_ROOT / "data" / "options_metrics_processed" / "clean_options_data.csv"
CALIBRATION_DIR = REPO_ROOT / "data" / "results" / "model_calibration"


def infer_option_type(cp_flag: str) -> str:
    """Convert OptionMetrics cp_flag to 'call' or 'put'."""
    cp = str(cp_flag).strip().upper()
    if cp == "C":
        return "call"
    if cp == "P":
        return "put"
    raise ValueError(f"Unrecognized cp_flag: {cp_flag}")


class HestonCalibrator:
    """Calibrate Heston model parameters to market data. Optimizes parameters {kappa, theta, xi, rho, v0} to minimize: sum_i w_i * (V_model^i - V_market^i)^2"""

    def __init__(self, r: float, q: float = 0.0, scheme: str = "modified_craig_sneyd", grid_params: Optional[Dict] = None) -> None:
        self.r = r
        self.q = q
        self.scheme = scheme
        self.grid_params = grid_params or {"N_S": 80, "N_v": 40, "N_t": 40}
        self.market_data = None
        self.S0 = None

    def load_and_prepare_data(self,data_path: str,max_rows: Optional[int] = None,option_type: str = "call",min_moneyness: float = 0.80,max_moneyness: float = 1.20,min_tau: float = 0.05,max_tau: float = 1.0,ticker: Optional[str] = None) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load and prepare market data for calibration."""
        df = pd.read_csv(data_path)
        if ticker is not None and "security_name" in df.columns:
            df = df[df["security_name"] == ticker].reset_index(drop=True)
            if df.empty:
                raise ValueError(f"No rows found for ticker '{ticker}' in {data_path}")
        if max_rows is not None:
            df = df.head(max_rows)

        S0 = float(df["spot_price"].iloc[0])
        print(f"Spot price: ${S0:.2f}")

        if "cp_flag" in df.columns:
            flag = "C" if option_type == "call" else "P"
            df = df[df["cp_flag"] == flag]
            print(f"After filtering for {option_type}s: {len(df)} rows")

        df["moneyness"] = df["strike_price"] / S0

        print(f"\nMoneyness distribution (before filter):")
        print(f"  Min: {df['moneyness'].min():.3f}  Max: {df['moneyness'].max():.3f}")
        print(f"\nMaturity distribution (years, before filter):")
        print(f"  Min: {df['tau (time to maturity)'].min():.4f}  "
              f"Max: {df['tau (time to maturity)'].max():.4f}")

        df = df[
            (df["moneyness"] >= min_moneyness)
            & (df["moneyness"] <= max_moneyness)
            & (df["tau (time to maturity)"] >= min_tau)
            & (df["tau (time to maturity)"] <= max_tau)
        ]

        if len(df) == 0:
            raise ValueError(
                "No options remain after filtering. Relax filter criteria."
            )

        df = df.drop_duplicates(subset=["strike_price", "tau (time to maturity)"])

        strikes       = df["strike_price"].values
        maturities    = df["tau (time to maturity)"].values
        market_prices = df["mid_price"].values

        weights = 1.0 / (market_prices + 1.0)
        weights = weights / weights.sum()

        self.market_data = df
        self.S0 = S0

        print(f"\nCALIBRATION DATASET SUMMARY")
        print(f"  Spot price:     ${S0:.2f}")
        print(f"  Options:        {len(strikes)}")
        print(f"  Strike range:   [{strikes.min():.2f}, {strikes.max():.2f}]")
        print(f"  Moneyness:      [{(strikes/S0).min():.3f}, {(strikes/S0).max():.3f}]")
        print(f"  Maturity range: [{maturities.min():.4f}, {maturities.max():.4f}] yrs")
        print(f"  Price range:    [${market_prices.min():.4f}, ${market_prices.max():.4f}]")
        print(f"  Unique maturities: {len(np.unique(maturities))}")
        print()

        return S0, strikes, maturities, market_prices, weights

    
    def objective_function(self, params: np.ndarray, S0: float, strikes: np.ndarray, maturities: np.ndarray, market_prices: np.ndarray, weights: np.ndarray, option_type: str = "call") -> np.ndarray:
        """Compute weighted residuals using CF pricing."""
        kappa, theta, xi, rho, v0 = params

        if not self._check_constraints(kappa, theta, xi, rho, v0):
            return 1e6 * np.ones(len(market_prices))

        try:
            model_prices = np.array([
                HestonModel.characteristic_function_price(
                    S0=S0, K=K, T=T,
                    r=self.r, q=self.q,
                    kappa=kappa, theta=theta, xi=xi,
                    rho=rho, v0=v0,
                    option_type=option_type,
                )
                for K, T in zip(strikes, maturities)
            ])

            return np.sqrt(weights) * (model_prices - market_prices)

        except Exception as e:
            print(f"  Error in objective: {e}")
            return 1e6 * np.ones(len(market_prices))

    def calibrate(self,S0: float,strikes: np.ndarray,maturities: np.ndarray,market_prices: np.ndarray,weights: np.ndarray,option_type: str = "call",
    initial_guess: Optional[Dict[str, float]] = None,bounds: Optional[Dict] = None,max_iterations: int = 100, verbose: int = 2,) -> Dict:
        """Calibrate Heston parameters via least squares on CF prices. After convergence, recomputes final prices with the PDE solver for validation."""
        if initial_guess is None:
            avg_iv = (
                self.market_data["impl_volatility"].mean()
                if self.market_data is not None
                else 0.20
            )
            initial_guess = {
                "kappa": 2.0,
                "theta": avg_iv ** 2,
                "xi":    0.3,
                "rho":   -0.5,
                "v0":    avg_iv ** 2,
            }

        x0 = np.array([
            initial_guess["kappa"],
            initial_guess["theta"],
            initial_guess["xi"],
            initial_guess["rho"],
            initial_guess["v0"],
        ])

        if bounds is None:
            bounds = {
                "kappa": (0.1,  15.0),
                "theta": (0.001, 0.5),
                "xi":    (0.01,  1.0),
                "rho":   (-0.99, 0.0),
                "v0":    (0.001, 0.5),
            }

        lower = np.array([bounds["kappa"][0], bounds["theta"][0],
                          bounds["xi"][0],    bounds["rho"][0], bounds["v0"][0]])
        upper = np.array([bounds["kappa"][1], bounds["theta"][1],
                          bounds["xi"][1],    bounds["rho"][1], bounds["v0"][1]])

        print(f"Initial guess:")
        print(f"  kappa = {x0[0]:.4f}")
        print(f"  theta = {x0[1]:.6f}  (vol = {np.sqrt(x0[1])*100:.2f}%)")
        print(f"  xi    = {x0[2]:.4f}")
        print(f"  rho   = {x0[3]:.4f}")
        print(f"  v0    = {x0[4]:.6f}  (vol = {np.sqrt(x0[4])*100:.2f}%)")
        print(f"Max iterations: {max_iterations}\n")

        start_time = time.time()

        result = least_squares(
            fun=self.objective_function,
            x0=x0,
            bounds=(lower, upper),
            method="trf",
            verbose=verbose,
            args=(S0, strikes, maturities, market_prices, weights, option_type),
            max_nfev=max_iterations,
            ftol=1e-6,
            xtol=1e-6,
            gtol=1e-6,
        )

        cf_elapsed = time.time() - start_time
        kappa_opt, theta_opt, xi_opt, rho_opt, v0_opt = result.x

        print(f"\nOptimized parameters (from CF calibration):")
        print(f"  kappa = {kappa_opt:.6f}  (mean reversion speed)")
        print(f"  theta = {theta_opt:.6f}  (long-run vol = {np.sqrt(theta_opt)*100:.2f}%)")
        print(f"  xi    = {xi_opt:.6f}  (vol of vol)")
        print(f"  rho   = {rho_opt:.6f}  (correlation)")
        print(f"  v0    = {v0_opt:.6f}  (initial vol = {np.sqrt(v0_opt)*100:.2f}%)")
        print(f"\nFeller condition satisfied: "
              f"{HestonModel.feller_condition_satisfied(kappa_opt, theta_opt, xi_opt)}")
        print(f"CF calibration cost:  {result.cost:.6e}")
        print(f"CF calibration time:  {cf_elapsed:.2f} seconds")
        print(f"Function evaluations: {result.nfev}")

        tol = 1e-3
        bound_hits = []
        names = ["kappa", "theta", "xi", "rho", "v0"]
        for name, val, lo, hi in zip(names, result.x, lower, upper):
            if abs(val - lo) < tol * (hi - lo + 1e-10):
                bound_hits.append(f"{name} hit lower bound ({lo})")
            if abs(val - hi) < tol * (hi - lo + 1e-10):
                bound_hits.append(f"{name} hit upper bound ({hi})")
        if bound_hits:
            print(f"\nWarning: parameters at bounds so solution may not be a true minimum:")
            for msg in bound_hits:
                print(f"  {msg}")
            print("  Consider widening bounds and rerunning.")

        cf_prices = np.array([
            HestonModel.characteristic_function_price(
                S0=S0, K=K, T=T,
                r=self.r, q=self.q,
                kappa=kappa_opt, theta=theta_opt, xi=xi_opt,
                rho=rho_opt, v0=v0_opt,
                option_type=option_type,
            )
            for K, T in zip(strikes, maturities)
        ])

        cf_errors = cf_prices - market_prices
        cf_rmse   = np.sqrt(np.mean(cf_errors ** 2))
        cf_mae    = np.mean(np.abs(cf_errors))
        cf_mape   = np.mean(np.abs(cf_errors / market_prices)) * 100

        print(f"\nCF pricing errors at optimum:")
        print(f"  RMSE: ${cf_rmse:.6f}")
        print(f"  MAE:  ${cf_mae:.6f}")
        print(f"  MAPE: {cf_mape:.2f}%")

        print(f"\nValidating with PDE solver ({self.scheme})...")
        pde_start = time.time()

        model_opt = HestonModel(
            r=self.r, q=self.q,
            kappa=kappa_opt, theta=theta_opt, xi=xi_opt,
            rho=rho_opt, v0=v0_opt,
        )

        S_max = max(3 * S0, 2 * strikes.max())
        v_max = max(10 * v0_opt, 10 * theta_opt, 0.15)

        pde_prices = model_opt.price_multiple_options(
            S0=S0,
            strikes=strikes,
            maturities=maturities,
            option_type=option_type,
            scheme=self.scheme,
            S_max=S_max,
            v_max=v_max,
            **self.grid_params,
        )

        pde_elapsed = time.time() - pde_start
        pde_errors  = pde_prices - market_prices
        pde_rmse    = np.sqrt(np.mean(pde_errors ** 2))
        pde_mae     = np.mean(np.abs(pde_errors))
        pde_mape    = np.mean(np.abs(pde_errors / market_prices)) * 100

        print(f"\nPDE pricing errors at optimum:")
        print(f"  RMSE: ${pde_rmse:.6f}")
        print(f"  MAE:  ${pde_mae:.6f}")
        print(f"  MAPE: {pde_mape:.2f}%")
        print(f"  PDE validation time: {pde_elapsed:.2f} seconds")

        return {
            "parameters": {
                "kappa": kappa_opt,
                "theta": theta_opt,
                "xi":    xi_opt,
                "rho":   rho_opt,
                "v0":    v0_opt,
            },
            "initial_guess":    initial_guess,
            "cf_prices":        cf_prices,
            "pde_prices":       pde_prices,
            "market_prices":    market_prices,
            "cf_errors":        cf_errors,
            "pde_errors":       pde_errors,
            "cf_rmse":          cf_rmse,
            "cf_mae":           cf_mae,
            "cf_mape":          cf_mape,
            "pde_rmse":         pde_rmse,
            "pde_mae":          pde_mae,
            "pde_mape":         pde_mape,
            "cf_elapsed":       cf_elapsed,
            "pde_elapsed":      pde_elapsed,
            "success":          result.success,
            "n_evaluations":    result.nfev,
            "strikes":          strikes,
            "maturities":       maturities,
            "bound_hits":       bound_hits,
        }

    
    def _check_constraints(self, kappa: float, theta: float, xi: float, rho: float, v0: float) -> bool:
        if kappa <= 0 or theta <= 0 or xi <= 0 or v0 <= 0:
            return False
        if not (-1.0 < rho < 1.0):
            return False
        return True

    def save_results(self, results: Dict, output_dir: Path, ticker: Optional[str] = None) -> None:
        """Save calibration results to CSV files."""
        output_dir.mkdir(parents=True, exist_ok=True)

        params_df = pd.DataFrame([results["parameters"]])
        params_df["cf_rmse"]       = results["cf_rmse"]
        params_df["cf_mae"]        = results["cf_mae"]
        params_df["cf_mape"]       = results["cf_mape"]
        params_df["pde_rmse"]      = results["pde_rmse"]
        params_df["pde_mae"]       = results["pde_mae"]
        params_df["pde_mape"]      = results["pde_mape"]
        params_df["cf_elapsed"]    = results["cf_elapsed"]
        params_df["pde_elapsed"]   = results["pde_elapsed"]
        params_df["success"]       = results["success"]
        params_df["n_evaluations"] = results["n_evaluations"]
        if ticker is not None:
            params_df["ticker"] = ticker

        suffix = f"_{ticker}" if ticker else ""
        params_file = output_dir / f"heston_calibrated_parameters{suffix}.csv"
        params_df.to_csv(params_file, index=False)
        print(f"\nSaved parameters to: {params_file}")

        comparison_df = pd.DataFrame({
            "strike":       results["strikes"],
            "maturity":     results["maturities"],
            "market_price": results["market_prices"],
            "cf_price":     results["cf_prices"],
            "pde_price":    results["pde_prices"],
            "cf_error":     results["cf_errors"],
            "pde_error":    results["pde_errors"],
            "cf_abs_error": np.abs(results["cf_errors"]),
            "pde_abs_error": np.abs(results["pde_errors"]),
            "cf_pct_error": results["cf_errors"] / results["market_prices"] * 100,
            "pde_pct_error": results["pde_errors"] / results["market_prices"] * 100,
        })

        comparison_suffix = f"_{ticker}" if ticker else ""
        comparison_file = output_dir / f"heston_price_comparison{comparison_suffix}.csv"
        comparison_df.to_csv(comparison_file, index=False)
        print(f"Saved price comparison to: {comparison_file}")


def main(ticker: Optional[str] = None) -> Dict:
    """Run Heston calibration on OptionMetrics data for a specific ticker."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Data file not found: {DATA_PATH}")

    df_temp = pd.read_csv(DATA_PATH, nrows=1)
    r = float(df_temp["rate"].iloc[0]) / 100.0
    label = ticker if ticker else "all securities"
    print(f"Running Heston calibration for: {label}")
    print(f"Using risk-free rate: {r*100:.4f}%\n")

    calibrator = HestonCalibrator(
        r=r,
        q=0.0,
        scheme="craig_sneyd",
        grid_params={"N_S": 80, "N_v": 40, "N_t": 40},
    )

    S0, strikes, maturities, market_prices, weights = calibrator.load_and_prepare_data(
        data_path=str(DATA_PATH),
        option_type="call",
        min_moneyness=0.80,
        max_moneyness=1.20,
        min_tau=0.05,
        max_tau=1.0,
        ticker=ticker,
    )

    avg_iv = calibrator.market_data["impl_volatility"].mean()

    initial_guess = {
        "kappa": 2.0,
        "theta": avg_iv ** 2,
        "xi":    0.3,
        "rho":   -0.7,
        "v0":    avg_iv ** 2,
    }

    bounds = {
        "kappa": (0.1,   15.0),
        "theta": (0.001,  0.5),
        "xi":    (0.01,   1.0),
        "rho":   (-0.99,  0.0),
        "v0":    (0.001,  0.5),
    }

    results = calibrator.calibrate(
        S0=S0,
        strikes=strikes,
        maturities=maturities,
        market_prices=market_prices,
        weights=weights,
        option_type="call",
        initial_guess=initial_guess,
        bounds=bounds,
        max_iterations=200,
        verbose=2,
    )

    calibrator.save_results(results, CALIBRATION_DIR, ticker=ticker)
    return results


if __name__ == "__main__":
    results = main()