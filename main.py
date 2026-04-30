from __future__ import annotations

import multiprocessing as mp
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config.config import Config
from src.parallel_processing import ParallelPricingQueue
from scripts.calibrate_heston_params import main as calibrate_heston
from scripts.calibrate_merton_jump_params import main as calibrate_merton
from scripts.run_convergence_test import main as convergence_test

RESULTS_DIR = REPO_ROOT / "data" / "results"
CALIBRATION_DIR = RESULTS_DIR / "model_calibration"


def calibrate_all_securities() -> None:
    """Run Heston and Merton calibrations for every security in Config.SECURITIES."""
    print("Calibration Phase")
    for _, ticker in Config.SECURITIES:
        print(f"\n  {ticker}")
        heston_out = CALIBRATION_DIR / f"heston_calibrated_parameters_{ticker}.csv"
        if heston_out.exists():
            print(f"    Heston - {heston_out.name} already exists.")
        else:
            print(f"    Calibrating Heston...")
            try:
                calibrate_heston(ticker=ticker)
            except Exception as exc:
                print(f"  Heston Calibration failed for {ticker}: {exc}")

        merton_out = CALIBRATION_DIR / f"merton_jump_calibration_results_{ticker}.csv"
        if merton_out.exists():
            print(f"    Merton - {merton_out.name} already exists.")
        else:
            print(f"    Calibrating Merton...")
            try:
                calibrate_merton(ticker=ticker)
            except Exception as exc:
                print(f"    Merton Calibration failed for {ticker}: {exc}")



def load_heston_params(ticker: str) -> dict:
    candidates = [
        CALIBRATION_DIR / f"heston_calibrated_parameters_{ticker}.csv",
        CALIBRATION_DIR / "heston_calibrated_parameters.csv",
    ]
    for path in candidates:
        if path.exists():
            row = pd.read_csv(path).iloc[0]
            params = {
                "kappa": float(row["kappa"]),
                "theta": float(row["theta"]),
                "xi":    float(row["xi"]),
                "rho":   float(row["rho"]),
                "v0":    float(row["v0"]),
            }
            return params

    params = {"kappa": 2.0, "theta": 0.04, "xi": 0.3, "rho": -0.7, "v0": 0.04}
    print(f"    No Heston calibration file found for {ticker} so use literature defaults.")
    return params


def load_merton_params(ticker: str) -> dict:
    candidates = [
        CALIBRATION_DIR / f"merton_jump_calibration_results_{ticker}.csv",
        CALIBRATION_DIR / "merton_jump_calibration_results.csv",
    ]
    for path in candidates:
        if path.exists():
            df  = pd.read_csv(path)
            row = df.sort_values("mae").iloc[0]
            params = {
                "lambda_jump": float(row["lambda_jump"]),
                "jump_mean":   float(row["jump_mean"]),
                "jump_std":    float(row["jump_std"]),
            }
            return params

    params = {"lambda_jump": 0.01, "jump_mean": -0.005, "jump_std": 0.03}
    print(f"    No Merton calibration file found for {ticker} so use literature defaults.")
    return params


def build_task_list() -> list[dict]:
    tasks: list[dict] = []

    for _, ticker in Config.SECURITIES:
        heston_params = load_heston_params(ticker)
        merton_params = load_merton_params(ticker)
        print()

        for scheme, n_t in [("explicit", 3000), ("implicit", 500), ("crank_nicolson", 500)]:
            tasks.append(dict(
                ticker=ticker, model="black_scholes", scheme=scheme,
                n_s=200, n_t=n_t, max_rows=75,
            ))

        for scheme in ["douglas", "craig_sneyd", "modified_craig_sneyd", "hundsdorfer_verwer"]:
            tasks.append(dict(
                ticker=ticker, model="heston", scheme=scheme,
                model_params=heston_params,
                n_s=80, n_v=40, n_t=40, max_rows=75,
            ))

        tasks.append(dict(
            ticker=ticker, model="merton", scheme="imex_euler",
            model_params=merton_params,
            n_s=120, n_t=400, max_rows=75,
        ))

        tasks.append(dict(
            ticker=ticker, model="black_scholes", scheme="monte_carlo",
            max_rows=75,
        ))
        tasks.append(dict(
            ticker=ticker, model="heston", scheme="monte_carlo",
            model_params=heston_params, max_rows=75,
        ))
        tasks.append(dict(
            ticker=ticker, model="merton", scheme="monte_carlo",
            model_params=merton_params, max_rows=75,
        ))

    return tasks


def main() -> None:
    securities = [name for _, name in Config.SECURITIES]

    print("\nOptions Pricing Suite: by Pascal Bermeo Neumann, Sarang Deshpande, and Michael Waltuch")
    print(f"\nSecurities: {securities}")
    
    calibrate_all_securities() 

    print("\nPricing")
    
    queue = ParallelPricingQueue(n_workers=mp.cpu_count())
    print(f"    Worker processes available: {queue.n_workers}\n")

    for cfg in build_task_list():
        queue.add_task(**cfg)
        print(f"    Queued  {cfg['ticker']} - {cfg['model']:<15} with {cfg['scheme']} scheme")

    total = len(queue)
    print()
    print(f"    Total queue depth: {total} task(s)\n")

    queue.run()

    print("\nSaving results")
    queue.save_results(prefix="pricing")

    print("\nSummary")
    summary = queue.summary()
    if not summary.empty:
        pd.set_option("display.float_format", "{:.6f}".format)
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 120)
        print(summary.to_string(index=False))
        summary.to_csv(RESULTS_DIR / "pricing_summary.csv", index=False)
    else:
        print("No results available.")

    print("\nConvergence Analysis")
    convergence_test()
    

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
