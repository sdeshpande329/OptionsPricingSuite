from __future__ import annotations

import multiprocessing as mp
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.black_scholes import BlackScholesModel, BlackScholesParams, EuropeanOption
from src.models.heston import HestonModel
from src.models.merton_jump_diffusion import (
        MertonJumpDiffusionModel, MertonJumpParams, EuropeanOption,
    )
from src.monte_carlo.mc_black_scholes import BlackScholesMonteCarlo
from src.monte_carlo.mc_heston import HestonMonteCarlo
from src.monte_carlo.mc_merton import MertonMonteCarlo



DATA_PATH   = REPO_ROOT / "data" / "options_metrics_processed" / "clean_options_data.csv"
RESULTS_DIR = REPO_ROOT / "data" / "results" / "pricing_results"


VALID_MODELS   = {"black_scholes", "heston", "merton"}
BS_SCHEMES     = {"explicit", "implicit", "crank_nicolson", "monte_carlo"}
HESTON_SCHEMES = {"douglas", "craig_sneyd", "modified_craig_sneyd", "hundsdorfer_verwer", "monte_carlo"}
MERTON_SCHEMES = {"imex_euler", "monte_carlo"}

# Literature defaults used when model_params are not fully specified
_DEFAULT_HESTON_PARAMS: Dict[str, float] = {
    "kappa": 2.0, "theta": 0.04, "xi": 0.3, "rho": -0.7, "v0": 0.04, "q": 0.0,
}
_DEFAULT_MERTON_PARAMS: Dict[str, float] = {
    "lambda_jump": 0.01, "jump_mean": -0.005, "jump_std": 0.03, "q": 0.0,
}


def _valid_schemes_for(model: str) -> set:
    return {"black_scholes": BS_SCHEMES, "heston": HESTON_SCHEMES, "merton": MERTON_SCHEMES}[model]



@dataclass
class PricingTask:
    """Represents a single pricing job: (ticker, model, scheme) + grid settings."""

    ticker: str
    model: str
    scheme: str
    model_params: Dict[str, Any] = field(default_factory=dict)
    # PDE grid dimensions
    n_s: int = 200
    n_t: int = 100
    n_v: int = 50
    # Monte Carlo path count
    n_paths: Optional[int] = None
    max_rows: Optional[int] = None
    theta_cn: float = 0.5

    def __post_init__(self) -> None:
        if self.model not in VALID_MODELS:
            raise ValueError(f"model must be one of {sorted(VALID_MODELS)}, got '{self.model}'")
        valid = _valid_schemes_for(self.model)
        if self.scheme not in valid:
            raise ValueError(
                f"scheme '{self.scheme}' is not valid for model '{self.model}'. "
                f"Valid schemes: {sorted(valid)}"
            )



def _infer_option_type(cp_flag: str) -> str:
    cp = str(cp_flag).strip().upper()
    if cp == "C":
        return "call"
    if cp == "P":
        return "put"
    raise ValueError(f"Unrecognized cp_flag: {cp_flag!r}")


def _bs_grid_bounds(
    spot: float, strike: float, r: float, q: float, sigma: float, tau: float, n_std: float = 4.0
) -> Tuple[float, float]:
    drift  = (r - q - 0.5 * sigma ** 2) * tau
    spread = n_std * sigma * np.sqrt(tau)
    s_min  = max(1e-8, spot * np.exp(drift - spread))
    s_max  = max(spot * np.exp(drift + spread), 2.0 * strike, s_min + 1.0)
    return s_min, s_max


def _heston_grid_bounds(
    spot: float, strike: float, v0: float, theta_lrv: float
) -> Tuple[float, float, float, float]:
    s_min = max(1e-8, 0.5 * min(spot, strike))
    s_max = max(2.0 * max(spot, strike), s_min + 1.0)
    v_max = max(3.0 * max(v0, theta_lrv), 0.5)
    return s_min, s_max, 0.0, v_max


def _build_result_row(row: pd.Series, model_price: float, mktpx: float, scheme: str) -> dict:
    abs_err = abs(model_price - mktpx)
    return {
        "date":                   row["date"],
        "exdate":                 row["exdate"],
        "cp_flag":                row["cp_flag"],
        "spot_price":             float(row["spot_price"]),
        "strike_price":           float(row["strike_price"]),
        "tau (time to maturity)": float(row["tau (time to maturity)"]),
        "market_mid_price":       mktpx,
        "model_price":            model_price,
        "abs_error":              abs_err,
        "sq_error":               abs_err ** 2,
        "scheme":                 scheme,
    }


def _build_mc_result_row(
    row: pd.Series,
    model_price: float,
    std_error: float,
    diagnostics: dict,
    mktpx: float,
) -> dict:
    abs_err = abs(model_price - mktpx)
    ci_lower, ci_upper = diagnostics["ci_95"]
    return {
        "date":                   row["date"],
        "exdate":                 row["exdate"],
        "cp_flag":                row["cp_flag"],
        "spot_price":             float(row["spot_price"]),
        "strike_price":           float(row["strike_price"]),
        "tau (time to maturity)": float(row["tau (time to maturity)"]),
        "market_mid_price":       mktpx,
        "model_price":            model_price,
        "std_error":              std_error,
        "ci_95_lower":            ci_lower,
        "ci_95_upper":            ci_upper,
        "n_paths":                diagnostics["n_paths"],
        "n_steps":                diagnostics["n_steps"],
        "antithetic_used":        diagnostics["antithetic_used"],
        "abs_error":              abs_err,
        "sq_error":               abs_err ** 2,
        "scheme":                 "monte_carlo",
    }


def _price_bs_row(
    row: pd.Series, scheme: str, n_s: int, n_t: int, theta_cn: float, params: dict
) -> dict:

    spot     = float(row["spot_price"])
    strike   = float(row["strike_price"])
    tau      = float(row["tau (time to maturity)"])
    mktpx    = float(row["mid_price"])
    opt_type = _infer_option_type(row["cp_flag"])
    r        = float(row["rate"]) / 100.0
    sigma    = float(row["impl_volatility"])
    q        = params.get("q", 0.0)

    bs_params = BlackScholesParams(r=r, sigma=sigma, q=q)
    contract  = EuropeanOption(K=strike, T=tau, option_type=opt_type)
    model     = BlackScholesModel(bs_params)

    s_min, s_max = _bs_grid_bounds(spot, strike, r, q, sigma, tau)
    model_price  = model.price(
        contract=contract, spot=spot, scheme=scheme,
        S_min=s_min, S_max=s_max, N_S=n_s, N_t=n_t, theta_cn=theta_cn,
    )
    return _build_result_row(row, model_price, mktpx, scheme)


def _price_heston_row(
    row: pd.Series, scheme: str, n_s: int, n_v: int, n_t: int, params: dict,
    ticker: str = "", feller_warned: Optional[set] = None,
) -> dict:

    spot     = float(row["spot_price"])
    strike   = float(row["strike_price"])
    tau      = float(row["tau (time to maturity)"])
    mktpx    = float(row["mid_price"])
    opt_type = _infer_option_type(row["cp_flag"])
    r        = float(row["rate"]) / 100.0

    hp    = {**_DEFAULT_HESTON_PARAMS, **params}
    label = f"{ticker} / {scheme}" if ticker else scheme
    model = HestonModel(
        r=r, q=hp["q"],
        kappa=hp["kappa"], theta=hp["theta"],
        xi=hp["xi"], rho=hp["rho"], v0=hp["v0"],
        label=label, _feller_warned=feller_warned,
    )

    _, s_max, _, v_max = _heston_grid_bounds(spot, strike, hp["v0"], hp["theta"])
    model_price = model.price_european_option(
        S0=spot, K=strike, T=tau, option_type=opt_type, scheme=scheme,
        N_S=n_s, N_v=n_v, N_t=n_t, S_max=s_max, v_max=v_max,
    )
    return _build_result_row(row, model_price, mktpx, scheme)


def _price_merton_row(
    row: pd.Series, scheme: str, n_s: int, n_t: int, params: dict
) -> dict:
    

    spot     = float(row["spot_price"])
    strike   = float(row["strike_price"])
    tau      = float(row["tau (time to maturity)"])
    mktpx    = float(row["mid_price"])
    opt_type = _infer_option_type(row["cp_flag"])
    r        = float(row["rate"]) / 100.0
    sigma    = float(row["impl_volatility"])

    mp_cfg    = {**_DEFAULT_MERTON_PARAMS, **params}
    mj_params = MertonJumpParams(
        r=r, sigma=sigma, q=mp_cfg["q"],
        lambda_jump=mp_cfg["lambda_jump"],
        jump_mean=mp_cfg["jump_mean"],
        jump_std=mp_cfg["jump_std"],
    )
    contract = EuropeanOption(K=strike, T=tau, option_type=opt_type)
    model    = MertonJumpDiffusionModel(mj_params)

    s_min, s_max = _bs_grid_bounds(spot, strike, r, mj_params.q, sigma, tau)
    model_price  = model.price(
        contract=contract, spot=spot, scheme=scheme,
        S_min=s_min, S_max=s_max, N_S=n_s, N_t=n_t,
    )
    return _build_result_row(row, model_price, mktpx, scheme)


def _price_mc_bs_row(row: pd.Series, n_paths: Optional[int], params: dict) -> dict:

    spot     = float(row["spot_price"])
    strike   = float(row["strike_price"])
    tau      = float(row["tau (time to maturity)"])
    mktpx    = float(row["mid_price"])
    opt_type = _infer_option_type(row["cp_flag"])
    r        = float(row["rate"]) / 100.0
    sigma    = float(row["impl_volatility"])
    q        = params.get("q", 0.0)

    model = BlackScholesMonteCarlo(r=r, q=q, sigma=sigma)
    price, std_error, diagnostics = model.price_european_option(
        S0=spot, K=strike, T=tau, option_type=opt_type, n_paths=n_paths,
    )
    return _build_mc_result_row(row, price, std_error, diagnostics, mktpx)


def _price_mc_heston_row(row: pd.Series, n_paths: Optional[int], params: dict) -> dict:
    

    spot     = float(row["spot_price"])
    strike   = float(row["strike_price"])
    tau      = float(row["tau (time to maturity)"])
    mktpx    = float(row["mid_price"])
    opt_type = _infer_option_type(row["cp_flag"])
    r        = float(row["rate"]) / 100.0

    hp    = {**_DEFAULT_HESTON_PARAMS, **params}
    model = HestonMonteCarlo(
        r=r, q=hp["q"],
        kappa=hp["kappa"], theta=hp["theta"],
        xi=hp["xi"], rho=hp["rho"], v0=hp["v0"],
    )
    price, std_error, diagnostics = model.price_european_option(
        S0=spot, K=strike, T=tau, option_type=opt_type, n_paths=n_paths,
    )
    return _build_mc_result_row(row, price, std_error, diagnostics, mktpx)


def _price_mc_merton_row(row: pd.Series, n_paths: Optional[int], params: dict) -> dict:
    spot     = float(row["spot_price"])
    strike   = float(row["strike_price"])
    tau      = float(row["tau (time to maturity)"])
    mktpx    = float(row["mid_price"])
    opt_type = _infer_option_type(row["cp_flag"])
    r        = float(row["rate"]) / 100.0
    sigma    = float(row["impl_volatility"])

    mp_cfg = {**_DEFAULT_MERTON_PARAMS, **params}
    model  = MertonMonteCarlo(
        r=r, q=mp_cfg["q"], sigma=sigma,
        lambda_jump=mp_cfg["lambda_jump"],
        jump_mean=mp_cfg["jump_mean"],
        jump_std=mp_cfg["jump_std"],
    )
    price, std_error, diagnostics = model.price_european_option(
        S0=spot, K=strike, T=tau, option_type=opt_type, n_paths=n_paths,
    )
    return _build_mc_result_row(row, price, std_error, diagnostics, mktpx)



def _worker(task_dict: dict) -> Tuple[str, str, str, List[dict]]:
    """Worker executed in a subprocess. Loads the options CSV, filters by ticker, 
    prices every row with the requested model and scheme, and returns a flat list of 
    result dicts.
    """
    
    repo = str(REPO_ROOT)
    if repo not in sys.path:
        sys.path.insert(0, repo)

    ticker       = task_dict["ticker"]
    model        = task_dict["model"]
    scheme       = task_dict["scheme"]
    model_params = task_dict.get("model_params", {})
    n_s          = task_dict.get("n_s", 200)
    n_t          = task_dict.get("n_t", 100)
    n_v          = task_dict.get("n_v", 50)
    n_paths      = task_dict.get("n_paths")
    max_rows     = task_dict.get("max_rows")
    theta_cn     = task_dict.get("theta_cn", 0.5)
    data_path    = task_dict.get("data_path", str(DATA_PATH))

    df = pd.read_csv(data_path)

    if "security_name" in df.columns:
        df = df[df["security_name"] == ticker].reset_index(drop=True)

    if max_rows is not None:
        df = df.head(max_rows).copy()

    feller_warned: set = set()

    if scheme == "monte_carlo":
        _pricers = {
            "black_scholes": lambda row: _price_mc_bs_row(row, n_paths, model_params),
            "heston":        lambda row: _price_mc_heston_row(row, n_paths, model_params),
            "merton":        lambda row: _price_mc_merton_row(row, n_paths, model_params),
        }
    else:
        _pricers = {
            "black_scholes": lambda row: _price_bs_row(row, scheme, n_s, n_t, theta_cn, model_params),
            "heston":        lambda row: _price_heston_row(
                row, scheme, n_s, n_v, n_t, model_params,
                ticker=ticker, feller_warned=feller_warned,
            ),
            "merton":        lambda row: _price_merton_row(row, scheme, n_s, n_t, model_params),
        }

    pricer = _pricers[model]

    results: List[dict] = []
    for _, row in df.iterrows():
        try:
            results.append(pricer(row))
        except Exception as exc:
            results.append({
                "date":                   row.get("date"),
                "exdate":                 row.get("exdate"),
                "cp_flag":                row.get("cp_flag"),
                "spot_price":             row.get("spot_price"),
                "strike_price":           row.get("strike_price"),
                "tau (time to maturity)": row.get("tau (time to maturity)"),
                "market_mid_price":       row.get("mid_price"),
                "model_price":            float("nan"),
                "abs_error":              float("nan"),
                "sq_error":               float("nan"),
                "scheme":                 scheme,
                "error_message":          str(exc),
            })

    return ticker, model, scheme, results


class ParallelPricingQueue:
    """Manages a pool of option-pricing jobs and executes them in parallel."""
    def __init__(
        self,
        n_workers: Optional[int] = None,
        data_path: Optional[Path] = None,
        results_dir: Optional[Path] = None,
    ) -> None:
        self.n_workers   = n_workers or mp.cpu_count()
        self.data_path   = Path(data_path)   if data_path   else DATA_PATH
        self.results_dir = Path(results_dir) if results_dir else RESULTS_DIR

        self._queue:   List[PricingTask]                            = []
        self._results: Dict[Tuple[str, str, str], pd.DataFrame]    = {}


    def add_task(
        self,
        ticker: str,
        model: str,
        scheme: str,
        model_params: Optional[Dict[str, Any]] = None,
        n_s: int = 200,
        n_t: int = 100,
        n_v: int = 50,
        n_paths: Optional[int] = None,
        max_rows: Optional[int] = None,
        theta_cn: float = 0.5,
    ) -> "ParallelPricingQueue":
        """Enqueue a pricing job."""
        task = PricingTask(
            ticker=ticker,
            model=model,
            scheme=scheme,
            model_params=model_params or {},
            n_s=n_s, n_t=n_t, n_v=n_v,
            n_paths=n_paths,
            max_rows=max_rows,
            theta_cn=theta_cn,
        )
        self._queue.append(task)
        return self

    def run(self) -> "ParallelPricingQueue":
        """Dispatch all queued tasks to a ``multiprocessing.Pool`` and block until every task completes."""
        if not self._queue:
            print("Queue is empty — nothing to run.")
            return self

        task_dicts = [self._task_to_dict(t) for t in self._queue]
        n_workers  = min(self.n_workers, len(task_dicts))

        print(
            f"Dispatching {len(task_dicts)} task(s) across "
            f"{n_workers} worker process(es)..."
        )

        with mp.Pool(processes=n_workers) as pool:
            async_handles = [pool.apply_async(_worker, (td,)) for td in task_dicts]

            for handle in async_handles:
                ticker, model, scheme, rows = handle.get()
                key = (ticker, model, scheme)
                df  = pd.DataFrame(rows)
                self._results[key] = df

                n_ok = int(df["model_price"].notna().sum()) if "model_price" in df.columns else 0
                print(f"    {ticker} - {model}/{scheme}: {n_ok}/{len(rows)} rows priced.")

        self._queue.clear()
        return self

    def get_results(self) -> Dict[Tuple[str, str, str], pd.DataFrame]:
        """Return the results dict (copy) keyed by ``(ticker, model, scheme)``."""
        return dict(self._results)

    def save_results(self, prefix: str = "parallel") -> "ParallelPricingQueue":
        """Write one CSV per completed task to ``results_dir``."""
        if not self._results:
            print("No results to save — call run() first.")
            return self

        self.results_dir.mkdir(parents=True, exist_ok=True)
        for (ticker, model, scheme), df in self._results.items():
            fname = self.results_dir / f"{prefix}_{ticker}_{model}_{scheme}.csv"
            df.to_csv(fname, index=False)
        return self

    def summary(self) -> pd.DataFrame:
        """Return a summary DataFrame with per-task MAE, RMSE, and row counts."""
        rows = []
        for (ticker, model, scheme), df in self._results.items():
            ok    = df["model_price"].notna() if "model_price" in df.columns else pd.Series([], dtype=bool)
            n_ok  = int(ok.sum())
            n_tot = len(df)
            mae   = float(df.loc[ok, "abs_error"].mean())              if n_ok > 0 else float("nan")
            rmse  = float(np.sqrt(df.loc[ok, "sq_error"].mean()))      if n_ok > 0 else float("nan")
            rows.append(
                dict(ticker=ticker, model=model, scheme=scheme,
                     n_priced=n_ok, n_total=n_tot, MAE=mae, RMSE=rmse)
            )
        return pd.DataFrame(rows)

    
    def _task_to_dict(self, task: PricingTask) -> dict:
        return {
            "ticker":       task.ticker,
            "model":        task.model,
            "scheme":       task.scheme,
            "model_params": task.model_params,
            "n_s":          task.n_s,
            "n_t":          task.n_t,
            "n_v":          task.n_v,
            "n_paths":      task.n_paths,
            "max_rows":     task.max_rows,
            "theta_cn":     task.theta_cn,
            "data_path":    str(self.data_path),
        }

    def __repr__(self) -> str:
        return (
            f"ParallelPricingQueue("
            f"queued={len(self._queue)}, "
            f"completed={len(self._results)}, "
            f"n_workers={self.n_workers})"
        )

    def __len__(self) -> int:
        return len(self._queue)