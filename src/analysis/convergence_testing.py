from __future__ import annotations

from dataclasses import dataclass, field
from math import erf, exp, log, sqrt
from time import perf_counter
from typing import Any, Callable, Dict, Optional, Sequence

import numpy as np
import pandas as pd


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def black_scholes_closed_form(
    spot: float,
    strike: float,
    maturity: float,
    r: float,
    q: float,
    sigma: float,
    option_type: str = "call",
) -> float:
    """Closed-form Black-Scholes price for a European option."""
    if maturity <= 0.0:
        if option_type == "call":
            return max(spot - strike, 0.0)
        return max(strike - spot, 0.0)

    sqrt_t = sqrt(maturity)
    d1 = (log(spot / strike) + (r - q + 0.5 * sigma**2) * maturity) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t

    if option_type == "call":
        return spot * exp(-q * maturity) * _normal_cdf(d1) - strike * exp(-r * maturity) * _normal_cdf(d2)
    return strike * exp(-r * maturity) * _normal_cdf(-d2) - spot * exp(-q * maturity) * _normal_cdf(-d1)


def choose_lognormal_grid_bounds(
    spot: float,
    strike: float,
    r: float,
    q: float,
    sigma: float,
    maturity: float,
    n_std: float = 4.0,
) -> tuple[float, float]:
    """Choose a 1D stock-price domain using a lognormal heuristic."""
    drift = (r - q - 0.5 * sigma**2) * maturity
    spread = n_std * sigma * sqrt(maturity)

    s_min = spot * exp(drift - spread)
    s_max = spot * exp(drift + spread)

    s_min = max(1e-8, s_min)
    s_max = max(s_max, 2.0 * strike, s_min + 1.0)
    return s_min, s_max


def choose_heston_grid_bounds(
    spot: float,
    strike: float,
    v0: float,
    theta: float,
    n_std_v: float = 3.0,
) -> tuple[float, float, float, float]:
    """Choose a simple stock/variance domain for the Heston grid."""
    s_min = max(1e-8, 0.5 * min(spot, strike))
    s_max = max(2.0 * max(spot, strike), s_min + 1.0)

    v_min = 0.0
    v_max = max(n_std_v * max(v0, theta), 0.5)
    return s_min, s_max, v_min, v_max


def compute_observed_order(
    previous_error: Optional[float],
    current_error: Optional[float],
    previous_resolution: Optional[float],
    current_resolution: Optional[float],
) -> Optional[float]:
    """Estimate the empirical convergence order between two refinement levels."""
    if previous_error is None or current_error is None:
        return None
    if previous_resolution is None or current_resolution is None:
        return None
    if previous_error <= 0.0 or current_error <= 0.0:
        return None
    if previous_resolution <= 0.0 or current_resolution <= previous_resolution:
        return None

    return log(previous_error / current_error) / log(current_resolution / previous_resolution)


@dataclass
class ConvergenceStudy:
    """One convergence or stability study for a specific model and scheme."""

    model: str
    scheme: str
    reference_label: str
    reference_price: float
    price_fn: Callable[[Dict[str, Any]], float]
    refinements: Sequence[Dict[str, Any]]
    primary_resolution_key: str
    objective: str = "convergence"
    scenario: str = "baseline"
    resolution_label: Optional[str] = None
    resolution_value_fn: Optional[Callable[[Dict[str, Any]], float]] = None
    contract_params: Dict[str, Any] = field(default_factory=dict)
    model_params: Dict[str, Any] = field(default_factory=dict)


def _get_resolution_value(study: ConvergenceStudy, refinement: Dict[str, Any]) -> Optional[float]:
    if study.resolution_value_fn is not None:
        return study.resolution_value_fn(refinement)
    value = refinement.get(study.primary_resolution_key)
    return float(value) if value is not None else None


def run_convergence_study(study: ConvergenceStudy) -> pd.DataFrame:
    """Run one study and return a row per refinement level."""
    rows = []
    previous_price = None
    previous_error = None
    previous_resolution = None

    for level, refinement in enumerate(study.refinements, start=1):
        resolution_value = _get_resolution_value(study, refinement)
        row: Dict[str, Any] = {
            "model": study.model,
            "scheme": study.scheme,
            "objective": study.objective,
            "scenario": study.scenario,
            "level": level,
            "reference_label": study.reference_label,
            "reference_price": study.reference_price,
            "primary_resolution_key": study.primary_resolution_key,
            "resolution_label": study.resolution_label or study.primary_resolution_key,
            "resolution_value": resolution_value,
        }
        row.update(study.contract_params)
        row.update(study.model_params)
        row.update(refinement)

        start = perf_counter()
        try:
            price = study.price_fn(refinement)
            runtime_sec = perf_counter() - start

            abs_error = abs(price - study.reference_price)
            rel_error = abs_error / abs(study.reference_price) if study.reference_price != 0.0 else np.nan
            delta_from_previous = abs(price - previous_price) if previous_price is not None else np.nan
            observed_order = compute_observed_order(
                previous_error=previous_error,
                current_error=abs_error,
                previous_resolution=previous_resolution,
                current_resolution=resolution_value,
            )

            row.update(
                {
                    "status": "success",
                    "price": price,
                    "runtime_sec": runtime_sec,
                    "abs_error": abs_error,
                    "rel_error": rel_error,
                    "delta_from_previous": delta_from_previous,
                    "observed_order": observed_order,
                    "error_message": None,
                }
            )

            previous_price = price
            previous_error = abs_error
            previous_resolution = resolution_value
        except Exception as exc:
            runtime_sec = perf_counter() - start
            row.update(
                {
                    "status": "failed",
                    "price": np.nan,
                    "runtime_sec": runtime_sec,
                    "abs_error": np.nan,
                    "rel_error": np.nan,
                    "delta_from_previous": np.nan,
                    "observed_order": np.nan,
                    "error_message": str(exc),
                }
            )

        rows.append(row)

    return pd.DataFrame(rows)


def run_convergence_suite(studies: Sequence[ConvergenceStudy]) -> pd.DataFrame:
    """Run multiple studies and combine them into one results table."""
    frames = [run_convergence_study(study) for study in studies]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def classify_convergence(errors: Sequence[float], prices: Sequence[float], success_ratio: float) -> str:
    """Classify convergence behavior conservatively."""
    if len(errors) == 0:
        return "all failed"
    if not np.all(np.isfinite(errors)) or not np.all(np.isfinite(prices)):
        return "failed / non-finite"
    if np.any(np.abs(np.asarray(prices)) > 1e8):
        return "unstable / blow-up"
    if success_ratio < 1.0:
        return "partial convergence with failures"

    monotone = all(curr < prev for prev, curr in zip(errors[:-1], errors[1:]))
    if monotone:
        return "clear convergence"
    if errors[-1] < errors[0]:
        return "overall improvement but non-monotone"
    return "no clear convergence"


def classify_stability(
    prices: Sequence[float],
    errors: Sequence[float],
    success_ratio: float,
    reference_price: float,
) -> str:
    """Classify stress-test stability behavior."""
    if len(prices) == 0:
        return "all failed"
    if success_ratio < 1.0:
        return "unstable / failed cases"
    if not np.all(np.isfinite(prices)) or not np.all(np.isfinite(errors)):
        return "unstable / non-finite"

    abs_prices = np.abs(np.asarray(prices))
    if np.any(abs_prices > 1e8):
        return "unstable / blow-up"

    max_error = float(np.max(errors))
    if reference_price != 0.0 and max_error / abs(reference_price) > 0.5:
        return "stable but inaccurate under stress"
    return "stable under stress"


def summarize_convergence_results(results_df: pd.DataFrame) -> pd.DataFrame:
    """Build one summary row per model/scheme/scenario."""
    if results_df.empty:
        return pd.DataFrame()

    summary_rows = []
    sort_cols = ["model", "objective", "scenario", "scheme", "level"]
    grouped = results_df.sort_values(sort_cols).groupby(
        ["model", "objective", "scenario", "scheme"],
        dropna=False,
    )

    for (model, objective, scenario, scheme), group in grouped:
        successful = group[group["status"] == "success"].copy()
        total_levels = len(group)
        successful_levels = len(successful)
        failed_levels = total_levels - successful_levels
        success_ratio = successful_levels / total_levels if total_levels > 0 else 0.0

        if successful.empty:
            summary_rows.append(
                {
                    "model": model,
                    "objective": objective,
                    "scenario": scenario,
                    "scheme": scheme,
                    "assessment": "all failed",
                    "reference_label": group["reference_label"].iloc[0],
                    "reference_price": group["reference_price"].iloc[0],
                    "initial_price": np.nan,
                    "final_price": np.nan,
                    "initial_abs_error": np.nan,
                    "final_abs_error": np.nan,
                    "best_abs_error": np.nan,
                    "final_observed_order": np.nan,
                    "average_observed_order": np.nan,
                    "initial_runtime_sec": np.nan,
                    "final_runtime_sec": np.nan,
                    "successful_levels": successful_levels,
                    "failed_levels": failed_levels,
                    "total_levels": total_levels,
                }
            )
            continue

        errors = successful["abs_error"].to_numpy(dtype=float)
        prices = successful["price"].to_numpy(dtype=float)
        orders = successful["observed_order"].dropna().to_numpy(dtype=float)
        initial_row = successful.iloc[0]
        final_row = successful.iloc[-1]
        reference_price = float(successful["reference_price"].iloc[0])

        if objective == "stability":
            assessment = classify_stability(
                prices=prices,
                errors=errors,
                success_ratio=success_ratio,
                reference_price=reference_price,
            )
        else:
            assessment = classify_convergence(
                errors=errors,
                prices=prices,
                success_ratio=success_ratio,
            )

        summary_rows.append(
            {
                "model": model,
                "objective": objective,
                "scenario": scenario,
                "scheme": scheme,
                "assessment": assessment,
                "reference_label": successful["reference_label"].iloc[0],
                "reference_price": reference_price,
                "initial_price": initial_row["price"],
                "final_price": final_row["price"],
                "initial_abs_error": initial_row["abs_error"],
                "final_abs_error": final_row["abs_error"],
                "best_abs_error": successful["abs_error"].min(),
                "final_observed_order": float(orders[-1]) if len(orders) > 0 else np.nan,
                "average_observed_order": float(np.mean(orders)) if len(orders) > 0 else np.nan,
                "initial_runtime_sec": initial_row["runtime_sec"],
                "final_runtime_sec": final_row["runtime_sec"],
                "successful_levels": successful_levels,
                "failed_levels": failed_levels,
                "total_levels": total_levels,
            }
        )

    return pd.DataFrame(summary_rows).sort_values(
        ["model", "objective", "scenario", "scheme"]
    ).reset_index(drop=True)
