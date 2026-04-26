from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analysis.convergence_testing import (
    ConvergenceStudy,
    black_scholes_closed_form,
    choose_heston_grid_bounds,
    choose_lognormal_grid_bounds,
    run_convergence_suite,
    summarize_convergence_results,
)
from src.models.black_scholes import BlackScholesModel, BlackScholesParams, EuropeanOption as BSEuropeanOption
from src.models.heston import HestonModel
from src.models.merton_jump_diffusion import (
    EuropeanOption as MertonEuropeanOption,
    MertonJumpDiffusionModel,
    MertonJumpParams,
)

RESULTS_DIR = REPO_ROOT / "data" / "results"
RESULTS_PATH = RESULTS_DIR / "convergence_analysis_results.csv"
SUMMARY_PATH = RESULTS_DIR / "convergence_summary_results.csv"
PLOTS_DIR = RESULTS_DIR / "convergence_plots"

OPTION = {
    "spot": 100.0,
    "strike": 100.0,
    "maturity": 1.0,
    "option_type": "call",
}

BSM_PARAMS = {
    "r": 0.05,
    "q": 0.0,
    "sigma": 0.20,
}

HESTON_PARAMS = {
    "r": 0.05,
    "q": 0.0,
    "kappa": 10.0,
    "theta": 0.01,
    "xi": 0.19,
    "rho": -0.90,
    "v0": 0.01,
}

HESTON_STRESS_PARAMS = {
    "r": 0.05,
    "q": 0.0,
    "kappa": 8.0,
    "theta": 0.02,
    "xi": 0.30,
    "rho": -0.95,
    "v0": 0.02,
}

MERTON_PARAMS = {
    "r": 0.05,
    "q": 0.0,
    "sigma": 0.20,
    "lambda_jump": 0.01,
    "jump_mean": -0.005,
    "jump_std": 0.03,
}

MERTON_STRESS_PARAMS = {
    "r": 0.05,
    "q": 0.0,
    "sigma": 0.20,
    "lambda_jump": 1.00,
    "jump_mean": -0.02,
    "jump_std": 0.15,
}

BSM_EXPLICIT_STABLE_REFINEMENTS = [
    {"N_S": 25, "N_t": 625},
    {"N_S": 35, "N_t": 1225},
    {"N_S": 50, "N_t": 2500},
    {"N_S": 70, "N_t": 4900},
]

BSM_SPATIAL_REFINEMENTS = [
    {"N_S": 50, "N_t": 1600},
    {"N_S": 75, "N_t": 1600},
    {"N_S": 100, "N_t": 1600},
    {"N_S": 150, "N_t": 1600},
]

BSM_TEMPORAL_REFINEMENTS = [
    {"N_S": 200, "N_t": 100},
    {"N_S": 200, "N_t": 200},
    {"N_S": 200, "N_t": 400},
    {"N_S": 200, "N_t": 800},
]

HESTON_SPATIAL_REFINEMENTS = [
    {"N_S": 20, "N_v": 15, "N_t": 120},
    {"N_S": 40, "N_v": 30, "N_t": 120},
    {"N_S": 60, "N_v": 45, "N_t": 120},
    {"N_S": 80, "N_v": 60, "N_t": 120},
]

HESTON_TEMPORAL_REFINEMENTS = [
    {"N_S": 80, "N_v": 60, "N_t": 20},
    {"N_S": 80, "N_v": 60, "N_t": 40},
    {"N_S": 80, "N_v": 60, "N_t": 80},
    {"N_S": 80, "N_v": 60, "N_t": 120},
]

HESTON_STRESS_REFINEMENTS = [
    {"N_S": 20, "N_v": 15, "N_t": 20},
    {"N_S": 40, "N_v": 30, "N_t": 40},
    {"N_S": 60, "N_v": 45, "N_t": 60},
    {"N_S": 80, "N_v": 60, "N_t": 80},
]

MERTON_SPATIAL_REFINEMENTS = [
    {"N_S": 50, "N_t": 1600},
    {"N_S": 100, "N_t": 1600},
    {"N_S": 150, "N_t": 1600},
    {"N_S": 200, "N_t": 1600},
]

MERTON_TEMPORAL_REFINEMENTS = [
    {"N_S": 200, "N_t": 200},
    {"N_S": 200, "N_t": 400},
    {"N_S": 200, "N_t": 800},
    {"N_S": 200, "N_t": 1600},
]

MERTON_STRESS_REFINEMENTS = [
    {"N_S": 50, "N_t": 400},
    {"N_S": 100, "N_t": 800},
    {"N_S": 150, "N_t": 1200},
    {"N_S": 200, "N_t": 1600},
]


def build_bsm_pricer(scheme: str, params: dict):
    model = BlackScholesModel(BlackScholesParams(**params))
    contract = BSEuropeanOption(
        K=OPTION["strike"],
        T=OPTION["maturity"],
        option_type=OPTION["option_type"],
    )
    s_min, s_max = choose_lognormal_grid_bounds(
        spot=OPTION["spot"],
        strike=OPTION["strike"],
        r=params["r"],
        q=params["q"],
        sigma=params["sigma"],
        maturity=OPTION["maturity"],
    )

    def price_fn(refinement: dict) -> float:
        return model.price(
            contract=contract,
            spot=OPTION["spot"],
            scheme=scheme,
            S_min=s_min,
            S_max=s_max,
            N_S=refinement["N_S"],
            N_t=refinement["N_t"],
        )

    return price_fn


def build_heston_pricer(scheme: str, params: dict):
    model = HestonModel(**params)
    choose_heston_grid_bounds(
        spot=OPTION["spot"],
        strike=OPTION["strike"],
        v0=params["v0"],
        theta=params["theta"],
        n_std_v=5.0 if params["rho"] <= -0.8 else 3.0,
    )
    if params["rho"] <= -0.8 or params["xi"] >= 0.5:
        s_max = 200.0
        v_max = 0.6
    else:
        s_max = 150.0
        v_max = 0.2

    def price_fn(refinement: dict) -> float:
        return model.price_european_option(
            S0=OPTION["spot"],
            K=OPTION["strike"],
            T=OPTION["maturity"],
            option_type=OPTION["option_type"],
            scheme=scheme,
            N_S=refinement["N_S"],
            N_v=refinement["N_v"],
            N_t=refinement["N_t"],
            S_max=s_max,
            v_max=v_max,
        )

    return price_fn


def build_merton_pricer(params: dict):
    model = MertonJumpDiffusionModel(MertonJumpParams(**params))
    contract = MertonEuropeanOption(
        K=OPTION["strike"],
        T=OPTION["maturity"],
        option_type=OPTION["option_type"],
    )
    s_min, s_max = choose_lognormal_grid_bounds(
        spot=OPTION["spot"],
        strike=OPTION["strike"],
        r=params["r"],
        q=params["q"],
        sigma=params["sigma"],
        maturity=OPTION["maturity"],
        n_std=5.0 if params["lambda_jump"] >= 0.5 else 4.0,
    )
    s_max = max(s_max, 3.0 * OPTION["spot"])

    def price_fn(refinement: dict) -> float:
        return model.price(
            contract=contract,
            spot=OPTION["spot"],
            scheme="imex_euler",
            S_min=s_min,
            S_max=s_max,
            N_S=refinement["N_S"],
            N_t=refinement["N_t"],
        )

    return price_fn


def build_studies() -> list[ConvergenceStudy]:
    studies: list[ConvergenceStudy] = []

    bsm_reference = black_scholes_closed_form(
        spot=OPTION["spot"],
        strike=OPTION["strike"],
        maturity=OPTION["maturity"],
        r=BSM_PARAMS["r"],
        q=BSM_PARAMS["q"],
        sigma=BSM_PARAMS["sigma"],
        option_type=OPTION["option_type"],
    )

    studies.append(
        ConvergenceStudy(
            model="black_scholes",
            scheme="explicit",
            objective="convergence",
            scenario="stable_coupled_refinement",
            reference_label="black_scholes_closed_form",
            reference_price=bsm_reference,
            price_fn=build_bsm_pricer("explicit", BSM_PARAMS),
            refinements=BSM_EXPLICIT_STABLE_REFINEMENTS,
            primary_resolution_key="N_S",
            resolution_label="N_S",
            contract_params=OPTION,
            model_params=BSM_PARAMS,
        )
    )

    for scheme in ["implicit", "crank_nicolson"]:
        studies.append(
            ConvergenceStudy(
                model="black_scholes",
                scheme=scheme,
                objective="convergence",
                scenario="spatial_refinement",
                reference_label="black_scholes_closed_form",
                reference_price=bsm_reference,
                price_fn=build_bsm_pricer(scheme, BSM_PARAMS),
                refinements=BSM_SPATIAL_REFINEMENTS,
                primary_resolution_key="N_S",
                resolution_label="N_S",
                contract_params=OPTION,
                model_params=BSM_PARAMS,
            )
        )
        studies.append(
            ConvergenceStudy(
                model="black_scholes",
                scheme=scheme,
                objective="convergence",
                scenario="temporal_refinement",
                reference_label="black_scholes_closed_form",
                reference_price=bsm_reference,
                price_fn=build_bsm_pricer(scheme, BSM_PARAMS),
                refinements=BSM_TEMPORAL_REFINEMENTS,
                primary_resolution_key="N_t",
                resolution_label="N_t",
                contract_params=OPTION,
                model_params=BSM_PARAMS,
            )
        )

    heston_reference = build_heston_pricer("hundsdorfer_verwer", HESTON_PARAMS)(
        {"N_S": 120, "N_v": 90, "N_t": 160}
    )
    heston_stress_reference = build_heston_pricer("hundsdorfer_verwer", HESTON_STRESS_PARAMS)(
        {"N_S": 100, "N_v": 75, "N_t": 140}
    )
    for scheme in ["douglas", "craig_sneyd", "modified_craig_sneyd", "hundsdorfer_verwer"]:
        studies.append(
            ConvergenceStudy(
                model="heston",
                scheme=scheme,
                objective="convergence",
                scenario="spatial_refinement",
                reference_label="heston_fine_grid_hundsdorfer_verwer",
                reference_price=heston_reference,
                price_fn=build_heston_pricer(scheme, HESTON_PARAMS),
                refinements=HESTON_SPATIAL_REFINEMENTS,
                primary_resolution_key="N_S",
                resolution_label="N_S",
                contract_params=OPTION,
                model_params=HESTON_PARAMS,
            )
        )
        studies.append(
            ConvergenceStudy(
                model="heston",
                scheme=scheme,
                objective="convergence",
                scenario="temporal_refinement",
                reference_label="heston_fine_grid_hundsdorfer_verwer",
                reference_price=heston_reference,
                price_fn=build_heston_pricer(scheme, HESTON_PARAMS),
                refinements=HESTON_TEMPORAL_REFINEMENTS,
                primary_resolution_key="N_t",
                resolution_label="N_t",
                contract_params=OPTION,
                model_params=HESTON_PARAMS,
            )
        )
        studies.append(
            ConvergenceStudy(
                model="heston",
                scheme=scheme,
                objective="stability",
                scenario="stress_rho_minus_0_90_xi_0_60",
                reference_label="heston_stress_fine_grid_hundsdorfer_verwer",
                reference_price=heston_stress_reference,
                price_fn=build_heston_pricer(scheme, HESTON_STRESS_PARAMS),
                refinements=HESTON_STRESS_REFINEMENTS,
                primary_resolution_key="N_S",
                resolution_label="N_S",
                contract_params=OPTION,
                model_params=HESTON_STRESS_PARAMS,
            )
        )

    merton_reference = build_merton_pricer(MERTON_PARAMS)({"N_S": 300, "N_t": 2400})
    merton_stress_reference = build_merton_pricer(MERTON_STRESS_PARAMS)({"N_S": 300, "N_t": 2400})

    studies.append(
        ConvergenceStudy(
            model="merton_jump_diffusion",
            scheme="imex_euler",
            objective="convergence",
            scenario="spatial_refinement",
            reference_label="merton_fine_grid_imex_euler",
            reference_price=merton_reference,
            price_fn=build_merton_pricer(MERTON_PARAMS),
            refinements=MERTON_SPATIAL_REFINEMENTS,
            primary_resolution_key="N_S",
            resolution_label="N_S",
            contract_params=OPTION,
            model_params=MERTON_PARAMS,
        )
    )
    studies.append(
        ConvergenceStudy(
            model="merton_jump_diffusion",
            scheme="imex_euler",
            objective="convergence",
            scenario="temporal_refinement",
            reference_label="merton_fine_grid_imex_euler",
            reference_price=merton_reference,
            price_fn=build_merton_pricer(MERTON_PARAMS),
            refinements=MERTON_TEMPORAL_REFINEMENTS,
            primary_resolution_key="N_t",
            resolution_label="N_t",
            contract_params=OPTION,
            model_params=MERTON_PARAMS,
        )
    )
    studies.append(
        ConvergenceStudy(
            model="merton_jump_diffusion",
            scheme="imex_euler",
            objective="stability",
            scenario="stress_lambda_1_00",
            reference_label="merton_stress_fine_grid_imex_euler",
            reference_price=merton_stress_reference,
            price_fn=build_merton_pricer(MERTON_STRESS_PARAMS),
            refinements=MERTON_STRESS_REFINEMENTS,
            primary_resolution_key="N_S",
            resolution_label="N_S",
            contract_params=OPTION,
            model_params=MERTON_STRESS_PARAMS,
        )
    )

    return studies


def _safe_filename(value: str) -> str:
    return value.replace(" ", "_").replace("/", "_")


def save_convergence_plots(results_df: pd.DataFrame) -> None:
    """Save convergence and stability plots grouped by model/objective/scenario."""
    successful = results_df[results_df["status"] == "success"].copy()
    if successful.empty:
        return

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    group_cols = ["model", "objective", "scenario"]
    for (model, objective, scenario), study_df in successful.groupby(group_cols, dropna=False):
        resolution_label = study_df["resolution_label"].iloc[0]
        stem = "_".join(_safe_filename(str(value)) for value in (model, objective, scenario))

        fig, ax = plt.subplots(figsize=(8, 5))
        for scheme, scheme_df in study_df.groupby("scheme", dropna=False):
            scheme_df = scheme_df.sort_values("resolution_value")
            if objective == "stability":
                ax.plot(
                    scheme_df["resolution_value"],
                    scheme_df["price"],
                    marker="o",
                    linewidth=2,
                    label=scheme,
                )
                ax.set_ylabel("Price")
                ax.set_title(f"{model}: {scenario} price response")
            else:
                ax.plot(
                    scheme_df["resolution_value"],
                    scheme_df["abs_error"],
                    marker="o",
                    linewidth=2,
                    label=scheme,
                )
                ax.set_ylabel("Absolute Error")
                ax.set_yscale("log")
                ax.set_title(f"{model}: {scenario} convergence")
            ax.set_xlabel(resolution_label)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / f"{stem}_primary.png", dpi=200)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 5))
        for scheme, scheme_df in study_df.groupby("scheme", dropna=False):
            scheme_df = scheme_df.sort_values("runtime_sec")
            ax.plot(
                scheme_df["runtime_sec"],
                scheme_df["abs_error"],
                marker="o",
                linewidth=2,
                label=scheme,
            )
        ax.set_title(f"{model}: {scenario} error vs runtime")
        ax.set_xlabel("Runtime (sec)")
        ax.set_ylabel("Absolute Error")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / f"{stem}_error_vs_runtime.png", dpi=200)
        plt.close(fig)


def main() -> None:
    studies = build_studies()
    results_df = run_convergence_suite(studies)
    summary_df = summarize_convergence_results(results_df)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(RESULTS_PATH, index=False)
    summary_df.to_csv(SUMMARY_PATH, index=False)
    save_convergence_plots(results_df)

    successful = int((results_df["status"] == "success").sum()) if "status" in results_df.columns else 0

    print(f"Saved convergence results to: {RESULTS_PATH}")
    print(f"Saved convergence summary to: {SUMMARY_PATH}")
    print(f"Saved convergence plots to: {PLOTS_DIR}")
    print(f"Studies run: {len(studies)}")
    print(f"Rows written: {len(results_df)}")
    print(f"Successful rows: {successful}")

    if not summary_df.empty:
        print("\nSummary by study:")
        for _, row in summary_df.iterrows():
            print(
                f"model: {row['model']}, "
                f"objective: {row['objective']}, "
                f"scenario: {row['scenario']}, "
                f"scheme: {row['scheme']}, "
                f"assessment: {row['assessment']}, "
                f"initial_abs_error: {row['initial_abs_error']:.6g}, "
                f"final_abs_error: {row['final_abs_error']:.6g}, "
                f"avg_order: {row['average_observed_order']:.4g}, "
                f"failed_levels: {int(row['failed_levels'])}"
            )


if __name__ == "__main__":
    main()
