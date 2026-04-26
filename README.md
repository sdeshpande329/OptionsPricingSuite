# Options Pricing Suite

## Project By: Pascal Bermeo Neumann, Sarang Deshpande, Michael Waltuch

Our goal is to develop a comprehensive option pricing and calibration suite that implements and compares multiple numerical PDE methods across different stochastic models.


## Startup Instructions:
1. Create a virtual environment (instructions may differ based on your operating system)
```bash
python -m venv venv
```
or 
```sh
python3 -m venv venv
```
2. Make sure virtual environment is active, then update pip and install required packages
```
pip install --upgrade pip && pip install -r requirements.txt
```


## Repository Structure:
```sh
OptionsPricingSuite
├── config
│   ├── config.py
│   └── mc_config.py
├── data
│   ├── options_metrics_processed
│   │   └── clean_options_data.csv
│   ├── options_metrics_raw
│   │   ├── raw_options_data.csv
│   │   ├── raw_rate_data.csv
│   │   └── raw_spot_data.csv
│   └── results
│       ├── convergence_plots
│       │   ├── black_scholes_convergence_spatial_refinement_error_vs_runtime.png
│       │   ├── black_scholes_convergence_spatial_refinement_primary.png
│       │   ├── black_scholes_convergence_stable_coupled_refinement_error_vs_runtime.png
│       │   ├── black_scholes_convergence_stable_coupled_refinement_primary.png
│       │   ├── black_scholes_convergence_temporal_refinement_error_vs_runtime.png
│       │   ├── black_scholes_convergence_temporal_refinement_primary.png
│       │   ├── heston_convergence_spatial_refinement_error_vs_runtime.png
│       │   ├── heston_convergence_spatial_refinement_primary.png
│       │   ├── heston_convergence_temporal_refinement_error_vs_runtime.png
│       │   ├── heston_convergence_temporal_refinement_primary.png
│       │   ├── heston_stability_stress_rho_minus_0_90_xi_0_60_error_vs_runtime.png
│       │   ├── heston_stability_stress_rho_minus_0_90_xi_0_60_primary.png
│       │   ├── merton_jump_diffusion_convergence_spatial_refinement_error_vs_runtime.png
│       │   ├── merton_jump_diffusion_convergence_spatial_refinement_primary.png
│       │   ├── merton_jump_diffusion_convergence_temporal_refinement_error_vs_runtime.png
│       │   ├── merton_jump_diffusion_convergence_temporal_refinement_primary.png
│       │   ├── merton_jump_diffusion_stability_stress_lambda_1_00_error_vs_runtime.png
│       │   └── merton_jump_diffusion_stability_stress_lambda_1_00_primary.png
│       ├── black_scholes_finite_difference_PDE_terminal_output.JPG
│       ├── black_scholes_monte_carlo_results.csv
│       ├── black_scholes_pricing_results_crank_nicolson.csv
│       ├── black_scholes_pricing_results_explicit.csv
│       ├── black_scholes_pricing_results_implicit.csv
│       ├── black_scholes_terminal_output.JPG
│       ├── convergence_analysis_results.csv
│       ├── convergence_summary_results.csv
│       ├── heston_calibrated_parameters.csv
│       ├── heston_price_comparison.csv
│       ├── heston_pricing_results_craig-sneyd.csv
│       ├── heston_pricing_results_hv.csv
│       ├── heston_pricing_results_mcs.csv
│       ├── merton_jump_calibration_results.csv
│       └── merton_pide_pricing_results_imex_euler.csv
├── docs
│   ├── reference_papers
│   │   ├── A Finite Difference Scheme for Option Pricing in Jump Diffusion and Exponential Lévy Models.pdf
│   │   ├── ADI finite difference schemes for option pricing in the Heston model with correlation.pdf
│   │   └── Calibration of Heston.pdf
│   ├── CSE 6730 Checkpoint 1.pdf
│   ├── CSE 6730 Checkpoint 2.pdf
│   └── Literature Review.pdf
├── notebooks
│   └── data_download.ipynb
├── scripts
│   ├── calibrate_heston_params.py
│   ├── calibrate_merton_jump_params.py
│   ├── download_data.py
│   ├── run_black_scholes_pricing.py
│   ├── run_convergence_test.py
│   ├── run_heston_pricing.py
│   ├── run_merton_pide_pricing.py
│   └── run_monte_carlo.py
├── src
│   ├── analysis
│   │   ├── __init__.py
│   │   └── convergence_testing.py
│   ├── data
│   │   ├── __init__.py
│   │   └── data_downloader.py
│   ├── models
│   │   ├── __init__.py
│   │   ├── black_scholes.py
│   │   ├── heston.py
│   │   └── merton_jump_diffusion.py
│   ├── monte_carlo
│   │   ├── __init__.py
│   │   ├── mc_black_scholes.py
│   │   ├── mc_heston.py
│   │   ├── mc_merton.py
│   │   └── mc_utils.py
│   ├── numerical
│   │   ├── __init__.py
│   │   ├── adi_schemes.py
│   │   ├── finite_difference.py
│   │   ├── imex_schemes.py
│   │   └── linear_solvers.py
│   └── __init__.py
└── requirements.txt
```

AI has been used thus far in the project for the purpose of ideating (specifically conducting feasibility analysis for ideas the group had come up with) and for enhancing the readability of the literature review and checkpoints. While coding, AI tools such as Claude, ChatGPT, and Cursor were used for debugging and adding documentation.