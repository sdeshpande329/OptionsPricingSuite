# Options Pricing Suite

## Project By: Pascal Bermeo Neumann, Sarang Deshpande, Michael Waltuch

Our goal is to develop a comprehensive option pricing and calibration suite that implements and compares multiple numerical PDE methods across different stochastic models.


## Startup Instructions:
1. Create a virtual environment (instructions may differ based on your operating system).
```bash
python -m venv venv
```
or 
```sh
python3 -m venv venv
```
2. Make sure virtual environment is active, then update pip and install required packages.
```sh
pip install --upgrade pip && pip install -r requirements.txt
```
3. To pull data, you will need an account with Wharton Research Data Services (WRDS). For convenience, we have already pulled the data and stored it in the repository.
4. To calibrate Heston and Jump-Diffusion Models, the current calibrated parameter csvs need to be removed. These can be found in the data/results/model_calibration folder.
```sh
rm -f data/results/model_calibration/heston_calibrated_parameters_*.csv \
      data/results/model_calibration/heston_price_comparison_*.csv \
      data/results/model_calibration/merton_jump_calibration_results_*.csv
```
5. Run main orchestration, which will price securities in parallel and end with convergence analysis. Make sure your current directory is the project root directory.
```sh
python main.py
```
or 
```sh
python3 main.py
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
│       ├── convergence_analysis
│       │   ├── convergence_analysis_results.csv
│       │   └── convergence_summary_results.csv
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
│       ├── model_calibration
│       │   ├── heston_calibrated_parameters_PLTR.csv
│       │   ├── heston_calibrated_parameters_RUT.csv
│       │   ├── heston_calibrated_parameters_SPX.csv
│       │   ├── heston_calibrated_parameters_TSLA.csv
│       │   ├── heston_price_comparison_PLTR.csv
│       │   ├── heston_price_comparison_RUT.csv
│       │   ├── heston_price_comparison_SPX.csv
│       │   ├── heston_price_comparison_TSLA.csv
│       │   ├── merton_jump_calibration_results_PLTR.csv
│       │   ├── merton_jump_calibration_results_RUT.csv
│       │   ├── merton_jump_calibration_results_SPX.csv
│       │   └── merton_jump_calibration_results_TSLA.csv
│       ├── pricing_results
│       │   ├── pricing_PLTR_black_scholes_crank_nicolson.csv
│       │   ├── pricing_PLTR_black_scholes_explicit.csv
│       │   ├── pricing_PLTR_black_scholes_implicit.csv
│       │   ├── pricing_PLTR_black_scholes_monte_carlo.csv
│       │   ├── pricing_PLTR_heston_craig_sneyd.csv
│       │   ├── pricing_PLTR_heston_douglas.csv
│       │   ├── pricing_PLTR_heston_hundsdorfer_verwer.csv
│       │   ├── pricing_PLTR_heston_modified_craig_sneyd.csv
│       │   ├── pricing_PLTR_heston_monte_carlo.csv
│       │   ├── pricing_PLTR_merton_imex_euler.csv
│       │   ├── pricing_PLTR_merton_monte_carlo.csv
│       │   ├── pricing_RUT_black_scholes_crank_nicolson.csv
│       │   ├── pricing_RUT_black_scholes_explicit.csv
│       │   ├── pricing_RUT_black_scholes_implicit.csv
│       │   ├── pricing_RUT_black_scholes_monte_carlo.csv
│       │   ├── pricing_RUT_heston_craig_sneyd.csv
│       │   ├── pricing_RUT_heston_douglas.csv
│       │   ├── pricing_RUT_heston_hundsdorfer_verwer.csv
│       │   ├── pricing_RUT_heston_modified_craig_sneyd.csv
│       │   ├── pricing_RUT_heston_monte_carlo.csv
│       │   ├── pricing_RUT_merton_imex_euler.csv
│       │   ├── pricing_RUT_merton_monte_carlo.csv
│       │   ├── pricing_SPX_black_scholes_crank_nicolson.csv
│       │   ├── pricing_SPX_black_scholes_explicit.csv
│       │   ├── pricing_SPX_black_scholes_implicit.csv
│       │   ├── pricing_SPX_black_scholes_monte_carlo.csv
│       │   ├── pricing_SPX_heston_craig_sneyd.csv
│       │   ├── pricing_SPX_heston_douglas.csv
│       │   ├── pricing_SPX_heston_hundsdorfer_verwer.csv
│       │   ├── pricing_SPX_heston_modified_craig_sneyd.csv
│       │   ├── pricing_SPX_heston_monte_carlo.csv
│       │   ├── pricing_SPX_merton_imex_euler.csv
│       │   ├── pricing_SPX_merton_monte_carlo.csv
│       │   ├── pricing_TSLA_black_scholes_crank_nicolson.csv
│       │   ├── pricing_TSLA_black_scholes_explicit.csv
│       │   ├── pricing_TSLA_black_scholes_implicit.csv
│       │   ├── pricing_TSLA_black_scholes_monte_carlo.csv
│       │   ├── pricing_TSLA_heston_craig_sneyd.csv
│       │   ├── pricing_TSLA_heston_douglas.csv
│       │   ├── pricing_TSLA_heston_hundsdorfer_verwer.csv
│       │   ├── pricing_TSLA_heston_modified_craig_sneyd.csv
│       │   ├── pricing_TSLA_heston_monte_carlo.csv
│       │   ├── pricing_TSLA_merton_imex_euler.csv
│       │   └── pricing_TSLA_merton_monte_carlo.csv
│       └── pricing_summary.csv
├── docs
│   ├── reference_papers
│   │   ├── A Finite Difference Scheme for Option Pricing in Jump Diffusion and Exponential Lévy Models.pdf
│   │   ├── ADI finite difference schemes for option pricing in the Heston model with correlation.pdf
│   │   └── Calibration of Heston.pdf
│   ├── CSE 6730 Checkpoint 1.pdf
│   ├── CSE 6730 Checkpoint 2.pdf
│   ├── Final_Presentation_Group_12.pdf
│   ├── Final_Project_Group_12.pdf
│   └── Literature Review.pdf
├── notebooks
│   └── data_download.ipynb
├── scripts
│   ├── calibrate_heston_params.py
│   ├── calibrate_merton_jump_params.py
│   ├── download_data.py
│   ├── run_black_scholes_pricing.py
│   ├── run_convergence_test.py
│   ├── run_greeks.py
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
│   ├── __init__.py
│   └── parallel_processing.py
├── main.py
└── requirements.txt
```

AI has been used thus far in the project for the purpose of ideating (specifically conducting feasibility analysis for ideas the group had come up with) and for enhancing the readability of the literature review, checkpoints, and final report. While coding, AI tools such as Claude, ChatGPT, and Cursor were used for debugging and adding documentation.