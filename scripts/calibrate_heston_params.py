"""
Calibrate Heston model parameters to market implied volatility surface.

Uses Levenberg-Marquardt optimization to minimize squared differences
between model and market option prices.

Based on:
- Cui, del Baño Rollin & Germano (2017). Full and fast calibration of 
  the Heston stochastic volatility model.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.heston import HestonModel

DATA_PATH = REPO_ROOT / "data" / "options_metrics_processed" / "clean_options_data.csv"
RESULTS_DIR = REPO_ROOT / "data" / "results"


def infer_option_type(cp_flag: str) -> str:
    """Convert OptionMetrics cp_flag to 'call' or 'put'."""
    cp = str(cp_flag).strip().upper()
    if cp == "C":
        return "call"
    if cp == "P":
        return "put"
    raise ValueError(f"Unrecognized cp_flag: {cp_flag}")


class HestonCalibrator:
    """
    Calibrate Heston model parameters to market data.
    
    Optimizes parameters {kappa, theta, xi, rho, v0} to minimize:
        sum_i w_i * (V_model^i - V_market^i)^2
    """
    
    def __init__(self, r: float, q: float = 0.0,
                 scheme: str = 'craig_sneyd',
                 grid_params: Optional[Dict] = None):
        """
        Initialize calibrator.
        
        Parameters:
        -----------
        r : float
            Risk-free rate (fixed, not calibrated)
        q : float
            Dividend yield (fixed, not calibrated)
        scheme : str
            ADI scheme to use for pricing
        grid_params : dict, optional
            Grid parameters for PDE solver
        """
        self.r = r
        self.q = q
        self.scheme = scheme
        self.grid_params = grid_params or {
            'N_S': 60,   # Coarser for speed during calibration
            'N_v': 30,
            'N_t': 30
        }
        
        self.market_data = None
        self.S0 = None
        
    def load_and_prepare_data(self, 
                             data_path: str,
                             max_rows: Optional[int] = None,
                             option_type: str = 'call',
                             min_moneyness: float = 0.85,
                             max_moneyness: float = 1.15,
                             min_tau: float = 0.05,
                             max_tau: float = 0.5) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load and prepare market data for calibration.
        
        Parameters:
        -----------
        data_path : str
            Path to market data CSV
        max_rows : int, optional
            Maximum rows to load (for testing)
        option_type : str
            'call' or 'put'
        min_moneyness, max_moneyness : float
            Filter by K/S0 range
        min_tau, max_tau : float
            Filter by maturity range (years)
        
        Returns:
        --------
        S0, strikes, maturities, market_prices, weights : arrays
        """
        # Load data
        df = pd.read_csv(data_path)
        
        print(f"Loaded {len(df)} rows from data")
        
        if max_rows is not None:
            df = df.head(max_rows)
            print(f"Using first {len(df)} rows")
        
        # Get spot price (use first row)
        S0 = float(df['spot_price'].iloc[0])
        print(f"Spot price: ${S0:.2f}")
        
        # Filter by option type
        if 'cp_flag' in df.columns:
            flag = 'C' if option_type == 'call' else 'P'
            df = df[df['cp_flag'] == flag]
            print(f"After filtering for {option_type}s: {len(df)} rows")
        
        # Check moneyness distribution BEFORE filtering
        df['moneyness'] = df['strike_price'] / S0
        print(f"\nMoneyness distribution:")
        print(f"  Min: {df['moneyness'].min():.3f}")
        print(f"  25%: {df['moneyness'].quantile(0.25):.3f}")
        print(f"  50%: {df['moneyness'].quantile(0.50):.3f}")
        print(f"  75%: {df['moneyness'].quantile(0.75):.3f}")
        print(f"  Max: {df['moneyness'].max():.3f}")
        
        # Check maturity distribution BEFORE filtering
        print(f"\nMaturity distribution (years):")
        print(f"  Min: {df['tau (time to maturity)'].min():.4f}")
        print(f"  25%: {df['tau (time to maturity)'].quantile(0.25):.4f}")
        print(f"  50%: {df['tau (time to maturity)'].quantile(0.50):.4f}")
        print(f"  75%: {df['tau (time to maturity)'].quantile(0.75):.4f}")
        print(f"  Max: {df['tau (time to maturity)'].max():.4f}")
        
        # Filter by moneyness
        df = df[(df['moneyness'] >= min_moneyness) & 
                (df['moneyness'] <= max_moneyness)]
        print(f"\nAfter moneyness filter [{min_moneyness}, {max_moneyness}]: {len(df)} rows")
        
        # Filter by maturity
        df = df[(df['tau (time to maturity)'] >= min_tau) & 
                (df['tau (time to maturity)'] <= max_tau)]
        print(f"After maturity filter [{min_tau}, {max_tau}]: {len(df)} rows")
        
        if len(df) == 0:
            raise ValueError("No options remain after filtering! Relax your filter criteria.")
        
        # Remove duplicates (same strike and maturity)
        df = df.drop_duplicates(subset=['strike_price', 'tau (time to maturity)'])
        print(f"After removing duplicates: {len(df)} rows")
        
        # Extract arrays
        strikes = df['strike_price'].values
        maturities = df['tau (time to maturity)'].values
        market_prices = df['mid_price'].values
        
        # Compute weights (inverse of price for normalization)
        weights = 1.0 / (market_prices + 1.0)
        weights = weights / weights.sum()
        
        # Store for later
        self.market_data = df
        self.S0 = S0
        
        print(f"\n" + "=" * 70)
        print(f"CALIBRATION DATASET SUMMARY")
        print(f"=" * 70)
        print(f"  Spot price: ${S0:.2f}")
        print(f"  Number of options: {len(strikes)}")
        print(f"  Strike range: [{strikes.min():.2f}, {strikes.max():.2f}]")
        print(f"  Moneyness range: [{(strikes/S0).min():.3f}, {(strikes/S0).max():.3f}]")
        print(f"  Maturity range: [{maturities.min():.4f}, {maturities.max():.4f}] years")
        print(f"  Price range: [{market_prices.min():.4f}, {market_prices.max():.4f}]")
        print()
        
        return S0, strikes, maturities, market_prices, weights
    
    def objective_function(self,
                          params: np.ndarray,
                          S0: float,
                          strikes: np.ndarray,
                          maturities: np.ndarray,
                          market_prices: np.ndarray,
                          weights: np.ndarray,
                          option_type: str = 'call') -> np.ndarray:
        """
        Objective function for optimization.
        
        Parameters:
        -----------
        params : np.ndarray
            [kappa, theta, xi, rho, v0]
        S0 : float
            Spot price
        strikes, maturities, market_prices, weights : np.ndarray
            Market data
        option_type : str
            'call' or 'put'
        
        Returns:
        --------
        np.ndarray
            Weighted residuals
        """
        kappa, theta, xi, rho, v0 = params
        
        # Check constraints
        if not self._check_constraints(kappa, theta, xi, rho, v0):
            return 1e6 * np.ones(len(market_prices))
        
        try:
            # Create model
            model = HestonModel(
                r=self.r, q=self.q,
                kappa=kappa, theta=theta, xi=xi,
                rho=rho, v0=v0
            )
            
            # Price all options
            model_prices = np.zeros(len(strikes))
            for i, (K, T) in enumerate(zip(strikes, maturities)):
                # Choose grid bounds
                S_max = max(3 * S0, 2 * K)
                v_max = max(5 * theta, 5 * v0, 0.5)
                
                model_prices[i] = model.price_european_option(
                    S0=S0,
                    K=K,
                    T=T,
                    option_type=option_type,
                    scheme=self.scheme,
                    S_max=S_max,
                    v_max=v_max,
                    **self.grid_params
                )
            
            # Compute weighted residuals
            residuals = np.sqrt(weights) * (model_prices - market_prices)
            
            return residuals
            
        except Exception as e:
            print(f"  Error in objective: {e}")
            return 1e6 * np.ones(len(market_prices))
    
    def calibrate(self,
                  S0: float,
                  strikes: np.ndarray,
                  maturities: np.ndarray,
                  market_prices: np.ndarray,
                  weights: np.ndarray,
                  option_type: str = 'call',
                  initial_guess: Optional[Dict[str, float]] = None,
                  bounds: Optional[Dict] = None,
                  max_iterations: int = 20,
                  verbose: int = 2) -> Dict:
        """
        Calibrate Heston parameters.
        
        Parameters:
        -----------
        S0 : float
            Spot price
        strikes, maturities, market_prices, weights : np.ndarray
            Market data
        option_type : str
            'call' or 'put'
        initial_guess : dict, optional
            Initial parameter values
        bounds : dict, optional
            Parameter bounds
        max_iterations : int
            Maximum function evaluations
        verbose : int
            Verbosity level (0, 1, 2)
        
        Returns:
        --------
        dict
            Calibration results
        """
        # Default initial guess
        if initial_guess is None:
            # Use implied volatility to estimate initial variance
            avg_iv = self.market_data['impl_volatility'].mean() if self.market_data is not None else 0.20
            initial_guess = {
                'kappa': 2.0,
                'theta': avg_iv**2,
                'xi': 0.3,
                'rho': -0.5,
                'v0': avg_iv**2
            }
        
        # Pack initial guess
        x0 = np.array([
            initial_guess['kappa'],
            initial_guess['theta'],
            initial_guess['xi'],
            initial_guess['rho'],
            initial_guess['v0']
        ])
        
        # Set bounds
        if bounds is None:
            bounds = {
                'kappa': (0.1, 10.0),
                'theta': (0.01, 1.0),
                'xi': (0.01, 2.0),
                'rho': (-0.95, 0.95),
                'v0': (0.01, 1.0)
            }
        
        lower = np.array([bounds['kappa'][0], bounds['theta'][0], 
                         bounds['xi'][0], bounds['rho'][0], bounds['v0'][0]])
        upper = np.array([bounds['kappa'][1], bounds['theta'][1], 
                         bounds['xi'][1], bounds['rho'][1], bounds['v0'][1]])
        
        print("=" * 70)
        print("STARTING HESTON CALIBRATION")
        print("=" * 70)
        print(f"\nInitial guess:")
        print(f"  kappa = {x0[0]:.4f}")
        print(f"  theta = {x0[1]:.4f}  (vol = {np.sqrt(x0[1])*100:.2f}%)")
        print(f"  xi    = {x0[2]:.4f}")
        print(f"  rho   = {x0[3]:.4f}")
        print(f"  v0    = {x0[4]:.4f}  (vol = {np.sqrt(x0[4])*100:.2f}%)")
        print(f"\nOptimization settings:")
        print(f"  Scheme: {self.scheme}")
        print(f"  Grid: {self.grid_params['N_S']}×{self.grid_params['N_v']}×{self.grid_params['N_t']}")
        print(f"  Max iterations: {max_iterations}")
        print()
        
        start_time = time.time()
        
        # Run optimization
        result = least_squares(
            fun=self.objective_function,
            x0=x0,
            bounds=(lower, upper),
            method='trf',
            verbose=verbose,
            args=(S0, strikes, maturities, market_prices, weights, option_type),
            max_nfev=max_iterations,
            ftol=1e-4,
            xtol=1e-4
        )
        
        elapsed = time.time() - start_time
        
        # Extract optimized parameters
        kappa_opt, theta_opt, xi_opt, rho_opt, v0_opt = result.x
        
        print(f"\n" + "=" * 70)
        print("CALIBRATION COMPLETED")
        print("=" * 70)
        print(f"\nOptimized parameters:")
        print(f"  kappa = {kappa_opt:.6f}  (mean reversion speed)")
        print(f"  theta = {theta_opt:.6f}  (long-run vol = {np.sqrt(theta_opt)*100:.2f}%)")
        print(f"  xi    = {xi_opt:.6f}  (vol of vol)")
        print(f"  rho   = {rho_opt:.6f}  (correlation)")
        print(f"  v0    = {v0_opt:.6f}  (initial vol = {np.sqrt(v0_opt)*100:.2f}%)")
        print(f"\nFeller condition: {HestonModel.feller_condition_satisfied(kappa_opt, theta_opt, xi_opt)}")
        print(f"Final cost: {result.cost:.6e}")
        print(f"Iterations: {result.nfev}")
        print(f"Elapsed time: {elapsed:.2f} seconds")
        
        # Compute final model prices
        model_opt = HestonModel(
            r=self.r, q=self.q,
            kappa=kappa_opt, theta=theta_opt, xi=xi_opt,
            rho=rho_opt, v0=v0_opt
        )
        
        print(f"\nComputing final prices...")
        final_prices = np.zeros(len(strikes))
        for i, (K, T) in enumerate(zip(strikes, maturities)):
            S_max = max(3 * S0, 2 * K)
            v_max = max(5 * theta_opt, 5 * v0_opt, 0.5)
            
            final_prices[i] = model_opt.price_european_option(
                S0=S0, K=K, T=T,
                option_type=option_type,
                scheme=self.scheme,
                S_max=S_max, v_max=v_max,
                **self.grid_params
            )
        
        # Compute error metrics
        errors = final_prices - market_prices
        rmse = np.sqrt(np.mean(errors**2))
        mae = np.mean(np.abs(errors))
        mape = np.mean(np.abs(errors / market_prices)) * 100
        
        print(f"\nPricing errors:")
        print(f"  RMSE: ${rmse:.6f}")
        print(f"  MAE:  ${mae:.6f}")
        print(f"  MAPE: {mape:.2f}%")
        
        # Show some comparisons
        print(f"\nSample price comparisons:")
        print(f"{'Strike':<10} {'Maturity':<10} {'Market':<12} {'Model':<12} {'Error':<12}")
        print("-" * 70)
        for i in range(min(10, len(strikes))):
            print(f"{strikes[i]:<10.2f} {maturities[i]:<10.4f} "
                  f"${market_prices[i]:<11.4f} ${final_prices[i]:<11.4f} "
                  f"${errors[i]:<11.4f}")
        
        return {
            'parameters': {
                'kappa': kappa_opt,
                'theta': theta_opt,
                'xi': xi_opt,
                'rho': rho_opt,
                'v0': v0_opt
            },
            'initial_guess': initial_guess,
            'model_prices': final_prices,
            'market_prices': market_prices,
            'errors': errors,
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'elapsed_time': elapsed,
            'success': result.success,
            'n_evaluations': result.nfev,
            'strikes': strikes,
            'maturities': maturities
        }
    
    def _check_constraints(self, kappa: float, theta: float, 
                          xi: float, rho: float, v0: float) -> bool:
        """Check if parameters satisfy constraints."""
        if kappa <= 0 or theta <= 0 or xi <= 0 or v0 <= 0:
            return False
        if rho <= -1 or rho >= 1:
            return False
        # Relaxed Feller condition
        if 2*kappa*theta <= 0.5*xi**2:
            return False
        return True
    
    def save_results(self, results: Dict, output_dir: Path):
        """Save calibration results to CSV files."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save parameters
        params_df = pd.DataFrame([results['parameters']])
        params_df['rmse'] = results['rmse']
        params_df['mae'] = results['mae']
        params_df['mape'] = results['mape']
        params_df['elapsed_time'] = results['elapsed_time']
        params_df['success'] = results['success']
        params_df['n_evaluations'] = results['n_evaluations']
        
        params_file = output_dir / "heston_calibrated_parameters.csv"
        params_df.to_csv(params_file, index=False)
        print(f"\nSaved parameters to: {params_file}")
        
        # Save price comparisons
        comparison_df = pd.DataFrame({
            'strike': results['strikes'],
            'maturity': results['maturities'],
            'market_price': results['market_prices'],
            'model_price': results['model_prices'],
            'error': results['errors'],
            'abs_error': np.abs(results['errors']),
            'pct_error': results['errors'] / results['market_prices'] * 100
        })
        
        comparison_file = output_dir / "heston_price_comparison.csv"
        comparison_df.to_csv(comparison_file, index=False)
        print(f"Saved price comparison to: {comparison_file}")


def main():
    """Run Heston calibration on OptionMetrics data."""
    
    # Check data exists
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Could not find data file: {DATA_PATH}")
    
    # Get risk-free rate (use average from data or set manually)
    df_temp = pd.read_csv(DATA_PATH, nrows=1)
    r = float(df_temp['rate'].iloc[0]) / 100.0
    
    print(f"Using risk-free rate: {r*100:.2f}%\n")
    
    # Initialize calibrator
    calibrator = HestonCalibrator(
        r=r,
        q=0.0,
        scheme='craig_sneyd',
        grid_params={
            'N_S': 80,  # Coarser grid for speed
            'N_v': 40,
            'N_t': 40
        }
    )
    
    # Load and prepare data with RELAXED filters
    S0, strikes, maturities, market_prices, weights = calibrator.load_and_prepare_data(
        data_path=str(DATA_PATH),
        max_rows= None,
        option_type='call',
        min_moneyness=0.92,  # Wider range
        max_moneyness=1.08,
        min_tau=0.08, 
        max_tau=0.5  
    )

    avg_iv = calibrator.market_data['impl_volatility'].mean()
    
    initial_guess = {
        'kappa': 4.0,           # Moderate mean reversion
        'theta': (avg_iv**2) * 0.9,  # Slightly below current IV
        'xi': 0.15,             # Conservative vol-of-vol
        'rho': -0.7,            # Standard leverage effect
        'v0': avg_iv**2         # Current IV level
    }
    
    bounds = {
        'kappa': (2.0, 10.0),   # Strong mean reversion
        'theta': (0.01, 0.40),  # Reasonable variance range
        'xi': (0.05, 0.25),     # Lower max for Feller
        'rho': (-0.90, -0.20),  # Force negative correlation
        'v0': (0.01, 0.40)
    }
    
    # Run calibration
    results = calibrator.calibrate(
        S0=S0,
        strikes=strikes,
        maturities=maturities,
        market_prices=market_prices,
        weights=weights,
        option_type='call',
        initial_guess=initial_guess,
        bounds=bounds,
        max_iterations=30,
        verbose=2
    )
    
    # Save results
    calibrator.save_results(results, RESULTS_DIR)
    
    print("\n" + "=" * 70)
    print("CALIBRATION COMPLETE!")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    results = main()