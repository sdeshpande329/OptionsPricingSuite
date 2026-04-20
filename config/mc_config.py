class MonteCarloConfig:
    """Configuration parameters for Monte Carlo simulations."""
    
    # Simulation parameters
    N_SIMULATIONS = 10000
    N_TIME_STEPS = 252
    
    # Random number generation
    RANDOM_SEED = 42
    
    # Variance reduction
    USE_ANTITHETIC = True
    
    # Output
    SAVE_PATHS = False
    SAVE_STATISTICS = True
    
    # Convergence testing
    CONVERGENCE_TEST_SIZES = [1000, 5000, 10000, 20000]
    
    # Numerical stability
    MIN_VARIANCE = 1e-8
    MAX_VARIANCE = 10.0
    
    @classmethod
    def get_config(cls) -> dict:
        """Return configuration as dictionary."""
        return {
            'n_simulations': cls.N_SIMULATIONS,
            'n_time_steps': cls.N_TIME_STEPS,
            'random_seed': cls.RANDOM_SEED,
            'use_antithetic': cls.USE_ANTITHETIC,
            'save_paths': cls.SAVE_PATHS,
            'save_statistics': cls.SAVE_STATISTICS,
            'min_variance': cls.MIN_VARIANCE,
            'max_variance': cls.MAX_VARIANCE
        }
    
    @classmethod
    def update_config(cls, **kwargs):
        """Update configuration parameters."""
        for key, value in kwargs.items():
            key_upper = key.upper()
            if hasattr(cls, key_upper):
                setattr(cls, key_upper, value)
            else:
                raise ValueError(f"Unknown configuration parameter: {key}")