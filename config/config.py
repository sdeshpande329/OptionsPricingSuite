import yaml

class Config:
    """Configurations for the Options Pricing Suite"""

    # Data Download Parameters
    SECURITIES = [
        (108105, "SPX"),
        (102434, "RUT"),
        (143439, "TSLA"),
        (214684, "PLTR"),
    ]

    START_DATE = "2025-07-01"
    END_DATE = "2025-08-29"

    DAYS_TO_EXPIRY_MIN = 30
    DAYS_TO_EXPIRY_MAX = 60

    # Global fallback filters (used if no per-security override exists)
    DEFAULT_FILTERS = {
        "max_spread_pct": 0.10,
        "min_volume": 1000,
        "min_implied_volatility": 0.05,
        "max_implied_volatility": 0.60,
        "min_option_price": 0.01,
        "max_abs_log_moneyness": 0.30,  # |ln(K/S)| <= 0.30
    }

    # Per-security overrides -- only specify what differs from DEFAULT_FILTERS
    SECURITY_FILTERS = {
        "SPX": {
            "max_spread_pct": 0.10,
            "min_volume": 1000,
            "min_implied_volatility": 0.05,
            "max_implied_volatility": 0.60,
            "min_option_price": 0.01,
            "max_abs_log_moneyness": 0.30,  # |ln(K/S)| <= 0.30
        },
        "RUT": {
            "max_spread_pct": 0.10,
            "min_volume": 1000,
            "min_implied_volatility": 0.05,
            "max_implied_volatility": 0.60,
            "min_option_price": 0.01,
            "max_abs_log_moneyness": 0.30,  # |ln(K/S)| <= 0.30
        },  
        "TSLA": {
            "max_spread_pct": 0.20,
            "min_volume": 200,
            "max_implied_volatility": 1.50,
        },
        "PLTR": {
            "max_spread_pct": 0.25,
            "min_volume": 100,
            "max_implied_volatility": 2.00,
        },
    }

    # SQL pre-filter bounds (loosest values across all securities)
    # These are intentionally permissive -- per-security tightening happens in Python
    SQL_MIN_VOLUME = 100
    SQL_MIN_IMPLIED_VOLATILITY = 0.05
    SQL_MAX_IMPLIED_VOLATILITY = 2.00

    @classmethod
    def get_filters(cls, security_name: str) -> dict:
        """Return the merged filter dict for a given security name."""
        overrides = cls.SECURITY_FILTERS.get(security_name, {})
        return {**cls.DEFAULT_FILTERS, **overrides}

    # Output Parameters
    OUTPUT_DIR = "data/options_metrics_raw"
    SAVE_CSV = True
    SAVE_PLOT = True
    SAVE_SUMMARY = True

    @staticmethod
    def load_credentials(credentials_path: str = "config/credentials.yaml") -> dict:
        with open(credentials_path, "r") as file:
            return yaml.safe_load(file)

    @classmethod
    def get_wrds_username(cls, credentials_path: str = "config/credentials.yaml") -> str:
        credentials = cls.load_credentials(credentials_path)
        return credentials["wrds"]["username"]