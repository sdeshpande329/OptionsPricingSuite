import yaml

class Config:
    """Configurations for the Options Pricing Suite"""

    # Data Download Parameters

    SECURITY_ID = 108105
    SECURITY_NAME = "SPX" # S&P 500 Index

    START_DATE = "2025-08-18"
    END_DATE = "2025-08-20"

    DAYS_TO_EXPIRY_MIN = 30
    DAYS_TO_EXPIRY_MAX = 60

    # Data Filtering Parameters

    MAX_SPREAD_PCT = 0.1 # Maximum spread allowed as % of mid price
    MIN_VOLUME = 1000 # Minimum volume required for the option to be considered
    MIN_IMPLIED_VOLATILITY = 0.1 # Minimum implied volatility required for the option to be considered
    MAX_IMPLIED_VOLATILITY = 0.5 # Maximum implied volatility required for the option to be considered
    MIN_OPTION_PRICE = 0.01 # Minimum option price required for the option to be considered
    MAX_MONEYNESS = 0.5 # Maximum moneyness required for the option to be considered (log-moneyness = ln(strike/spot))

    # Output Parameters
    OUTPUT_DIR = 'data/options_metrics_raw'
    SAVE_CSV = True
    SAVE_PLOT = True
    SAVE_SUMMARY = True

    @staticmethod
    def load_credentials(credentials_path: str = 'config/credentials.yaml') -> dict:
        with open(credentials_path, 'r') as file:
            return yaml.safe_load(file)
    
    
    @classmethod
    def get_wrds_username(cls, credentials_path: str = "config/credentials.yaml") -> str:
        credentials = cls.load_credentials(credentials_path)
        return credentials['wrds']['username']