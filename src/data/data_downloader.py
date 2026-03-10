from config.config import Config
import wrds  # pyright: ignore[reportMissingImports]
import pandas as pd
import numpy as np
import yaml
from datetime import datetime
from pathlib import Path

class DataDownloader:
    def __init__(self):
        self.config = Config()
        self.db = wrds.Connection(wrds_username=self.config.get_wrds_username())
        self.security_id = self.config.SECURITY_ID
        self.security_name = self.config.SECURITY_NAME

    def download_options_data(self):
        option_query = f"""
        SELECT
            date, exdate, strike_price/1000 as strike_price,
            impl_volatility, best_bid, best_offer,
            cp_flag, volume, open_interest,
            delta, gamma, theta, vega
        FROM optionm.opprcd2025
        WHERE secid = {self.security_id}
            AND date BETWEEN '{self.config.START_DATE}' AND '{self.config.END_DATE}'
            AND (exdate - date) BETWEEN {self.config.DAYS_TO_EXPIRY_MIN} AND {self.config.DAYS_TO_EXPIRY_MAX}
            AND volume >= {self.config.MIN_VOLUME}
            AND impl_volatility BETWEEN {self.config.MIN_IMPLIED_VOLATILITY} AND {self.config.MAX_IMPLIED_VOLATILITY}
        """
        options_data = self.db.raw_sql(option_query)
        options_data.to_csv('data/options_metrics_raw/raw_options_data.csv', index=False)
        return options_data

    def download_spot_data(self):
        spot_query = f"""
        SELECT date, close as spot_price
        FROM optionm.secprd2025
        WHERE secid = {self.security_id}
            AND date BETWEEN '{self.config.START_DATE}' AND '{self.config.END_DATE}'
        """
        spot_data = self.db.raw_sql(spot_query)
        spot_data.to_csv('data/options_metrics_raw/raw_spot_data.csv', index=False)
        return spot_data
    
    def download_rate_data(self):
        rate_query = f"""
        SELECT *
        FROM optionm.zerocd
        WHERE date BETWEEN '{self.config.START_DATE}' AND '{self.config.END_DATE}'
            AND days BETWEEN {self.config.DAYS_TO_EXPIRY_MIN} AND {self.config.DAYS_TO_EXPIRY_MAX}
        """
        rate_data = self.db.raw_sql(rate_query)
        rate_data.to_csv('data/options_metrics_raw/raw_rate_data.csv', index=False)
        return rate_data

    def clean_data(self) -> pd.DataFrame:
        options_data = self.download_options_data()
        spot_data = self.download_spot_data()
        rate_data = self.download_rate_data()
        df = options_data.merge(spot_data, on='date', how='left')
        df = df.merge(rate_data, on='date', how='left')
        df['mid_price'] = (df['best_bid'] + df['best_offer']) / 2
        df['spread'] = (df['best_offer'] - df['best_bid'])
        df['spread_pct'] = df['spread'] / df['mid_price']
        df['moneyness'] = df['strike_price'] / df['spot_price']
        df['date'] = pd.to_datetime(df['date'])
        df['exdate'] = pd.to_datetime(df['exdate'])
        df['tau (time to maturity)'] = (df['exdate'] - df['date']).dt.days / 365
        df_processed = df[
            (df['spread_pct'] <= self.config.MAX_SPREAD_PCT) &
            (df['volume'] >= self.config.MIN_VOLUME) &
            (df['impl_volatility'] >= self.config.MIN_IMPLIED_VOLATILITY) &
            (df['impl_volatility'] <= self.config.MAX_IMPLIED_VOLATILITY) &
            (df['moneyness'] <= self.config.MAX_MONEYNESS)
        ]
        df_processed.to_csv('data/options_metrics_processed/clean_options_data.csv', index=False)
        return df_processed