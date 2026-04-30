from config.config import Config
import wrds
import pandas as pd
import numpy as np
import yaml
from datetime import datetime
from pathlib import Path

class DataDownloader:
    def __init__(self):
        self.config = Config()
        if not self.config.SECURITIES:
            raise ValueError("Config.SECURITIES must contain at least one (secid, name) pair")
        self.db = wrds.Connection(wrds_username=self.config.get_wrds_username())
        self._secid_list = [s[0] for s in self.config.SECURITIES]
        self._secid_to_name = {s[0]: s[1] for s in self.config.SECURITIES}

    def _secid_in_clause(self) -> str:
        return ", ".join(str(sid) for sid in self._secid_list)

    def download_options_data(self):
        option_query = f"""
        SELECT
            secid,
            date, exdate, strike_price/1000 as strike_price,
            impl_volatility, best_bid, best_offer,
            cp_flag, volume, open_interest,
            delta, gamma, theta, vega
        FROM optionm.opprcd2025
        WHERE secid IN ({self._secid_in_clause()})
            AND date BETWEEN '{self.config.START_DATE}' AND '{self.config.END_DATE}'
            AND (exdate - date) BETWEEN {self.config.DAYS_TO_EXPIRY_MIN} AND {self.config.DAYS_TO_EXPIRY_MAX}
            AND volume >= {self.config.SQL_MIN_VOLUME}
            AND impl_volatility BETWEEN {self.config.SQL_MIN_IMPLIED_VOLATILITY} AND {self.config.SQL_MAX_IMPLIED_VOLATILITY}
        """
        options_data = self.db.raw_sql(option_query)
        options_data["security_name"] = options_data["secid"].map(self._secid_to_name)
        options_data = (
            options_data
            .sort_values("volume", ascending=False)
            .drop_duplicates(subset=["secid", "date", "exdate", "strike_price", "cp_flag"])
            .sort_values(["secid", "date", "exdate", "strike_price", "cp_flag"])
            .reset_index(drop=True)
        )
        options_data.to_csv('data/options_metrics_raw/raw_options_data.csv', index=False)
        return options_data

    def download_spot_data(self):
        spot_query = f"""
        SELECT secid, date, close as spot_price
        FROM optionm.secprd2025
        WHERE secid IN ({self._secid_in_clause()})
            AND date BETWEEN '{self.config.START_DATE}' AND '{self.config.END_DATE}'
        """
        spot_data = self.db.raw_sql(spot_query)
        spot_data["security_name"] = spot_data["secid"].map(self._secid_to_name)
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

        df = options_data.merge(spot_data, on=["secid", "date"], how="left", suffixes=("", "_spot"))
        if "security_name_spot" in df.columns:
            df = df.drop(columns=["security_name_spot"])
        df = df.merge(rate_data, on="date", how="left")

        df["mid_price"] = (df["best_bid"] + df["best_offer"]) / 2
        df["spread"] = df["best_offer"] - df["best_bid"]
        df["spread_pct"] = df["spread"] / df["mid_price"]
        df["date"] = pd.to_datetime(df["date"])
        df["exdate"] = pd.to_datetime(df["exdate"])
        df["tau (time to maturity)"] = (df["exdate"] - df["date"]).dt.days / 365

        # Moneyness as log-moneyness for symmetry
        df["log_moneyness"] = np.log(df["strike_price"] / df["spot_price"])

        # Apply per-security filters
        filtered_groups = []
        for security_name, group in df.groupby("security_name"):
            f = self.config.get_filters(security_name)
            mask = (
                (group["spread_pct"] <= f["max_spread_pct"]) &
                (group["volume"] >= f["min_volume"]) &
                (group["impl_volatility"] >= f["min_implied_volatility"]) &
                (group["impl_volatility"] <= f["max_implied_volatility"]) &
                (group["mid_price"] >= f["min_option_price"]) &
                (group["log_moneyness"].abs() <= f["max_abs_log_moneyness"])
            )
            filtered_groups.append(group[mask])

        df_processed = pd.concat(filtered_groups, ignore_index=True)
        df_processed.to_csv("data/options_metrics_processed/clean_options_data.csv", index=False)
        return df_processed