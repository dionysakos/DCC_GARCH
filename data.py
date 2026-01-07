import numpy as np
import pandas as pd
import yfinance as yf
from typing import List, Tuple

def fetch_prices(tickers: List[str], start_date: str, end_date: str, auto_adjust: bool = True) -> pd.DataFrame:
    px = yf.download( tickers, start=start_date, end=end_date, auto_adjust=auto_adjust,progress=False)["Close"]

    if isinstance(px, pd.Series):
        px = px.to_frame(name=tickers[0])

    px = px.dropna(how="any")
    return px

def log_returns(prices: pd.DataFrame, scale: float = 100.0) -> pd.DataFrame:
    rets = np.log(prices / prices.shift(1)).dropna()
    return rets * scale
