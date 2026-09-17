"""Market-data adapter used by the future API/dashboard."""

import pandas as pd


def fetch_daily_ohlcv(symbol: str = "GOOGL", period: str = "2y") -> pd.DataFrame:
    """Fetch daily OHLCV data and return the project's canonical columns.

    This function needs the optional ``yfinance`` dependency and an internet
    connection. It deliberately returns data only; model decisions stay local.
    """
    import yfinance as yf

    data = yf.download(symbol, period=period, interval="1d", auto_adjust=False, progress=False)
    if data.empty:
        raise ValueError(f"No market data returned for {symbol!r}")
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError(f"Market data missing columns: {missing}")
    result = data[required].copy().dropna()
    result.index = pd.to_datetime(result.index)
    return result.sort_index()
