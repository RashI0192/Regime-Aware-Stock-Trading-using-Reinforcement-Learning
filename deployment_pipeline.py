"""Reusable feature and model definitions extracted from the ML/RL study.
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ALL_FEATURES = [
    "return_lag1", "return_lag2", "return_lag3", "return_lag5",
    "momentum5", "momentum10", "momentum20", "trend_gap",
    "price_sma20", "macd_pct", "rsi", "volatility5", "volatility20",
    "vol_ratio", "bb_width", "range_pct", "intraday", "overnight",
    "relative_volume", "log_volume_change", "signed_volume",
]

COMPACT_FEATURES = [
    "return_lag1", "momentum5", "trend_gap", "rsi", "volatility20",
    "range_pct", "relative_volume",
]


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Build predictors using information available before the next open.
    """
    p = frame["Close"]
    r = p.pct_change(fill_method=None)
    f = pd.DataFrame(index=frame.index)
    for lag in [0, 1, 2, 4]:
        f[f"return_lag{lag + 1}"] = r.shift(lag)
    for window in [5, 10, 20]:
        f[f"momentum{window}"] = p.pct_change(window, fill_method=None)
    sma20 = p.rolling(20).mean()
    sma50 = p.rolling(50).mean()
    f["trend_gap"] = sma20 / sma50 - 1
    f["price_sma20"] = p / sma20 - 1
    f["macd_pct"] = (
        p.ewm(span=12, adjust=False, min_periods=12).mean()
        - p.ewm(span=26, adjust=False, min_periods=26).mean()
    ) / p
    delta = p.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    f["rsi"] = 100 - 100 / (1 + gain / loss)
    f.loc[(gain == 0) & (loss == 0), "rsi"] = 50
    f["volatility5"] = r.rolling(5).std()
    f["volatility20"] = r.rolling(20).std()
    f["vol_ratio"] = f.volatility5 / f.volatility20
    f["bb_width"] = 4 * p.rolling(20).std(ddof=0) / sma20
    f["range_pct"] = (frame["High"] - frame["Low"]) / p
    f["intraday"] = p / frame["Open"] - 1
    f["overnight"] = frame["Open"] / p.shift(1) - 1
    f["relative_volume"] = frame["Volume"] / frame["Volume"].shift(1).rolling(20).mean()
    f["log_volume_change"] = np.log(frame["Volume"]).diff()
    f["signed_volume"] = np.sign(r) * f.relative_volume
    return f.shift(1).replace([np.inf, -np.inf], np.nan)


def make_logistic_top8() -> Pipeline:
    """Create the unfitted selected logistic_top8 pipeline."""
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("select", SelectKBest(f_classif, k=8)),
        ("model", LogisticRegression(max_iter=2000, random_state=7)),
    ])


def predict_local(frame: pd.DataFrame, model: Pipeline, threshold: float = 0.53) -> dict:
    """Predict the next open-to-open direction from an OHLCV frame.

    The frame must contain Open, High, Low, Close and Volume, in chronological
    order.
    """
    features = build_features(frame)
    row = features.iloc[[-1]][ALL_FEATURES]
    if row.isna().any().any():
        missing = row.columns[row.isna().iloc[0]].tolist()
        raise ValueError(f"Insufficient history or missing features: {missing}")
    probability = float(model.predict_proba(row)[:, 1][0])
    return {
        "probability_up": probability,
        "signal": "LONG" if probability >= threshold else "CASH",
        "threshold": float(threshold),
    }
