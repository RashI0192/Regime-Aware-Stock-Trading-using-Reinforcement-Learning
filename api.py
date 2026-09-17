"""FastAPI backend for the GOOGL research model."""
from fastapi import FastAPI, HTTPException, Query
from finance_data import fetch_daily_ohlcv
from local_prediction import predict_frame, load_model
from deployment_pipeline import ALL_FEATURES, build_features
from paper_trading import record_signal, recent_trades

app = FastAPI(title="GOOGL Research Model API", version="0.1.0")

@app.get("/health")
def health():
    return {"status": "ok", "service": "research-model-api"}

@app.get("/quote/{symbol}")
def quote(symbol: str, period: str = Query("3mo", pattern=r"^[0-9]+[dmy]$")):
    try:
        data = fetch_daily_ohlcv(symbol.upper(), period); row = data.iloc[-1]
        return {"symbol": symbol.upper(), "date": data.index[-1].isoformat(),
                "open": float(row.Open), "high": float(row.High), "low": float(row.Low),
                "close": float(row.Close), "volume": int(row.Volume)}
    except Exception as exc:
        raise HTTPException(502, f"Market data unavailable: {exc}") from exc

@app.get("/predict/{symbol}")
def predict(symbol: str, period: str = Query("2y", pattern=r"^[0-9]+[dmy]$")):
    try:
        result = predict_frame(fetch_daily_ohlcv(symbol.upper(), period), symbol.upper())
        result["data_source"] = "yfinance"; return result
    except Exception as exc:
        raise HTTPException(502, f"Prediction unavailable: {exc}") from exc

@app.post("/paper-trading/record/{symbol}")
def record_paper_signal(symbol: str, period: str = Query("2y", pattern=r"^[0-9]+[dmy]$")):
    try:
        data = fetch_daily_ohlcv(symbol.upper(), period); result = predict_frame(data, symbol.upper())
        model, meta = load_model(); trade_id = record_signal(symbol.upper(), float(data.Close.iloc[-1]), result["signal"], result["probability_up"], meta["model_version"])
        return {"trade_id": trade_id, **result, "paper_only": True}
    except Exception as exc:
        raise HTTPException(502, f"Paper signal unavailable: {exc}") from exc

@app.get("/paper-trading/trades")
def paper_trades(limit: int = Query(50, ge=1, le=500)):
    return {"trades": recent_trades(limit), "paper_only": True}

@app.get("/backtest/{symbol}")
def backtest(symbol: str, period: str = Query("2y", pattern=r"^[0-9]+[dmy]$")):
    try:
        data = fetch_daily_ohlcv(symbol.upper(), period); model, meta = load_model()
        valid = build_features(data)[ALL_FEATURES].dropna()
        probs = model.predict_proba(valid)[:, 1]
        signals = ["LONG" if p >= meta["decision_threshold"] else "CASH" for p in probs]
        return {"symbol": symbol.upper(), "model": meta["model_version"], "rows": len(valid),
                "signals": [{"date": d.isoformat(), "probability_up": float(p), "signal": s}
                            for d, p, s in zip(valid.index, probs, signals)],
                "note": "Signal history only; no fees, fills, slippage, or portfolio accounting."}
    except Exception as exc:
        raise HTTPException(502, f"Backtest unavailable: {exc}") from exc
