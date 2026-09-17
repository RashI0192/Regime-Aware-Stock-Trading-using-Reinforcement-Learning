"""Interactive Streamlit research and paper-trading dashboard."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from deployment_pipeline import ALL_FEATURES, build_features
from finance_data import fetch_daily_ohlcv
from local_prediction import load_model
from paper_trading import record_signal, recent_trades

st.set_page_config(page_title="Research Model Dashboard", layout="wide")
st.title("GOOGL Research Model Dashboard")
st.caption("Research and paper-trading interface — not financial advice. No real trades are placed.")

@st.cache_data(ttl=900, show_spinner=False)
def load_prices(symbol, period):
    return fetch_daily_ohlcv(symbol, period)

def add_indicators(data):
    out = data.copy()
    out["SMA 20"] = out.Close.rolling(20).mean()
    out["SMA 50"] = out.Close.rolling(50).mean()
    delta = out.Close.diff(); gain = delta.clip(lower=0).rolling(14).mean(); loss = -delta.clip(upper=0).rolling(14).mean()
    out["RSI 14"] = 100 - 100 / (1 + gain / loss)
    return out

def get_history(data, model, threshold):
    valid = build_features(data)[ALL_FEATURES].dropna()
    result = pd.DataFrame({"Probability up": model.predict_proba(valid)[:, 1]}, index=valid.index)
    result["Signal"] = np.where(result["Probability up"] >= threshold, "LONG", "CASH")
    return result

with st.sidebar:
    st.header("Controls")
    symbol = st.text_input("Primary symbol", "GOOGL").upper().strip()
    compare = st.text_input("Compare symbols (comma-separated)", "AAPL,MSFT")
    period = st.selectbox("History", ["1y", "2y", "5y"], index=1)
    threshold = st.slider("LONG decision threshold", 0.50, 0.80, 0.53, 0.01)
    fee = st.number_input("Backtest fee (%)", 0.0, 2.0, 0.10, 0.01) / 100
    show_sma = st.checkbox("Show moving averages", True)
    show_signals = st.checkbox("Show signal markers", True)
    if st.button("Refresh market data"):
        load_prices.clear(); st.rerun()

try:
    model, meta = load_model(); data = load_prices(symbol, period); table = add_indicators(data)
    history = get_history(data, model, threshold); latest_prob = float(history.iloc[-1]["Probability up"]); latest_signal = history.iloc[-1]["Signal"]
    a, b, c, d = st.columns(4)
    a.metric("Latest close", f"{data.Close.iloc[-1]:.2f}"); b.metric("Probability up", f"{latest_prob:.1%}"); c.metric("Signal", latest_signal); d.metric("Data as of", str(data.index[-1].date()))
    st.info(f"Threshold: {threshold:.0%} (saved default: {meta['decision_threshold']:.0%}) · Model: {meta['model_version']}")
    chart_tab, compare_tab, backtest_tab, paper_tab = st.tabs(["Charts", "Compare", "Backtest", "Paper trading"])

    with chart_tab:
        chart = go.Figure(go.Scatter(x=table.index, y=table.Close, name="Close"))
        if show_sma:
            chart.add_trace(go.Scatter(x=table.index, y=table["SMA 20"], name="SMA 20")); chart.add_trace(go.Scatter(x=table.index, y=table["SMA 50"], name="SMA 50"))
        if show_signals:
            longs = history[history.Signal == "LONG"]
            chart.add_trace(go.Scatter(x=longs.index, y=table.loc[longs.index, "Close"], name="LONG", mode="markers", marker=dict(color="green", size=9, symbol="triangle-up")))
        chart.update_layout(height=480, hovermode="x unified", xaxis_rangeslider_visible=True); st.plotly_chart(chart, use_container_width=True)
        rsi = go.Figure(go.Scatter(x=table.index, y=table["RSI 14"], name="RSI 14")); rsi.add_hline(y=70, line_dash="dot"); rsi.add_hline(y=30, line_dash="dot"); rsi.update_layout(height=220, yaxis_range=[0, 100]); st.plotly_chart(rsi, use_container_width=True)
        st.download_button("Download prices and indicators", table.to_csv().encode(), f"{symbol}_prices.csv", "text/csv"); st.download_button("Download signal history", history.to_csv().encode(), f"{symbol}_signals.csv", "text/csv")

    with compare_tab:
        symbols = [symbol] + [x.strip().upper() for x in compare.split(",") if x.strip() and x.strip().upper() != symbol]; prices = pd.DataFrame()
        for item in symbols:
            try: prices[item] = load_prices(item, period).Close
            except Exception as exc: st.warning(f"Could not load {item}: {exc}")
        if not prices.empty:
            st.line_chart(prices / prices.iloc[0] * 100, y_label="Indexed price (start = 100)"); st.dataframe(prices.tail(1).T.rename(columns={prices.index[-1]: "Latest close"}))

    with backtest_tab:
        bt = data.loc[history.index].copy(); bt["probability_up"] = history["Probability up"]; bt["signal"] = history.Signal
        daily_return = bt.Close.pct_change().fillna(0)
        bt["strategy_return"] = np.where(bt.signal == "LONG", daily_return, 0.0) - np.where(bt.signal == "LONG", fee, 0); bt["buy_hold_return"] = daily_return
        bt["strategy_value"] = (1 + bt.strategy_return).cumprod(); bt["buy_hold_value"] = (1 + bt.buy_hold_return).cumprod()
        x, y, z = st.columns(3); x.metric("Strategy return", f"{bt.strategy_value.iloc[-1]-1:.1%}"); y.metric("Buy & hold", f"{bt.buy_hold_value.iloc[-1]-1:.1%}"); z.metric("LONG days", f"{(bt.signal == 'LONG').sum():,}")
        st.line_chart(bt[["strategy_value", "buy_hold_value"]], y_label="Growth of $1"); st.download_button("Download backtest", bt.to_csv().encode(), f"{symbol}_backtest.csv", "text/csv")

    with paper_tab:
        st.warning("Paper trading only — signals are saved locally and never sent to a broker.")
        if st.button(f"Record {latest_signal} paper signal for {symbol}"):
            trade_id = record_signal(symbol, float(data.Close.iloc[-1]), latest_signal, latest_prob, meta["model_version"]); st.success(f"Recorded paper signal #{trade_id}")
        st.dataframe(pd.DataFrame(recent_trades()), use_container_width=True)
except Exception as exc:
    st.error(f"Unable to load {symbol}: {exc}")
