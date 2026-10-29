import streamlit as st
import yfinance as yf
import pandas as pd

def analyze_trend(ticker: str):
    # Fetch 1.5 years of daily split/dividend-adjusted data
    df = yf.download(ticker, period="18mo", interval="1d", auto_adjust=True, progress=False)
    
    if df.empty or len(df) < 200:
        return None, "Insufficient data (requires at least 200 trading days)."

    # Flatten multi-index columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Calculate Moving Averages
    df["EMA_20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["SMA_50"] = df["Close"].rolling(window=50).mean()
    df["SMA_200"] = df["Close"].rolling(window=200).mean()

    latest = df.iloc[-1]
    prev = df.iloc[-2]

    price = float(latest["Close"])
    ema20 = float(latest["EMA_20"])
    sma50 = float(latest["SMA_50"])
    sma200 = float(latest["SMA_200"])

    # Calculate percentage distance from 50-day SMA
    dist_from_sma50 = ((price - sma50) / sma50) * 100
    golden_cross_recent = (sma50 > sma200) and (float(prev["SMA_50"]) <= float(prev["SMA_200"]))

    # Decision Matrix
    if price > sma200 and sma50 > sma200:
        if golden_cross_recent:
            signal = "STRONG BUY — Fresh Golden Cross"
        elif price >= ema20 and dist_from_sma50 <= 12.0:
            signal = "BUY — Confirmed Uptrend (Healthy Range)"
        elif price < ema20 and price >= sma50:
            signal = "BUY ON PULLBACK — Testing 50-Day SMA Support"
        elif dist_from_sma50 > 12.0:
            signal = f"HOLD / WAIT — Uptrend Overextended (+{dist_from_sma50:.1f}% above 50 SMA)"
        else:
            signal = "HOLD / NEUTRAL — Losing Short-Term Momentum"
    elif price > sma50 and sma50 <= sma200:
        signal = "WATCH — Early Bullish Rebound (Below 200-Day SMA)"
    else:
        signal = "SELL / AVOID — Bearish Regime (Below Key Moving Averages)"

    metrics = {
        "Current Price": round(price, 2),
        "20-Day EMA": round(ema20, 2),
        "50-Day SMA": round(sma50, 2),
        "200-Day SMA": round(sma200, 2),
        "Spread vs 50 SMA": f"{dist_from_sma50:+.2f}%"
    }
    return df, (signal, metrics)

# Streamlit UI
st.title("Trend-Following Signal Analyzer")
ticker_input = st.text_input("Enter Ticker (e.g., AAPL, MSFT, RELIANCE.NS):", "AAPL").upper().strip()

if st.button("Analyze Ticker") and ticker_input:
    df, result = analyze_trend(ticker_input)
    if df is None:
        st.error(result)
    else:
        signal, metrics = result
        st.subheader(f"Verdict: {signal}")
        st.write(metrics)
        st.line_chart(df[["Close", "EMA_20", "SMA_50", "SMA_200"]].tail(250))
