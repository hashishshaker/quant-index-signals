import streamlit as st
import yfinance as yf
import pandas as pd
import altair as alt

st.set_page_config(page_title="Trend Signal Analyzer", layout="wide")

# --- HEADER & METHODOLOGY SECTION ---
st.title("Trend-Following Signal Analyzer")

st.markdown("""
This tool evaluates stocks using institutional moving average crossovers and extension rules to determine whether a ticker is in a healthy buy zone, overextended, or in a downtrend.
""")

with st.expander("How the Buy / Sell Signal Process Works", expanded=True):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**1. Macro Trend Filter (200-Day SMA)**")
        st.caption(
            "First, the model checks if both the **Current Price** and the **50-Day SMA** are above the "
            "**200-Day SMA** (a post-Golden Cross state). If the price is below both averages, the stock is in a "
            "macro downtrend and triggers a **SELL / AVOID** signal."
        )
        
    with col2:
        st.markdown("**2. Tactical Trigger (20 EMA & 50 SMA)**")
        st.caption(
            "In a confirmed uptrend, the model checks short-term momentum. Trading above the **20-Day EMA** signals "
            "active strength (**BUY**), while dipping below the 20-Day EMA but holding above the **50-Day SMA** "
            "identifies a lower-risk dip (**BUY ON PULLBACK**)."
        )
        
    with col3:
        st.markdown("**3. Anti-Chasing Filter (12% Rule)**")
        st.caption(
            "Even strong uptrends snap back when stretched too far. If the current price is more than **12% above "
            "its 50-Day SMA**, the system overrides the buy signal to **HOLD / WAIT** so you avoid buying at an "
            "overextended peak."
        )

    st.markdown("""
    **Signal Summary:**
    * **STRONG BUY / BUY:** Price & 50 SMA > 200 SMA, price > 20 EMA, and stretched less than 12% above the 50 SMA.
    * **BUY ON PULLBACK:** Long-term uptrend intact, with price pulling back between the 20 EMA and 50 SMA support zone.
    * **HOLD / WAIT:** Uptrend intact, but price is stretched >12% above the 50 SMA (high pullback risk) or losing short-term momentum.
    * **WATCH:** Price has climbed back above the 50 SMA, but remains below the long-term 200 SMA.
    * **SELL / AVOID:** Price has broken below key moving averages, indicating a bearish trend regime.
    """)

st.divider()

# --- DATA FETCHING & SIGNAL LOGIC ---
@st.cache_data(ttl=3600, show_spinner="Fetching latest market data...")
def analyze_trend(ticker: str):
    df = yf.download(ticker, period="18mo", interval="1d", auto_adjust=True, progress=False)
    
    if df.empty or len(df) < 200:
        return None, "Insufficient data (requires at least 200 trading days)."

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df["EMA_20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["SMA_50"] = df["Close"].rolling(window=50).mean()
    df["SMA_200"] = df["Close"].rolling(window=200).mean()

    latest = df.iloc[-1]
    prev = df.iloc[-2]

    price = float(latest["Close"])
    ema20 = float(latest["EMA_20"])
    sma50 = float(latest["SMA_50"])
    sma200 = float(latest["SMA_200"])

    dist_from_sma50 = ((price - sma50) / sma50) * 100
    golden_cross_recent = (sma50 > sma200) and (float(prev["SMA_50"]) <= float(prev["SMA_200"]))

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

# --- USER INPUT & OUTPUT DISPLAY ---
ticker_input = st.text_input("Enter Ticker (e.g., AAPL, MSFT, RELIANCE.NS):", "AAPL").upper().strip()

if st.button("Analyze Ticker", type="primary") and ticker_input:
    df, result = analyze_trend(ticker_input)
    if df is None:
        st.error(result)
    else:
        signal, metrics = result
        st.subheader(f"Verdict for {ticker_input}: {signal}")
        
        # Display key metrics in clean horizontal cards
        m_cols = st.columns(len(metrics))
        for col, (label, value) in zip(m_cols, metrics.items()):
            col.metric(label=label, value=value)

        # Prepare last 250 trading days for dynamic scaling chart
        chart_df = df[["Close", "EMA_20", "SMA_50", "SMA_200"]].tail(250).reset_index()
        chart_df = chart_df.rename(columns={
            "Close": "Close Price",
            "EMA_20": "20-Day EMA",
            "SMA_50": "50-Day SMA",
            "SMA_200": "200-Day SMA"
        })
        
        melted_df = chart_df.melt("Date", var_name="Indicator", value_name="Price")

        # Dynamic Y-Axis Chart (zero=False scales tightly to the data range)
        chart = (
            alt.Chart(melted_df)
            .mark_line(strokeWidth=2)
            .encode(
                x=alt.X("Date:T", title="Date"),
                y=alt.Y(
                    "Price:Q",
                    title="Price",
                    scale=alt.Scale(zero=False, padding=15)
                ),
                color=alt.Color(
                    "Indicator:N",
                    scale=alt.Scale(
                        domain=["Close Price", "20-Day EMA", "50-Day SMA", "200-Day SMA"],
                        range=["#2563eb", "#10b981", "#f59e0b", "#ef4444"]
                    ),
                    legend=alt.Legend(orient="top", title=None)
                ),
                tooltip=[
                    alt.Tooltip("Date:T", title="Date", format="%Y-%m-%d"),
                    alt.Tooltip("Indicator:N", title="Line"),
                    alt.Tooltip("Price:Q", title="Value", format=",.2f")
                ]
            )
            .properties(height=450)
            .interactive()
        )

        st.altair_chart(chart, use_container_width=True)
