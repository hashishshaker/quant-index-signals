import streamlit as st
import yfinance as yf
import pandas as pd
import altair as alt

st.set_page_config(page_title="Trend & Bottom Reversal Analyzer", layout="wide")

# --- HEADER & METHODOLOGY SECTION ---
st.title("Trend & Multi-Stage Reversal Analyzer")

st.markdown("""
This tool combines **Volatility (Bollinger Bands)**, **Volume Capitulation**, and **Momentum (RSI & MACD Histogram)** to spot market bottoms early, alongside **Moving Averages (20 EMA, 50 SMA, 200 SMA)** to ride confirmed uptrends.
""")

with st.expander("How the Buy / Sell Signal Process Works", expanded=True):
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("**1. Exact Low (Bollinger + MACD + Vol)**")
        st.caption(
            "Before price even reaches the 20 EMA, the model checks if the stock pierced its **Lower Bollinger Band** "
            "with oversold **RSI (<38)** and is now bouncing with a rising **MACD Histogram** or **high volume (>1.3x avg)**. "
            "This triggers a **STAGE 1 BOTTOM BUY** targeting the 20 EMA."
        )

    with col2:
        st.markdown("**2. Early Recovery (20 EMA to 50 SMA)**")
        st.caption(
            "Once price climbs back above the **20-Day EMA** after an oversold dip and RSI rises above 45, the stock "
            "enters the mean-reversion pocket. This triggers a **STAGE 2 EARLY BUY** to capture the move up to the "
            "**50-Day SMA**."
        )
        
    with col3:
        st.markdown("**3. Confirmed Trend (50 & 200 SMA)**")
        st.caption(
            "When price and the **50-Day SMA** are above the **200-Day SMA**, the macro uptrend is active. Holding "
            "above the 20 EMA triggers a **BUY**, while dipping between the 20 EMA and 50 SMA triggers a "
            "**BUY ON PULLBACK**."
        )
        
    with col4:
        st.markdown("**4. Anti-Chasing Filter (12% Rule)**")
        st.caption(
            "To prevent buying at exhausted peaks, if the price stretches more than **12% above its 50-Day SMA**, "
            "the model overrides the buy signal to **HOLD / WAIT**. Falling below all key averages without reversal "
            "confirmation triggers **SELL / AVOID**."
        )

st.divider()

# --- DATA FETCHING & SIGNAL LOGIC ---
@st.cache_data(ttl=3600, show_spinner="Fetching latest market data...")
def analyze_trend(ticker: str):
    df = yf.download(ticker, period="18mo", interval="1d", auto_adjust=True, progress=False)
    
    if df.empty or len(df) < 200:
        return None, "Insufficient data (requires at least 200 trading days)."

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # 1. Moving Averages
    df["EMA_20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["SMA_50"] = df["Close"].rolling(window=50).mean()
    df["SMA_200"] = df["Close"].rolling(window=200).mean()

    # 2. Bollinger Bands (20-day SMA +/- 2 Standard Deviations)
    sma_20 = df["Close"].rolling(window=20).mean()
    std_20 = df["Close"].rolling(window=20).std()
    df["BB_Lower"] = sma_20 - (2 * std_20)

    # 3. 14-Day RSI (Wilder's Smoothing)
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss
    df["RSI_14"] = 100 - (100 / (1 + rs))

    # 4. MACD (12, 26, 9) Histogram
    ema_12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["Close"].ewm(span=26, adjust=False).mean()
    macd_line = ema_12 - ema_26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = macd_line - signal_line

    # 5. Relative Volume (vs 20-Day Average)
    df["Vol_Avg_20"] = df["Volume"].rolling(window=20).mean()
    df["Vol_Ratio"] = df["Volume"] / df["Vol_Avg_20"]

    latest = df.iloc[-1]
    prev = df.iloc[-2]

    price = float(latest["Close"])
    prev_price = float(prev["Close"])
    ema20 = float(latest["EMA_20"])
    sma50 = float(latest["SMA_50"])
    sma200 = float(latest["SMA_200"])
    bb_lower = float(latest["BB_Lower"])
    rsi14 = float(latest["RSI_14"])
    macd_hist = float(latest["MACD_Hist"])
    prev_macd_hist = float(prev["MACD_Hist"])
    vol_ratio = float(latest["Vol_Ratio"]) if pd.notna(latest["Vol_Ratio"]) else 1.0

    # Bottom Detection Conditions (Lookback 5-10 days)
    recent_min_rsi = float(df["RSI_14"].tail(10).min())
    touched_bb_lower = bool((df["Low"].tail(5) <= df["BB_Lower"].tail(5) * 1.01).any())
    macd_ticking_up = macd_hist > prev_macd_hist
    volume_surge = vol_ratio >= 1.3
    green_day_rebound = (price > prev_price) and (price >= bb_lower)

    # Distance Calculations
    dist_from_sma50 = ((price - sma50) / sma50) * 100
    upside_to_ema20 = ((ema20 - price) / price) * 100
    upside_to_sma50 = ((sma50 - price) / price) * 100
    golden_cross_recent = (sma50 > sma200) and (float(prev["SMA_50"]) <= float(prev["SMA_200"]))

    # Multi-Stage Decision Matrix
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
            if price >= ema20 and rsi14 >= 45:
                signal = f"STAGE 2 EARLY BUY — Reclaimed 20 EMA (+{upside_to_sma50:.1f}% room to 50 SMA)"
            elif touched_bb_lower and recent_min_rsi < 38 and green_day_rebound and (macd_ticking_up or volume_surge):
                signal = f"STAGE 1 BOTTOM BUY — Oversold Bounce (+{upside_to_ema20:.1f}% to 20 EMA, +{upside_to_sma50:.1f}% to 50 SMA)"
            else:
                signal = "HOLD / NEUTRAL — Pulling Back Below 50-Day SMA"
    elif price >= sma50 and sma50 <= sma200:
        signal = "BUY / WATCH — Bullish Rebound Above 50-Day SMA (Targeting 200 SMA)"
    else:
        # Below 50-Day SMA: Evaluate Stage 1 (Exact Bottom) & Stage 2 (20 EMA Reclaim)
        if price >= ema20 and recent_min_rsi < 40 and rsi14 >= 45:
            signal = f"STAGE 2 EARLY BUY (20 EMA Reclaimed) — Targeting 50 SMA (+{upside_to_sma50:.1f}% upside)"
        elif price >= ema20 and rsi14 >= 50:
            signal = f"WATCH / EARLY RECOVERY — Holding 20 EMA (+{upside_to_sma50:.1f}% room to 50 SMA)"
        elif price < ema20 and touched_bb_lower and recent_min_rsi < 38 and green_day_rebound and (macd_ticking_up or volume_surge):
            signal = f"STAGE 1 BOTTOM BUY (Low-Point Reversal) — Bounce off Lower BB (+{upside_to_ema20:.1f}% to 20 EMA)"
        else:
            signal = "SELL / AVOID — Active Downtrend (No Bottom Confirmation Yet)"

    metrics = {
        "Current Price": round(price, 2),
        "Lower Bollinger": round(bb_lower, 2),
        "20-Day EMA": round(ema20, 2),
        "50-Day SMA": round(sma50, 2),
        "14-Day RSI": round(rsi14, 1),
        "Volume vs Avg": f"{vol_ratio:.2f}x"
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
        
        m_cols = st.columns(len(metrics))
        for col, (label, value) in zip(m_cols, metrics.items()):
            col.metric(label=label, value=value)

        chart_df = df[["Close", "BB_Lower", "EMA_20", "SMA_50", "SMA_200"]].tail(250).reset_index()
        chart_df = chart_df.rename(columns={
            "Close": "Close Price",
            "BB_Lower": "Lower Bollinger Band",
            "EMA_20": "20-Day EMA",
            "SMA_50": "50-Day SMA",
            "SMA_200": "200-Day SMA"
        })
        
        melted_df = chart_df.melt("Date", var_name="Indicator", value_name="Price")

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
                        domain=["Close Price", "Lower Bollinger Band", "20-Day EMA", "50-Day SMA", "200-Day SMA"],
                        range=["#2563eb", "#8b5cf6", "#10b981", "#f59e0b", "#ef4444"]
                    ),
                    legend=alt.Legend(orient="top", title=None)
                ),
                strokeDash=alt.condition(
                    alt.datum.Indicator == "Lower Bollinger Band",
                    alt.value([4, 4]),
                    alt.value([0])
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
