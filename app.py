import io
import re
import urllib.request
import streamlit as st
import yfinance as yf
import pandas as pd
import altair as alt

st.set_page_config(page_title="Multi-Ticker Trend & Reversal Screener", layout="wide")

DEFAULT_CSV_URL = "https://raw.githubusercontent.com/hashishshaker/quant-index-signals/refs/heads/main/input_tickers.csv"

# --- HEADER & METHODOLOGY SECTION ---
st.title("Multi-Ticker Trend & Reversal Screener")

st.markdown("""
Screens your watchlist using **Volatility (Bollinger Bands)**, **Volume Capitulation**, **Momentum (RSI & MACD)**, and **Moving Averages (20 EMA, 50 SMA, 200 SMA)**, automatically resolving human-readable names for every symbol.
""")

with st.expander("How the Buy / Sell Signal Process Works", expanded=False):
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("**1. Exact Low (Stage 1 Bottom)**")
        st.caption(
            "Below the 20 EMA, if price pierced the **Lower Bollinger Band** with oversold **RSI (<38)** and bounces "
            "with a rising **MACD Histogram** or **high volume (>1.3x avg)**, it triggers a **STAGE 1 BOTTOM BUY** "
            "targeting the 20 EMA."
        )

    with col2:
        st.markdown("**2. Early Recovery (Stage 2)**")
        st.caption(
            "Once price reclaims the **20-Day EMA** after an oversold dip and RSI rises above 45, it triggers a "
            "**STAGE 2 EARLY BUY** to capture the climb up to the **50-Day SMA**."
        )
        
    with col3:
        st.markdown("**3. Confirmed Trend (50 & 200 SMA)**")
        st.caption(
            "When price and the **50-Day SMA** are above the **200-Day SMA**, holding above the 20 EMA triggers "
            "**BUY**, while dipping between the 20 EMA and 50 SMA triggers **BUY ON PULLBACK**."
        )
        
    with col4:
        st.markdown("**4. Anti-Chasing & Risk Filter**")
        st.caption(
            "If price stretches more than **12% above its 50-Day SMA**, the signal switches to **HOLD / WAIT**. "
            "Trading below key averages without reversal confirmation triggers **SELL / AVOID**."
        )

st.divider()

# --- HELPER FUNCTIONS: CSV PARSING & NAME LOOKUP ---
@st.cache_data(ttl=900, show_spinner=False)
def fetch_remote_csv(url: str) -> str:
    with urllib.request.urlopen(url, timeout=10) as response:
        return response.read().decode("utf-8")

def parse_tickers_and_names(raw_text: str) -> dict[str, str]:
    """
    Supports both:
    1) Headerless comma/newline lists: ^NSEI, ^NSMIDCP, HDFCGOLD.NS
    2) Structured CSVs with headers: Ticker,Name
    Returns {TICKER: custom_name_or_empty_string}
    """
    ticker_map: dict[str, str] = {}
    cleaned = raw_text.strip()
    if not cleaned:
        return ticker_map

    first_line = cleaned.splitlines()[0].lower()
    has_header = any(h in first_line for h in ("ticker", "symbol"))

    if has_header:
        df_in = pd.read_csv(io.StringIO(cleaned))
        cols_lower = {c.strip().lower(): c for c in df_in.columns}
        ticker_col = next((cols_lower[k] for k in ("ticker", "symbol", "tickers", "symbols") if k in cols_lower), df_in.columns[0])
        name_col = next((cols_lower[k] for k in ("name", "company", "index", "description", "label") if k in cols_lower), None)

        for _, row in df_in.iterrows():
            t = str(row[ticker_col]).strip().upper()
            if t and t != "NAN":
                custom_name = str(row[name_col]).strip() if (name_col and pd.notna(row[name_col])) else ""
                ticker_map[t] = custom_name
    else:
        for token in re.split(r"[,\s\n]+", cleaned):
            t = token.strip().upper()
            if t:
                ticker_map[t] = ""

    return ticker_map

@st.cache_data(ttl=86400, show_spinner=False)
def fetch_yahoo_name(ticker: str) -> str:
    """Fetches shortName/longName from Yahoo Finance and caches it for 24 hours."""
    try:
        info = yf.Ticker(ticker).info
        if isinstance(info, dict):
            return info.get("shortName") or info.get("longName") or ticker
    except Exception:
        pass
    return ticker

# --- DATA FETCHING & SIGNAL LOGIC ---
@st.cache_data(ttl=3600, show_spinner=False)
def analyze_ticker(ticker: str, custom_name: str = ""):
    display_name = custom_name if custom_name else fetch_yahoo_name(ticker)

    try:
        df = yf.download(ticker, period="18mo", interval="1d", auto_adjust=True, progress=False)
    except Exception as e:
        return None, {"Ticker": ticker, "Name": display_name, "Action": "ERROR", "Details": f"Download failed: {e}"}
    
    if df.empty or len(df) < 200:
        return None, {"Ticker": ticker, "Name": display_name, "Action": "ERROR", "Details": "Insufficient data (<200 trading days) or invalid symbol."}

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # 1. Moving Averages
    df["EMA_20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["SMA_50"] = df["Close"].rolling(window=50).mean()
    df["SMA_200"] = df["Close"].rolling(window=200).mean()

    # 2. Bollinger Bands (20-day SMA +/- 2 Std Dev)
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

    # Bottom Detection Conditions
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
            action = "STRONG BUY"
            details = "Fresh Golden Cross (50 SMA crossed above 200 SMA)"
        elif price >= ema20 and dist_from_sma50 <= 12.0:
            action = "BUY"
            details = "Confirmed uptrend in healthy range above 20 EMA"
        elif price < ema20 and price >= sma50:
            action = "BUY ON PULLBACK"
            details = "Uptrend intact; testing 50-Day SMA support"
        elif dist_from_sma50 > 12.0:
            action = "HOLD / WAIT"
            details = f"Overextended (+{dist_from_sma50:.1f}% above 50 SMA)"
        else:
            if price >= ema20 and rsi14 >= 45:
                action = "STAGE 2 EARLY BUY"
                details = f"Reclaimed 20 EMA (+{upside_to_sma50:.1f}% room to 50 SMA)"
            elif touched_bb_lower and recent_min_rsi < 38 and green_day_rebound and (macd_ticking_up or volume_surge):
                action = "STAGE 1 BOTTOM BUY"
                details = f"Bounce off Lower BB (+{upside_to_ema20:.1f}% to 20 EMA)"
            else:
                action = "HOLD / NEUTRAL"
                details = "Pulling back below 50-Day SMA"
    elif price >= sma50 and sma50 <= sma200:
        action = "WATCH / ACCUMULATE"
        details = "Bullish rebound above 50 SMA (below 200 SMA)"
    else:
        if price >= ema20 and recent_min_rsi < 40 and rsi14 >= 45:
            action = "STAGE 2 EARLY BUY"
            details = f"Reclaimed 20 EMA (+{upside_to_sma50:.1f}% room to 50 SMA)"
        elif price >= ema20 and rsi14 >= 50:
            action = "WATCH"
            details = f"Holding 20 EMA (+{upside_to_sma50:.1f}% room to 50 SMA)"
        elif price < ema20 and touched_bb_lower and recent_min_rsi < 38 and green_day_rebound and (macd_ticking_up or volume_surge):
            action = "STAGE 1 BOTTOM BUY"
            details = f"Low-point bounce off Lower BB (+{upside_to_ema20:.1f}% to 20 EMA)"
        else:
            action = "SELL / AVOID"
            details = "Active downtrend below key moving averages"

    summary_row = {
        "Ticker": ticker,
        "Name": display_name,
        "Action": action,
        "Details": details,
        "Price": round(price, 2),
        "20 EMA": round(ema20, 2),
        "50 SMA": round(sma50, 2),
        "200 SMA": round(sma200, 2),
        "vs 50 SMA": f"{dist_from_sma50:+.1f}%",
        "14D RSI": round(rsi14, 1),
        "Vol Ratio": f"{vol_ratio:.2f}x"
    }
    return df, summary_row

# --- INPUT SECTION ---
csv_url = st.text_input("GitHub Raw CSV URL:", value=DEFAULT_CSV_URL)

# Pre-load tickers from GitHub URL on first load
default_text = ""
if csv_url.strip():
    try:
        default_text = fetch_remote_csv(csv_url.strip())
    except Exception as e:
        st.warning(f"Could not fetch URL automatically ({e}). You can still paste tickers or upload a CSV below.")

input_col1, input_col2 = st.columns([2, 1])

with input_col1:
    raw_tickers = st.text_area(
        "Watchlist Tickers (auto-loaded from URL above, or edit directly):",
        value=default_text,
        height=110
    )

with input_col2:
    uploaded_file = st.file_uploader("Or upload a local CSV / TXT file", type=["csv", "txt"])

# Merge text area and uploaded file into {Ticker: CustomName}
ticker_map = parse_tickers_and_names(raw_tickers)
if uploaded_file is not None:
    file_text = uploaded_file.read().decode("utf-8")
    ticker_map.update(parse_tickers_and_names(file_text))

# Run Screener
if st.button("Screen Watchlist", type="primary") and ticker_map:
    results = []
    valid_dfs = {}
    resolved_names = {}
    
    total = len(ticker_map)
    progress_bar = st.progress(0, text="Fetching names & analyzing tickers...")
    for idx, (t, custom_name) in enumerate(ticker_map.items()):
        df, row = analyze_ticker(t, custom_name)
        results.append(row)
        resolved_names[t] = row["Name"]
        if df is not None:
            valid_dfs[t] = df
        progress_bar.progress((idx + 1) / total, text=f"Analyzed {t} — {row['Name']} ({idx + 1}/{total})")
    progress_bar.empty()

    st.session_state["results_df"] = pd.DataFrame(results)
    st.session_state["valid_dfs"] = valid_dfs
    st.session_state["resolved_names"] = resolved_names

# --- DISPLAY RESULTS TABLE & DRILL-DOWN CHART ---
if "results_df" in st.session_state:
    results_df = st.session_state["results_df"]
    valid_dfs = st.session_state["valid_dfs"]
    resolved_names = st.session_state.get("resolved_names", {})

    st.subheader(f"Screener Results ({len(results_df)} Tickers)")

    filter_col, dl_col = st.columns([3, 1])
    with filter_col:
        action_filter = st.radio(
            "Filter by Verdict:",
            options=["All", "Buy Signals Only", "Hold / Watch", "Sell / Avoid"],
            horizontal=True
        )

    filtered_df = results_df.copy()
    if action_filter == "Buy Signals Only":
        filtered_df = filtered_df[filtered_df["Action"].str.contains("BUY", na=False)]
    elif action_filter == "Hold / Watch":
        filtered_df = filtered_df[filtered_df["Action"].str.contains("HOLD|WATCH", na=False)]
    elif action_filter == "Sell / Avoid":
        filtered_df = filtered_df[filtered_df["Action"].str.contains("SELL|AVOID", na=False)]

    with dl_col:
        csv_data = filtered_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Table (CSV)",
            data=csv_data,
            file_name="ticker_signals.csv",
            mime="text/csv"
        )

    def style_action(val: str) -> str:
        if "BUY" in val:
            return "background-color: rgba(16, 185, 129, 0.2); color: #059669; font-weight: bold;"
        elif "SELL" in val or "AVOID" in val:
            return "background-color: rgba(239, 68, 68, 0.2); color: #dc2626; font-weight: bold;"
        elif "HOLD" in val or "WATCH" in val:
            return "background-color: rgba(245, 158, 11, 0.2); color: #d97706; font-weight: bold;"
        return ""

    styled_table = filtered_df.style.map(style_action, subset=["Action"])
    st.dataframe(styled_table, use_container_width=True, hide_index=True)

    # --- INTERACTIVE CHART INSPECTOR ---
    if valid_dfs:
        st.divider()
        st.subheader("Chart Inspector")
        selected_ticker = st.selectbox(
            "Select a ticker from your watchlist to inspect its chart:",
            options=list(valid_dfs.keys()),
            format_func=lambda t: f"{t} — {resolved_names.get(t, t)}"
        )

        if selected_ticker:
            df = valid_dfs[selected_ticker]
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
                .properties(height=420)
                .interactive()
            )

            st.altair_chart(chart, use_container_width=True)
