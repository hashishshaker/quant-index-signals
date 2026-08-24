import pandas as pd
import yfinance as yf
import ta
from datetime import datetime
import numpy as np

# ---------------------------------------------------------
# 1. DEFINE YOUR TICKERS HERE
# ---------------------------------------------------------
TICKERS = {
    "^GSPC": "S&P 500"
}
# Add/remove tickers as needed


# ---------------------------------------------------------
# 2. FETCH DATA FOR ONE TICKER
# ---------------------------------------------------------
def fetch_index_data(ticker):
    start_date = datetime.datetime.fromtimestamp(1000000000).strftime("%Y-%m-%d")
    end_date = datetime.datetime.now().strftime("%Y-%m-%d")
    df = yf.download(ticker, start=start_date, end=end_date)
    df = df[['Date', 'Close', 'Volume']]
    return df


# ---------------------------------------------------------
# 3. APPLY STRATEGY TO ONE TICKER
# ---------------------------------------------------------
def generate_daily_signal(df):
    df['SMA200'] = df['Close'].rolling(window=200).mean()
    df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()

    macd = ta.trend.MACD(df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_signal'] = macd.macd_signal()

    bb = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2)
    df['BB_low'] = bb.bollinger_lband()

    df['Avg_Volume'] = df['Volume'].rolling(20).mean()

    latest = df.iloc[-1]

    buy_signal = (
        (latest['Close'] > latest['SMA200']) and
        (latest['RSI'] < 70) and
        (latest['MACD'] > latest['MACD_signal']) and
        (latest['Close'] > latest['BB_low']) and
        (latest['Volume'] > latest['Avg_Volume'])
    )

    return latest['Date'].date(), buy_signal


# ---------------------------------------------------------
# 4. RUN FOR ALL TICKERS AND BUILD TABLE
# ---------------------------------------------------------
def run_all_signals():
    results = []

    for ticker, name in TICKERS.items():
        df = fetch_index_data(ticker)
        date, signal = generate_daily_signal(df)

        results.append({
            "Date": date,
            "Ticker": ticker,
            "Name": name,
            "Signal": "BUY" if signal else "NO BUY"
        })

    return pd.DataFrame(results)


def save_results_to_csv(table, filename="signals_log.csv"):
    try:
        existing = pd.read_csv(filename)
        updated = pd.concat([existing, table], ignore_index=True)
    except FileNotFoundError:
        updated = table

    updated.to_csv(filename, index=False)


# ---------------------------------------------------------
# 5. MAIN EXECUTION
# ---------------------------------------------------------
if __name__ == "__main__":
    table = run_all_signals()
    print(table)
