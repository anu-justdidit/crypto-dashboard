
# app.py — Crypto Dashboard (bulletproof, Dash-only, no html.Style)
import os
import time
import requests
import numpy as np
import pandas as pd
import yfinance as yf

import dash
from dash import html, dcc, Input, Output
import plotly.graph_objects as go

# ======================
# CONFIG
# ======================
COINS = [
    {"label": "Bitcoin (BTC)", "value": "bitcoin"},
    {"label": "Ethereum (ETH)", "value": "ethereum"},
    {"label": "Solana (SOL)", "value": "solana"},
    {"label": "Cardano (ADA)", "value": "cardano"},
    {"label": "Polkadot (DOT)", "value": "polkadot"},
]

YF_SYMBOLS = {
    "bitcoin": "BTC-USD",
    "ethereum": "ETH-USD",
    "solana": "SOL-USD",
    "cardano": "ADA-USD",
    "polkadot": "DOT-USD",
}

DEFAULT_COIN = "bitcoin"
REFRESH_INTERVAL = 120 * 1000  # 2 minutes
CACHE_DURATION = 300           # 5 minutes
MIN_ROWS_REQUIRED = 20         # validation threshold

# ======================
# SMALL UTILS / CACHES
# ======================
def _now_ts() -> int:
    return int(time.time())

# simple TTL cache for CoinGecko
_cg_cache = {}
def _cg_key(coin_id: str, vs: str, days: int) -> str:
    return f"{coin_id}|{vs}|{days}"

def cg_market_chart(coin_id: str, vs_currency: str, days: int):
    """TTL cached CoinGecko market_chart."""
    key = _cg_key(coin_id, vs_currency, int(days))
    entry = _cg_cache.get(key)
    if entry and _now_ts() - entry["ts"] < CACHE_DURATION:
        return entry["data"]
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {"vs_currency": vs_currency, "days": int(days)}
    try:
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
        _cg_cache[key] = {"ts": _now_ts(), "data": data}
        return data
    except Exception as e:
        print(f"[CG ERROR] {e}")
        return None

def get_yf(symbol: str, period: str) -> pd.DataFrame:
    """yfinance with CSV cache (no parse_dates issues)."""
    os.makedirs("data/raw", exist_ok=True)
    cache_file = f"data/raw/yfinance_{symbol}.csv"

    if os.path.exists(cache_file) and (time.time() - os.path.getmtime(cache_file) < CACHE_DURATION):
        try:
            return pd.read_csv(cache_file, index_col="Date", parse_dates=True)
        except Exception as e:
            print(f"[CACHE READ WARN] {e}")

    try:
        df = yf.download(symbol, period=period, progress=False, auto_adjust=False)
        if not df.empty:
            df.to_csv(cache_file, index_label="Date")
        return df
    except Exception as e:
        print(f"[YF ERROR] {e}")
        return pd.DataFrame()

# ======================
# DATA PIPELINE
# ======================
def validate_df(df: pd.DataFrame) -> bool:
    if df is None or df.empty:
        return False
    if "Close" not in df.columns:
        return False
    return df["Close"].dropna().shape[0] >= MIN_ROWS_REQUIRED

def fetch_from_cg(coin_id: str, days: int) -> pd.DataFrame:
    """Fetch & align CoinGecko data; daily resample for stability."""
    data = cg_market_chart(coin_id, "usd", int(days))
    if not data:
        return pd.DataFrame()
    try:
        # Raw lists
        prices_raw = data.get("prices", [])
        vols_raw = data.get("total_volumes", [])

        # Build DFs
        prices = pd.DataFrame(prices_raw, columns=["timestamp", "price"])
        volumes = pd.DataFrame(vols_raw, columns=["timestamp", "volume"])

        # Align by timestamp (handles mismatched lengths)
        df = pd.merge(prices, volumes, on="timestamp", how="inner")

        # If still mismatched somehow, trim to shortest safely
        min_len = min(len(prices), len(volumes))
        if len(df) == 0 and min_len > 0:
            prices = prices.iloc[:min_len].copy()
            volumes = volumes.iloc[:min_len].copy()
            df = pd.merge(prices, volumes, on="timestamp", how="inner")

        if df.empty:
            return pd.DataFrame()

        df["date"] = pd.to_datetime(df["timestamp"], unit="ms")
        df = df.set_index("date").sort_index()

        # Resample daily → last price, sum volume
        df = df.resample("1D").agg({"price": "last", "volume": "sum"})
        df.rename(columns={"price": "Close", "volume": "Volume"}, inplace=True)
        return df
    except Exception as e:
        print(f"[CG PARSE ERROR] {e}")
        return pd.DataFrame()

def fetch_from_yf(coin_id: str, days: int) -> pd.DataFrame:
    symbol = YF_SYMBOLS.get(coin_id, "BTC-USD")
    period = {30: "1mo", 90: "3mo", 365: "1y"}.get(int(days), "1mo")
    df = get_yf(symbol, period)
    if df is None or df.empty:
        return pd.DataFrame()

    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            pass

    # Keep only needed columns; create Volume if missing
    cols = df.columns
    out = pd.DataFrame(index=df.index)
    out["Close"] = df["Close"] if "Close" in cols else np.nan
    out["Volume"] = df["Volume"] if "Volume" in cols else np.nan
    return out

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "Close" not in df.columns:
        return df.copy()

    out = df.copy()

    # RSI (14) safe
    delta = out["Close"].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(14, min_periods=14).mean()
    avg_loss = loss.rolling(14, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out["RSI"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = out["Close"].ewm(span=12, adjust=False).mean()
    ema26 = out["Close"].ewm(span=26, adjust=False).mean()
    out["MACD"] = ema12 - ema26
    out["Signal"] = out["MACD"].ewm(span=9, adjust=False).mean()

    # MAs
    out["SMA_50"] = out["Close"].rolling(50, min_periods=25).mean()
    out["EMA_20"] = out["Close"].ewm(span=20, adjust=False).mean()

    return out

def compute_kpis(df: pd.DataFrame, coin_id: str, source_used: str):
    closes = df["Close"].dropna() if "Close" in df.columns else pd.Series(dtype=float)
    last_close = closes.iloc[-1] if not closes.empty else np.nan

    if closes.shape[0] >= 2:
        prev_close = closes.iloc[-2]
        change_24h = ((last_close - prev_close) / prev_close) * 100
    else:
        change_24h = np.nan

    # Volume safe
    if "Volume" in df.columns and not df["Volume"].dropna().empty:
        volume = df["Volume"].dropna().iloc[-1]
    else:
        volume = np.nan

    # If source is CG, try live KPIs (guarded)
    try:
        if source_used == "coingecko":
            url = "https://api.coingecko.com/api/v3/simple/price"
            params = {"ids": coin_id, "vs_currencies": "usd", "include_24hr_change": "true"}
            j = requests.get(url, params=params, timeout=10).json().get(coin_id, {})
            if "usd" in j:
                last_close = j["usd"]
            if "usd_24h_change" in j and j["usd_24h_change"] is not None:
                change_24h = j["usd_24h_change"]
    except Exception as e:
        print(f"[KPI WARN] {e}")

    rsi_series = df["RSI"].dropna() if "RSI" in df.columns else pd.Series(dtype=float)
    rsi_last = rsi_series.iloc[-1] if not rsi_series.empty else np.nan

    def kpi_card(title, value, tooltip, color=None):
        return html.Div(
            [
                html.Div(title, className="kpi-title", title=tooltip),
                html.Div(value, className="kpi-value", style={"color": color} if color else None),
            ],
            className="kpi-card",
        )

    color_change = "var(--green)" if (isinstance(change_24h, (int, float)) and not np.isnan(change_24h) and change_24h >= 0) else "var(--red)"
    rsi_color = None
    if isinstance(rsi_last, (int, float)) and not np.isnan(rsi_last):
        rsi_color = "var(--red)" if rsi_last > 70 else ("var(--green)" if rsi_last < 30 else None)

    cards = [
        kpi_card("Price", f"${last_close:,.2f}" if pd.notna(last_close) else "N/A", "Current price"),
        kpi_card("24h Change", f"{change_24h:+.2f}%" if pd.notna(change_24h) else "N/A", "24-hour change", color_change if pd.notna(change_24h) else None),
        kpi_card("Volume", f"${volume:,.0f}" if pd.notna(volume) else "N/A", "Latest period volume"),
        kpi_card("RSI", f"{rsi_last:.1f}" if pd.notna(rsi_last) else "N/A", "Relative Strength Index", rsi_color),
        html.Div(
            [html.Span("Data Source:", className="kpi-title"), html.Span(source_used.upper(), className="kpi-value")],
            className="kpi-card",
        ),
    ]
    return cards

# ======================
# DASH APP
# ======================
app = dash.Dash(__name__, title="Crypto Analytics Pro")
server = app.server

app.layout = html.Div(
    [
        html.Div(
            [html.H1("📊 Crypto Analytics Dashboard", className="header-title"),
             html.P("Live market tracking with automatic data-source fallback", className="header-subtitle")],
            className="header",
        ),
        html.Div(
            [
                dcc.Dropdown(id="coin-selector", options=COINS, value=DEFAULT_COIN, clearable=False, className="dropdown"),
                dcc.RadioItems(
                    id="data-source",
                    options=[
                        {"label": " CoinGecko", "value": "coingecko"},
                        {"label": " Yahoo Finance", "value": "yfinance"},
                        {"label": " Auto (fallback)", "value": "auto"},
                    ],
                    value="auto",
                    inline=True,
                    className="radio-items",
                ),
                dcc.Dropdown(
                    id="timeframe",
                    options=[
                        {"label": "1 Month", "value": "30"},
                        {"label": "3 Months", "value": "90"},
                        {"label": "1 Year", "value": "365"},
                    ],
                    value="90",
                    clearable=False,
                    className="dropdown",
                ),
                dcc.Checklist(
                    id="indicators",
                    options=[
                        {"label": " RSI", "value": "RSI"},
                        {"label": " MACD", "value": "MACD"},
                        {"label": " Moving Averages", "value": "MA"},
                    ],
                    value=["RSI", "MACD"],
                    inline=True,
                    className="checklist",
                ),
            ],
            className="controls",
        ),
        html.Div(id="kpi-cards", className="kpi-container"),
        dcc.Graph(id="price-chart", className="chart"),
        dcc.Graph(id="indicator-chart", className="chart"),
        dcc.Store(id="data-store"),
        dcc.Store(id="source-used"),
        dcc.Interval(id="refresh-interval", interval=REFRESH_INTERVAL),
    ],
    className="page",
)

# ======================
# CALLBACKS
# ======================
@app.callback(
    Output("data-store", "data"),
    Output("kpi-cards", "children"),
    Output("source-used", "data"),
    Input("coin-selector", "value"),
    Input("data-source", "value"),
    Input("timeframe", "value"),
    Input("refresh-interval", "n_intervals"),
)
def update_data(coin_id, data_source, days, _):
    try:
        days = int(days) if days is not None else 90

        # Decide fetch order
        if data_source == "auto":
            order = ["coingecko", "yfinance"]
        elif data_source == "coingecko":
            order = ["coingecko", "yfinance"]
        else:
            order = ["yfinance", "coingecko"]

        used = None
        df = pd.DataFrame()

        for src in order:
            if src == "coingecko":
                df = fetch_from_cg(coin_id, days)
            else:
                df = fetch_from_yf(coin_id, days)

            if validate_df(df):
                used = src
                break

        if used is None:
            raise ValueError("No usable data from either source after validation.")

        df = add_indicators(df)

        kpis = compute_kpis(df, coin_id, used)
        return df.to_json(date_format="iso"), kpis, used

    except Exception as e:
        print(f"[update_data ERROR] {e}")
        def error_card(title, value, tooltip, color="var(--red)"):
            return html.Div(
                [html.Div(title, className="kpi-title", title=tooltip),
                 html.Div(value, className="kpi-value", style={"color": color})],
                className="kpi-card",
            )
        return None, [error_card("Error", str(e)[:160], "Data fetch failed")], None

@app.callback(
    Output("price-chart", "figure"),
    Output("indicator-chart", "figure"),
    Input("data-store", "data"),
    Input("indicators", "value"),
)
def update_charts(json_data, indicators):
    if not json_data:
        return go.Figure(), go.Figure()
    try:
        df = pd.read_json(json_data)

        # Ensure datetime index
        if "index" in df.columns:
            df.index = pd.to_datetime(df["index"])
        else:
            df.index = pd.to_datetime(df.index)

        # Price figure
        price_fig = go.Figure()
        price_fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Price", line=dict(color="#64b5f6")))
        if "MA" in indicators:
            if "SMA_50" in df.columns:
                price_fig.add_trace(go.Scatter(x=df.index, y=df["SMA_50"], name="50D SMA", line=dict(dash="dot")))
            if "EMA_20" in df.columns:
                price_fig.add_trace(go.Scatter(x=df.index, y=df["EMA_20"], name="20D EMA", line=dict(dash="dash")))
        price_fig.update_layout(title="Price", xaxis_title="Date", yaxis_title="USD", hovermode="x unified", template="plotly_white")

        # Indicators figure
        indicator_fig = go.Figure()
        if "RSI" in indicators and "RSI" in df.columns:
            indicator_fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI", line=dict(color="#ffb74d")))
            indicator_fig.add_hline(y=70, line_dash="dot", line_color="red")
            indicator_fig.add_hline(y=30, line_dash="dot", line_color="green")
        if "MACD" in indicators and "MACD" in df.columns and "Signal" in df.columns:
            indicator_fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD", line=dict(color="#66bb6a")))
            indicator_fig.add_trace(go.Scatter(x=df.index, y=df["Signal"], name="Signal", line=dict(color="#ab63fa")))
            hist = (df["MACD"] - df["Signal"]).fillna(0)
            indicator_fig.add_trace(go.Bar(x=df.index, y=hist, name="MACD Hist", opacity=0.25))
        indicator_fig.update_layout(title="Technical Indicators", xaxis_title="Date", hovermode="x unified", template="plotly_white")

        return price_fig, indicator_fig

    except Exception as e:
        print(f"[update_charts ERROR] {e}")
        return go.Figure(), go.Figure()

# ======================
# RUN
# ======================
if __name__ == "__main__":
    os.makedirs("data/raw", exist_ok=True)
    app.run(debug=True, port=8050)

