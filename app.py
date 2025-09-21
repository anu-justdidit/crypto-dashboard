"""
app.py — Multi-coin Full-Mode Crypto Dashboard (Dash)
Features:
- Multi-select up to 5 coins
- Full mode: 5 charts per coin (Price+BB, Moving Averages, Volume, RSI, MACD)
- CoinGecko primary, Yahoo Finance fallback
- Lightweight TTL cache + HTTP retries
- Single callback generates per-coin chart sets (easy to maintain)
- Responsive CSS grid layout
"""
import os
import time
import logging
from typing import Dict, Optional

import requests
import numpy as np
import pandas as pd
import yfinance as yf

import dash
from dash import html, dcc, Input, Output, State
import plotly.graph_objects as go

# ---------------------------
# CONFIG
# ---------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("crypto_dashboard")

SESSION = requests.Session()
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
retry = Retry(total=3, backoff_factor=0.4, status_forcelist=[429,502,503,504], allowed_methods=["GET"])
SESSION.mount("https://", HTTPAdapter(max_retries=retry))
SESSION.mount("http://", HTTPAdapter(max_retries=retry))

# Supported coins (CoinGecko id, Yahoo ticker)
COIN_META = {
    "bitcoin":     {"label":"Bitcoin (BTC)",     "cg":"bitcoin",      "yf":"BTC-USD"},
    "ethereum":    {"label":"Ethereum (ETH)",    "cg":"ethereum",     "yf":"ETH-USD"},
    "solana":      {"label":"Solana (SOL)",      "cg":"solana",       "yf":"SOL-USD"},
    "cardano":     {"label":"Cardano (ADA)",     "cg":"cardano",      "yf":"ADA-USD"},
    "polkadot":    {"label":"Polkadot (DOT)",    "cg":"polkadot",     "yf":"DOT-USD"},
}

DEFAULT_SELECTION = ["bitcoin", "ethereum", "solana"]
MAX_COINS = 5

REFRESH_INTERVAL_MS = 120 * 1000   # dashboard refresh
CACHE_TTL = 300                    # seconds
MIN_ROWS_REQUIRED = 5              # adjust for less-liquid coins

# Simple in-memory cache
_CACHE: Dict[str, Dict] = {}

def cache_get(key: str) -> Optional[dict]:
    ent = _CACHE.get(key)
    if not ent: 
        return None
    if time.time() - ent["ts"] > CACHE_TTL:
        _CACHE.pop(key, None)
        return None
    return ent["data"]

def cache_set(key: str, data: dict):
    _CACHE[key] = {"ts": time.time(), "data": data}

# ---------------------------
# DATA FETCH & PROCESS
# ---------------------------
def cg_market_chart(coin_id: str, days: int) -> Optional[dict]:
    key = f"cg|{coin_id}|{days}"
    cached = cache_get(key)
    if cached:
        return cached
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {"vs_currency":"usd", "days": days}
    try:
        r = SESSION.get(url, params=params, timeout=12)
        r.raise_for_status()
        j = r.json()
        # basic validation
        if not isinstance(j, dict) or "prices" not in j:
            logger.warning("CoinGecko returned unexpected payload for %s", coin_id)
            return None
        cache_set(key, j)
        return j
    except Exception as e:
        logger.warning("CoinGecko error %s: %s", coin_id, e)
        return None

def fetch_yf_df(ticker: str, period: str = "3mo") -> pd.DataFrame:
    key = f"yf|{ticker}|{period}"
    cached = cache_get(key)
    if cached:
        return pd.DataFrame.from_dict(cached)
    try:
        df = yf.download(ticker, period=period, progress=False, threads=False, auto_adjust=False)
        if df is None or df.empty:
            return pd.DataFrame()
        # ensure Close & Volume present
        for col in ["Close", "Volume"]:
            if col not in df.columns:
                df[col] = np.nan
        df.index = pd.to_datetime(df.index)
        cache_set(key, df[["Close","Volume"]].to_dict(orient="list"))
        return df
    except Exception as e:
        logger.warning("yfinance error %s: %s", ticker, e)
        return pd.DataFrame()

def process_cg_to_df(coin_cg_id: str, days: int) -> pd.DataFrame:
    j = cg_market_chart(coin_cg_id, days)
    if not j:
        return pd.DataFrame()
    try:
        prices = pd.DataFrame(j.get("prices", []), columns=["ts","price"])
        vols = pd.DataFrame(j.get("total_volumes", []), columns=["ts","volume"])
        # convert ms -> datetime
        prices["date"] = pd.to_datetime(prices["ts"], unit="ms")
        vols["date"] = pd.to_datetime(vols["ts"], unit="ms")
        # merge on date (nearest)
        df = pd.merge_asof(prices.sort_values("date"), vols.sort_values("date"), on="date")
        df = df.set_index("date").sort_index()
        df = df[["price","volume"]].rename(columns={"price":"Close","volume":"Volume"})
        # resample daily (fill gaps)
        df = df.resample("D").last().ffill()
        return df
    except Exception as e:
        logger.warning("parse cg payload fail %s: %s", coin_cg_id, e)
        return pd.DataFrame()

def load_coin_df(coin_key: str, days: int = 90) -> pd.DataFrame:
    """Try CoinGecko first, if inadequate fallback to Yahoo Finance"""
    meta = COIN_META.get(coin_key)
    if not meta:
        return pd.DataFrame()
    # Try CoinGecko
    df = process_cg_to_df(meta["cg"], days)
    if _is_valid_df(df):
        return df
    # Fallback: Yahoo
    yf_period = {7:"7d", 30:"1mo", 90:"3mo", 365:"1y"}.get(days, "3mo")
    df_yf = fetch_yf_df(meta["yf"], period=yf_period)
    if df_yf is None or df_yf.empty:
        return pd.DataFrame()
    # normalize to Close/Volume with daily resample / forward fill
    df2 = df_yf[["Close","Volume"]].copy()
    df2 = df2.resample("D").last().ffill()
    if _is_valid_df(df2):
        return df2
    return pd.DataFrame()

def _is_valid_df(df: pd.DataFrame) -> bool:
    if df is None or df.empty or "Close" not in df.columns:
        return False
    valid = df["Close"].dropna()
    return len(valid) >= MIN_ROWS_REQUIRED and valid.max() > 0

# ---------------------------
# INDICATORS
# ---------------------------
def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "Close" not in df.columns:
        return df
    out = df.copy()
    out["SMA_20"] = out["Close"].rolling(20).mean()
    out["SMA_50"] = out["Close"].rolling(50).mean()
    out["EMA_12"] = out["Close"].ewm(span=12, adjust=False).mean()
    out["EMA_26"] = out["Close"].ewm(span=26, adjust=False).mean()
    out["MACD"] = out["EMA_12"] - out["EMA_26"]
    out["MACD_Signal"] = out["MACD"].ewm(span=9, adjust=False).mean()
    out["MACD_Hist"] = out["MACD"] - out["MACD_Signal"]
    # RSI
    delta = out["Close"].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    out["RSI"] = 100 - (100 / (1 + rs))
    # Bollinger
    out["BB_MID"] = out["Close"].rolling(20).mean()
    bb_std = out["Close"].rolling(20).std()
    out["BB_UP"] = out["BB_MID"] + 2 * bb_std
    out["BB_LO"] = out["BB_MID"] - 2 * bb_std
    # Volume SMA
    out["VOL_SMA20"] = out["Volume"].rolling(20).mean()
    return out

# ---------------------------
# DASH APP
# ---------------------------
app = dash.Dash(__name__, title="Multi-Coin Crypto Dashboard (Full Mode)")
server = app.server

# Inline CSS for clean responsive grid
app.index_string = """
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            body{font-family:Inter,Segoe UI,Roboto,Arial; margin:0; padding:18px; background:#0f172a; color:#e6eef8;}
            .header{display:flex; align-items:center; justify-content:space-between; gap:12px; margin-bottom:14px}
            .title{font-size:22px; font-weight:600}
            .subtitle{font-size:12px; color:#cbd5e1}
            .controls{background:#ffffff0f; padding:14px; border-radius:10px; margin-bottom:16px}
            .grid{display:grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap:16px}
            .card{background:linear-gradient(180deg,#07133066,#0b1b3b88); padding:12px; border-radius:10px; box-shadow:0 6px 20px rgba(2,6,23,0.6)}
            .coin-title{font-weight:600; color:#f8fafc; margin-bottom:6px}
            .kpis{display:flex; gap:8px; flex-wrap:wrap; margin-bottom:8px}
            .kpi{background:#ffffff10; padding:8px 10px; border-radius:8px; min-width:106px; text-align:center}
            .kpi .val{font-weight:700}
            footer{color:#94a3b8; margin-top:18px; font-size:12px}
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>{%config%}{%scripts%}{%renderer%}</footer>
    </body>
</html>
"""

controls_children = html.Div([
    html.Div([
        html.Div([html.Div("📊 Multi-Coin Full Mode", className="title"), html.Div("Pick up to 5 coins — full indicator set", className="subtitle")]),
    ], style={"display":"flex","justifyContent":"space-between","alignItems":"center","gap":"12px","marginBottom":"8px"}),
    html.Div([
        dcc.Dropdown(
            id="multi-coins",
            options=[{"label": COIN_META[k]["label"], "value": k} for k in COIN_META],
            value=DEFAULT_SELECTION,
            multi=True,
            placeholder="Select coins (max 5)"
        ),
        html.Div(style={"height":"8px"}),
        dcc.Dropdown(
            id="timeframe",
            options=[{"label":"7 Days","value":"7"},{"label":"30 Days","value":"30"},{"label":"90 Days","value":"90"},{"label":"1 Year","value":"365"}],
            value="90",
            clearable=False
        ),
        html.Div(style={"height":"8px"}),
        dcc.RadioItems(id="data-source", value="auto", options=[
            {"label":" Auto (CoinGecko -> Yahoo)", "value":"auto"},
            {"label":" CoinGecko only", "value":"coingecko"},
            {"label":" Yahoo only", "value":"yfinance"},
        ], inline=True),
    ], style={"display":"grid","gridTemplateColumns":"1fr","gap":"8px"})
], className="controls")

app.layout = html.Div([
    html.Div([controls_children], className="controls"),
    html.Div(id="coins-container"),
    dcc.Interval(id="refresh", interval=REFRESH_INTERVAL_MS, n_intervals=0),
    html.Div(id="debug", style={"display":"none"})
])

# ---------------------------
# Helper to build per-coin card (returns html.Div)
# ---------------------------
def make_coin_card(coin_key: str, df: pd.DataFrame, used_source: str) -> html.Div:
    meta = COIN_META[coin_key]
    title = meta["label"]
    # KPI values (guarded)
    price = df["Close"].dropna().iloc[-1] if "Close" in df.columns and not df["Close"].dropna().empty else None
    prev = df["Close"].dropna().iloc[-2] if "Close" in df.columns and df["Close"].dropna().shape[0] >= 2 else price
    change = ((price - prev) / prev * 100) if (price is not None and prev not in (None, 0)) else 0
    vol = df["Volume"].dropna().iloc[-1] if "Volume" in df.columns and not df["Volume"].dropna().empty else None
    rsi = df["RSI"].dropna().iloc[-1] if "RSI" in df.columns and not df["RSI"].dropna().empty else None

    # Price + BB figure
    fig_price = go.Figure()
    fig_price.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Close", line=dict(width=2)))
    if "BB_UP" in df.columns:
        fig_price.add_trace(go.Scatter(x=df.index, y=df["BB_UP"], name="BB UP", line=dict(dash="dot"), opacity=0.6))
        fig_price.add_trace(go.Scatter(x=df.index, y=df["BB_LO"], name="BB LO", line=dict(dash="dot"), opacity=0.6, fill='tonexty'))
    fig_price.update_layout(margin=dict(l=20,r=20,t=30,b=20), height=260, template="plotly_dark", title="Price + Bollinger")

    # Moving averages panel
    fig_ma = go.Figure()
    if "SMA_20" in df.columns:
        fig_ma.add_trace(go.Scatter(x=df.index, y=df["SMA_20"], name="SMA 20"))
    if "SMA_50" in df.columns:
        fig_ma.add_trace(go.Scatter(x=df.index, y=df["SMA_50"], name="SMA 50"))
    if "EMA_12" in df.columns:
        fig_ma.add_trace(go.Scatter(x=df.index, y=df["EMA_12"], name="EMA 12"))
    fig_ma.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Close", line=dict(width=1), opacity=0.6))
    fig_ma.update_layout(margin=dict(l=20,r=20,t=30,b=20), height=200, template="plotly_dark", title="Moving Averages")

    # Volume chart
    fig_vol = go.Figure()
    if "Volume" in df.columns:
        fig_vol.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume"))
    if "VOL_SMA20" in df.columns:
        fig_vol.add_trace(go.Scatter(x=df.index, y=df["VOL_SMA20"], name="Vol SMA20", line=dict(color="yellow")))
    fig_vol.update_layout(margin=dict(l=20,r=20,t=30,b=20), height=160, template="plotly_dark", title="Volume")

    # RSI chart
    fig_rsi = go.Figure()
    if "RSI" in df.columns:
        fig_rsi.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI"))
        fig_rsi.add_hline(y=70, line_dash="dot", line_color="red")
        fig_rsi.add_hline(y=30, line_dash="dot", line_color="green")
    fig_rsi.update_layout(margin=dict(l=20,r=20,t=30,b=20), height=140, template="plotly_dark", title="RSI (14)")

    # MACD chart
    fig_macd = go.Figure()
    if "MACD" in df.columns and "MACD_Signal" in df.columns:
        fig_macd.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD"))
        fig_macd.add_trace(go.Scatter(x=df.index, y=df["MACD_Signal"], name="Signal"))
        fig_macd.add_trace(go.Bar(x=df.index, y=df.get("MACD_Hist", []), name="Hist", opacity=0.3))
    fig_macd.update_layout(margin=dict(l=20,r=20,t=30,b=20), height=140, template="plotly_dark", title="MACD")

    # KPI cards
    kpis = html.Div([
        html.Div([
            html.Div(f"${price:,.2f}" if price is not None else "N/A", className="val"),
            html.Div("Price")
        ], className="kpi"),
        html.Div([
            html.Div(f"{change:+.2f}%" if change is not None else "N/A", className="val"),
            html.Div("Change")
        ], className="kpi"),
        html.Div([
            html.Div(f"${vol:,.0f}" if vol is not None else "N/A", className="val"),
            html.Div("Volume")
        ], className="kpi"),
        html.Div([
            html.Div(f"{rsi:.1f}" if rsi is not None else "N/A", className="val"),
            html.Div("RSI")
        ], className="kpi"),
        html.Div([
            html.Div(used_source.upper() if used_source else "N/A", className="val"),
            html.Div("Source")
        ], className="kpi"),
    ], className="kpis")

    return html.Div([
        html.Div([html.Div(title, className="coin-title"), kpis]),
        html.Div(className="card", children=[
            dcc.Graph(figure=fig_price, config={"displayModeBar": False}),
            dcc.Graph(figure=fig_ma, config={"displayModeBar": False}),
            dcc.Graph(figure=fig_vol, config={"displayModeBar": False}),
            dcc.Graph(figure=fig_rsi, config={"displayModeBar": False}),
            dcc.Graph(figure=fig_macd, config={"displayModeBar": False}),
        ])
    ], style={"marginBottom":"12px"})

# ---------------------------
# MAIN CALLBACK — builds grid of coin cards
# ---------------------------
@app.callback(
    Output("coins-container", "children"),
    Input("multi-coins", "value"),
    Input("timeframe", "value"),
    Input("data-source", "value"),
    Input("refresh", "n_intervals")
)
def render_selected_coins(selected, tf_value, data_source, n_intervals):
    # ensure list
    if not selected:
        selected = DEFAULT_SELECTION
    if isinstance(selected, str):
        selected = [selected]
    # enforce max
    selected = selected[:MAX_COINS]
    days = int(tf_value) if tf_value else 90

    cards = []
    debug_items = []
    for coin_key in selected:
        if coin_key not in COIN_META:
            continue
        df = None
        used = None
        # choose data source behavior
        order = []
        if data_source == "auto":
            order = ["coingecko","yfinance"]
        else:
            order = [data_source]

        for src in order:
            if src == "coingecko":
                df = process_cg_to_df(COIN_META[coin_key]["cg"], days)
                if _is_valid_df(df):
                    used = "coingecko"
                    break
            elif src == "yfinance":
                # map days to yfinance period
                yf_period = {7:"7d", 30:"1mo", 90:"3mo", 365:"1y"}.get(days, "3mo")
                df_tmp = fetch_yf_df(COIN_META[coin_key]["yf"], period=yf_period)
                if df_tmp is not None and not df_tmp.empty:
                    df = df_tmp[["Close","Volume"]].copy()
                    df = df.resample("D").last().ffill()
                    if _is_valid_df(df):
                        used = "yfinance"
                        break
                df = pd.DataFrame()  # ensure consistent type

        if df is None or df.empty or not _is_valid_df(df):
            # show small card with error message instead of charts
            err_card = html.Div([
                html.Div(COIN_META[coin_key]["label"], className="coin-title"),
                html.Div("⚠️ Data unavailable for this coin; try toggling data source or timeframe.", style={"color":"#ffb4a2","fontSize":"13px"})
            ], className="card")
            cards.append(err_card)
            debug_items.append({coin_key: "no-data"})
            continue

        # enrich with indicators
        df_ind = add_all_indicators(df)
        card = make_coin_card(coin_key, df_ind, used)
        cards.append(card)
        debug_items.append({coin_key: f"rows={len(df_ind)}, src={used}"})

    # assemble grid
    grid = html.Div(children=cards, className="grid")
    # attach hidden debug for quick troubleshooting
    logger.debug("render debug: %s", debug_items)
    return html.Div([grid, html.Div(str(debug_items), id="debug", style={"display":"none"})])

# ---------------------------
# RUN APPLICATION (Render compatible)
# ---------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    logger.info("Starting app on port %s", port)
    app.run_server(debug=False, host="0.0.0.0", port=port)