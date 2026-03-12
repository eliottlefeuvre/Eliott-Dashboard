"""
╔══════════════════════════════════════════════════════════════╗
║              QUANTDESK — Live US Tech Dashboard              ║
║         Powered by Yahoo Finance · Dash · Plotly             ║
╠══════════════════════════════════════════════════════════════╣
║  Install:  pip install yfinance dash plotly pandas numpy     ║
║            scipy dash-bootstrap-components                   ║
║  Run:      python quantdesk.py                               ║
║  Open:     http://127.0.0.1:8050                             ║
╚══════════════════════════════════════════════════════════════╝
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats
import yfinance as yf
from datetime import datetime, timedelta
import dash
from dash import dcc, html, Input, Output, callback_context
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
TICKERS = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "AVGO", "ORCL", "AMD"]
NAMES   = {"AAPL":"Apple","MSFT":"Microsoft","NVDA":"NVIDIA","GOOGL":"Alphabet",
           "AMZN":"Amazon","META":"Meta","TSLA":"Tesla","AVGO":"Broadcom","ORCL":"Oracle","AMD":"AMD"}
COLORS  = ["#00f5d4","#f72585","#fee440","#4cc9f0","#7bed9f",
           "#ff6b6b","#a29bfe","#fd79a8","#55efc4","#fdcb6e"]
PERIOD  = "1y"
MC_SIMS = 500
MC_DAYS = 63

# Macro / commodities tickers
MACRO_TICKERS = {
    "^VIX":  "VIX Fear Index",
    "GC=F":  "Gold",
    "CL=F":  "Crude Oil",
    "SI=F":  "Silver",
    "^TNX":  "10Y Treasury Yield",
    "DX-Y.NYB": "US Dollar Index",
    "BTC-USD": "Bitcoin",
}

BG      = "#040d1a"
BG2     = "#060f1f"
BG3     = "#0a1628"
BORDER  = "#0d2040"
TXT     = "#c9d4e8"
DIM     = "#556070"
ACCENT  = "#00f5d4"
RED     = "#f72585"
YELLOW  = "#fee440"
BLUE    = "#4cc9f0"

def hex_rgba(hex_color, alpha=0.1):
    """Convert '#rrggbb' to 'rgba(r,g,b,alpha)' — required by Plotly 5.x fillcolor."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

PLOT_CFG = dict(
    paper_bgcolor=BG2, plot_bgcolor=BG,
    font=dict(family="IBM Plex Mono, Courier New, monospace", color=TXT, size=11),
    margin=dict(l=50, r=20, t=40, b=40),
    xaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER),
    yaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER),
)

# ─────────────────────────────────────────────
#  DATA LAYER
# ─────────────────────────────────────────────

def fetch_all():
    """Download 1y OHLCV for all tickers + SPY + QQQ."""
    end   = datetime.today()
    start = end - timedelta(days=400)
    raw = yf.download(TICKERS + ["SPY","QQQ"], start=start, end=end,
                      auto_adjust=True, progress=False)

    # yfinance ≥0.2 returns MultiIndex columns: (field, ticker)
    # Flatten to a simple (date × ticker) close-price DataFrame
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"]
    else:
        close = raw[["Close"]] if "Close" in raw.columns else raw

    # Ensure plain column names (drop any remaining MultiIndex level)
    if isinstance(close.columns, pd.MultiIndex):
        close.columns = close.columns.get_level_values(-1)

    close = close.dropna(how="all")
    return close

def compute_metrics(close):
    """Return dict of per-ticker analytics + cross-asset data."""
    metrics  = {}
    # Ensure every column is a plain Series before computing returns
    close    = close.copy()
    for col in close.columns:
        close[col] = pd.to_numeric(close[col], errors="coerce")

    rets     = close.pct_change().dropna()
    log_rets = np.log(close / close.shift(1)).dropna()

    spy_ret  = rets["SPY"].squeeze() if "SPY" in rets.columns else None
    qqq_ret  = rets["QQQ"].squeeze() if "QQQ" in rets.columns else None

    for t in TICKERS:
        if t not in close.columns:
            continue
        p   = close[t].squeeze().dropna()
        r   = log_rets[t].squeeze().dropna()
        pct = rets[t].squeeze().dropna()

        # Skip ticker if rate-limited or empty
        if len(p) < 10 or len(r) < 10:
            print(f"⚠  Skipping {t} — insufficient data ({len(r)} rows)")
            continue

        ann_ret   = float(r.mean() * 252)
        ann_vol   = float(r.std()  * np.sqrt(252))
        sharpe    = ann_ret / ann_vol if ann_vol else 0

        # Max drawdown
        roll_max  = p.cummax()
        dd        = (p - roll_max) / roll_max
        max_dd    = float(dd.min())

        # Beta vs SPY
        if spy_ret is not None and t in rets:
            cov   = np.cov(pct.values, spy_ret.reindex(pct.index).fillna(0).values)
            beta  = cov[0,1] / cov[1,1] if cov[1,1] else 1.0
        else:
            beta  = 1.0

        # VaR & CVaR (95%)
        var95  = float(np.percentile(r, 5))
        cvar95 = float(r[r <= var95].mean())

        # Sortino
        neg_r     = r[r < 0]
        sortino   = ann_ret / (neg_r.std() * np.sqrt(252)) if len(neg_r) else 0

        # Calmar
        calmar    = -ann_ret / max_dd if max_dd else 0

        # Rolling 30d vol
        roll_vol  = r.rolling(30).std() * np.sqrt(252)

        # Correlation with SPY
        spy_corr  = float(pct.corr(spy_ret.reindex(pct.index).fillna(0))) if spy_ret is not None else 0

        # Fundamental (live from yfinance)
        try:
            info = yf.Ticker(t).fast_info
            mkt_cap  = getattr(info, "market_cap",  None)
            pe_ratio = getattr(info, "pe_ratio",     None)
            last_px  = float(p.iloc[-1])
            prev_px  = float(p.iloc[-2])
        except Exception:
            mkt_cap  = None
            pe_ratio = None
            last_px  = float(p.iloc[-1])
            prev_px  = float(p.iloc[-2])

        metrics[t] = dict(
            prices   = p,
            returns  = r,
            pct_rets = pct,
            ann_ret  = ann_ret,
            ann_vol  = ann_vol,
            sharpe   = sharpe,
            sortino  = sortino,
            calmar   = calmar,
            max_dd   = max_dd,
            beta     = beta,
            var95    = var95,
            cvar95   = cvar95,
            roll_vol = roll_vol,
            spy_corr = spy_corr,
            mkt_cap  = mkt_cap,
            pe_ratio = pe_ratio,
            last_px  = last_px,
            chg_1d   = (last_px - prev_px) / prev_px * 100,
        )

    metrics["_close"]    = close
    metrics["_log_rets"] = log_rets
    metrics["_spy"]      = close["SPY"].squeeze()  if "SPY" in close.columns else None
    metrics["_qqq"]      = close["QQQ"].squeeze()  if "QQQ" in close.columns else None
    return metrics


def monte_carlo(returns, current_price, days=MC_DAYS, sims=MC_SIMS):
    mu  = returns.mean()
    sig = returns.std()
    rng = np.random.default_rng(42)
    shocks = rng.normal(mu, sig, (sims, days))
    paths  = current_price * np.exp(np.cumsum(shocks, axis=1))
    paths  = np.hstack([np.full((sims,1), current_price), paths])
    return paths


def corr_matrix(metrics):
    tickers = [t for t in TICKERS if t in metrics]
    rets    = pd.DataFrame({t: metrics[t]["pct_rets"] for t in tickers})
    return rets.corr()


def fetch_macro():
    """Fetch macro / commodity data."""
    macro = {}
    end   = datetime.today()
    start = end - timedelta(days=400)
    for sym, name in MACRO_TICKERS.items():
        try:
            raw = yf.download(sym, start=start, end=end, auto_adjust=True, progress=False)
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            if "Close" in raw.columns and len(raw) > 5:
                p = raw["Close"].squeeze().dropna()
                last  = float(p.iloc[-1])
                prev  = float(p.iloc[-2])
                chg   = (last - prev) / prev * 100
                ytd_start = p[p.index >= pd.Timestamp(f"{datetime.today().year}-01-01")]
                ytd   = float((last / ytd_start.iloc[0] - 1) * 100) if len(ytd_start) > 0 else 0
                macro[sym] = {"name": name, "price": last, "chg_1d": chg, "ytd": ytd, "history": p}
        except Exception:
            pass
    return macro


def compute_invest_signal(ticker, metrics):
    """
    Composite BUY / HOLD / SELL signal combining 8 factors.
    Returns score 0–100, verdict, and factor breakdown.
    """
    m = metrics[ticker]
    factors = {}

    # 1. Momentum — annualised return vs peers
    peer_rets = [metrics[t]["ann_ret"] for t in TICKERS if t in metrics]
    pct_rank  = sum(m["ann_ret"] > r for r in peer_rets) / len(peer_rets)
    factors["Momentum"]      = round(pct_rank * 100, 1)

    # 2. Risk-adjusted return (Sharpe)
    sharpe_score = min(100, max(0, (m["sharpe"] + 0.5) / 3.0 * 100))
    factors["Sharpe Ratio"]  = round(sharpe_score, 1)

    # 3. Drawdown risk (lower is better)
    dd_score = max(0, 100 + m["max_dd"] * 200)   # max_dd is negative
    factors["Drawdown Risk"] = round(dd_score, 1)

    # 4. Volatility (lower vol relative to peers = better)
    peer_vols = [metrics[t]["ann_vol"] for t in TICKERS if t in metrics]
    vol_rank  = 1 - sum(m["ann_vol"] > v for v in peer_vols) / len(peer_vols)
    factors["Low Volatility"] = round(vol_rank * 100, 1)

    # 5. Beta — penalise high beta in uncertain markets
    beta_score = max(0, min(100, (2.5 - m["beta"]) / 2.0 * 100))
    factors["Beta Score"]    = round(beta_score, 1)

    # 6. VaR efficiency
    var_score = max(0, min(100, (0.04 + m["var95"]) / 0.04 * 100))
    factors["VaR Efficiency"] = round(var_score, 1)

    # 7. Sortino (downside protection)
    sortino_score = min(100, max(0, (m["sortino"] + 0.5) / 3.5 * 100))
    factors["Sortino"]       = round(sortino_score, 1)

    # 8. Monte Carlo upside — P95/current vs P5/current
    try:
        paths  = monte_carlo(m["returns"], m["last_px"])
        finals = paths[:, -1]
        p5, p95 = np.percentile(finals, [5, 95])
        upside   = (p95 / m["last_px"] - 1) * 100
        downside = (1 - p5 / m["last_px"]) * 100
        asymmetry = min(100, max(0, upside / (upside + downside + 1e-6) * 100))
        factors["MC Asymmetry"] = round(asymmetry, 1)
    except Exception:
        factors["MC Asymmetry"] = 50.0

    # Weighted composite
    weights = {
        "Momentum": 0.18, "Sharpe Ratio": 0.18, "Drawdown Risk": 0.12,
        "Low Volatility": 0.10, "Beta Score": 0.10, "VaR Efficiency": 0.10,
        "Sortino": 0.12, "MC Asymmetry": 0.10
    }
    score = sum(factors[k] * weights[k] for k in factors)

    if score >= 68:
        verdict, color = "BUY", "#00c853"
    elif score >= 42:
        verdict, color = "HOLD", "#ffab00"
    else:
        verdict, color = "SELL", "#ff1744"

    return round(score, 1), verdict, color, factors


def portfolio_optimize(weights_pct, metrics):
    """
    Given user weight dict {ticker: pct}, compute portfolio stats
    and also find max-Sharpe and min-variance portfolios.
    """
    tickers = [t for t in weights_pct if t in metrics and weights_pct[t] > 0]
    if not tickers:
        return None
    w = np.array([weights_pct[t] / 100 for t in tickers])
    w = w / w.sum()

    rets_df = pd.DataFrame({t: metrics[t]["pct_rets"] for t in tickers}).dropna()
    mu_vec  = rets_df.mean().values * 252
    cov_mat = rets_df.cov().values * 252

    port_ret = float(w @ mu_vec)
    port_vol = float(np.sqrt(w @ cov_mat @ w))
    port_sharpe = port_ret / port_vol if port_vol else 0

    # Monte-Carlo frontier
    n = len(tickers)
    frontier = []
    rng = np.random.default_rng(0)
    for _ in range(3000):
        rw = rng.dirichlet(np.ones(n))
        r  = float(rw @ mu_vec)
        v  = float(np.sqrt(rw @ cov_mat @ rw))
        s  = r / v if v else 0
        frontier.append((v * 100, r * 100, s, rw))

    # Max Sharpe
    best_sharpe = max(frontier, key=lambda x: x[2])
    # Min Variance
    best_minvar = min(frontier, key=lambda x: x[0])

    return {
        "tickers": tickers, "weights": w,
        "ret": port_ret * 100, "vol": port_vol * 100, "sharpe": port_sharpe,
        "frontier": frontier,
        "max_sharpe": {"vol": best_sharpe[0], "ret": best_sharpe[1], "w": best_sharpe[3]},
        "min_var":    {"vol": best_minvar[0], "ret": best_minvar[1], "w": best_minvar[3]},
    }


# ─────────────────────────────────────────────
#  INITIAL LOAD
# ─────────────────────────────────────────────
import time
print("⬇  Fetching live data from Yahoo Finance …")
for attempt in range(3):
    try:
        CLOSE   = fetch_all()
        METRICS = compute_metrics(CLOSE)
        CORR    = corr_matrix(METRICS)
        loaded  = [t for t in TICKERS if t in METRICS]
        print(f"✅  Data loaded — {len(loaded)} tickers · {len(CLOSE)} days")
        if len(loaded) < 5:
            raise ValueError(f"Too few tickers ({len(loaded)}), retrying…")
        break
    except Exception as e:
        print(f"⚠  Attempt {attempt+1}/3 failed: {e}")
        if attempt < 2:
            print("   Waiting 15s before retry…")
            time.sleep(15)
        else:
            print("❌  Could not load data after 3 attempts.")
            raise

print("⬇  Fetching macro / commodity data …")
try:
    MACRO = fetch_macro()
    print(f"✅  Macro loaded — {len(MACRO)} instruments")
except Exception as e:
    print(f"⚠  Macro fetch failed: {e}")
    MACRO = {}

VALID_TICKERS = [t for t in TICKERS if t in METRICS]

# ─────────────────────────────────────────────
#  DASH APP
# ─────────────────────────────────────────────
app = dash.Dash(__name__, title="QuantDesk")
app.index_string = """
<!DOCTYPE html>
<html>
<head>
  {%metas%}
  <title>{%title%}</title>
  {%favicon%}
  {%css%}
  <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;600&family=Bebas+Neue&display=swap" rel="stylesheet">
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { background: #040d1a; font-family: 'IBM Plex Mono', monospace; color: #c9d4e8; }
    ::-webkit-scrollbar { width: 4px; height: 4px; }
    ::-webkit-scrollbar-track { background: #040d1a; }
    ::-webkit-scrollbar-thumb { background: #1a2744; border-radius: 2px; }
    .ticker-chip { cursor: pointer; border-radius: 3px; padding: 6px 10px; border: 1px solid #0d2040;
                   transition: all .15s; min-width: 110px; }
    .ticker-chip:hover { border-color: #00f5d4 !important; background: #0d2040 !important; }
    .tab { cursor: pointer; padding: 8px 20px; border: none; background: none;
           border-bottom: 2px solid transparent; font-family: inherit;
           font-size: 11px; letter-spacing: 2px; text-transform: uppercase;
           transition: color .15s; color: #556070; }
    .tab.active { color: #00f5d4; border-bottom-color: #00f5d4; }
    .tab:hover { color: #00f5d4; }
    .metric-card { border-radius: 4px; padding: 12px; border: 1px solid #0d2040;
                   background: #060f1f; transition: all .15s; }
    .metric-card:hover { border-color: #00f5d4; background: #0a1628; }
    .section { background: #060f1f; border: 1px solid #0d2040; border-radius: 4px; padding: 16px; }
    .section-title { color: #4cc9f0; font-size: 11px; letter-spacing: 2px; margin-bottom: 12px; }
    input[type=range] { -webkit-appearance:none; width:100%; height:4px; background:#0d2040; border-radius:2px; outline:none; }
    input[type=range]::-webkit-slider-thumb { -webkit-appearance:none; width:14px; height:14px; border-radius:50%; background:#00f5d4; cursor:pointer; }
    .signal-buy  { color:#00c853; font-weight:600; }
    .signal-hold { color:#ffab00; font-weight:600; }
    .signal-sell { color:#ff1744; font-weight:600; }
  </style>
</head>
<body>
  {%app_entry%}
  <footer>{%config%}{%scripts%}{%renderer%}</footer>
</body>
</html>
"""

# ─────────────────────────────────────────────
#  LAYOUT
# ─────────────────────────────────────────────
def make_layout():
    ticker_chips = []
    for i, t in enumerate(VALID_TICKERS):
        m = METRICS[t]
        color = COLORS[TICKERS.index(t)]
        chg_color = ACCENT if m["chg_1d"] >= 0 else RED
        ticker_chips.append(
            html.Div([
                html.Div([
                    html.Span(t, style={"color": color, "fontWeight": "600", "letterSpacing": "1px"}),
                    html.Span(f"{m['chg_1d']:+.2f}%", style={"color": chg_color, "fontSize": "10px"}),
                ], style={"display":"flex","justifyContent":"space-between"}),
                html.Div(f"${m['last_px']:.2f}", style={"color": TXT, "marginTop":"2px", "fontSize":"11px"}),
            ], className="ticker-chip", id={"type":"ticker-chip","index":t},
               style={"background":"transparent","borderColor":BORDER},
               n_clicks=0)
        )

    return html.Div([
        # ── HEADER ──────────────────────────────────────
        html.Div([
            html.Div([
                html.Span("QUANT", style={"color":ACCENT,"fontFamily":"Bebas Neue","fontSize":"28px","letterSpacing":"3px"}),
                html.Span("DESK", style={"color":RED,"fontFamily":"Bebas Neue","fontSize":"28px","letterSpacing":"3px"}),
                html.Div(style={"width":"1px","height":"24px","background":BORDER,"margin":"0 16px"}),
                html.Div("US MEGA-CAP TECH  ·  LIVE DATA", style={"color":BLUE,"fontSize":"10px","letterSpacing":"2px"}),
            ], style={"display":"flex","alignItems":"center","gap":"0"}),
            html.Div([
                html.Span("LIVE ", style={"color":BLUE}),
                html.Span("●", style={"color":ACCENT,"fontSize":"8px","marginRight":"16px"}),
                html.Span(datetime.now().strftime("%Y-%m-%d  %H:%M EST"), style={"color":DIM,"fontSize":"10px","letterSpacing":"1px"}),
            ], style={"display":"flex","alignItems":"center"}),
        ], style={"background":BG,"borderBottom":f"1px solid {BORDER}","padding":"12px 24px",
                  "display":"flex","justifyContent":"space-between","alignItems":"center",
                  "position":"sticky","top":"0","zIndex":"100"}),

        # ── TICKER BAR ──────────────────────────────────
        html.Div(ticker_chips, id="ticker-bar",
                 style={"background":"#060f1f","borderBottom":f"1px solid {BORDER}",
                        "padding":"8px 24px","display":"flex","gap":"8px","overflowX":"auto","flexWrap":"nowrap"}),

        # ── CONTENT ─────────────────────────────────────
        html.Div([
            # TAB BAR
            html.Div([
                html.Button("Overview",    id="tab-overview",    className="tab active", n_clicks=0),
                html.Button("Candlestick", id="tab-candlestick", className="tab",        n_clicks=0),
                html.Button("Monte Carlo", id="tab-montecarlo",  className="tab",        n_clicks=0),
                html.Button("Correlation", id="tab-correlation", className="tab",        n_clicks=0),
                html.Button("Risk",        id="tab-risk",        className="tab",        n_clicks=0),
                html.Button("Portfolio",   id="tab-portfolio",   className="tab",        n_clicks=0),
                html.Button("Macro",       id="tab-macro",       className="tab",        n_clicks=0),
                html.Button("Comparison",  id="tab-comparison",  className="tab",        n_clicks=0),
                html.Button("🎯 Signal",   id="tab-signal",      className="tab",        n_clicks=0),
            ], style={"display":"flex","borderBottom":f"1px solid {BORDER}","marginBottom":"20px","flexWrap":"wrap"}),

            # PANELS
            html.Div(id="panel-overview"),
            html.Div(id="panel-candlestick", style={"display":"none"}),
            html.Div(id="panel-montecarlo",  style={"display":"none"}),
            html.Div(id="panel-correlation", style={"display":"none"}),
            html.Div(id="panel-risk",        style={"display":"none"}),
            html.Div(id="panel-portfolio",   style={"display":"none"}),
            html.Div(id="panel-macro",       style={"display":"none"}),
            html.Div(id="panel-comparison",  style={"display":"none"}),
            html.Div(id="panel-signal",      style={"display":"none"}),

        ], style={"padding":"16px 24px"}),

        # Hidden state stores
        dcc.Store(id="selected-ticker", data=VALID_TICKERS[2] if len(VALID_TICKERS)>2 else VALID_TICKERS[0]),
        dcc.Store(id="active-tab", data="overview"),
        dcc.Store(id="portfolio-weights", data={t: round(100/len(VALID_TICKERS),1) for t in VALID_TICKERS}),

        # Footer
        html.Div([
            html.Span("QUANTDESK v3.0 — LIVE YAHOO FINANCE DATA"),
            html.Span("FOR EDUCATIONAL PURPOSES ONLY · NOT FINANCIAL ADVICE"),
            html.Span(f"© {datetime.now().year} QUANTDESK ANALYTICS"),
        ], style={"borderTop":f"1px solid {BORDER}","padding":"12px 24px","display":"flex",
                  "justifyContent":"space-between","color":"#2a3a55","fontSize":"9px",
                  "letterSpacing":"1px","marginTop":"24px"}),
    ])

app.layout = make_layout


# ─────────────────────────────────────────────
#  CALLBACKS — ticker selection
# ─────────────────────────────────────────────
@app.callback(
    Output("selected-ticker","data"),
    [Input({"type":"ticker-chip","index":t},"n_clicks") for t in VALID_TICKERS],
    prevent_initial_call=True
)
def select_ticker(*args):
    ctx = callback_context
    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate
    tid = ctx.triggered[0]["prop_id"].split(".")[0]
    import json
    return json.loads(tid)["index"]


# ─────────────────────────────────────────────
#  CALLBACKS — tab switching
# ─────────────────────────────────────────────
TABS = ["overview","candlestick","montecarlo","correlation","risk","portfolio","macro","comparison","signal"]

@app.callback(
    Output("active-tab","data"),
    [Input(f"tab-{t}","n_clicks") for t in TABS],
    prevent_initial_call=True
)
def switch_tab(*args):
    ctx = callback_context
    if not ctx.triggered: raise dash.exceptions.PreventUpdate
    tid = ctx.triggered[0]["prop_id"].replace(".n_clicks","").replace("tab-","")
    return tid

@app.callback(
    [Output(f"tab-{t}","className") for t in TABS],
    Input("active-tab","data")
)
def update_tab_classes(active):
    return ["tab active" if t==active else "tab" for t in TABS]

@app.callback(
    [Output(f"panel-{t}","style") for t in TABS],
    Input("active-tab","data")
)
def show_panel(active):
    return [{"display":"block"} if t==active else {"display":"none"} for t in TABS]


# ─────────────────────────────────────────────
#  HELPERS — plot style
# ─────────────────────────────────────────────
def styled_fig(fig, height=300, title=""):
    fig.update_layout(
        **PLOT_CFG, height=height,
        title=dict(text=title, font=dict(size=11, color=BLUE), x=0),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=BORDER, font=dict(size=10)),
        hovermode="x unified",
    )
    fig.update_xaxes(showgrid=True, gridcolor=BORDER, linecolor=BORDER)
    fig.update_yaxes(showgrid=True, gridcolor=BORDER, linecolor=BORDER)
    return fig

def sign_color(v): return ACCENT if v >= 0 else RED

def kpi_card(label, value, sub="", value_color=TXT):
    return html.Div([
        html.Div(label, style={"color":DIM,"fontSize":"9px","letterSpacing":"2px","marginBottom":"6px"}),
        html.Div(value, style={"fontSize":"16px","fontWeight":"600","color":value_color,"lineHeight":"1"}),
        html.Div(sub,   style={"color":DIM,"fontSize":"10px","marginTop":"4px"}),
    ], className="metric-card")


# ─────────────────────────────────────────────
#  PANEL — OVERVIEW
# ─────────────────────────────────────────────
@app.callback(Output("panel-overview","children"), Input("selected-ticker","data"))
def render_overview(ticker):
    try:
        m   = METRICS[ticker]
        col = COLORS[TICKERS.index(ticker)]

        kpis = html.Div([
            kpi_card("PRICE",       f"${m['last_px']:.2f}", f"{m['chg_1d']:+.2f}%", sign_color(m["chg_1d"])),
            kpi_card("ANN. RETURN", f"{m['ann_ret']*100:+.1f}%", "252d", sign_color(m["ann_ret"])),
            kpi_card("VOLATILITY",  f"{m['ann_vol']*100:.1f}%",  "Ann. σ", YELLOW),
            kpi_card("SHARPE",      f"{m['sharpe']:.2f}", "Ratio", ACCENT if m["sharpe"]>1 else RED),
            kpi_card("SORTINO",     f"{m['sortino']:.2f}", "Downside adj.", ACCENT if m["sortino"]>1 else RED),
            kpi_card("MAX DRAWDOWN",f"{m['max_dd']*100:.1f}%", "Peak→Trough", RED),
            kpi_card("BETA",        f"{m['beta']:.2f}", "vs S&P 500", RED if m["beta"]>1.5 else YELLOW if m["beta"]>1 else ACCENT),
            kpi_card("VAR 95%",     f"{m['var95']*100:.2f}%", "Daily", RED),
        ], style={"display":"grid","gridTemplateColumns":"repeat(8,1fr)","gap":"8px","marginBottom":"16px"})

        # ── Price chart (simple AreaChart, no subplots to avoid axis issues) ──
        prices = m["prices"]

        # Safely extract volume from global CLOSE download
        vol_series = None
        try:
            raw_v = yf.download(ticker, period=PERIOD, auto_adjust=True, progress=False)
            # yfinance ≥0.2 may return MultiIndex columns
            if isinstance(raw_v.columns, pd.MultiIndex):
                raw_v.columns = raw_v.columns.get_level_values(0)
            if "Volume" in raw_v.columns:
                vol_series = raw_v["Volume"].squeeze()
        except Exception:
            pass

        # Build subplot figure safely
        fig_price = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            row_heights=[0.72, 0.28], vertical_spacing=0.04,
            subplot_titles=("", "")
        )
        fig_price.add_trace(go.Scatter(
            x=list(prices.index), y=list(prices.values),
            mode="lines", line=dict(color=col, width=1.5),
            fill="tozeroy", fillcolor=hex_rgba(col, 0.1),
            name=ticker, hovertemplate="$%{y:.2f}<extra></extra>"
        ), row=1, col=1)

        if vol_series is not None and len(vol_series) > 0:
            fig_price.add_trace(go.Bar(
                x=list(vol_series.index), y=list(vol_series.values),
                name="Volume", marker_color=hex_rgba(col, 0.4),
                hovertemplate="%{y:,.0f}<extra>Volume</extra>"
            ), row=2, col=1)

        fig_price.update_layout(
            paper_bgcolor=BG2, plot_bgcolor=BG,
            font=dict(family="IBM Plex Mono, Courier New, monospace", color=TXT, size=11),
            margin=dict(l=50, r=20, t=40, b=40),
            height=310,
            title=dict(text=f"PRICE & VOLUME — {ticker}  (1Y)", font=dict(size=11, color=BLUE), x=0),
            showlegend=False, hovermode="x unified",
        )
        fig_price.update_xaxes(showgrid=True, gridcolor=BORDER, linecolor=BORDER)
        fig_price.update_yaxes(showgrid=True, gridcolor=BORDER, linecolor=BORDER)

        # ── Sharpe ranking bar ──
        sharpe_data = sorted([(t, METRICS[t]["sharpe"]) for t in VALID_TICKERS], key=lambda x: -x[1])
        fig_sharpe = go.Figure(go.Bar(
            x=[s for _, s in sharpe_data],
            y=[t for t, _ in sharpe_data],
            orientation="h",
            marker=dict(color=[COLORS[TICKERS.index(t)] for t, _ in sharpe_data]),
            text=[f"{s:.2f}" for _, s in sharpe_data],
            textposition="outside",
            hovertemplate="%{y}: %{x:.2f}<extra>Sharpe</extra>"
        ))
        styled_fig(fig_sharpe, 280, "SHARPE RATIO RANKING")

        # ── Metrics table ──
        rows = []
        for i, t in enumerate(VALID_TICKERS):
            mm  = METRICS[t]
            cap = f"${mm['mkt_cap']/1e9:.0f}B" if mm.get("mkt_cap") else "—"
            pe  = f"{mm['pe_ratio']:.1f}x"      if mm.get("pe_ratio") else "—"
            rows.append(html.Tr([
                html.Td(t,  style={"color":COLORS[i],"fontWeight":"600","letterSpacing":"1px","padding":"7px 10px"}),
                html.Td(f"${mm['last_px']:.2f}",        style={"textAlign":"right","padding":"7px 10px"}),
                html.Td(f"{mm['chg_1d']:+.2f}%",        style={"textAlign":"right","padding":"7px 10px","color":sign_color(mm["chg_1d"])}),
                html.Td(f"{mm['ann_ret']*100:+.1f}%",   style={"textAlign":"right","padding":"7px 10px","color":sign_color(mm["ann_ret"])}),
                html.Td(f"{mm['ann_vol']*100:.1f}%",    style={"textAlign":"right","padding":"7px 10px","color":YELLOW}),
                html.Td(f"{mm['sharpe']:.2f}",          style={"textAlign":"right","padding":"7px 10px","color":ACCENT if mm["sharpe"]>1 else RED}),
                html.Td(f"{mm['sortino']:.2f}",         style={"textAlign":"right","padding":"7px 10px","color":ACCENT if mm["sortino"]>1 else RED}),
                html.Td(f"{mm['max_dd']*100:.1f}%",     style={"textAlign":"right","padding":"7px 10px","color":RED}),
                html.Td(f"{mm['beta']:.2f}",            style={"textAlign":"right","padding":"7px 10px","color":RED if mm["beta"]>1.5 else YELLOW if mm["beta"]>1 else ACCENT}),
                html.Td(f"{mm['var95']*100:.2f}%",      style={"textAlign":"right","padding":"7px 10px","color":RED}),
                html.Td(cap,                            style={"textAlign":"right","padding":"7px 10px"}),
                html.Td(pe,                             style={"textAlign":"right","padding":"7px 10px"}),
            ], style={"borderBottom":f"1px solid {BORDER}",
                      "background":BG3 if t==ticker else "transparent"}))

        table = html.Div([
            html.Div("COMPARATIVE METRICS TABLE", className="section-title"),
            html.Div(style={"overflowX":"auto","fontSize":"11px"}, children=[
                html.Table([
                    html.Thead(html.Tr([
                        html.Th(h, style={"padding":"6px 10px","color":DIM,"fontSize":"9px","letterSpacing":"1px",
                                          "textAlign":"right","fontWeight":"500","whiteSpace":"nowrap",
                                          "borderBottom":f"1px solid {BORDER}"})
                        for h in ["TICKER","PRICE","1D","ANN RET","VOL","SHARPE","SORTINO","MAX DD","BETA","VAR95","MKT CAP","P/E"]
                    ])),
                    html.Tbody(rows),
                ], style={"width":"100%","borderCollapse":"collapse"})
            ])
        ], className="section", style={"marginTop":"16px"})

        return html.Div([
            kpis,
            html.Div([
                html.Div([dcc.Graph(figure=fig_price,  config={"displayModeBar":False})], className="section"),
                html.Div([dcc.Graph(figure=fig_sharpe, config={"displayModeBar":False})], className="section"),
            ], style={"display":"grid","gridTemplateColumns":"1fr 340px","gap":"16px","marginBottom":"16px"}),
            table,
        ])
    except Exception as e:
        return html.Div(f"⚠ Overview error: {e}",
                        style={"color":RED,"padding":"40px","fontFamily":"monospace"})


# ─────────────────────────────────────────────
#  PANEL — MONTE CARLO
# ─────────────────────────────────────────────
@app.callback(Output("panel-montecarlo","children"), Input("selected-ticker","data"))
def render_mc(ticker):
    m     = METRICS[ticker]
    col   = COLORS[TICKERS.index(ticker)]
    paths = monte_carlo(m["returns"], m["last_px"])
    finals = paths[:, -1]
    p5, p25, p50, p75, p95 = np.percentile(finals, [5,25,50,75,95])
    days_ax = np.arange(MC_DAYS + 1)

    # KPIs
    exp_ret = (p50 - m["last_px"]) / m["last_px"] * 100
    prob_profit = (finals > m["last_px"]).mean() * 100

    kpis = html.Div([
        kpi_card("CURRENT PRICE",  f"${m['last_px']:.2f}", "Entry point", TXT),
        kpi_card("BEAR CASE P5",   f"${p5:.2f}", f"{(p5/m['last_px']-1)*100:+.1f}%", RED),
        kpi_card("BASE CASE P50",  f"${p50:.2f}", f"{(p50/m['last_px']-1)*100:+.1f}%", YELLOW),
        kpi_card("BULL CASE P95",  f"${p95:.2f}", f"{(p95/m['last_px']-1)*100:+.1f}%", ACCENT),
        kpi_card("EXPECTED RETURN",f"{exp_ret:+.1f}%", "63-day horizon", sign_color(exp_ret)),
        kpi_card("PROB. OF PROFIT",f"{prob_profit:.0f}%", "P(price > entry)", ACCENT if prob_profit>50 else RED),
    ], style={"display":"grid","gridTemplateColumns":"repeat(6,1fr)","gap":"8px","marginBottom":"16px"})

    # Paths chart
    fig_mc = go.Figure()
    # Plot 100 sampled paths
    sample_idx = np.random.choice(MC_SIMS, size=min(100, MC_SIMS), replace=False)
    for i in sample_idx:
        fig_mc.add_trace(go.Scatter(
            x=days_ax, y=paths[i],
            mode="lines", line=dict(color="#1a3060", width=0.5),
            showlegend=False, hoverinfo="skip"
        ))
    # Percentile bands
    p5_path  = np.percentile(paths, 5,  axis=0)
    p50_path = np.percentile(paths, 50, axis=0)
    p95_path = np.percentile(paths, 95, axis=0)
    fig_mc.add_trace(go.Scatter(x=days_ax, y=p95_path, mode="lines",
        line=dict(color=ACCENT, width=2, dash="dash"), name="P95 Bull"))
    fig_mc.add_trace(go.Scatter(x=days_ax, y=p50_path, mode="lines",
        line=dict(color=YELLOW, width=2), name="P50 Base"))
    fig_mc.add_trace(go.Scatter(x=days_ax, y=p5_path, mode="lines",
        line=dict(color=RED, width=2, dash="dash"), name="P5 Bear"))
    fig_mc.add_hline(y=m["last_px"], line_dash="dot", line_color=DIM, line_width=1)
    styled_fig(fig_mc, 350, f"MONTE CARLO SIMULATION — {ticker}  ({MC_SIMS} PATHS · {MC_DAYS} TRADING DAYS)")
    fig_mc.update_xaxes(title_text="Trading Days")
    fig_mc.update_yaxes(title_text="Price ($)", tickprefix="$")

    # Distribution chart
    fig_dist = go.Figure()
    fig_dist.add_trace(go.Histogram(
        x=finals, nbinsx=60,
        marker=dict(color=col, opacity=0.8, line=dict(color=BG, width=0.3)),
        name="Final Price Distribution"
    ))
    fig_dist.add_vline(x=p5,  line_color=RED,    line_dash="dash", annotation_text="P5",  annotation_font=dict(color=RED))
    fig_dist.add_vline(x=p50, line_color=YELLOW, line_dash="dash", annotation_text="P50", annotation_font=dict(color=YELLOW))
    fig_dist.add_vline(x=p95, line_color=ACCENT, line_dash="dash", annotation_text="P95", annotation_font=dict(color=ACCENT))
    fig_dist.add_vline(x=m["last_px"], line_color=DIM, line_dash="dot",
                       annotation_text="Today", annotation_font=dict(color=DIM))
    styled_fig(fig_dist, 250, "FINAL PRICE DISTRIBUTION (63-DAY HORIZON)")
    fig_dist.update_xaxes(title_text="Price ($)", tickprefix="$")
    fig_dist.update_yaxes(title_text="Frequency")

    return html.Div([
        kpis,
        html.Div([dcc.Graph(figure=fig_mc,   config={"displayModeBar":False})], className="section", style={"marginBottom":"16px"}),
        html.Div([dcc.Graph(figure=fig_dist, config={"displayModeBar":False})], className="section"),
    ])


# ─────────────────────────────────────────────
#  PANEL — CORRELATION
# ─────────────────────────────────────────────
@app.callback(Output("panel-correlation","children"), Input("selected-ticker","data"))
def render_corr(_ticker):
    try:
        tickers = VALID_TICKERS

        # Re-compute corr safely from stored pct_rets
        rets_df = pd.DataFrame({t: METRICS[t]["pct_rets"] for t in tickers})
        corr    = rets_df.corr()
        # Keep only tickers that made it into corr
        tickers = [t for t in tickers if t in corr.columns]
        corr    = corr.loc[tickers, tickers]

        z_vals  = np.round(corr.values.astype(float), 2)
        text_vals = [[f"{v:.2f}" for v in row] for row in z_vals]

        fig_heat = go.Figure(go.Heatmap(
            z=z_vals, x=tickers, y=tickers,
            colorscale=[[0,"#c0392b"],[0.5,"#1a2744"],[1,"#00f5d4"]],
            zmin=-1, zmax=1,
            text=text_vals, texttemplate="%{text}",
            textfont=dict(size=10, color=TXT),
            hovertemplate="%{y} ↔ %{x}: %{z:.3f}<extra></extra>",
            colorbar=dict(title=dict(text="Corr", font=dict(color=TXT)), tickfont=dict(color=TXT))
        ))
        fig_heat.update_layout(
            paper_bgcolor=BG2, plot_bgcolor=BG,
            font=dict(family="IBM Plex Mono, Courier New, monospace", color=TXT, size=11),
            margin=dict(l=60, r=20, t=50, b=60), height=460,
            title=dict(text="RETURN CORRELATION MATRIX (1Y DAILY RETURNS)", font=dict(size=11, color=BLUE), x=0),
        )

        # ── Risk-Return scatter ──
        scatter_data = [
            (t, float(METRICS[t]["ann_vol"])*100, float(METRICS[t]["ann_ret"])*100, float(METRICS[t]["sharpe"]))
            for t in tickers
        ]
        fig_scatter = go.Figure()
        for t, vol, ret, sharpe in scatter_data:
            i = TICKERS.index(t)
            fig_scatter.add_trace(go.Scatter(
                x=[vol], y=[ret], mode="markers+text",
                marker=dict(size=max(8, 12 + abs(sharpe)*4), color=COLORS[i], opacity=0.85,
                            line=dict(color=BG, width=1)),
                text=[t], textposition="top center",
                textfont=dict(color=COLORS[i], size=10),
                name=t,
                hovertemplate=f"<b>{t}</b><br>Vol: {vol:.1f}%<br>Return: {ret:+.1f}%<br>Sharpe: {sharpe:.2f}<extra></extra>"
            ))
        vols = [v for _, v, _, _ in scatter_data]
        fig_scatter.add_shape(type="line", x0=min(vols)-2, x1=max(vols)+2, y0=0, y1=0,
                              line=dict(color=DIM, dash="dot", width=1))
        fig_scatter.update_layout(
            paper_bgcolor=BG2, plot_bgcolor=BG,
            font=dict(family="IBM Plex Mono, Courier New, monospace", color=TXT, size=11),
            margin=dict(l=60, r=20, t=50, b=50), height=320,
            title=dict(text="RISK–RETURN SCATTER  (bubble size ∝ |Sharpe|)", font=dict(size=11, color=BLUE), x=0),
            showlegend=False, hovermode="closest",
            xaxis=dict(title="Ann. Volatility (%)", gridcolor=BORDER, ticksuffix="%"),
            yaxis=dict(title="Ann. Return (%)",     gridcolor=BORDER, ticksuffix="%"),
        )

        return html.Div([
            html.Div([dcc.Graph(figure=fig_heat,    config={"displayModeBar":False})], className="section", style={"marginBottom":"16px"}),
            html.Div([dcc.Graph(figure=fig_scatter, config={"displayModeBar":False})], className="section"),
        ])
    except Exception as e:
        return html.Div(f"⚠ Correlation error: {e}",
                        style={"color":RED,"padding":"40px","fontFamily":"monospace"})


# ─────────────────────────────────────────────
#  PANEL — RISK
# ─────────────────────────────────────────────
@app.callback(Output("panel-risk","children"), Input("selected-ticker","data"))
def render_risk(ticker):
    try:
        m   = METRICS[ticker]
        col = COLORS[TICKERS.index(ticker)]

        # ── Rolling volatility ──
        roll_vol = m["roll_vol"].dropna() * 100
        roll_vol = roll_vol.squeeze()  # ensure Series not DataFrame
        mean_vol = float(roll_vol.mean())

        fig_rvol = go.Figure(go.Scatter(
            x=list(roll_vol.index), y=list(roll_vol.values),
            mode="lines", line=dict(color=YELLOW, width=1.5),
            fill="tozeroy", fillcolor=hex_rgba(YELLOW, 0.1),
            hovertemplate="%{y:.1f}%<extra>30d Rolling Vol</extra>"
        ))
        fig_rvol.add_hline(y=mean_vol, line_dash="dash", line_color=DIM, line_width=1,
                           annotation_text=f"Mean {mean_vol:.1f}%",
                           annotation_font=dict(color=DIM))
        fig_rvol.update_layout(
            paper_bgcolor=BG2, plot_bgcolor=BG,
            font=dict(family="IBM Plex Mono, Courier New, monospace", color=TXT, size=11),
            margin=dict(l=55, r=20, t=45, b=40), height=240,
            title=dict(text=f"30-DAY ROLLING VOLATILITY — {ticker}", font=dict(size=11, color=BLUE), x=0),
            showlegend=False,
            xaxis=dict(gridcolor=BORDER), yaxis=dict(gridcolor=BORDER, ticksuffix="%"),
        )

        # ── Return distribution ──
        daily_rets = m["returns"].squeeze() * 100
        mu, sig    = float(daily_rets.mean()), float(daily_rets.std())
        x_norm     = np.linspace(float(daily_rets.min()), float(daily_rets.max()), 200)
        y_norm     = stats.norm.pdf(x_norm, mu, sig)
        var95_line = float(m["var95"]) * 100

        fig_rdist = go.Figure()
        fig_rdist.add_trace(go.Histogram(
            x=list(daily_rets.values), nbinsx=80, histnorm="probability density",
            marker=dict(color=col, opacity=0.7), name="Actual Returns"
        ))
        fig_rdist.add_trace(go.Scatter(
            x=list(x_norm), y=list(y_norm), mode="lines",
            line=dict(color=ACCENT, width=2, dash="dash"), name="Normal Fit"
        ))
        fig_rdist.add_vline(x=var95_line, line_color=RED, line_dash="dot",
                            annotation_text=f"VaR95 {var95_line:.2f}%",
                            annotation_font=dict(color=RED))
        fig_rdist.update_layout(
            paper_bgcolor=BG2, plot_bgcolor=BG,
            font=dict(family="IBM Plex Mono, Courier New, monospace", color=TXT, size=11),
            margin=dict(l=55, r=20, t=45, b=40), height=240,
            title=dict(text=f"DAILY RETURN DISTRIBUTION — {ticker}", font=dict(size=11, color=BLUE), x=0),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
            xaxis=dict(gridcolor=BORDER, ticksuffix="%"),
            yaxis=dict(gridcolor=BORDER),
        )

        # ── VaR / CVaR bar ──
        var_data = sorted(
            [(t, -float(METRICS[t]["var95"])*100, -float(METRICS[t]["cvar95"])*100)
             for t in VALID_TICKERS],
            key=lambda x: x[1], reverse=True
        )
        fig_var = go.Figure()
        fig_var.add_trace(go.Bar(
            x=[t for t,_,_ in var_data], y=[v for _,v,_ in var_data],
            name="VaR 95%",
            marker_color=[COLORS[TICKERS.index(t)] for t,_,_ in var_data],
            hovertemplate="%{x}: %{y:.2f}%<extra>VaR 95%</extra>"
        ))
        fig_var.add_trace(go.Bar(
            x=[t for t,_,_ in var_data], y=[c for _,_,c in var_data],
            name="CVaR 95%", marker=dict(color=RED, opacity=0.5),
            hovertemplate="%{x}: %{y:.2f}%<extra>CVaR 95%</extra>"
        ))
        fig_var.update_layout(
            paper_bgcolor=BG2, plot_bgcolor=BG,
            font=dict(family="IBM Plex Mono, Courier New, monospace", color=TXT, size=11),
            margin=dict(l=55, r=20, t=45, b=40), height=260,
            title=dict(text="VALUE AT RISK & CONDITIONAL VAR (95%)", font=dict(size=11, color=BLUE), x=0),
            barmode="group",
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
            xaxis=dict(gridcolor=BORDER),
            yaxis=dict(gridcolor=BORDER, ticksuffix="%"),
        )

        # ── Quant scorecards ──
        def quant_score(t):
            mm = METRICS[t]
            s  = (min(2, max(0, mm["sharpe"]))
                + min(2, max(0, mm["sortino"] / 2))
                + min(2, max(0, (1 - abs(mm["max_dd"])) * 2))
                + min(2, max(0, (1 - mm["beta"] / 3) * 2))
                + min(2, max(0, (mm["ann_ret"] + 0.2) / 0.8 * 2)))
            return round(min(10, max(0, s)), 1)

        score_cards = []
        for i, t in enumerate(VALID_TICKERS):
            score = quant_score(t)
            sc    = ACCENT if score >= 7 else YELLOW if score >= 5 else RED
            score_cards.append(html.Div([
                html.Div(t, style={"color":COLORS[i],"fontWeight":"600","letterSpacing":"2px",
                                   "fontSize":"13px","marginBottom":"8px","textAlign":"center"}),
                html.Div(f"{score}", style={"fontSize":"30px","fontWeight":"600","color":sc,
                                            "textAlign":"center","lineHeight":"1"}),
                html.Div("QUANT SCORE", style={"color":DIM,"fontSize":"9px","letterSpacing":"1px",
                                               "textAlign":"center","marginTop":"4px"}),
                html.Div(style={"marginTop":"8px","height":"3px","background":BORDER,"borderRadius":"2px"},
                         children=[html.Div(style={"width":f"{score*10}%","height":"100%",
                                                   "background":sc,"borderRadius":"2px"})]),
                html.Div([
                    html.Span(f"β {METRICS[t]['beta']:.1f}", style={"color":DIM}),
                    html.Span(f"σ {METRICS[t]['ann_vol']*100:.0f}%", style={"color":DIM}),
                ], style={"display":"flex","justifyContent":"space-between","marginTop":"8px","fontSize":"9px"}),
            ], className="metric-card",
               style={"background":BG3 if t==ticker else BG, "textAlign":"center"}))

        return html.Div([
            html.Div([
                html.Div([dcc.Graph(figure=fig_rvol,  config={"displayModeBar":False})], className="section"),
                html.Div([dcc.Graph(figure=fig_rdist, config={"displayModeBar":False})], className="section"),
            ], style={"display":"grid","gridTemplateColumns":"1fr 1fr","gap":"16px","marginBottom":"16px"}),
            html.Div([dcc.Graph(figure=fig_var, config={"displayModeBar":False})],
                     className="section", style={"marginBottom":"16px"}),
            html.Div([
                html.Div("QUANT SCORECARDS", className="section-title"),
                html.Div(score_cards, style={"display":"grid","gridTemplateColumns":"repeat(5,1fr)","gap":"8px"}),
            ], className="section"),
        ])
    except Exception as e:
        return html.Div(f"⚠ Risk error: {e}",
                        style={"color":RED,"padding":"40px","fontFamily":"monospace"})


# ─────────────────────────────────────────────
#  PANEL — COMPARISON
# ─────────────────────────────────────────────
@app.callback(Output("panel-comparison","children"), Input("selected-ticker","data"))
def render_comparison(ticker):
    m   = METRICS[ticker]
    col = COLORS[TICKERS.index(ticker)]

    # vs SPY & QQQ cumulative return
    prices  = m["prices"]
    cum_sel = (prices / prices.iloc[0] - 1) * 100

    fig_cmp = go.Figure()
    fig_cmp.add_trace(go.Scatter(
        x=cum_sel.index, y=cum_sel.values, mode="lines",
        line=dict(color=col, width=2), name=ticker,
        hovertemplate="%{y:+.2f}%<extra>" + ticker + "</extra>"
    ))
    if METRICS.get("_spy") is not None:
        spy = METRICS["_spy"].reindex(prices.index).ffill()
        cum_spy = (spy / spy.iloc[0] - 1) * 100
        fig_cmp.add_trace(go.Scatter(
            x=cum_spy.index, y=cum_spy.values, mode="lines",
            line=dict(color=YELLOW, width=1.5, dash="dash"), name="S&P 500",
            hovertemplate="%{y:+.2f}%<extra>S&P 500</extra>"
        ))
    if METRICS.get("_qqq") is not None:
        qqq = METRICS["_qqq"].reindex(prices.index).ffill()
        cum_qqq = (qqq / qqq.iloc[0] - 1) * 100
        fig_cmp.add_trace(go.Scatter(
            x=cum_qqq.index, y=cum_qqq.values, mode="lines",
            line=dict(color=BLUE, width=1.5, dash="dot"), name="NASDAQ (QQQ)",
            hovertemplate="%{y:+.2f}%<extra>NASDAQ</extra>"
        ))
    fig_cmp.add_hline(y=0, line_color=DIM, line_dash="dot", line_width=1)
    styled_fig(fig_cmp, 300, f"{ticker} vs S&P 500 vs NASDAQ — CUMULATIVE RETURN (1Y)")
    fig_cmp.update_yaxes(ticksuffix="%")

    # All tickers cumulative returns overlay
    fig_all = go.Figure()
    for i, t in enumerate(VALID_TICKERS):
        p = METRICS[t]["prices"]
        cum = (p / p.iloc[0] - 1) * 100
        fig_all.add_trace(go.Scatter(
            x=cum.index, y=cum.values, mode="lines",
            line=dict(color=COLORS[i], width=1.5 if t==ticker else 1,
                      dash="solid" if t==ticker else "solid"),
            name=t, opacity=1.0 if t==ticker else 0.55,
            hovertemplate=f"%{{y:+.2f}}%<extra>{t}</extra>"
        ))
    styled_fig(fig_all, 280, "ALL TICKERS — CUMULATIVE RETURN COMPARISON (1Y)")
    fig_all.update_yaxes(ticksuffix="%")

    # Annual return bar
    ret_data = sorted([(t, METRICS[t]["ann_ret"]*100) for t in VALID_TICKERS], key=lambda x:-x[1])
    fig_ret = go.Figure(go.Bar(
        x=[t for t,_ in ret_data], y=[r for _,r in ret_data],
        marker_color=[ACCENT if r>=0 else RED for _,r in ret_data],
        text=[f"{r:+.1f}%" for _,r in ret_data], textposition="outside",
        hovertemplate="%{x}: %{y:+.1f}%<extra>Ann. Return</extra>"
    ))
    fig_ret.add_hline(y=0, line_color=DIM, line_width=1)
    styled_fig(fig_ret, 250, "ANNUALISED RETURN — ALL TICKERS")
    fig_ret.update_yaxes(ticksuffix="%")

    # Market cap bar
    cap_data = sorted([(t, METRICS[t]["mkt_cap"]/1e9) for t in VALID_TICKERS
                       if METRICS[t]["mkt_cap"]], key=lambda x:-x[1])
    fig_cap = go.Figure(go.Bar(
        x=[t for t,_ in cap_data], y=[c for _,c in cap_data],
        marker_color=[COLORS[TICKERS.index(t)] for t,_ in cap_data],
        text=[f"${c:.0f}B" for _,c in cap_data], textposition="outside",
        hovertemplate="%{x}: $%{y:.0f}B<extra>Market Cap</extra>"
    ))
    styled_fig(fig_cap, 250, "MARKET CAPITALISATION ($B)")
    fig_cap.update_yaxes(tickprefix="$", ticksuffix="B")

    return html.Div([
        html.Div([dcc.Graph(figure=fig_cmp, config={"displayModeBar":False})], className="section", style={"marginBottom":"16px"}),
        html.Div([dcc.Graph(figure=fig_all, config={"displayModeBar":False})], className="section", style={"marginBottom":"16px"}),
        html.Div([
            html.Div([dcc.Graph(figure=fig_ret, config={"displayModeBar":False})], className="section"),
            html.Div([dcc.Graph(figure=fig_cap, config={"displayModeBar":False})], className="section"),
        ], style={"display":"grid","gridTemplateColumns":"1fr 1fr","gap":"16px"}),
    ])


# ─────────────────────────────────────────────
#  PANEL — CANDLESTICK + INDICATORS
# ─────────────────────────────────────────────
@app.callback(Output("panel-candlestick","children"), Input("selected-ticker","data"))
def render_candlestick(ticker):
    try:
        col = COLORS[TICKERS.index(ticker)]
        end   = datetime.today()
        start = end - timedelta(days=400)
        raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw = raw.dropna()

        o = raw["Open"].squeeze()
        h = raw["High"].squeeze()
        l = raw["Low"].squeeze()
        c = raw["Close"].squeeze()
        v = raw["Volume"].squeeze() if "Volume" in raw.columns else None

        # ── Indicators ──
        ema20  = c.ewm(span=20).mean()
        ema50  = c.ewm(span=50).mean()
        bb_mid = c.rolling(20).mean()
        bb_std = c.rolling(20).std()
        bb_up  = bb_mid + 2 * bb_std
        bb_dn  = bb_mid - 2 * bb_std

        delta  = c.diff()
        gain   = delta.clip(lower=0).rolling(14).mean()
        loss   = (-delta.clip(upper=0)).rolling(14).mean()
        rs     = gain / (loss + 1e-10)
        rsi    = 100 - 100 / (1 + rs)

        ema12  = c.ewm(span=12).mean()
        ema26  = c.ewm(span=26).mean()
        macd_line   = ema12 - ema26
        macd_signal = macd_line.ewm(span=9).mean()
        macd_hist   = macd_line - macd_signal

        fig = make_subplots(
            rows=4, cols=1, shared_xaxes=True,
            row_heights=[0.50, 0.18, 0.16, 0.16],
            vertical_spacing=0.02,
        )

        # Candlestick
        fig.add_trace(go.Candlestick(
            x=raw.index, open=o, high=h, low=l, close=c,
            increasing_line_color="#00c853", decreasing_line_color="#ff1744",
            increasing_fillcolor="#00c853", decreasing_fillcolor="#ff1744",
            name="OHLC", showlegend=False,
        ), row=1, col=1)

        # Bollinger Bands
        fig.add_trace(go.Scatter(x=list(bb_up.index), y=list(bb_up.values), mode="lines",
            line=dict(color=YELLOW, width=0.8, dash="dot"), name="BB Upper", showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=list(bb_dn.index), y=list(bb_dn.values), mode="lines",
            line=dict(color=YELLOW, width=0.8, dash="dot"), name="BB Lower",
            fill="tonexty", fillcolor=hex_rgba(YELLOW, 0.04), showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=list(ema20.index), y=list(ema20.values), mode="lines",
            line=dict(color=ACCENT, width=1.2), name="EMA20", showlegend=True), row=1, col=1)
        fig.add_trace(go.Scatter(x=list(ema50.index), y=list(ema50.values), mode="lines",
            line=dict(color=BLUE, width=1.2), name="EMA50", showlegend=True), row=1, col=1)

        # Volume
        if v is not None:
            colors_v = [hex_rgba("#00c853", 0.6) if c.iloc[i] >= o.iloc[i] else hex_rgba("#ff1744", 0.6)
                        for i in range(len(c))]
            fig.add_trace(go.Bar(x=list(v.index), y=list(v.values),
                marker_color=colors_v, name="Volume", showlegend=False), row=2, col=1)

        # RSI
        fig.add_trace(go.Scatter(x=list(rsi.index), y=list(rsi.values), mode="lines",
            line=dict(color=col, width=1.2), name="RSI", showlegend=False), row=3, col=1)
        fig.add_hline(y=70, line_color="#ff1744", line_dash="dot", line_width=0.8, row=3, col=1)
        fig.add_hline(y=30, line_color="#00c853", line_dash="dot", line_width=0.8, row=3, col=1)

        # MACD
        fig.add_trace(go.Scatter(x=list(macd_line.index), y=list(macd_line.values), mode="lines",
            line=dict(color=ACCENT, width=1.2), name="MACD", showlegend=False), row=4, col=1)
        fig.add_trace(go.Scatter(x=list(macd_signal.index), y=list(macd_signal.values), mode="lines",
            line=dict(color=YELLOW, width=1.2), name="Signal", showlegend=False), row=4, col=1)
        hist_colors = [hex_rgba("#00c853", 0.7) if v >= 0 else hex_rgba("#ff1744", 0.7)
                       for v in macd_hist.values]
        fig.add_trace(go.Bar(x=list(macd_hist.index), y=list(macd_hist.values),
            marker_color=hist_colors, name="Histogram", showlegend=False), row=4, col=1)

        fig.update_layout(
            paper_bgcolor=BG2, plot_bgcolor=BG,
            font=dict(family="IBM Plex Mono, Courier New, monospace", color=TXT, size=11),
            margin=dict(l=55, r=20, t=45, b=40), height=700,
            title=dict(text=f"CANDLESTICK + INDICATORS — {ticker}", font=dict(size=11, color=BLUE), x=0),
            legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", x=0, y=1.01, font=dict(size=10)),
            xaxis_rangeslider_visible=False,
            hovermode="x unified",
        )
        fig.update_xaxes(showgrid=True, gridcolor=BORDER, linecolor=BORDER)
        fig.update_yaxes(showgrid=True, gridcolor=BORDER, linecolor=BORDER)
        fig.update_yaxis(title_text="Price", row=1, col=1)
        fig.update_yaxis(title_text="Vol",   row=2, col=1)
        fig.update_yaxis(title_text="RSI",   row=3, col=1, range=[0,100])
        fig.update_yaxis(title_text="MACD",  row=4, col=1)

        # Last RSI reading
        last_rsi = float(rsi.iloc[-1])
        rsi_txt  = "OVERBOUGHT" if last_rsi > 70 else "OVERSOLD" if last_rsi < 30 else "NEUTRAL"
        rsi_col  = "#ff1744" if last_rsi > 70 else "#00c853" if last_rsi < 30 else YELLOW

        last_macd = float(macd_line.iloc[-1])
        macd_txt  = "BULLISH" if last_macd > 0 else "BEARISH"
        macd_col  = "#00c853" if last_macd > 0 else "#ff1744"

        ema_cross = "ABOVE EMA50 ✓" if float(c.iloc[-1]) > float(ema50.iloc[-1]) else "BELOW EMA50 ✗"
        ema_col   = "#00c853" if float(c.iloc[-1]) > float(ema50.iloc[-1]) else "#ff1744"

        info_bar = html.Div([
            html.Div([html.Div("RSI(14)", style={"color":DIM,"fontSize":"9px","letterSpacing":"2px","marginBottom":"4px"}),
                      html.Div(f"{last_rsi:.1f} — {rsi_txt}", style={"color":rsi_col,"fontWeight":"600"})],
                     className="metric-card"),
            html.Div([html.Div("MACD", style={"color":DIM,"fontSize":"9px","letterSpacing":"2px","marginBottom":"4px"}),
                      html.Div(f"{last_macd:.3f} — {macd_txt}", style={"color":macd_col,"fontWeight":"600"})],
                     className="metric-card"),
            html.Div([html.Div("EMA TREND", style={"color":DIM,"fontSize":"9px","letterSpacing":"2px","marginBottom":"4px"}),
                      html.Div(ema_cross, style={"color":ema_col,"fontWeight":"600"})],
                     className="metric-card"),
            html.Div([html.Div("BB WIDTH", style={"color":DIM,"fontSize":"9px","letterSpacing":"2px","marginBottom":"4px"}),
                      html.Div(f"{float((bb_up.iloc[-1]-bb_dn.iloc[-1])/bb_mid.iloc[-1]*100):.1f}% — {'WIDE' if (bb_up.iloc[-1]-bb_dn.iloc[-1])/bb_mid.iloc[-1]>0.1 else 'TIGHT'}",
                               style={"color":YELLOW,"fontWeight":"600"})],
                     className="metric-card"),
        ], style={"display":"grid","gridTemplateColumns":"repeat(4,1fr)","gap":"8px","marginBottom":"16px"})

        return html.Div([
            info_bar,
            html.Div([dcc.Graph(figure=fig, config={"displayModeBar":False})], className="section"),
        ])
    except Exception as e:
        return html.Div(f"⚠ Candlestick error: {e}", style={"color":RED,"padding":"40px","fontFamily":"monospace"})


# ─────────────────────────────────────────────
#  PANEL — PORTFOLIO BUILDER
# ─────────────────────────────────────────────
@app.callback(
    Output("panel-portfolio","children"),
    [Input("selected-ticker","data"), Input("portfolio-weights","data")]
)
def render_portfolio(ticker, weights):
    try:
        if not weights:
            weights = {t: round(100/len(VALID_TICKERS),1) for t in VALID_TICKERS}

        result = portfolio_optimize(weights, METRICS)

        sliders = []
        for i, t in enumerate(VALID_TICKERS):
            w = weights.get(t, 0)
            sliders.append(html.Div([
                html.Div([
                    html.Span(t, style={"color":COLORS[i],"fontWeight":"600","letterSpacing":"1px","fontSize":"11px"}),
                    html.Span(f"{w:.0f}%", id=f"w-label-{t}",
                              style={"color":TXT,"fontSize":"11px","marginLeft":"8px"}),
                ], style={"display":"flex","justifyContent":"space-between","marginBottom":"4px"}),
                dcc.Slider(id=f"slider-{t}", min=0, max=100, step=1, value=w,
                           marks=None, tooltip={"always_visible":False}),
            ], style={"marginBottom":"12px"}))

        if result:
            # Efficient frontier scatter
            fig_ef = go.Figure()
            xs = [p[0] for p in result["frontier"]]
            ys = [p[1] for p in result["frontier"]]
            ss = [p[2] for p in result["frontier"]]
            fig_ef.add_trace(go.Scatter(
                x=xs, y=ys, mode="markers",
                marker=dict(color=ss, colorscale=[[0,"#ff1744"],[0.5,YELLOW],[1,"#00c853"]],
                            size=3, opacity=0.5, showscale=True,
                            colorbar=dict(title=dict(text="Sharpe",font=dict(color=TXT)),
                                          tickfont=dict(color=TXT), thickness=10)),
                name="Simulated Portfolios", hovertemplate="Vol: %{x:.1f}%<br>Ret: %{y:.1f}%<extra></extra>"
            ))
            # User portfolio
            fig_ef.add_trace(go.Scatter(
                x=[result["vol"]], y=[result["ret"]], mode="markers+text",
                marker=dict(color=ACCENT, size=14, symbol="star",
                            line=dict(color=BG, width=1)),
                text=["YOUR PORTFOLIO"], textposition="top center",
                textfont=dict(color=ACCENT, size=10),
                name="Your Portfolio"
            ))
            # Max Sharpe
            fig_ef.add_trace(go.Scatter(
                x=[result["max_sharpe"]["vol"]], y=[result["max_sharpe"]["ret"]],
                mode="markers+text",
                marker=dict(color="#00c853", size=12, symbol="diamond"),
                text=["MAX SHARPE"], textposition="top right",
                textfont=dict(color="#00c853", size=10),
                name="Max Sharpe"
            ))
            # Min Variance
            fig_ef.add_trace(go.Scatter(
                x=[result["min_var"]["vol"]], y=[result["min_var"]["ret"]],
                mode="markers+text",
                marker=dict(color=BLUE, size=12, symbol="diamond"),
                text=["MIN VARIANCE"], textposition="top right",
                textfont=dict(color=BLUE, size=10),
                name="Min Variance"
            ))
            fig_ef.update_layout(
                paper_bgcolor=BG2, plot_bgcolor=BG,
                font=dict(family="IBM Plex Mono, Courier New, monospace", color=TXT, size=11),
                margin=dict(l=55, r=20, t=45, b=40), height=380,
                title=dict(text="EFFICIENT FRONTIER (3,000 RANDOM PORTFOLIOS)", font=dict(size=11,color=BLUE), x=0),
                legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
                hovermode="closest",
                xaxis=dict(title="Volatility (%)", gridcolor=BORDER, ticksuffix="%"),
                yaxis=dict(title="Return (%)", gridcolor=BORDER, ticksuffix="%"),
            )

            # Weights bar for max sharpe and min var
            ms_w = {result["tickers"][i]: round(result["max_sharpe"]["w"][i]*100,1) for i in range(len(result["tickers"]))}
            mv_w = {result["tickers"][i]: round(result["min_var"]["w"][i]*100,1) for i in range(len(result["tickers"]))}

            fig_wts = go.Figure()
            fig_wts.add_trace(go.Bar(
                x=list(ms_w.keys()), y=list(ms_w.values()),
                name="Max Sharpe", marker_color="#00c853", opacity=0.85,
            ))
            fig_wts.add_trace(go.Bar(
                x=list(mv_w.keys()), y=list(mv_w.values()),
                name="Min Variance", marker_color=BLUE, opacity=0.85,
            ))
            fig_wts.update_layout(
                paper_bgcolor=BG2, plot_bgcolor=BG,
                font=dict(family="IBM Plex Mono, Courier New, monospace", color=TXT, size=11),
                margin=dict(l=55, r=20, t=45, b=40), height=240,
                title=dict(text="OPTIMAL WEIGHT ALLOCATION", font=dict(size=11,color=BLUE), x=0),
                barmode="group", legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
                xaxis=dict(gridcolor=BORDER), yaxis=dict(gridcolor=BORDER, ticksuffix="%"),
            )

            port_kpis = html.Div([
                html.Div([html.Div("YOUR PORTFOLIO", style={"color":DIM,"fontSize":"9px","letterSpacing":"2px","marginBottom":"6px"}),
                          html.Div(f"{result['ret']:+.1f}%", style={"fontSize":"20px","fontWeight":"600","color":sign_color(result['ret'])}),
                          html.Div("Ann. Return", style={"color":DIM,"fontSize":"10px","marginTop":"4px"})], className="metric-card"),
                html.Div([html.Div("VOLATILITY", style={"color":DIM,"fontSize":"9px","letterSpacing":"2px","marginBottom":"6px"}),
                          html.Div(f"{result['vol']:.1f}%", style={"fontSize":"20px","fontWeight":"600","color":YELLOW}),
                          html.Div("Ann. σ", style={"color":DIM,"fontSize":"10px","marginTop":"4px"})], className="metric-card"),
                html.Div([html.Div("SHARPE", style={"color":DIM,"fontSize":"9px","letterSpacing":"2px","marginBottom":"6px"}),
                          html.Div(f"{result['sharpe']:.2f}", style={"fontSize":"20px","fontWeight":"600","color":ACCENT if result['sharpe']>1 else RED}),
                          html.Div("Ratio", style={"color":DIM,"fontSize":"10px","marginTop":"4px"})], className="metric-card"),
                html.Div([html.Div("MAX SHARPE WT", style={"color":DIM,"fontSize":"9px","letterSpacing":"2px","marginBottom":"6px"}),
                          html.Div(f"{result['max_sharpe']['ret']:+.1f}% / {result['max_sharpe']['vol']:.1f}%",
                                   style={"fontSize":"14px","fontWeight":"600","color":"#00c853"}),
                          html.Div("Ret / Vol", style={"color":DIM,"fontSize":"10px","marginTop":"4px"})], className="metric-card"),
                html.Div([html.Div("MIN VAR WT", style={"color":DIM,"fontSize":"9px","letterSpacing":"2px","marginBottom":"6px"}),
                          html.Div(f"{result['min_var']['ret']:+.1f}% / {result['min_var']['vol']:.1f}%",
                                   style={"fontSize":"14px","fontWeight":"600","color":BLUE}),
                          html.Div("Ret / Vol", style={"color":DIM,"fontSize":"10px","marginTop":"4px"})], className="metric-card"),
            ], style={"display":"grid","gridTemplateColumns":"repeat(5,1fr)","gap":"8px","marginBottom":"16px"})
        else:
            fig_ef = go.Figure()
            fig_wts = go.Figure()
            port_kpis = html.Div()

        return html.Div([
            port_kpis,
            html.Div([
                html.Div([
                    html.Div("ADJUST WEIGHTS", className="section-title"),
                    html.Div("(weights are indicative — adjust sliders to explore)", style={"color":DIM,"fontSize":"9px","marginBottom":"16px"}),
                    *sliders
                ], className="section", style={"flex":"0 0 260px"}),
                html.Div([
                    html.Div([dcc.Graph(figure=fig_ef, config={"displayModeBar":False})], className="section", style={"marginBottom":"16px"}),
                    html.Div([dcc.Graph(figure=fig_wts, config={"displayModeBar":False})], className="section"),
                ], style={"flex":"1"}),
            ], style={"display":"flex","gap":"16px"}),
        ])
    except Exception as e:
        return html.Div(f"⚠ Portfolio error: {e}", style={"color":RED,"padding":"40px","fontFamily":"monospace"})


# ─────────────────────────────────────────────
#  PANEL — MACRO & COMMODITIES
# ─────────────────────────────────────────────
@app.callback(Output("panel-macro","children"), Input("selected-ticker","data"))
def render_macro(_ticker):
    try:
        if not MACRO:
            return html.Div("⚠ No macro data available. Check your internet connection.",
                            style={"color":YELLOW,"padding":"40px","fontFamily":"monospace"})

        # KPI cards for each macro instrument
        macro_cards = []
        for sym, d in MACRO.items():
            chg_col = "#00c853" if d["chg_1d"] >= 0 else "#ff1744"
            ytd_col = "#00c853" if d["ytd"] >= 0 else "#ff1744"
            macro_cards.append(html.Div([
                html.Div(d["name"], style={"color":DIM,"fontSize":"9px","letterSpacing":"1px","marginBottom":"4px"}),
                html.Div(f"{d['price']:.2f}", style={"fontSize":"18px","fontWeight":"600","color":TXT,"lineHeight":"1"}),
                html.Div([
                    html.Span(f"{d['chg_1d']:+.2f}% 1D", style={"color":chg_col,"fontSize":"10px","marginRight":"8px"}),
                    html.Span(f"{d['ytd']:+.1f}% YTD", style={"color":ytd_col,"fontSize":"10px"}),
                ], style={"marginTop":"4px"}),
            ], className="metric-card"))

        kpi_grid = html.Div(macro_cards,
                            style={"display":"grid","gridTemplateColumns":"repeat(4,1fr)","gap":"8px","marginBottom":"16px"})

        # Macro history chart
        fig_macro = go.Figure()
        macro_colors_list = [ACCENT, YELLOW, RED, BLUE, "#00c853", "#a29bfe", "#fd79a8"]
        for idx, (sym, d) in enumerate(MACRO.items()):
            p = d["history"]
            norm = (p / p.iloc[0] - 1) * 100
            fig_macro.add_trace(go.Scatter(
                x=list(norm.index), y=list(norm.values), mode="lines",
                line=dict(color=macro_colors_list[idx % len(macro_colors_list)], width=1.5),
                name=d["name"],
                hovertemplate=f"%{{y:+.2f}}%<extra>{d['name']}</extra>"
            ))
        fig_macro.add_hline(y=0, line_color=DIM, line_dash="dot", line_width=1)
        fig_macro.update_layout(
            paper_bgcolor=BG2, plot_bgcolor=BG,
            font=dict(family="IBM Plex Mono, Courier New, monospace", color=TXT, size=11),
            margin=dict(l=55, r=20, t=45, b=40), height=320,
            title=dict(text="MACRO INSTRUMENTS — NORMALISED RETURN (1Y)", font=dict(size=11,color=BLUE), x=0),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
            hovermode="x unified",
            xaxis=dict(gridcolor=BORDER), yaxis=dict(gridcolor=BORDER, ticksuffix="%"),
        )

        # VIX vs Tech correlation note
        vix_note = ""
        if "^VIX" in MACRO:
            vix_val = MACRO["^VIX"]["price"]
            if vix_val > 30:
                vix_note = f"⚠ VIX at {vix_val:.1f} — HIGH FEAR. Equity risk elevated."
                vix_col  = "#ff1744"
            elif vix_val > 20:
                vix_note = f"◈ VIX at {vix_val:.1f} — MODERATE volatility regime."
                vix_col  = YELLOW
            else:
                vix_note = f"✓ VIX at {vix_val:.1f} — LOW fear. Risk-on environment."
                vix_col  = "#00c853"
        else:
            vix_note, vix_col = "", DIM

        # Individual macro charts
        charts = []
        for sym, d in MACRO.items():
            p = d["history"]
            fig_s = go.Figure(go.Scatter(
                x=list(p.index), y=list(p.values), mode="lines",
                line=dict(color=macro_colors_list[list(MACRO.keys()).index(sym) % len(macro_colors_list)], width=1.5),
                fill="tozeroy", fillcolor=hex_rgba(macro_colors_list[list(MACRO.keys()).index(sym) % len(macro_colors_list)], 0.08),
                hovertemplate="%{y:.2f}<extra></extra>"
            ))
            fig_s.update_layout(
                paper_bgcolor=BG2, plot_bgcolor=BG,
                font=dict(family="IBM Plex Mono, Courier New, monospace", color=TXT, size=10),
                margin=dict(l=40, r=8, t=30, b=30), height=160,
                title=dict(text=d["name"], font=dict(size=10,color=BLUE), x=0),
                showlegend=False, hovermode="x unified",
                xaxis=dict(gridcolor=BORDER, showticklabels=False),
                yaxis=dict(gridcolor=BORDER),
            )
            charts.append(html.Div([dcc.Graph(figure=fig_s, config={"displayModeBar":False})], className="section"))

        macro_keys = list(MACRO_TICKERS.keys())
        top_charts = [charts[i] for i in range(len(charts)) if macro_keys[i] not in ["CL=F", "BTC-USD"]]
        bottom_charts = [charts[i] for i in range(len(charts)) if macro_keys[i] in ["CL=F", "BTC-USD"]]
        chart_grid = html.Div([
            html.Div(top_charts, style={"display":"grid","gridTemplateColumns":f"repeat({len(top_charts)},1fr)","gap":"8px","marginTop":"16px"}),
            html.Div(bottom_charts, style={"display":"grid","gridTemplateColumns":"repeat(2,1fr)","gap":"8px","marginTop":"8px"}),
        ])

        return html.Div([
            kpi_grid,
            html.Div(vix_note, style={"color":vix_col,"fontSize":"11px","padding":"10px 16px",
                                       "background":BG2,"border":f"1px solid {BORDER}","borderRadius":"4px",
                                       "marginBottom":"16px","letterSpacing":"1px"}) if vix_note else html.Div(),
            html.Div([dcc.Graph(figure=fig_macro, config={"displayModeBar":False})], className="section"),
            chart_grid,
        ])
    except Exception as e:
        return html.Div(f"⚠ Macro error: {e}", style={"color":RED,"padding":"40px","fontFamily":"monospace"})


# ─────────────────────────────────────────────
#  PANEL — INVEST SIGNAL
# ─────────────────────────────────────────────
@app.callback(Output("panel-signal","children"), Input("selected-ticker","data"))
def render_signal(ticker):
    try:
        score, verdict, v_color, factors = compute_invest_signal(ticker, METRICS)
        m = METRICS[ticker]

        # Big signal card
        signal_card = html.Div([
            html.Div(f"{NAMES.get(ticker, ticker)} ({ticker})",
                     style={"color":DIM,"fontSize":"10px","letterSpacing":"2px","marginBottom":"8px"}),
            html.Div(verdict,
                     style={"fontSize":"52px","fontWeight":"600","color":v_color,
                            "letterSpacing":"6px","lineHeight":"1","fontFamily":"Bebas Neue"}),
            html.Div(f"Composite Score: {score}/100",
                     style={"color":TXT,"fontSize":"13px","marginTop":"8px"}),
            # Score bar
            html.Div(style={"marginTop":"12px","height":"6px","background":BORDER,"borderRadius":"3px"},
                     children=[html.Div(style={"width":f"{score}%","height":"100%",
                                               "background":v_color,"borderRadius":"3px",
                                               "transition":"width 0.5s"})]),
            html.Div([
                html.Span("0", style={"color":DIM,"fontSize":"9px"}),
                html.Span("SELL < 42 | 42 ≤ HOLD < 68 | BUY ≥ 68", style={"color":DIM,"fontSize":"9px"}),
                html.Span("100", style={"color":DIM,"fontSize":"9px"}),
            ], style={"display":"flex","justifyContent":"space-between","marginTop":"4px"}),
        ], style={"background":BG3,"border":f"2px solid {v_color}","borderRadius":"6px",
                  "padding":"24px 32px","textAlign":"center","marginBottom":"20px"})

        # Factor breakdown
        factor_cards = []
        for fname, fscore in factors.items():
            fc = "#00c853" if fscore >= 65 else YELLOW if fscore >= 40 else "#ff1744"
            factor_cards.append(html.Div([
                html.Div(fname.upper(), style={"color":DIM,"fontSize":"9px","letterSpacing":"1px","marginBottom":"6px"}),
                html.Div(f"{fscore:.0f}", style={"fontSize":"22px","fontWeight":"600","color":fc,"lineHeight":"1"}),
                html.Div(style={"marginTop":"6px","height":"3px","background":BORDER,"borderRadius":"2px"},
                         children=[html.Div(style={"width":f"{fscore}%","height":"100%",
                                                   "background":fc,"borderRadius":"2px"})]),
                html.Div("/100", style={"color":DIM,"fontSize":"9px","marginTop":"3px"}),
            ], className="metric-card", style={"textAlign":"center"}))

        factor_grid = html.Div(factor_cards,
                               style={"display":"grid","gridTemplateColumns":"repeat(4,1fr)","gap":"8px","marginBottom":"20px"})

        # All tickers signal comparison
        all_signals = []
        for t in VALID_TICKERS:
            try:
                s, v, vc, _ = compute_invest_signal(t, METRICS)
                all_signals.append((t, s, v, vc))
            except Exception:
                pass
        all_signals.sort(key=lambda x: -x[1])

        fig_sig = go.Figure(go.Bar(
            x=[x[0] for x in all_signals],
            y=[x[1] for x in all_signals],
            marker_color=[x[3] for x in all_signals],
            text=[x[2] for x in all_signals],
            textposition="outside",
            hovertemplate="%{x}: %{y:.1f}<extra>Signal Score</extra>"
        ))
        fig_sig.add_hline(y=68, line_color="#00c853", line_dash="dash", line_width=1,
                          annotation_text="BUY threshold", annotation_font=dict(color="#00c853", size=9))
        fig_sig.add_hline(y=42, line_color="#ff1744", line_dash="dash", line_width=1,
                          annotation_text="SELL threshold", annotation_font=dict(color="#ff1744", size=9))
        fig_sig.update_layout(
            paper_bgcolor=BG2, plot_bgcolor=BG,
            font=dict(family="IBM Plex Mono, Courier New, monospace", color=TXT, size=11),
            margin=dict(l=55, r=20, t=45, b=40), height=280,
            title=dict(text="COMPOSITE INVEST SIGNAL — ALL TICKERS", font=dict(size=11,color=BLUE), x=0),
            xaxis=dict(gridcolor=BORDER), yaxis=dict(gridcolor=BORDER, range=[0,110]),
            showlegend=False,
        )

        # Methodology note
        method_note = html.Div([
            html.Div("SIGNAL METHODOLOGY", style={"color":BLUE,"fontSize":"9px","letterSpacing":"2px","marginBottom":"8px"}),
            html.Div([
                html.Span("8 factors weighted: ", style={"color":DIM}),
                html.Span("Momentum (18%) · Sharpe (18%) · Sortino (12%) · Drawdown (12%) · "
                          "Beta (10%) · VaR (10%) · Volatility (10%) · MC Asymmetry (10%)",
                          style={"color":TXT}),
            ], style={"fontSize":"10px","lineHeight":"1.8"}),
            html.Div("⚠  This is a quantitative model for educational purposes only. "
                     "It does not constitute financial advice.",
                     style={"color":DIM,"fontSize":"9px","marginTop":"8px","fontStyle":"italic"}),
        ], style={"background":BG3,"border":f"1px solid {BORDER}","borderRadius":"4px",
                  "padding":"12px 16px","marginTop":"16px"})

        return html.Div([
            signal_card,
            factor_grid,
            html.Div([dcc.Graph(figure=fig_sig, config={"displayModeBar":False})], className="section"),
            method_note,
        ])
    except Exception as e:
        return html.Div(f"⚠ Signal error: {e}", style={"color":RED,"padding":"40px","fontFamily":"monospace"})


# ─────────────────────────────────────────────
#  RUN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8050))
    host = "0.0.0.0"
    print(f"\n🚀  QuantDesk is running at  http://{host}:{port}\n")
    app.run(debug=False, host=host, port=port)
