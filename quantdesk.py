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
PERIOD  = "1y"       # history window
MC_SIMS = 500        # Monte Carlo paths
MC_DAYS = 63         # ~3 months forecast

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


# ─────────────────────────────────────────────
#  INITIAL LOAD
# ─────────────────────────────────────────────
print("⬇  Fetching live data from Yahoo Finance …")
try:
    CLOSE   = fetch_all()
    METRICS = compute_metrics(CLOSE)
    CORR    = corr_matrix(METRICS)
    print(f"✅  Data loaded — {len([t for t in TICKERS if t in METRICS])} tickers · "
          f"{len(CLOSE)} days")
except Exception as e:
    print(f"❌  Error fetching data: {e}")
    raise

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
                html.Button("Overview",   id="tab-overview",   className="tab active", n_clicks=0),
                html.Button("Monte Carlo",id="tab-montecarlo", className="tab",        n_clicks=0),
                html.Button("Correlation",id="tab-correlation",className="tab",        n_clicks=0),
                html.Button("Risk",       id="tab-risk",       className="tab",        n_clicks=0),
                html.Button("Comparison", id="tab-comparison", className="tab",        n_clicks=0),
            ], style={"display":"flex","borderBottom":f"1px solid {BORDER}","marginBottom":"20px"}),

            # PANELS
            html.Div(id="panel-overview"),
            html.Div(id="panel-montecarlo",  style={"display":"none"}),
            html.Div(id="panel-correlation", style={"display":"none"}),
            html.Div(id="panel-risk",        style={"display":"none"}),
            html.Div(id="panel-comparison",  style={"display":"none"}),

        ], style={"padding":"16px 24px"}),

        # Hidden state stores
        dcc.Store(id="selected-ticker", data=VALID_TICKERS[2] if len(VALID_TICKERS)>2 else VALID_TICKERS[0]),
        dcc.Store(id="active-tab", data="overview"),

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
TABS = ["overview","montecarlo","correlation","risk","comparison"]

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
#  RUN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("\n🚀  QuantDesk is running at  http://127.0.0.1:8050\n")
    app.run(debug=False, port=8050)
