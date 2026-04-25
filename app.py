"""
SalesCast AI — Sales Forecasting Dashboard
Run: streamlit run app.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SalesCast AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background-color: #0d1117; }
[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#161b27 0%,#0d1117 100%);
    border-right: 1px solid #21262d;
}
.kpi { background: linear-gradient(135deg,#1c2333,#212a3e);
    border:1px solid #30363d; border-radius:14px; padding:22px 18px;
    text-align:center; margin-bottom:8px; }
.kpi-label { color:#8b949e; font-size:11px; font-weight:600;
    text-transform:uppercase; letter-spacing:1.2px; margin-bottom:6px; }
.kpi-value { color:#e6edf3; font-size:30px; font-weight:700; }
.kpi-sub { font-size:12px; margin-top:4px; }
.green { color:#3fb950; } .red { color:#f85149; } .blue { color:#58a6ff; }
h1,h2,h3 { color:#e6edf3 !important; }
div[data-testid="stMetricValue"] > div { color:#e6edf3 !important; }
.stTabs [data-baseweb="tab-list"] { gap:4px; background:#161b27;
    border-radius:10px; padding:4px; }
.stTabs [data-baseweb="tab"] { color:#8b949e; border-radius:7px;
    padding:6px 18px; font-weight:500; background:transparent; }
.stTabs [aria-selected="true"] { background:#1f6feb !important; color:#fff !important; }
footer,header,#MainMenu { visibility:hidden; }
</style>
""", unsafe_allow_html=True)

# ── Products ──────────────────────────────────────────────────────────────────
PRODUCTS = [
    {"id":"P001","name":"Laptop Pro 15",      "category":"Computers",   "price":1299.99,"base":45, "trend":0.30,"q4":1.80,"summer":1.40},
    {"id":"P002","name":"Wireless Headphones", "category":"Audio",       "price":149.99, "base":120,"trend":0.50,"q4":2.20,"summer":1.10},
    {"id":"P003","name":"Mechanical Keyboard", "category":"Peripherals", "price":89.99,  "base":85, "trend":0.20,"q4":1.50,"summer":1.20},
    {"id":"P004","name":"USB-C Hub 7-in-1",    "category":"Accessories", "price":49.99,  "base":200,"trend":0.40,"q4":1.60,"summer":1.00},
    {"id":"P005","name":"1080p Webcam",         "category":"Peripherals", "price":79.99,  "base":95, "trend":-0.1,"q4":1.30,"summer":0.90},
    {"id":"P006","name":"27-inch 4K Monitor",  "category":"Displays",    "price":499.99, "base":30, "trend":0.60,"q4":1.70,"summer":1.30},
    {"id":"P007","name":"Gaming Mouse RGB",     "category":"Peripherals", "price":59.99,  "base":110,"trend":0.30,"q4":1.40,"summer":1.50},
    {"id":"P008","name":"Ergonomic Desk Chair", "category":"Furniture",   "price":399.99, "base":20, "trend":0.80,"q4":1.20,"summer":0.80},
    {"id":"P009","name":"Portable SSD 1TB",     "category":"Storage",     "price":99.99,  "base":75, "trend":0.40,"q4":1.60,"summer":1.10},
    {"id":"P010","name":"Wireless Charger Pad", "category":"Accessories", "price":29.99,  "base":250,"trend":0.50,"q4":2.50,"summer":1.00},
]
PLOOKUP = {p["id"]: p for p in PRODUCTS}

# ── Data generation ───────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_or_generate_sales():
    csv = Path("data/sales_data.csv")
    if csv.exists():
        df = pd.read_csv(csv, parse_dates=["date"])
        return df
    return _generate_sales()

def _generate_sales():
    np.random.seed(42)
    dates = pd.date_range("2023-01-02", "2024-12-30", freq="W-MON")
    rows = []
    for p in PRODUCTS:
        n = len(dates)
        base = p["base"]
        woys = np.array([d.isocalendar()[1] for d in dates])
        trend = np.linspace(1.0, 1.0 + p["trend"], n)
        sea = np.ones(n)
        sea[(woys >= 44) & (woys <= 52)] = p["q4"]
        sea[(woys >= 24) & (woys <= 35)] *= p["summer"]
        sea[woys <= 3] = 1.15
        smooth = 1.0 + 0.12 * np.sin(2 * np.pi * woys / 52 - np.pi / 6)
        demand = np.maximum(1, np.round(
            base * trend * sea * smooth + np.random.normal(0, base * 0.08, n)
        )).astype(int)
        for i, d in enumerate(dates):
            rows.append({"date": d, "product_id": p["id"], "product_name": p["name"],
                         "category": p["category"], "quantity_sold": int(demand[i]),
                         "unit_price": p["price"],
                         "total_sales": round(float(demand[i]) * p["price"], 2)})
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)

# ── Feature engineering ───────────────────────────────────────────────────────
FCOLS = [
    "week_num","week_sin","week_cos","month_sin","month_cos",
    "quarter","is_q4","is_summer","is_newyear",
    "lag_1","lag_2","lag_3","lag_4","lag_8","lag_13",
    "roll_mean_4","roll_mean_8","roll_mean_13",
    "roll_std_4","roll_std_8","roll_max_4","roll_min_4",
]

def featurize(df_p: pd.DataFrame) -> pd.DataFrame:
    df = df_p.sort_values("date").reset_index(drop=True).copy()
    woy = df["date"].dt.isocalendar().week.astype(int)
    m   = df["date"].dt.month
    df["week_num"]   = np.arange(len(df))
    df["week_sin"]   = np.sin(2*np.pi*woy/52)
    df["week_cos"]   = np.cos(2*np.pi*woy/52)
    df["month_sin"]  = np.sin(2*np.pi*m/12)
    df["month_cos"]  = np.cos(2*np.pi*m/12)
    df["quarter"]    = df["date"].dt.quarter
    df["is_q4"]      = ((woy>=44)&(woy<=52)).astype(int)
    df["is_summer"]  = ((woy>=24)&(woy<=35)).astype(int)
    df["is_newyear"] = (woy<=3).astype(int)
    qs = df["quantity_sold"]
    for lag in [1,2,3,4,8,13]:
        df[f"lag_{lag}"] = qs.shift(lag)
    df["roll_mean_4"]  = qs.shift(1).rolling(4).mean()
    df["roll_mean_8"]  = qs.shift(1).rolling(8).mean()
    df["roll_mean_13"] = qs.shift(1).rolling(13).mean()
    df["roll_std_4"]   = qs.shift(1).rolling(4).std().fillna(0)
    df["roll_std_8"]   = qs.shift(1).rolling(8).std().fillna(0)
    df["roll_max_4"]   = qs.shift(1).rolling(4).max()
    df["roll_min_4"]   = qs.shift(1).rolling(4).min()
    return df

# ── Model training ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def train_models(_df_hash):
    sales = load_or_generate_sales()
    models, val_mapes = {}, {}
    for p in PRODUCTS:
        pid = p["id"]
        df_p  = featurize(sales[sales["product_id"]==pid]).dropna(subset=FCOLS)
        X, y  = df_p[FCOLS].values, df_p["quantity_sold"].values
        split = max(len(X)-8, int(len(X)*0.85))
        rf = RandomForestRegressor(n_estimators=400, max_depth=12,
                                   min_samples_leaf=2, random_state=42, n_jobs=-1)
        rf.fit(X[:split], y[:split])
        val_mapes[pid] = round(mean_absolute_percentage_error(y[split:], rf.predict(X[split:]))*100, 1)
        models[pid] = rf
    return models, val_mapes

@st.cache_data(show_spinner=False)
def load_or_generate_forecast(_df_hash, _models_key):
    csv = Path("data/forecast_data.csv")
    if csv.exists():
        return pd.read_csv(csv, parse_dates=["date"])
    return _make_forecast()

def _make_forecast():
    sales  = load_or_generate_sales()
    models, _ = train_models(id(sales))
    rows = []
    for p in PRODUCTS:
        pid = p["id"]
        rf  = models[pid]
        df_p = featurize(sales[sales["product_id"]==pid]).dropna(subset=FCOLS)
        history  = list(df_p["quantity_sold"].values)
        last_wn  = int(df_p["week_num"].iloc[-1])
        last_date= df_p["date"].iloc[-1]
        future   = pd.date_range(last_date + pd.Timedelta(weeks=1), periods=26, freq="W-MON")

        def lag(k):
            i = len(history)-k
            return history[i] if i >= 0 else history[0]
        def roll(k, fn):
            w = history[max(0,len(history)-k):]
            return fn(w) if w else history[-1]

        for i, fd in enumerate(future):
            woy = fd.isocalendar()[1]; mo = fd.month
            row = [
                last_wn+i+1,
                np.sin(2*np.pi*woy/52), np.cos(2*np.pi*woy/52),
                np.sin(2*np.pi*mo/12),  np.cos(2*np.pi*mo/12),
                (mo-1)//3+1,
                int(44<=woy<=52), int(24<=woy<=35), int(woy<=3),
                lag(1),lag(2),lag(3),lag(4),lag(8),lag(13),
                roll(4,np.mean),roll(8,np.mean),roll(13,np.mean),
                roll(4,np.std),roll(8,np.std),roll(4,max),roll(4,min),
            ]
            Xp = np.array([row])
            pt  = float(rf.predict(Xp)[0])
            tp  = np.array([t.predict(Xp)[0] for t in rf.estimators_])
            lo, hi = float(np.percentile(tp,10)), float(np.percentile(tp,90))
            history.append(max(1, round(pt)))
            rows.append({"product_id":pid,"product_name":p["name"],"date":fd,
                         "forecasted_quantity":max(1,round(pt)),
                         "lower_bound":max(1,round(lo)),
                         "upper_bound":max(1,round(hi))})
    return pd.DataFrame(rows)

# ── Load data ─────────────────────────────────────────────────────────────────
with st.spinner("🔄  Loading data & training models (first run ~30s)…"):
    sales_df   = load_or_generate_sales()
    df_hash    = len(sales_df)
    models, val_mapes = train_models(df_hash)
    forecast_df = load_or_generate_forecast(df_hash, id(models))

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📈 SalesCast AI")
    st.markdown("---")
    names    = [p["name"] for p in PRODUCTS]
    sel_name = st.selectbox("**Product**", names, index=0)
    sel_pid  = next(p["id"] for p in PRODUCTS if p["name"]==sel_name)
    meta     = PLOOKUP[sel_pid]
    st.markdown(f"**Category:** {meta['category']}")
    st.markdown(f"**Unit Price:** ${meta['price']:,.2f}")
    st.markdown("---")
    show_ci  = st.toggle("Show Confidence Interval", value=True)
    show_hist= st.toggle("Show Historical Data",     value=True)
    st.markdown("---")
    mape_val = val_mapes.get(sel_pid, 0)
    acc      = 100 - mape_val
    color    = "#3fb950" if acc >= 92 else "#d29922"
    st.markdown(f"**Model Accuracy**")
    st.markdown(f"<span style='font-size:28px;font-weight:700;color:{color}'>{acc:.1f}%</span>", unsafe_allow_html=True)
    st.caption(f"MAPE: {mape_val:.1f}% on held-out validation set")

# ── Filter data ───────────────────────────────────────────────────────────────
hist = sales_df[sales_df["product_id"]==sel_pid].copy()
fcast = forecast_df[forecast_df["product_id"]==sel_pid].copy()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(f"# 📦 {sel_name}")
st.caption(f"{meta['category']} · Unit price ${meta['price']:,.2f} · 2-year history + 6-month forecast")

# ── KPI cards ─────────────────────────────────────────────────────────────────
total_hist   = hist["quantity_sold"].sum()
total_rev    = hist["total_sales"].sum()
avg_weekly   = hist["quantity_sold"].mean()
fcast_total  = fcast["forecasted_quantity"].sum()
hist_last6w  = hist.tail(26)["quantity_sold"].sum()
growth_pct   = (fcast_total - hist_last6w) / max(hist_last6w, 1) * 100
peak_row     = hist.loc[hist["quantity_sold"].idxmax()]

c1, c2, c3, c4 = st.columns(4)
for col, label, val, sub, sub_cls in [
    (c1, "Total Units Sold",    f"{total_hist:,}",         "2-year historical",     "blue"),
    (c2, "Total Revenue",       f"${total_rev:,.0f}",      "2-year historical",     "blue"),
    (c3, "Avg Weekly Units",    f"{avg_weekly:.1f}",       "historical average",    "blue"),
    (c4, "6-Month Forecast",    f"{fcast_total:,} units",  f"{'▲' if growth_pct>=0 else '▼'} {abs(growth_pct):.1f}% vs prior period",
     "green" if growth_pct >= 0 else "red"),
]:
    col.markdown(f"""
    <div class="kpi">
      <div class="kpi-label">{label}</div>
      <div class="kpi-value">{val}</div>
      <div class="kpi-sub {sub_cls}">{sub}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📈  Forecast Chart", "📊  Data Explorer", "🔍  Model Insights"])

# ── Tab 1: Chart ──────────────────────────────────────────────────────────────
with tab1:
    fig = go.Figure()

    if show_ci and not fcast.empty:
        fig.add_trace(go.Scatter(
            x=pd.concat([fcast["date"], fcast["date"].iloc[::-1]]),
            y=pd.concat([fcast["upper_bound"], fcast["lower_bound"].iloc[::-1]]),
            fill="toself", fillcolor="rgba(31,111,235,0.15)",
            line=dict(color="rgba(0,0,0,0)"),
            name="80% Confidence Interval", hoverinfo="skip"
        ))

    if show_hist and not hist.empty:
        fig.add_trace(go.Scatter(
            x=hist["date"], y=hist["quantity_sold"],
            mode="lines", name="Historical",
            line=dict(color="#58a6ff", width=2),
            hovertemplate="<b>%{x|%b %d, %Y}</b><br>Units: %{y:,}<extra></extra>"
        ))

    if not fcast.empty:
        fig.add_trace(go.Scatter(
            x=fcast["date"], y=fcast["forecasted_quantity"],
            mode="lines+markers", name="Forecast",
            line=dict(color="#f0883e", width=2.5, dash="dot"),
            marker=dict(size=6),
            hovertemplate="<b>%{x|%b %d, %Y}</b><br>Forecast: %{y:,}<extra></extra>"
        ))
        # Vertical divider via shapes (avoids add_vline datetime bug)
        divider_str = str(hist["date"].max().date())
        fig.add_shape(type="line", x0=divider_str, x1=divider_str,
                      y0=0, y1=1, yref="paper",
                      line=dict(color="#484f58", dash="dash", width=1.5))
        fig.add_annotation(x=divider_str, y=1, yref="paper",
                           text="Forecast Start", showarrow=False,
                           font=dict(color="#8b949e", size=11),
                           xanchor="left", yanchor="bottom")

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        font=dict(family="Inter", color="#e6edf3"),
        xaxis=dict(title="Date", gridcolor="#21262d", showgrid=True),
        yaxis=dict(title="Units Sold", gridcolor="#21262d", showgrid=True),
        legend=dict(bgcolor="#161b27", bordercolor="#30363d", borderwidth=1),
        height=480, margin=dict(t=20, b=60, l=60, r=20),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### 📅 6-Month Forecast Table")
    fcast_display = fcast[["date","forecasted_quantity","lower_bound","upper_bound"]].copy()
    fcast_display.columns = ["Week","Forecast (Units)","Lower Bound","Upper Bound"]
    fcast_display["Week"] = fcast_display["Week"].dt.strftime("%b %d, %Y")
    st.dataframe(fcast_display, use_container_width=True, hide_index=True)

# ── Tab 2: Data Explorer ──────────────────────────────────────────────────────
with tab2:
    st.markdown("#### 📦 Historical Sales Data")
    show = hist[["date","quantity_sold","total_sales"]].copy()
    show.columns = ["Date","Units Sold","Revenue ($)"]
    show["Date"] = show["Date"].dt.strftime("%b %d, %Y")
    show["Revenue ($)"] = show["Revenue ($)"].apply(lambda x: f"${x:,.2f}")
    st.dataframe(show.sort_values("Date", ascending=False), use_container_width=True, hide_index=True)

    st.markdown("#### 📊 Monthly Aggregation")
    hist_m = hist.copy()
    hist_m["Month"] = hist_m["date"].dt.to_period("M").astype(str)
    monthly = hist_m.groupby("Month").agg(
        Units=("quantity_sold","sum"), Revenue=("total_sales","sum")
    ).reset_index()

    fig2 = go.Figure()
    fig2.add_bar(x=monthly["Month"], y=monthly["Units"],
                 marker_color="#1f6feb", name="Monthly Units")
    fig2.update_layout(
        template="plotly_dark", paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        height=300, margin=dict(t=10,b=40,l=50,r=10),
        xaxis=dict(gridcolor="#21262d"), yaxis=dict(gridcolor="#21262d"),
    )
    st.plotly_chart(fig2, use_container_width=True)

# ── Tab 3: Model Insights ─────────────────────────────────────────────────────
with tab3:
    st.markdown("#### 🧠 Model Performance — All Products")
    perf_data = []
    for p in PRODUCTS:
        pid = p["id"]
        mape = val_mapes.get(pid, 0)
        perf_data.append({
            "Product": p["name"],
            "Category": p["category"],
            "Validation MAPE (%)": mape,
            "Accuracy (%)": round(100 - mape, 1),
        })
    perf_df = pd.DataFrame(perf_data).sort_values("Accuracy (%)", ascending=False)
    st.dataframe(perf_df, use_container_width=True, hide_index=True)

    st.markdown("#### 📐 How It Works")
    st.markdown("""
    | Component | Detail |
    |---|---|
    | **Model** | Random Forest Regressor (400 trees, depth 12) |
    | **Features** | Lag windows (1–13w), rolling stats, cyclical time encoding, seasonal flags |
    | **Confidence Interval** | 10th–90th percentile across all decision trees |
    | **Validation** | Hold-out last 8 weeks per product |
    | **Forecast Horizon** | 26 weeks (6 months) recursive |
    | **Training Data** | 2 years weekly — `2023-01-02` to `2024-12-30` |
    """)

    st.markdown("#### 📈 Accuracy by Product")
    fig3 = go.Figure(go.Bar(
        x=[d["Product"] for d in perf_data],
        y=[d["Accuracy (%)"] for d in perf_data],
        marker_color=["#3fb950" if d["Accuracy (%)"]>=92 else "#d29922" for d in perf_data],
        text=[f"{d['Accuracy (%)']:.1f}%" for d in perf_data],
        textposition="outside"
    ))
    fig3.update_layout(
        template="plotly_dark", paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        yaxis=dict(range=[80,102], gridcolor="#21262d"),
        xaxis=dict(gridcolor="#21262d", tickangle=-30),
        height=360, margin=dict(t=20,b=100,l=50,r=10),
    )
    st.plotly_chart(fig3, use_container_width=True)
