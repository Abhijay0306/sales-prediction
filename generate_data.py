"""
generate_data.py
================
Run this script to (re)generate the two CSV files used by the dashboard.
    python generate_data.py
Outputs:
    data/sales_data.csv     — 2 years of weekly historical sales
    data/forecast_data.csv  — 6-month ahead forecasts with confidence intervals
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error

np.random.seed(42)

# ---------------------------------------------------------------------------
# Product catalogue
# ---------------------------------------------------------------------------
PRODUCTS = [
    {"id": "P001", "name": "Laptop Pro 15",       "category": "Computers",   "price": 1299.99, "base": 45,  "trend": 0.30, "q4": 1.80, "summer": 1.40},
    {"id": "P002", "name": "Wireless Headphones",  "category": "Audio",       "price": 149.99,  "base": 120, "trend": 0.50, "q4": 2.20, "summer": 1.10},
    {"id": "P003", "name": "Mechanical Keyboard",  "category": "Peripherals", "price": 89.99,   "base": 85,  "trend": 0.20, "q4": 1.50, "summer": 1.20},
    {"id": "P004", "name": "USB-C Hub 7-in-1",     "category": "Accessories", "price": 49.99,   "base": 200, "trend": 0.40, "q4": 1.60, "summer": 1.00},
    {"id": "P005", "name": "1080p Webcam",          "category": "Peripherals", "price": 79.99,   "base": 95,  "trend":-0.10, "q4": 1.30, "summer": 0.90},
    {"id": "P006", "name": "27-inch 4K Monitor",   "category": "Displays",    "price": 499.99,  "base": 30,  "trend": 0.60, "q4": 1.70, "summer": 1.30},
    {"id": "P007", "name": "Gaming Mouse RGB",      "category": "Peripherals", "price": 59.99,   "base": 110, "trend": 0.30, "q4": 1.40, "summer": 1.50},
    {"id": "P008", "name": "Ergonomic Desk Chair",  "category": "Furniture",   "price": 399.99,  "base": 20,  "trend": 0.80, "q4": 1.20, "summer": 0.80},
    {"id": "P009", "name": "Portable SSD 1TB",      "category": "Storage",     "price": 99.99,   "base": 75,  "trend": 0.40, "q4": 1.60, "summer": 1.10},
    {"id": "P010", "name": "Wireless Charger Pad",  "category": "Accessories", "price": 29.99,   "base": 250, "trend": 0.50, "q4": 2.50, "summer": 1.00},
]


def _seasonal_demand(product: dict, dates: pd.DatetimeIndex) -> np.ndarray:
    n = len(dates)
    base = product["base"]
    week_nums = np.array([d.isocalendar()[1] for d in dates])

    trend = np.linspace(1.0, 1.0 + product["trend"], n)
    seasonal = np.ones(n)
    seasonal[(week_nums >= 44) & (week_nums <= 52)] = product["q4"]
    seasonal[(week_nums >= 24) & (week_nums <= 35)] *= product["summer"]
    seasonal[week_nums <= 3] = 1.15
    smooth = 1.0 + 0.12 * np.sin(2 * np.pi * week_nums / 52 - np.pi / 6)

    demand = base * trend * seasonal * smooth
    noise = np.random.normal(0, base * 0.08, n)
    return np.maximum(1, np.round(demand + noise)).astype(int)


def generate_sales_data() -> pd.DataFrame:
    dates = pd.date_range("2023-01-02", "2024-12-30", freq="W-MON")
    rows = []
    for p in PRODUCTS:
        demand = _seasonal_demand(p, dates)
        for i, date in enumerate(dates):
            rows.append({
                "date": date,
                "product_id": p["id"],
                "product_name": p["name"],
                "category": p["category"],
                "quantity_sold": int(demand[i]),
                "unit_price": p["price"],
                "total_sales": round(float(demand[i]) * p["price"], 2),
            })
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
FEATURE_COLS = [
    "week_num", "week_sin", "week_cos", "month_sin", "month_cos",
    "quarter", "is_q4", "is_summer", "is_newyear",
    "lag_1", "lag_2", "lag_3", "lag_4", "lag_8", "lag_13",
    "roll_mean_4", "roll_mean_8", "roll_mean_13",
    "roll_std_4", "roll_std_8", "roll_max_4", "roll_min_4",
]


def make_features(df_prod: pd.DataFrame) -> pd.DataFrame:
    df = df_prod.sort_values("date").reset_index(drop=True).copy()
    woy = df["date"].dt.isocalendar().week.astype(int)
    df["week_of_year"] = woy
    df["month"]        = df["date"].dt.month
    df["quarter"]      = df["date"].dt.quarter
    df["week_num"]     = np.arange(len(df))
    df["week_sin"]     = np.sin(2 * np.pi * woy / 52)
    df["week_cos"]     = np.cos(2 * np.pi * woy / 52)
    df["month_sin"]    = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"]    = np.cos(2 * np.pi * df["month"] / 12)
    df["is_q4"]        = ((woy >= 44) & (woy <= 52)).astype(int)
    df["is_summer"]    = ((woy >= 24) & (woy <= 35)).astype(int)
    df["is_newyear"]   = (woy <= 3).astype(int)
    for lag in [1, 2, 3, 4, 8, 13]:
        df[f"lag_{lag}"] = df["quantity_sold"].shift(lag)
    df["roll_mean_4"]  = df["quantity_sold"].shift(1).rolling(4).mean()
    df["roll_mean_8"]  = df["quantity_sold"].shift(1).rolling(8).mean()
    df["roll_mean_13"] = df["quantity_sold"].shift(1).rolling(13).mean()
    df["roll_std_4"]   = df["quantity_sold"].shift(1).rolling(4).std().fillna(0)
    df["roll_std_8"]   = df["quantity_sold"].shift(1).rolling(8).std().fillna(0)
    df["roll_max_4"]   = df["quantity_sold"].shift(1).rolling(4).max()
    df["roll_min_4"]   = df["quantity_sold"].shift(1).rolling(4).min()
    return df


# ---------------------------------------------------------------------------
# Training + forecasting
# ---------------------------------------------------------------------------
def train_and_forecast(sales_df: pd.DataFrame):
    forecast_rows = []
    product_metrics = []

    for p in PRODUCTS:
        pid = p["id"]
        df_prod = sales_df[sales_df["product_id"] == pid].copy()
        df_feat = make_features(df_prod).dropna(subset=FEATURE_COLS)

        X = df_feat[FEATURE_COLS].values
        y = df_feat["quantity_sold"].values

        # Hold out last 8 weeks for validation
        split = len(X) - 8
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        model = RandomForestRegressor(
            n_estimators=400, max_depth=12,
            min_samples_leaf=2, random_state=42, n_jobs=-1
        )
        model.fit(X_train, y_train)

        val_preds = model.predict(X_val)
        mape = mean_absolute_percentage_error(y_val, val_preds) * 100
        product_metrics.append({"product_id": pid, "val_mape": round(mape, 2)})

        # Recursive 26-week forecast
        history = list(df_feat["quantity_sold"].values)
        hist_dates = list(df_feat["date"])
        last_week_num = int(df_feat["week_num"].iloc[-1])

        future_dates = pd.date_range(
            hist_dates[-1] + pd.Timedelta(weeks=1), periods=26, freq="W-MON"
        )

        for i, fdate in enumerate(future_dates):
            wn = last_week_num + i + 1
            woy = fdate.isocalendar()[1]
            month = fdate.month

            def lag(k):
                idx = len(history) - k
                return history[idx] if idx >= 0 else history[0]

            def roll_stat(k, fn):
                window = [history[j] for j in range(max(0, len(history) - k), len(history))]
                if not window:
                    return history[-1]
                return fn(window)

            row = {
                "week_num":   wn,
                "week_sin":   np.sin(2 * np.pi * woy / 52),
                "week_cos":   np.cos(2 * np.pi * woy / 52),
                "month_sin":  np.sin(2 * np.pi * month / 12),
                "month_cos":  np.cos(2 * np.pi * month / 12),
                "quarter":    (month - 1) // 3 + 1,
                "is_q4":      int(44 <= woy <= 52),
                "is_summer":  int(24 <= woy <= 35),
                "is_newyear": int(woy <= 3),
                "lag_1":      lag(1), "lag_2": lag(2), "lag_3": lag(3),
                "lag_4":      lag(4), "lag_8": lag(8), "lag_13": lag(13),
                "roll_mean_4":  roll_stat(4, np.mean),
                "roll_mean_8":  roll_stat(8, np.mean),
                "roll_mean_13": roll_stat(13, np.mean),
                "roll_std_4":   roll_stat(4, np.std),
                "roll_std_8":   roll_stat(8, np.std),
                "roll_max_4":   roll_stat(4, max),
                "roll_min_4":   roll_stat(4, min),
            }
            X_pred = np.array([[row[c] for c in FEATURE_COLS]])

            # Point estimate
            point = float(model.predict(X_pred)[0])

            # Confidence interval from individual tree predictions
            tree_preds = np.array([t.predict(X_pred)[0] for t in model.estimators_])
            lo = float(np.percentile(tree_preds, 10))
            hi = float(np.percentile(tree_preds, 90))

            history.append(max(1, round(point)))
            forecast_rows.append({
                "product_id":         pid,
                "product_name":       p["name"],
                "date":               fdate,
                "forecasted_quantity": max(1, round(point)),
                "lower_bound":        max(1, round(lo)),
                "upper_bound":        max(1, round(hi)),
            })

    return pd.DataFrame(forecast_rows), pd.DataFrame(product_metrics)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    out_dir = Path("data")
    out_dir.mkdir(exist_ok=True)

    print("Generating sales data...")
    sales_df = generate_sales_data()
    sales_df.to_csv(out_dir / "sales_data.csv", index=False)
    print(f"  Saved {len(sales_df)} rows → data/sales_data.csv")

    print("Training models & generating forecasts...")
    forecast_df, metrics_df = train_and_forecast(sales_df)
    forecast_df.to_csv(out_dir / "forecast_data.csv", index=False)
    print(f"  Saved {len(forecast_df)} rows → data/forecast_data.csv")

    print("\nValidation MAPE by product:")
    for _, row in metrics_df.iterrows():
        print(f"  {row['product_id']}: {row['val_mape']:.1f}%")
    print("\nDone! Run: streamlit run app.py")
