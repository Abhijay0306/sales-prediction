# 📈 SalesCast AI — Sales Forecasting Dashboard

A production-quality, self-contained sales forecasting dashboard built with **Streamlit**, **scikit-learn Random Forest**, and **Plotly**. Clone it, install dependencies, and run — no data setup required.

---

## ✨ Features

| Feature | Detail |
|---|---|
| 🤖 **ML Forecasting** | Random Forest with 400 trees + rich feature engineering |
| 📊 **10 Products** | Generic tech product catalogue with realistic seasonality |
| 📅 **6-Month Forecast** | Recursive 26-week ahead predictions |
| 🎯 **Confidence Intervals** | 10th–90th percentile from tree ensemble |
| 📈 **KPI Cards** | Revenue, units sold, forecast growth |
| 🗂️ **Tabbed UI** | Forecast Chart / Data Explorer / Model Insights |
| 🎨 **Dark Theme** | Premium dark UI with Inter font |
| ⚡ **Plug & Play** | Generates its own data on first run — no CSVs needed |

---

## 🚀 Quick Start

```bash
# 1. Clone
git clone https://github.com/Abhijay0306/sales-prediction.git
cd sales-prediction

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run
streamlit run app.py
```

> The app auto-generates synthetic sales data and trains the model on first launch (~30 seconds). Subsequent runs load from cache instantly.

---

## 📁 Project Structure

```
sales-prediction/
├── app.py                # Main dashboard (self-contained)
├── generate_data.py      # Optional: export CSVs to data/
├── requirements.txt
├── README.md
└── data/                 # Auto-created by generate_data.py
    ├── sales_data.csv
    └── forecast_data.csv
```

### Optionally Pre-generate CSVs

If you want to inspect or customise the data before running the app:

```bash
python generate_data.py
# Outputs: data/sales_data.csv, data/forecast_data.csv
# The app will use these files automatically if they exist.
```

---

## 📦 Data Schema

### `data/sales_data.csv`

| Column | Type | Description |
|---|---|---|
| `date` | datetime | Week start (Monday) |
| `product_id` | string | Product code (P001–P010) |
| `product_name` | string | Human-readable name |
| `category` | string | Product category |
| `quantity_sold` | int | Units sold that week |
| `unit_price` | float | Price per unit ($) |
| `total_sales` | float | Revenue for that week ($) |

### `data/forecast_data.csv`

| Column | Type | Description |
|---|---|---|
| `date` | datetime | Forecast week |
| `product_id` | string | Product code |
| `forecasted_quantity` | int | Predicted units |
| `lower_bound` | int | 10th-percentile prediction |
| `upper_bound` | int | 90th-percentile prediction |

---

## 🧠 Model Architecture

```
Historical Sales (2 years, weekly)
        │
        ▼
  Feature Engineering
  ├── Lag features: 1w, 2w, 3w, 4w, 8w, 13w
  ├── Rolling stats: mean/std/max/min (4w, 8w, 13w windows)
  ├── Cyclical encoding: week-of-year sin/cos, month sin/cos
  └── Seasonal flags: Q4 holiday, summer peak, New Year
        │
        ▼
  Random Forest Regressor
  ├── 400 estimators, max_depth=12
  └── Validated on last 8 weeks (held-out)
        │
        ▼
  26-Week Recursive Forecast
  ├── Point estimate: mean tree prediction
  └── Confidence interval: 10th–90th percentile across trees
```

### Typical Accuracy

| Product | Accuracy |
|---|---|
| Laptop Pro 15 | ~96% |
| Wireless Headphones | ~95% |
| Wireless Charger Pad | ~96% |
| All products | **92–97%** |

> Accuracy = 100% − MAPE, evaluated on 8-week held-out validation set.

---

## 🛠️ Customisation

### Add your own products
Edit the `PRODUCTS` list in `app.py` or `generate_data.py`:

```python
{"id":"P011","name":"Your Product","category":"Category",
 "price":199.99,"base":80,"trend":0.3,"q4":1.6,"summer":1.2}
```

### Bring your own data
Place a `data/sales_data.csv` with the schema above — the app will use it automatically and skip data generation.

### Tune the model
In `app.py`, find `train_models()` and adjust:
```python
RandomForestRegressor(n_estimators=400, max_depth=12, ...)
```

---

## 📋 Requirements

- Python 3.9+
- streamlit ≥ 1.32
- pandas ≥ 2.0
- numpy ≥ 1.24
- plotly ≥ 5.18
- scikit-learn ≥ 1.4

---

## 📄 License

MIT — free to use, modify, and distribute.
