import streamlit as st
import pandas as pd
import plotly.graph_objs as go
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load data
@st.cache_data
def load_data():
    original_df = pd.read_csv("Processed_Product_Sales.csv")
    forecast_df = pd.read_csv("multi_month_forecast.csv")
    original_df["date"] = pd.to_datetime(original_df["date"])
    forecast_df["date"] = pd.to_datetime(forecast_df["date"])
    return original_df, forecast_df

original_df, forecast_df = load_data()

# Sidebar - Product selector
product_ids = forecast_df["product_id"].unique()
product_id = st.sidebar.selectbox("Select Product ID", sorted(product_ids))

# Filter data for selected product
hist = original_df[original_df["product_id"] == product_id]
fut = forecast_df[forecast_df["product_id"] == product_id]

# Sidebar - Date range selector
start_date = st.sidebar.date_input("Start Date", hist["date"].min())
end_date = st.sidebar.date_input("End Date", hist["date"].max())

# Filter historical data based on date range
hist = hist[(hist["date"] >= pd.to_datetime(start_date)) & (hist["date"] <= pd.to_datetime(end_date))]

# Title
st.title("📦 Product-Level Sales Forecast Dashboard")
st.subheader(f"Product ID: {product_id}")

# Plotly chart
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=hist["date"], y=hist["quantity_sold"],
    mode="lines+markers", name="Historical",
    line=dict(color="blue")
))
fig.add_trace(go.Scatter(
    x=fut["date"], y=fut["forecasted_quantity"],
    mode="lines+markers", name="Forecast",
    line=dict(color="orange", dash="dash")
))

# Optional: Add confidence intervals if available
if 'lower_bound' in fut.columns and 'upper_bound' in fut.columns:
    fig.add_trace(go.Scatter(
        x=fut["date"], y=fut["lower_bound"],
        mode="lines", name="Lower Bound",
        line=dict(color="lightgray", dash="dash")
    ))
    fig.add_trace(go.Scatter(
        x=fut["date"], y=fut["upper_bound"],
        mode="lines", name="Upper Bound",
        line=dict(color="lightgray", dash="dash")
    ))

fig.update_layout(
    title="Sales Forecast (Next 6 Months)",
    xaxis_title="Date",
    yaxis_title="Quantity Sold",
    hovermode="x unified"
)

# Show chart
st.plotly_chart(fig, use_container_width=True)

# Show forecasted values
st.subheader("📊 Forecasted Quantities")
st.dataframe(fut[["date", "forecasted_quantity"]].reset_index(drop=True))

# Calculate and display accuracy metrics
if 'actual_quantity' in hist.columns:
    # Calculate Mean Absolute Percentage Error (MAPE)
    mape = (abs(hist["quantity_sold"].values - fut["forecasted_quantity"].iloc[:len(hist)].values) / hist["quantity_sold"].values).mean() * 100
    accuracy = 100 - mape

    st.write(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
    st.write(f"Forecast Accuracy: {accuracy:.2f}%")

    if accuracy < 90:
        st.warning("The forecast accuracy is below 90%. Consider reviewing the forecasting model or data.")
    else:
        st.success("The forecast accuracy is above 90%.")
