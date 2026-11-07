from __future__ import annotations

import pandas as pd
import streamlit as st

from enterprise_intelligence.analytics import compute_kpis, generate_business_insights
from enterprise_intelligence.business_assistant import BusinessAssistant
from enterprise_intelligence.data_pipeline import clean_business_data, generate_demo_dataframe
from enterprise_intelligence.ml import AnomalyDetector, CustomerSegmentationModel, SalesForecastModel

st.set_page_config(page_title="AI Enterprise Intelligence", page_icon="📊", layout="wide")
st.title("AI-Powered Enterprise Intelligence & Predictive Analytics System")

with st.sidebar:
    st.header("Controls")
    uploaded = st.file_uploader("Upload a CSV or Excel dataset", type=["csv", "xlsx", "xls"])
    st.caption("The app supports local business data uploads and demo analytics.")

if uploaded is not None:
    df = pd.read_csv(uploaded) if uploaded.name.lower().endswith(".csv") else pd.read_excel(uploaded)
    with st.spinner("Processing uploaded dataset..."):
        df = clean_business_data(df)
else:
    df = generate_demo_dataframe()

kpis = compute_kpis(df)
insights = generate_business_insights(df)

col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("Revenue", f"${kpis['total_revenue']:,.2f}")
col2.metric("Orders", f"{kpis['total_orders']}")
col3.metric("Units Sold", f"{kpis['total_units_sold']}")
col4.metric("Avg Order", f"${kpis['average_order_value']:,.2f}")
col5.metric("Customers", f"{kpis['total_customers']}")
col6.metric("Products", f"{kpis['total_products']}")

st.subheader("Sales overview")
if "InvoiceDate" in df.columns and "Revenue" in df.columns:
    monthly = df.set_index("InvoiceDate").resample("M")["Revenue"].sum()
    st.line_chart(monthly)

st.subheader("Top business insights")
for item in insights.get("top_products", [])[:3]:
    st.write(f"- {item['product']}: ${item['revenue']:.2f}")

st.subheader("Business assistant")
question = st.text_input("Ask a business question", value="What is the total revenue?")
if question:
    st.write(BusinessAssistant(df).answer(question))

st.subheader("Customer segmentation")
try:
    segments = CustomerSegmentationModel().segment(df)
    st.dataframe(segments["summary"], use_container_width=True)
except Exception as exc:  # pragma: no cover
    st.warning(f"Segment modelling unavailable: {exc}")

st.subheader("Anomaly detection")
try:
    anomaly_result = AnomalyDetector().detect(df)
    st.dataframe(anomaly_result["anomalies"].head(10), use_container_width=True)
except Exception as exc:  # pragma: no cover
    st.warning(f"Anomaly detection unavailable: {exc}")

st.subheader("Forecast preview")
if "InvoiceDate" in df.columns and "Revenue" in df.columns:
    series = df.set_index("InvoiceDate")["Revenue"].sort_index()
    forecast = SalesForecastModel().forecast(series, horizon=7, frequency="daily")
    st.line_chart(forecast["forecast"])

st.subheader("Data preview")
st.dataframe(df.head(10), use_container_width=True)
