import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

# --- Paths ---
BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data"

# --- Load CSVs ---
@st.cache_data
def load_data():
    customers = pd.read_csv(DATA / "customers.csv", parse_dates=["signup_date"])
    subs = pd.read_csv(
        DATA / "subscriptions.csv",
        parse_dates=["period_start", "period_end"]
    )
    events = pd.read_csv(DATA / "events_marketing.csv", parse_dates=["date"])
    return customers, subs, events

customers, subs, events = load_data()

st.set_page_config(page_title="KPI Dashboard", layout="wide")

# Tabs for each KPI section
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🏦 Revenue & Growth",
    "👥 Customer Metrics",
    "💰 Value Metrics",
    "📈 Retention & Cohorts",
    "📊 Funnel & Marketing",
    "🌍 Segmentation"
])

with tab1:
    st.header("🏦 Revenue & Growth")
    st.divider()


    # --- MRR (Monthly Recurring Revenue) ---

    mrr_by_month = (
        subs.groupby(subs["period_start"].dt.to_period("M"))["mrr"]
        .sum()
        .reset_index()
    )

    mrr_by_month["period_start"] = mrr_by_month["period_start"].astype(str)

    avg_mrr = mrr_by_month["mrr"].mean()
    st.markdown("# MRR (Monthly Recurring Revenue)")
    st.metric(label="Average MRR", value=f"€{avg_mrr:,.0f}")

    fig = px.line(
        mrr_by_month,
        x="period_start",
        y="mrr",
        markers=True,
        title="MRR Over Time"
    )
    fig.update_layout(xaxis_title="Month", yaxis_title="MRR (€)")

    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        mrr_by_month.rename(columns={"period_start": "Month", "mrr": "MRR (€)"}),
        use_container_width=True
    )
    
    st.divider()


    # --- ARR (Annual Recurring Revenue) ---

    arr_by_month = mrr_by_month.copy()
    arr_by_month["ARR"] = arr_by_month["mrr"] * 12

    latest_arr = arr_by_month["ARR"].iloc[-1]
    st.markdown("# ARR (Annual Recurring Revenue)")
    st.metric(label="ARR (run-rate of latest month)", value=f"€{latest_arr:,.0f}")

    fig_arr = px.line(
        arr_by_month,
        x="period_start",
        y="ARR",
        markers=True,
        title="ARR Over Time"
    )
    fig_arr.update_layout(xaxis_title="Month", yaxis_title="ARR (€)")
    st.plotly_chart(fig_arr, use_container_width=True)

    st.dataframe(
        arr_by_month.rename(columns={"period_start": "Month", "ARR": "ARR (€)"})[["Month", "ARR (€)"]],
        use_container_width=True
    )
    st.divider()



with tab2:
    st.header("👥 Customer Metrics")
    # show Active Customers, Churn Rate, ARPA
    # charts: customer count over time, churn trend

with tab3:
    st.header("💰 Value Metrics")
    # show LTV, CAC, LTV/CAC, Payback period
    # charts: CAC by channel bar chart

with tab4:
    st.header("📈 Retention & Cohorts")
    # show NRR, GRR
    # charts: cohort heatmap, retention curve

with tab5:
    st.header("📊 Funnel & Marketing")
    # show conversion rates, spend, ROI
    # charts: funnel diagram, spend vs conversions

with tab6:
    st.header("🌍 Segmentation")
    # show breakdowns: MRR by plan/country/channel
    # charts: stacked bar by segment
