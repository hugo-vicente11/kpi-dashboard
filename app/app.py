import streamlit as st
import pandas as pd
import numpy as np
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

    # --- MRR Movements (New, Expansion, Contraction, Churned) ---

    # 1) Monthly series per customer
    cm = (
        subs.assign(month=subs["period_start"].dt.to_period("M"))
            .groupby(["customer_id", "month"], as_index=False)["mrr"]
            .sum()
    )

    # 2) Align current month with previous month for the same customer
    prev = cm.rename(columns={"mrr": "mrr_prev"}).copy()
    prev["month"] = prev["month"] + 1  # shift forward to align prev->current

    cur_prev = cm.merge(prev, on=["customer_id", "month"], how="outer")

    # 3) Replace NaN with 0 for easier calculations
    cur_prev[["mrr", "mrr_prev"]] = cur_prev[["mrr", "mrr_prev"]].fillna(0)

    # 4) Classify movements row by row (per customer, month)
    cur_prev["new_mrr"]         = np.where((cur_prev["mrr_prev"] == 0) & (cur_prev["mrr"] > 0), cur_prev["mrr"], 0)
    cur_prev["expansion_mrr"]   = np.where((cur_prev["mrr"] > cur_prev["mrr_prev"]) & (cur_prev["mrr_prev"] > 0),
                                        cur_prev["mrr"] - cur_prev["mrr_prev"], 0)
    cur_prev["contraction_mrr"] = np.where((cur_prev["mrr_prev"] > cur_prev["mrr"]) & (cur_prev["mrr"] > 0),
                                        cur_prev["mrr_prev"] - cur_prev["mrr"], 0)
    cur_prev["churned_mrr"]     = np.where((cur_prev["mrr_prev"] > 0) & (cur_prev["mrr"] == 0),
                                        cur_prev["mrr_prev"], 0)

    # 5) Aggregate by month
    mov_month = (cur_prev
                .groupby("month", as_index=False)[["new_mrr","expansion_mrr","contraction_mrr","churned_mrr"]]
                .sum()
                .sort_values("month"))
    mov_month = mov_month.iloc[:-1] # Remove ghost month created when merging

    # 6) Net New MRR
    mov_month["net_new_mrr"] = (mov_month["new_mrr"] + mov_month["expansion_mrr"]
                                - mov_month["contraction_mrr"] - mov_month["churned_mrr"])

    # 7) Prepare for chart (positives vs negatives)
    plot_df = mov_month.copy()
    plot_df["month"] = plot_df["month"].astype(str)
    plot_long = (plot_df
                .assign(Contraction=-plot_df["contraction_mrr"], Churned=-plot_df["churned_mrr"],
                        New=plot_df["new_mrr"], Expansion=plot_df["expansion_mrr"])
                .melt(id_vars="month",
                    value_vars=["New","Expansion","Contraction","Churned"],
                    var_name="Movement", value_name="Amount"))

    st.markdown("# MRR Movements")
    # Stacked bar chart (expansion/new positive; contraction/churned negative)
    fig_mov = px.bar(plot_long, x="month", y="Amount", color="Movement",
                    title="MRR Movements by Month (New / Expansion / Contraction / Churned)")
    fig_mov.update_layout(xaxis_title="Month", yaxis_title="MRR Movement (€)")
    st.plotly_chart(fig_mov, use_container_width=True)

    # Summary table (last 6 months)
    st.dataframe(
        mov_month.rename(columns={
            "month":"Month",
            "new_mrr":"New",
            "expansion_mrr":"Expansion",
            "contraction_mrr":"Contraction",
            "churned_mrr":"Churned",
            "net_new_mrr":"Net New"
        }),
        use_container_width=True
    )
    st.divider()

    # --- Growth Rate (MoM % change in MRR) ---

    # Ensure months are sorted correctly
    mrr_by_month_sorted = mrr_by_month.sort_values("period_start").copy()

    # Calculate % change vs previous month
    mrr_by_month_sorted["growth_rate"] = mrr_by_month_sorted["mrr"].pct_change() * 100

    # KPI: latest growth rate
    latest_growth = mrr_by_month_sorted["growth_rate"].iloc[-1]
    st.markdown("# Growth Rate (MoM % change in MRR)")
    st.metric(
        label="Latest Growth Rate",
        value=f"{latest_growth:+.2f}%"
    )

    # Line chart of Growth Rate
    fig_growth = px.line(
        mrr_by_month_sorted,
        x="period_start",
        y="growth_rate",
        markers=True,
        title="MRR Growth Rate (Month-over-Month)"
    )
    fig_growth.update_layout(xaxis_title="Month", yaxis_title="Growth Rate (%)")
    st.plotly_chart(fig_growth, use_container_width=True)

    # Show last 6 months in table
    st.dataframe(
        mrr_by_month_sorted[["period_start", "mrr", "growth_rate"]].rename(
            columns={"period_start":"Month", "mrr":"MRR (€)", "growth_rate":"Growth Rate (%)"}
        ),
        use_container_width=True
    )

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
