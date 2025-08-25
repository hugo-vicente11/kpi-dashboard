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
    cust = pd.read_csv(DATA / "customers.csv", parse_dates=["signup_date"])
    subs = pd.read_csv(
        DATA / "subscriptions.csv",
        parse_dates=["period_start", "period_end"]
    )
    events = pd.read_csv(DATA / "events_marketing.csv", parse_dates=["date"])
    return cust, subs, events

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


    # === Top summary metrics (latest) ===
    mrr_series = (
        subs.groupby(subs["period_start"].dt.to_period("M"))["mrr"]
            .sum()
            .sort_index()
    )

    latest_period = mrr_series.index.max()
    latest_mrr = float(mrr_series.loc[latest_period])

    # previous month (for growth rate)
    prev_mrr = float(mrr_series.shift(1).loc[latest_period]) if latest_period in mrr_series.index else None
    growth_latest = ( (latest_mrr - prev_mrr) / prev_mrr * 100 ) if (prev_mrr and prev_mrr != 0) else None

    latest_arr = latest_mrr * 12

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("MRR (latest)", f"€{latest_mrr:,.0f}")
    with c2:
        st.metric("ARR (latest)", f"€{latest_arr:,.0f}")
    with c3:
        st.metric("Monthly Growth Rate (latest)", f"{growth_latest:+.2f}%" if growth_latest is not None else "—")



    # --- MRR (Monthly Recurring Revenue) ---
    st.subheader("MRR (Monthly Recurring Revenue)")

    mrr_by_month = (
        subs.groupby(subs["period_start"].dt.to_period("M"))["mrr"]
        .sum()
        .reset_index()
    )
    mrr_by_month["period_start"] = mrr_by_month["period_start"].astype(str)

    avg_mrr = mrr_by_month["mrr"].mean()
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
    st.subheader("ARR (Annual Recurring Revenue)")

    arr_by_month = mrr_by_month.copy()
    arr_by_month["ARR"] = arr_by_month["mrr"] * 12

    latest_arr = arr_by_month["ARR"].iloc[-1]
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

    # --- MRR Movements ---
    st.subheader("MRR Movements (New, Expansion, Contraction, Churned)")

    cm = (
        subs.assign(month=subs["period_start"].dt.to_period("M"))
            .groupby(["customer_id", "month"], as_index=False)["mrr"]
            .sum()
    )
    prev = cm.rename(columns={"mrr": "mrr_prev"}).copy()
    prev["month"] = prev["month"] + 1
    cur_prev = cm.merge(prev, on=["customer_id", "month"], how="outer")
    cur_prev[["mrr", "mrr_prev"]] = cur_prev[["mrr", "mrr_prev"]].fillna(0)

    cur_prev["new_mrr"]         = np.where((cur_prev["mrr_prev"] == 0) & (cur_prev["mrr"] > 0), cur_prev["mrr"], 0)
    cur_prev["expansion_mrr"]   = np.where((cur_prev["mrr"] > cur_prev["mrr_prev"]) & (cur_prev["mrr_prev"] > 0),
                                           cur_prev["mrr"] - cur_prev["mrr_prev"], 0)
    cur_prev["contraction_mrr"] = np.where((cur_prev["mrr_prev"] > cur_prev["mrr"]) & (cur_prev["mrr"] > 0),
                                           cur_prev["mrr_prev"] - cur_prev["mrr"], 0)
    cur_prev["churned_mrr"]     = np.where((cur_prev["mrr_prev"] > 0) & (cur_prev["mrr"] == 0),
                                           cur_prev["mrr_prev"], 0)

    mov_month = (cur_prev
                .groupby("month", as_index=False)[["new_mrr","expansion_mrr","contraction_mrr","churned_mrr"]]
                .sum()
                .sort_values("month"))
    mov_month = mov_month.iloc[:-1]
    mov_month["net_new_mrr"] = (mov_month["new_mrr"] + mov_month["expansion_mrr"]
                                - mov_month["contraction_mrr"] - mov_month["churned_mrr"])

    plot_df = mov_month.copy()
    plot_df["month"] = plot_df["month"].astype(str)
    plot_long = (plot_df
                .assign(Contraction=-plot_df["contraction_mrr"], Churned=-plot_df["churned_mrr"],
                        New=plot_df["new_mrr"], Expansion=plot_df["expansion_mrr"])
                .melt(id_vars="month",
                      value_vars=["New","Expansion","Contraction","Churned"],
                      var_name="Movement", value_name="Amount"))

    fig_mov = px.bar(plot_long, x="month", y="Amount", color="Movement",
                    title="MRR Movements by Month (New / Expansion / Contraction / Churned)")
    fig_mov.update_layout(xaxis_title="Month", yaxis_title="MRR Movement (€)")
    st.plotly_chart(fig_mov, use_container_width=True)

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

    # --- Growth Rate ---
    st.subheader("Growth Rate (MoM % change in MRR)")

    mrr_by_month_sorted = mrr_by_month.sort_values("period_start").copy()
    mrr_by_month_sorted["growth_rate"] = mrr_by_month_sorted["mrr"].pct_change() * 100

    latest_growth = mrr_by_month_sorted["growth_rate"].iloc[-1]
    st.metric(label="Latest Growth Rate", value=f"{latest_growth:+.2f}%")

    fig_growth = px.line(
        mrr_by_month_sorted,
        x="period_start",
        y="growth_rate",
        markers=True,
        title="MRR Growth Rate (Month-over-Month)"
    )
    fig_growth.update_layout(xaxis_title="Month", yaxis_title="Growth Rate (%)")
    st.plotly_chart(fig_growth, use_container_width=True)

    st.dataframe(
        mrr_by_month_sorted[["period_start", "mrr", "growth_rate"]].rename(
            columns={"period_start":"Month", "mrr":"MRR (€)", "growth_rate":"Growth Rate (%)"}
        ),
        use_container_width=True
    )


with tab2:
    st.header("👥 Customer Metrics")
    st.divider()

    # --- Build monthly keys ---
    subs["month"] = subs["period_start"].dt.to_period("M")
    customers["signup_month"] = customers["signup_date"].dt.to_period("M")

    # --- Active Customers per month ---
    active_by_month = (
        subs.groupby("month")["customer_id"]
            .nunique()
            .reset_index(name="active_customers")
            .sort_values("month")
    )

    # --- New Customers (signups) per month ---
    new_by_month = (
        customers.groupby("signup_month")["customer_id"]
                 .nunique()
                 .reset_index()
                 .rename(columns={"signup_month": "month", "customer_id": "new_customers"})
    )

    # --- Churned Customers per month (mrr_prev > 0 and current mrr == 0) ---
    # per-customer monthly MRR
    cm = (
        subs.groupby(["customer_id", "month"], as_index=False)["mrr"]
            .sum()
            .sort_values(["customer_id", "month"])
    )
    prev = cm.rename(columns={"mrr": "mrr_prev"}).copy()
    prev["month"] = prev["month"] + 1  # align prev->current
    cur_prev = cm.merge(prev, on=["customer_id", "month"], how="outer").fillna(0)

    churned_rows = cur_prev[(cur_prev["mrr_prev"] > 0) & (cur_prev["mrr"] == 0)]
    churned_by_month = (
        churned_rows.groupby("month")["customer_id"]
                    .nunique()
                    .reset_index(name="churned_customers")
    )

    # --- MRR per month (for ARPA) ---
    mrr_by_month_2 = (
        subs.groupby("month")["mrr"]
            .sum()
            .reset_index(name="mrr")
    )

    # --- Assemble one monthly table ---
    cust_month = (
        active_by_month
        .merge(new_by_month, on="month", how="left")
        .merge(churned_by_month, on="month", how="left")
        .merge(mrr_by_month_2, on="month", how="left")
        .fillna({"new_customers": 0, "churned_customers": 0})
        .sort_values("month")
    )

    # Churn rate = churned / active previous month
    cust_month["active_prev"] = cust_month["active_customers"].shift(1)
    cust_month["churn_rate_pct"] = np.where(
        cust_month["active_prev"] > 0,
        cust_month["churned_customers"] / cust_month["active_prev"] * 100,
        np.nan
    )

    # ARPA = MRR / active customers
    cust_month["arpa"] = np.where(
        cust_month["active_customers"] > 0,
        cust_month["mrr"] / cust_month["active_customers"],
        np.nan
    )

    # Customer Lifetime (months) = 1 / churn_rate (use decimal, protect divide-by-zero)
    churn_decimal = cust_month["churn_rate_pct"] / 100.0
    cust_month["lifetime_months"] = np.where(
        churn_decimal > 0, 1.0 / churn_decimal, np.nan
    )

    # Prepare for charts
    cust_month_plot = cust_month.copy()
    cust_month_plot["month"] = cust_month_plot["month"].astype(str)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Active Customers (latest)",
                  int(cust_month["active_customers"].iloc[-1]))
    with col2:
        latest_churn = cust_month["churn_rate_pct"].iloc[-1]
        st.metric("Churn Rate (latest)",
                  f"{latest_churn:.2f}%" if pd.notna(latest_churn) else "—")
    with col3:
        latest_arpa = cust_month["arpa"].iloc[-1]
        st.metric("ARPA (latest)", f"€{latest_arpa:,.2f}" if pd.notna(latest_arpa) else "—")

    st.subheader("Active / New / Churned Customers")
    fig_counts = px.line(
        cust_month_plot,
        x="month",
        y=["active_customers", "new_customers", "churned_customers"],
        markers=True,
        labels={"value": "Customers", "variable": "Metric"},
        title="Customer Counts Over Time"
    )
    st.plotly_chart(fig_counts, use_container_width=True)

    st.subheader("Churn Rate (%)")
    fig_churn = px.line(
        cust_month_plot, x="month", y="churn_rate_pct", markers=True,
        labels={"churn_rate_pct": "Churn Rate (%)"},
        title="Customer Churn Rate (MoM)"
    )
    st.plotly_chart(fig_churn, use_container_width=True)

    # --- ARPA chart ---
    st.subheader("ARPA (€)")
    fig_arpa_only = px.line(
        cust_month_plot,
        x="month",
        y="arpa",
        markers=True,
        labels={"arpa": "ARPA (€)", "month": "Month"},
        title="Average Revenue per Account (ARPA)"
    )
    fig_arpa_only.update_layout(yaxis_title="ARPA (€)")
    st.plotly_chart(fig_arpa_only, use_container_width=True)

    # --- Lifetime chart ---
    st.subheader("Estimated Customer Lifetime (months)")
    fig_lifetime_only = px.line(
        cust_month_plot,
        x="month",
        y="lifetime_months",
        markers=True,
        labels={"lifetime_months": "Lifetime (months)", "month": "Month"},
        title="Estimated Customer Lifetime"
    )
    fig_lifetime_only.update_layout(yaxis_title="Months")
    st.plotly_chart(fig_lifetime_only, use_container_width=True)

    st.dataframe(
        cust_month.rename(columns={
            "month": "Month",
            "active_customers": "Active",
            "new_customers": "New",
            "churned_customers": "Churned",
            "churn_rate_pct": "Churn Rate (%)",
            "arpa": "ARPA (€)",
            "lifetime_months": "Lifetime (months)",
            "mrr": "MRR (€)"
        }),
        use_container_width=True
    )


with tab3:
    st.header("💰 Value Metrics")
    st.divider()

    # ---- Controls ----
    gross_margin_pct = st.slider("Assumed Gross Margin (%)", 40, 99, 80, step=1)
    gm = gross_margin_pct / 100.0

    # ---- Pre-compute monthly keys ----
    subs["month"] = subs["period_start"].dt.to_period("M")
    events["month"] = events["date"].dt.to_period("M")
    customers["signup_month"] = customers["signup_date"].dt.to_period("M")

    # ---- ARPA (for LTV & Payback) ----
    active_by_month_v = (
        subs.groupby("month")["customer_id"].nunique().reset_index(name="active_customers")
    )
    mrr_by_month_v = subs.groupby("month")["mrr"].sum().reset_index(name="mrr")
    kpi_base = active_by_month_v.merge(mrr_by_month_v, on="month", how="left")
    kpi_base["arpa"] = np.where(kpi_base["active_customers"] > 0,
                                kpi_base["mrr"] / kpi_base["active_customers"], np.nan)

    # ---- Churn rate (customers) ----
    cm = subs.groupby(["customer_id", "month"], as_index=False)["mrr"].sum().sort_values(["customer_id","month"])
    prev = cm.rename(columns={"mrr":"mrr_prev"}).copy()
    prev["month"] = prev["month"] + 1  # align prev->current
    cur_prev = cm.merge(prev, on=["customer_id","month"], how="outer").fillna(0)
    churned_rows = cur_prev[(cur_prev["mrr_prev"] > 0) & (cur_prev["mrr"] == 0)]
    churned_by_month_v = churned_rows.groupby("month")["customer_id"].nunique().reset_index(name="churned")

    kpi_base = kpi_base.merge(churned_by_month_v, on="month", how="left").fillna({"churned":0})
    kpi_base["active_prev"] = kpi_base["active_customers"].shift(1)
    kpi_base["churn_rate"] = np.where(
        kpi_base["active_prev"] > 0, kpi_base["churned"] / kpi_base["active_prev"], np.nan
    )  # decimal (e.g., 0.025)

    # ---- CAC (overall) ----
    mkt_month = (events.groupby("month")[["spend","conversions"]].sum().reset_index())
    mkt_month["cac"] = np.where(mkt_month["conversions"] > 0,
                                mkt_month["spend"] / mkt_month["conversions"], np.nan)

    # ---- CAC by channel (Google Ads vs LinkedIn Ads, all months) ----
    channels_keep = ["Google Ads", "LinkedIn Ads"]

    cac_month_ch = (
        events.query("channel in @channels_keep")
              .groupby(["month", "channel"], as_index=False)[["spend", "conversions"]]
              .sum()
    )

    # ensure every month-channel pair exists
    all_months = pd.period_range(events["date"].min().to_period("M"),
                                 events["date"].max().to_period("M"),
                                 freq="M")
    full_index = pd.MultiIndex.from_product([all_months, channels_keep], names=["month","channel"])
    cac_month_ch = (
        cac_month_ch.set_index(["month","channel"])
                    .reindex(full_index)
                    .reset_index()
    )
    cac_month_ch[["spend","conversions"]] = cac_month_ch[["spend","conversions"]].fillna(0)
    cac_month_ch["cac"] = np.where(cac_month_ch["conversions"] > 0,
                                   cac_month_ch["spend"] / cac_month_ch["conversions"],
                                   np.nan)

    # ---- LTV, LTV/CAC, Payback ----
    vals = kpi_base.merge(mkt_month[["month","cac"]], on="month", how="left")
    vals["ltv"] = np.where(
        (vals["arpa"].notna()) & (vals["churn_rate"] > 0),
        vals["arpa"] * gm / vals["churn_rate"],
        np.nan
    )
    vals["ltv_cac"] = np.where((vals["ltv"].notna()) & (vals["cac"] > 0),
                               vals["ltv"] / vals["cac"], np.nan)
    vals["payback_months"] = np.where((vals["cac"] > 0) & (vals["arpa"] > 0),
                                      vals["cac"] / vals["arpa"], np.nan)

    # ---- Latest KPIs row ----
    latest_row = vals.loc[vals["month"].idxmax()]
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("CAC (latest)", f"€{latest_row['cac']:,.2f}" if pd.notna(latest_row["cac"]) else "—")
    with c2:
        st.metric("LTV (latest)", f"€{latest_row['ltv']:,.2f}" if pd.notna(latest_row["ltv"]) else "—")
    with c3:
        st.metric("LTV/CAC (latest)", f"{latest_row['ltv_cac']:.2f}" if pd.notna(latest_row["ltv_cac"]) else "—")
    with c4:
        st.metric("Payback (months, latest)", f"{latest_row['payback_months']:.2f}" if pd.notna(latest_row["payback_months"]) else "—")

    # ---- Charts ----
    vals_plot = vals.copy()
    vals_plot["month"] = vals_plot["month"].astype(str)

    st.subheader("CAC Over Time (All Channels)")
    fig_cac = px.line(vals_plot, x="month", y="cac", markers=True, title="CAC Over Time")
    fig_cac.update_layout(yaxis_title="CAC (€)")
    st.plotly_chart(fig_cac, use_container_width=True)

    st.subheader("CAC by Channel")

    plot_cac = cac_month_ch.copy()
    plot_cac["month"] = plot_cac["month"].astype(str)

    fig_cac_ch_all = px.bar(
        plot_cac,
        x="month",
        y="cac",
        color="channel",
        barmode="group",                     
        title="CAC by Channel Over Time",
        labels={"cac": "CAC (€)", "month": "Month", "channel": "Channel"}
    )

    # keep months in chronological order and readable
    fig_cac_ch_all.update_layout(
        xaxis={"type": "category", "categoryorder": "category ascending"},
        yaxis_title="CAC (€)"
    )

    st.plotly_chart(fig_cac_ch_all, use_container_width=True)


    # --- LTV chart (separate) ---
    st.subheader("LTV Over Time")
    fig_ltv_only = px.line(
        vals_plot,
        x="month",
        y="ltv",
        markers=True,
        title="Customer Lifetime Value (LTV)"
    )
    fig_ltv_only.update_layout(yaxis_title="LTV (€)")
    st.plotly_chart(fig_ltv_only, use_container_width=True)

    # --- Payback chart (separate) ---
    st.subheader("Payback Period Over Time")
    fig_payback_only = px.line(
        vals_plot,
        x="month",
        y="payback_months",
        markers=True,
        title="Payback Period (months)"
    )
    fig_payback_only.update_layout(yaxis_title="Months")
    st.plotly_chart(fig_payback_only, use_container_width=True)


    st.subheader("LTV/CAC Ratio Over Time")
    fig_ratio = px.line(vals_plot, x="month", y="ltv_cac", markers=True, title="LTV/CAC Ratio")
    fig_ratio.update_layout(yaxis_title="Ratio")
    st.plotly_chart(fig_ratio, use_container_width=True)

    # ---- Table ----
    st.dataframe(
        vals.rename(columns={
            "month":"Month",
            "arpa":"ARPA (€)",
            "churn_rate":"Churn (decimal)",
            "cac":"CAC (€)",
            "ltv":"LTV (€)",
            "ltv_cac":"LTV/CAC",
            "payback_months":"Payback (months)"
        }),
        use_container_width=True
    )



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
