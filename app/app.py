import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data"

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


    mrr_series = (
        subs.groupby(subs["period_start"].dt.to_period("M"))["mrr"]
            .sum()
            .sort_index()
    )

    latest_period = mrr_series.index.max()
    latest_mrr = float(mrr_series.loc[latest_period])

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

    st.divider()

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

    st.divider()

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

    st.divider()

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

    mrr_month = (
        subs.groupby(subs["period_start"].dt.to_period("M"))["mrr"]
            .sum()
            .reset_index()
            .rename(columns={"period_start": "month", "mrr": "MRR (€)"})
            .sort_values("month")
    )

    mrr_month["ARR (€)"] = mrr_month["MRR (€)"] * 12
    mrr_month["Growth Rate (%)"] = mrr_month["MRR (€)"].pct_change() * 100

    mov_renamed = mov_month.rename(columns={
        "new_mrr": "New (€)",
        "expansion_mrr": "Expansion (€)",
        "contraction_mrr": "Contraction (€)",
        "churned_mrr": "Churned (€)",
        "net_new_mrr": "Net New (€)"
    })

    summary = (
        mrr_month.merge(mov_renamed, on="month", how="left")
                .fillna({"New (€)": 0, "Expansion (€)": 0, "Contraction (€)": 0, "Churned (€)": 0, "Net New (€)": 0})
    )

    summary_display = summary.copy()
    summary_display["Month"] = summary_display["month"].astype(str)
    summary_display = summary_display.drop(columns=["month"])

    summary_display = summary_display[[
        "Month",
        "MRR (€)", "ARR (€)", "Growth Rate (%)",
        "New (€)", "Expansion (€)", "Contraction (€)", "Churned (€)", "Net New (€)"
    ]]

    for col in ["MRR (€)", "ARR (€)", "New (€)", "Expansion (€)", "Contraction (€)", "Churned (€)", "Net New (€)"]:
        summary_display[col] = summary_display[col].round(2)
    summary_display["Growth Rate (%)"] = summary_display["Growth Rate (%)"].round(2)

    st.dataframe(summary_display, use_container_width=True)



with tab2:
    st.header("👥 Customer Metrics")
    st.divider()

    subs["month"] = subs["period_start"].dt.to_period("M")
    customers["signup_month"] = customers["signup_date"].dt.to_period("M")

    active_by_month = (
        subs.groupby("month")["customer_id"]
            .nunique()
            .reset_index(name="active_customers")
            .sort_values("month")
    )

    new_by_month = (
        customers.groupby("signup_month")["customer_id"]
                 .nunique()
                 .reset_index()
                 .rename(columns={"signup_month": "month", "customer_id": "new_customers"})
    )

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

    mrr_by_month_2 = (
        subs.groupby("month")["mrr"]
            .sum()
            .reset_index(name="mrr")
    )

    cust_month = (
        active_by_month
        .merge(new_by_month, on="month", how="left")
        .merge(churned_by_month, on="month", how="left")
        .merge(mrr_by_month_2, on="month", how="left")
        .fillna({"new_customers": 0, "churned_customers": 0})
        .sort_values("month")
    )

    cust_month["active_prev"] = cust_month["active_customers"].shift(1)
    cust_month["churn_rate_pct"] = np.where(
        cust_month["active_prev"] > 0,
        cust_month["churned_customers"] / cust_month["active_prev"] * 100,
        np.nan
    )

    cust_month["arpa"] = np.where(
        cust_month["active_customers"] > 0,
        cust_month["mrr"] / cust_month["active_customers"],
        np.nan
    )

    churn_decimal = cust_month["churn_rate_pct"] / 100.0
    cust_month["lifetime_months"] = np.where(
        churn_decimal > 0, 1.0 / churn_decimal, np.nan
    )

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

    gross_margin_pct = st.slider("Assumed Gross Margin (%)", 40, 99, 80, step=1)
    gm = gross_margin_pct / 100.0

    subs["month"] = subs["period_start"].dt.to_period("M")
    events["month"] = events["date"].dt.to_period("M")
    customers["signup_month"] = customers["signup_date"].dt.to_period("M")

    active_by_month_v = (
        subs.groupby("month")["customer_id"].nunique().reset_index(name="active_customers")
    )
    mrr_by_month_v = subs.groupby("month")["mrr"].sum().reset_index(name="mrr")
    kpi_base = active_by_month_v.merge(mrr_by_month_v, on="month", how="left")
    kpi_base["arpa"] = np.where(kpi_base["active_customers"] > 0,
                                kpi_base["mrr"] / kpi_base["active_customers"], np.nan)

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

    mkt_month = (events.groupby("month")[["spend","conversions"]].sum().reset_index())
    mkt_month["cac"] = np.where(mkt_month["conversions"] > 0,
                                mkt_month["spend"] / mkt_month["conversions"], np.nan)

    channels_keep = ["Google Ads", "LinkedIn Ads"]

    cac_month_ch = (
        events.query("channel in @channels_keep")
              .groupby(["month", "channel"], as_index=False)[["spend", "conversions"]]
              .sum()
    )

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

    fig_cac_ch_all.update_layout(
        xaxis={"type": "category", "categoryorder": "category ascending"},
        yaxis_title="CAC (€)"
    )

    st.plotly_chart(fig_cac_ch_all, use_container_width=True)


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
    st.divider()

    events["month"] = events["date"].dt.to_period("M")
    subs["month"] = subs["period_start"].dt.to_period("M")
    customers["signup_month"] = customers["signup_date"].dt.to_period("M")

    funnel = (
        events.groupby("month")[["visits", "trials", "conversions", "spend"]]
              .sum()
              .reset_index()
              .sort_values("month")
    )
    funnel["visit_to_trial_rate"] = np.where(
        funnel["visits"] > 0, funnel["trials"] / funnel["visits"] * 100, np.nan
    )
    funnel["trial_to_paid_rate"] = np.where(
        funnel["trials"] > 0, funnel["conversions"] / funnel["trials"] * 100, np.nan
    )
    funnel["overall_conv_rate"] = np.where(
        funnel["visits"] > 0, funnel["conversions"] / funnel["visits"] * 100, np.nan
    )

    last_row = funnel.iloc[-1]
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Visits → Trials", f"{last_row['visit_to_trial_rate']:.2f}%")
    with c2:
        st.metric("Trials → Paid", f"{last_row['trial_to_paid_rate']:.2f}%")
    with c3:
        st.metric("Overall Conversion", f"{last_row['overall_conv_rate']:.2f}%")

    st.subheader("Conversion Rates Over Time")
    fplot = funnel.copy()
    fplot["month"] = fplot["month"].astype(str)

    conv_long = fplot.melt(
        id_vars="month",
        value_vars=["visit_to_trial_rate", "trial_to_paid_rate", "overall_conv_rate"],
        var_name="Metric",
        value_name="Rate"
    )

    fig_conv_bar = px.bar(
        conv_long,
        x="month",
        y="Rate",
        color="Metric",
        barmode="group",
        title="Funnel Conversion Rates (Grouped Bar Chart)",
        labels={"Rate": "Rate (%)", "month": "Month"}
    )
    fig_conv_bar.update_layout(
        xaxis={"type": "category", "categoryorder": "category ascending"},
        yaxis_title="Rate (%)"
    )
    st.plotly_chart(fig_conv_bar, use_container_width=True)

    st.subheader("Spend per Channel (All Months)")
    spend_ch = (
        events.groupby(["month", "channel"], as_index=False)["spend"]
              .sum()
              .sort_values(["month", "channel"])
    )
    spend_ch_plot = spend_ch.copy()
    spend_ch_plot["month"] = spend_ch_plot["month"].astype(str)

    fig_spend = px.bar(
        spend_ch_plot,
        x="month",
        y="spend",
        color="channel",
        barmode="group",
        title="Marketing Spend by Channel Over Time",
        labels={"spend": "Spend (€)", "month": "Month", "channel": "Channel"}
    )
    fig_spend.update_layout(xaxis={"type": "category", "categoryorder": "category ascending"})
    st.plotly_chart(fig_spend, use_container_width=True)

    spend_wide = spend_ch.pivot_table(index="month", columns="channel", values="spend", fill_value=0)

    summary = (
        funnel[["month", "visits", "trials", "conversions",
                "visit_to_trial_rate", "trial_to_paid_rate", "overall_conv_rate"]]
        .merge(spend_wide.add_prefix("spend_"), left_on="month", right_index=True, how="left")
        .sort_values("month")
    )

    summary_disp = summary.copy()
    summary_disp["Month"] = summary_disp["month"].astype(str)
    summary_disp = summary_disp.drop(columns=["month"])

    for col in ["visit_to_trial_rate", "trial_to_paid_rate", "overall_conv_rate"]:
        summary_disp[col] = summary_disp[col].round(2)

    st.subheader("Monthly Funnel & Marketing Summary")
    st.dataframe(summary_disp, use_container_width=True)




with tab6:
    st.header("🌍 Segmentation")
    st.divider()

    subs["month"] = subs["period_start"].dt.to_period("M")

    channel_col = (
        "channel" if "channel" in customers.columns
        else ("acquisition_channel" if "acquisition_channel" in customers.columns else None)
    )

    seg_cols = ["customer_id"]
    if "country" in customers.columns: seg_cols.append("country")
    if channel_col is not None: seg_cols.append(channel_col)

    cust_seg = customers[seg_cols].copy()
    subs_seg = subs.merge(cust_seg, on="customer_id", how="left")

    st.subheader("MRR by Plan (monthly)")
    mrr_plan = (
        subs.groupby(["month", "plan"], as_index=False)["mrr"].sum().sort_values("month")
    )
    mrr_plan_plot = mrr_plan.copy()
    mrr_plan_plot["month"] = mrr_plan_plot["month"].astype(str)

    fig_plan = px.bar(
        mrr_plan_plot, x="month", y="mrr", color="plan", barmode="group",
        title="MRR by Plan Over Time", labels={"mrr":"MRR (€)", "month":"Month", "plan":"Plan"}
    )
    fig_plan.update_layout(xaxis={"type":"category","categoryorder":"category ascending"})
    st.plotly_chart(fig_plan, use_container_width=True)

    if "country" in subs_seg.columns:
        st.subheader("MRR by Country (monthly)")
        mrr_country = (
            subs_seg.groupby(["month", "country"], as_index=False)["mrr"].sum().sort_values("month")
        )
        mrr_country_plot = mrr_country.copy()
        mrr_country_plot["month"] = mrr_country_plot["month"].astype(str)

        fig_country = px.bar(
            mrr_country_plot, x="month", y="mrr", color="country", barmode="group",
            title="MRR by Country Over Time", labels={"mrr":"MRR (€)", "month":"Month", "country":"Country"}
        )
        fig_country.update_layout(xaxis={"type":"category","categoryorder":"category ascending"})
        st.plotly_chart(fig_country, use_container_width=True)
    else:
        st.info("`country` column not found in customers.csv — skipping MRR by Country.")

    if channel_col is not None:
        st.subheader("MRR by Acquisition Channel (monthly)")
        mrr_channel = (
            subs_seg.groupby(["month", channel_col], as_index=False)["mrr"].sum().sort_values("month")
        )
        mrr_channel_plot = mrr_channel.rename(columns={channel_col:"channel"}).copy()
        mrr_channel_plot["month"] = mrr_channel_plot["month"].astype(str)

        fig_channel = px.bar(
            mrr_channel_plot, x="month", y="mrr", color="channel", barmode="group",
            title="MRR by Acquisition Channel Over Time",
            labels={"mrr":"MRR (€)", "month":"Month", "channel":"Channel"}
        )
        fig_channel.update_layout(xaxis={"type":"category","categoryorder":"category ascending"})
        st.plotly_chart(fig_channel, use_container_width=True)
    else:
        st.info("No acquisition channel column (`channel` or `acquisition_channel`) — skipping MRR by Channel.")

    st.divider()

    st.subheader("Churn by Segment (monthly)")

    # --- 1) Per-customer, per-month status with segments (correct .agg dict) ---
    agg_dict = {"mrr": "sum", "plan": "last"}
    if "country" in subs_seg.columns:
        agg_dict["country"] = "last"
    if channel_col is not None:
        agg_dict[channel_col] = "last"

    per_cust = (
        subs_seg.groupby(["customer_id", "month"], as_index=False)
                .agg(agg_dict)
                .sort_values(["customer_id", "month"])
    )

    # --- 2) Build previous-month snapshot and align (shift month forward) ---
    rename_prev = {"mrr": "mrr_prev", "plan": "plan_prev"}
    if "country" in per_cust.columns:
        rename_prev["country"] = "country_prev"
    if channel_col is not None:
        rename_prev[channel_col] = f"{channel_col}_prev"

    prev = per_cust.rename(columns=rename_prev).copy()
    prev["month"] = prev["month"] + 1  # align prev -> current

    # --- 3) Merge & clean ghost months created by the shift ---
    cur_prev = per_cust.merge(prev, on=["customer_id", "month"], how="outer")

    min_m, max_m = subs_seg["month"].min(), subs_seg["month"].max()
    cur_prev = cur_prev[(cur_prev["month"] >= min_m) & (cur_prev["month"] <= max_m)]

    # --- 4) Fill numeric NaNs for math ---
    cur_prev[["mrr", "mrr_prev"]] = cur_prev[["mrr", "mrr_prev"]].fillna(0)

    # --- 5) Choose segment (use previous month’s segment for churn attribution) ---
    seg_options = ["plan"]
    if "country" in customers.columns:
        seg_options.append("country")
    if channel_col is not None:
        seg_options.append(channel_col)

    seg_choice = st.selectbox(
        "Choose segment dimension",
        options=seg_options,
        index=0,
        format_func=lambda x: {"plan": "Plan", "country": "Country", channel_col: "Channel"}.get(x, x)
    )

    seg_prev_col = {"plan": "plan_prev", "country": "country_prev"}.get(seg_choice, f"{channel_col}_prev")

    # --- 6) Keep only rows active in the previous month and with a valid prev segment ---
    cur_prev = cur_prev[cur_prev["mrr_prev"] > 0]
    cur_prev = cur_prev.dropna(subset=[seg_prev_col])

    # --- 7) Aggregate churn by month & segment ---
    cur_prev["churned_flag"] = (cur_prev["mrr_prev"] > 0) & (cur_prev["mrr"] == 0)

    churn_seg = (
        cur_prev.groupby(["month", seg_prev_col], as_index=False)
                .agg(active_prev=("mrr_prev", lambda s: (s > 0).sum()),
                    churned=("churned_flag", "sum"))
                .rename(columns={seg_prev_col: "segment"})
                .sort_values("month")
    )

    churn_seg["churn_rate_pct"] = np.where(
        churn_seg["active_prev"] > 0,
        churn_seg["churned"] / churn_seg["active_prev"] * 100,
        np.nan
    )

    # --- 8) Plot & table ---
    churn_plot = churn_seg.copy()
    churn_plot["month"] = churn_plot["month"].astype(str)

    fig_churn_seg = px.line(
        churn_plot, x="month", y="churn_rate_pct", color="segment", markers=True,
        title=f"Churn Rate by { 'Plan' if seg_choice=='plan' else ('Country' if seg_choice=='country' else 'Channel') }",
        labels={"churn_rate_pct": "Churn Rate (%)", "month": "Month", "segment": "Segment"}
    )
    fig_churn_seg.update_layout(yaxis_title="Churn Rate (%)")
    st.plotly_chart(fig_churn_seg, use_container_width=True)

    st.dataframe(
        churn_seg.rename(columns={
            "month": "Month",
            "segment": "Segment",
            "active_prev": "Active (prev)",
            "churned": "Churned",
            "churn_rate_pct": "Churn Rate (%)"
        }).assign(**{"Churn Rate (%)": lambda d: d["Churn Rate (%)"].round(2)}),
        use_container_width=True
)



