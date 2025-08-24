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

"""
🏦 Revenue & Growth

	MRR (Monthly Recurring Revenue) → total subscription revenue per month.

	ARR (Annual Recurring Revenue) → MRR × 12.

	New MRR → revenue from brand new customers.

	Expansion MRR → upsells / add-ons / seat increases.

	Contraction MRR → downgrades / reduced usage.

	Churned MRR → lost revenue from cancellations.

	Net New MRR = New + Expansion − Contraction − Churn.

	Growth Rate → % change in MRR month-over-month.

👥 Customer Metrics

	Active Customers → count of distinct customers per month.

	New Customers → count of signups.

	Churned Customers (Logo Churn) → customers who left in a month.

	Customer Churn Rate = churned / active previous month.

	Average Revenue per Account (ARPA) = MRR / # active customers.

	Customer Lifetime (months) = 1 / churn rate.

💰 Value Metrics

	LTV (Customer Lifetime Value) = ARPA × Gross Margin ÷ Churn Rate.

	CAC (Customer Acquisition Cost) = spend ÷ conversions.

	CAC by Channel (from events_marketing.csv).

	LTV/CAC Ratio (efficiency benchmark, >3 = healthy).

	Payback Period = CAC ÷ ARPA (months to recover acquisition cost).

📈 Retention & Cohorts

	Cohort Retention → survival of customers grouped by signup month.

	Revenue Retention:

	Gross Revenue Retention (GRR) = 1 − (churned + contraction) ÷ starting MRR.

	Net Revenue Retention (NRR) = (starting MRR − churn + expansion) ÷ starting MRR.

	Logo Retention = % of customers still active from a cohort.

📊 Funnel & Marketing

	Visits → Trials Conversion Rate = trials ÷ visits.

	Trials → Paid Conversion Rate = conversions ÷ trials.

	Overall Funnel Conversion Rate = conversions ÷ visits.

	Spend per Channel (Google, LinkedIn, Organic, Referral).

	ROI per Channel = (MRR from converted customers − spend) ÷ spend.

🌍 Segmentation

	Using customers.csv:

	MRR by Plan (Basic, Pro, Enterprise).

	MRR by Country.

	MRR by Acquisition Channel.

	Churn by Segment (are Enterprise customers more loyal?).
"""



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
    # show metrics like MRR, ARR, New/Expansion/Churn MRR, Growth rate
    # charts: MRR over time, waterfall of movements

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
