import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path
import google.generativeai as genai
import json

# Project paths
BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data"

@st.cache_data
def load_data():
    """Load core CSVs once per session; parse dates for time ops."""
    cust = pd.read_csv(DATA / "customers.csv", parse_dates=["signup_date"])
    subs = pd.read_csv(DATA / "subscriptions.csv", parse_dates=["period_start", "period_end"])
    events = pd.read_csv(DATA / "events_marketing.csv", parse_dates=["date"])
    return cust, subs, events

customers, subs, events = load_data()

# Keep originals clean; add normalized time keys
subs = subs.copy()
events = events.copy()
customers = customers.copy()
subs["month"] = subs["period_start"].dt.to_period("M")
events["month"] = events["date"].dt.to_period("M")
customers["signup_month"] = customers["signup_date"].dt.to_period("M")
customers["cohort"] = customers["signup_month"]

# Support both possible channel column names; None if absent
CHANNEL_COL = "channel" if "channel" in customers.columns else ("acquisition_channel" if "acquisition_channel" in customers.columns else None)

# --- Revenue aggregates ---
mrr_by_month = (
    subs.groupby("month", as_index=False)["mrr"]
        .sum()
        .rename(columns={"mrr": "mrr_total"})
        .sort_values("month")
)
mrr_by_month["month_str"] = mrr_by_month["month"].astype(str)

active_by_month = (
    subs.groupby("month")["customer_id"]
        .nunique()
        .reset_index(name="active_customers")
        .sort_values("month")
)

# Per-customer per-month MRR to compute movements
cm = (
    subs.groupby(["customer_id", "month"], as_index=False)["mrr"]
        .sum()
        .sort_values(["customer_id", "month"])
)

# Lagged frame to compare current vs previous month MRR
cm_prev = cm.rename(columns={"mrr": "mrr_prev"}).copy()
cm_prev["month"] = cm_prev["month"] + 1  # shift forward to align month t with t-1

cur_prev = cm.merge(cm_prev, on=["customer_id", "month"], how="outer")
cur_prev[["mrr", "mrr_prev"]] = cur_prev[["mrr", "mrr_prev"]].fillna(0)

# Restrict to observed period range only
VALID_MIN, VALID_MAX = cm["month"].min(), cm["month"].max()
cur_prev = cur_prev[(cur_prev["month"] >= VALID_MIN) & (cur_prev["month"] <= VALID_MAX)]
VALID_MONTHS = set(cm["month"].unique())

# MRR movement buckets (new/expansion/contraction/churn)
mov_month = (
    cur_prev.assign(
        new_mrr=np.where((cur_prev["mrr_prev"] == 0) & (cur_prev["mrr"] > 0), cur_prev["mrr"], 0.0),
        expansion_mrr=np.where((cur_prev["mrr"] > cur_prev["mrr_prev"]) & (cur_prev["mrr_prev"] > 0), cur_prev["mrr"] - cur_prev["mrr_prev"], 0.0),
        contraction_mrr=np.where((cur_prev["mrr_prev"] > cur_prev["mrr"]) & (cur_prev["mrr"] > 0), cur_prev["mrr_prev"] - cur_prev["mrr"], 0.0),
        churned_mrr=np.where((cur_prev["mrr_prev"] > 0) & (cur_prev["mrr"] == 0), cur_prev["mrr_prev"], 0.0),
    )
    .groupby("month", as_index=False)[["new_mrr","expansion_mrr","contraction_mrr","churned_mrr"]]
    .sum()
    .sort_values("month")
)
# Exclude last month to avoid partial movement artifacts
if len(mov_month) > 0:
    mov_month = mov_month.iloc[:-1]
mov_month["net_new_mrr"] = mov_month["new_mrr"] + mov_month["expansion_mrr"] - mov_month["contraction_mrr"] - mov_month["churned_mrr"]

# --- Customer metrics preparation ---
# New and churned logo counts by month
new_by_month = (
    customers.groupby("signup_month")["customer_id"]
             .nunique()
             .reset_index()
             .rename(columns={"signup_month":"month","customer_id":"new_customers"})
             .sort_values("month")
)
churned_rows = cur_prev[(cur_prev["mrr_prev"] > 0) & (cur_prev["mrr"] == 0)]
churned_by_month = churned_rows.groupby("month")["customer_id"].nunique().reset_index(name="churned_customers")

mrr_by_month_2 = mrr_by_month.rename(columns={"mrr_total":"mrr"}).copy()

# Combine into single customer-month frame
cust_month = (
    active_by_month
    .merge(new_by_month, on="month", how="left")
    .merge(churned_by_month, on="month", how="left")
    .merge(mrr_by_month_2, on="month", how="left")
    .fillna({"new_customers":0, "churned_customers":0})
    .sort_values("month")
)

# Derived KPIs (guard NaNs)
cust_month["active_prev"] = cust_month["active_customers"].shift(1)
cust_month["churn_rate_pct"] = np.where(cust_month["active_prev"] > 0, cust_month["churned_customers"] / cust_month["active_prev"] * 100, np.nan)
cust_month["arpa"] = np.where(cust_month["active_customers"] > 0, cust_month["mrr"] / cust_month["active_customers"], np.nan)
churn_decimal = cust_month["churn_rate_pct"] / 100.0
cust_month["lifetime_months"] = np.where(churn_decimal > 0, 1.0 / churn_decimal, np.nan)

# --- Marketing aggregates ---
events_month = (
    events.groupby("month", as_index=False)[["visits","trials","conversions","spend"]]
          .sum()
          .sort_values("month")
)
# Guard against divide-by-zero
events_month["cac"] = np.where(events_month["conversions"] > 0, events_month["spend"] / events_month["conversions"], np.nan)

events_month_channel = (
    events.groupby(["month","channel"], as_index=False)[["spend","conversions","visits","trials"]]
          .sum()
          .sort_values(["month","channel"])
)
# Focus on primary paid channels for CAC split
CHANNELS_KEEP = ["Google Ads", "LinkedIn Ads"]
events_month_channel_gl = events_month_channel[events_month_channel["channel"].isin(CHANNELS_KEEP)].copy()
events_month_channel_gl["cac"] = np.where(events_month_channel_gl["conversions"] > 0, events_month_channel_gl["spend"] / events_month_channel_gl["conversions"], np.nan)

# --- Enrich subscriptions with segmentation columns (if present) ---
seg_cols = ["customer_id"]
if "country" in customers.columns:
    seg_cols.append("country")
if CHANNEL_COL is not None:
    seg_cols.append(CHANNEL_COL)

cust_seg = customers[seg_cols].copy()
subs_seg = subs.merge(cust_seg, on="customer_id", how="left")


mov_ret = (
    cur_prev.assign(
        base_mrr=np.where(cur_prev["mrr_prev"] > 0, cur_prev["mrr_prev"], 0.0),
        churn=np.where((cur_prev["mrr_prev"] > 0) & (cur_prev["mrr"] == 0), cur_prev["mrr_prev"], 0.0),
        contraction=np.where((cur_prev["mrr_prev"] > cur_prev["mrr"]) & (cur_prev["mrr"] > 0), cur_prev["mrr_prev"] - cur_prev["mrr"], 0.0),
        expansion=np.where((cur_prev["mrr"] > cur_prev["mrr_prev"]) & (cur_prev["mrr_prev"] > 0), cur_prev["mrr"] - cur_prev["mrr_prev"], 0.0)
    )
    .groupby("month", as_index=False)[["base_mrr","churn","contraction","expansion"]]
    .sum()
    .sort_values("month")
)
mov_ret = mov_ret[mov_ret["month"].isin(VALID_MONTHS)].copy()
mov_ret["grr"] = np.where(mov_ret["base_mrr"] > 0, (mov_ret["base_mrr"] - mov_ret["churn"] - mov_ret["contraction"]) / mov_ret["base_mrr"] * 100, np.nan)
mov_ret["nrr"] = np.where(mov_ret["base_mrr"] > 0, (mov_ret["base_mrr"] - mov_ret["churn"] - mov_ret["contraction"] + mov_ret["expansion"]) / mov_ret["base_mrr"] * 100, np.nan)

# Configure Gemini AI
def configure_ai():
    """Configure Gemini AI with API key"""
    api_key = st.secrets.get("GEMINI_API_KEY", "")
    if not api_key:
        st.sidebar.warning("⚠️ Add your Gemini API key to .streamlit/secrets.toml to enable the AI chatbot")
        return None
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-flash')
    return model

def get_data_context():
    """Prepare comprehensive data context for AI queries"""

    try:
        # Revenue metrics with trends
        mrr_growth_rates = mrr_by_month["mrr_total"].pct_change() * 100
        recent_growth = mrr_growth_rates.tail(3).mean() if len(mrr_growth_rates) >= 3 else None
        
        # Customer acquisition and churn trends - convert Period index to string
        monthly_new_customers = new_by_month.set_index('month')['new_customers'] if not new_by_month.empty else pd.Series()
        if not monthly_new_customers.empty:
            monthly_new_customers.index = monthly_new_customers.index.astype(str)
            
        monthly_churned = churned_by_month.set_index('month')['churned_customers'] if not churned_by_month.empty else pd.Series()
        if not monthly_churned.empty:
            monthly_churned.index = monthly_churned.index.astype(str)
        
        # Marketing efficiency
        marketing_summary = events_month.copy()
        total_spend = marketing_summary['spend'].sum() if not marketing_summary.empty else 0
        total_conversions = marketing_summary['conversions'].sum() if not marketing_summary.empty else 0
        avg_cac = total_spend / total_conversions if total_conversions > 0 else None
        
        # Channel performance
        channel_performance = {}
        if not events_month_channel.empty:
            for channel in events_month_channel['channel'].unique():
                channel_data = events_month_channel[events_month_channel['channel'] == channel]
                channel_spend = channel_data['spend'].sum()
                channel_conversions = channel_data['conversions'].sum()
                channel_cac = channel_spend / channel_conversions if channel_conversions > 0 else None
                channel_performance[channel] = {
                    'total_spend': float(channel_spend),
                    'total_conversions': int(channel_conversions),
                    'cac': float(channel_cac) if channel_cac else None
                }

        # Revenue segments
        plan_revenue = {}
        if not subs.empty:
            plan_summary = subs.groupby('plan')['mrr'].sum()
            for plan, revenue in plan_summary.items():
                plan_revenue[plan] = float(revenue)

        # Retention analysis
        retention_metrics = {}
        if not mov_ret.empty:
            latest_retention = mov_ret.iloc[-1] if len(mov_ret) > 0 else None
            if latest_retention is not None:
                retention_metrics = {
                    'latest_grr': float(latest_retention['grr']) if pd.notna(latest_retention['grr']) else None,
                    'latest_nrr': float(latest_retention['nrr']) if pd.notna(latest_retention['nrr']) else None,
                    'avg_grr': float(mov_ret['grr'].mean()) if not mov_ret['grr'].empty else None,
                    'avg_nrr': float(mov_ret['nrr'].mean()) if not mov_ret['nrr'].empty else None
                }

        # Cohort insights
        cohort_insights = {}
        if not customers.empty:
            cohort_sizes = customers.groupby('cohort').size()
            largest_cohort = cohort_sizes.idxmax() if not cohort_sizes.empty else None
            cohort_insights = {
                'total_cohorts': len(cohort_sizes),
                'largest_cohort': str(largest_cohort) if largest_cohort else None,
                'largest_cohort_size': int(cohort_sizes.max()) if not cohort_sizes.empty else None,
                'avg_cohort_size': float(cohort_sizes.mean()) if not cohort_sizes.empty else None
            }

        # Business health indicators
        latest_metrics = cust_month.iloc[-1] if not cust_month.empty else None
        prev_metrics = cust_month.iloc[-2] if len(cust_month) > 1 else None
        
        health_indicators = {}
        if latest_metrics is not None:
            health_indicators = {
                'mrr_trend': 'growing' if recent_growth and recent_growth > 0 else 'declining' if recent_growth and recent_growth < 0 else 'stable',
                'customer_growth': float(latest_metrics['new_customers'] - latest_metrics['churned_customers']) if pd.notna(latest_metrics['new_customers']) and pd.notna(latest_metrics['churned_customers']) else None,
                'churn_trend': 'improving' if prev_metrics is not None and pd.notna(latest_metrics['churn_rate_pct']) and pd.notna(prev_metrics['churn_rate_pct']) and latest_metrics['churn_rate_pct'] < prev_metrics['churn_rate_pct'] else 'worsening' if prev_metrics is not None and pd.notna(latest_metrics['churn_rate_pct']) and pd.notna(prev_metrics['churn_rate_pct']) and latest_metrics['churn_rate_pct'] > prev_metrics['churn_rate_pct'] else 'stable'
            }

        # Convert DataFrames to records with string conversion for Period columns
        def convert_periods_to_string(df):
            """Convert DataFrame with Period columns to JSON-serializable format"""
            df_copy = df.copy()
            for col in df_copy.columns:
                if df_copy[col].dtype.name.startswith('period'):
                    df_copy[col] = df_copy[col].astype(str)
            return df_copy.to_dict('records')

        # Comprehensive context
        context = {
            # Core metrics (existing) - convert Period columns
            "mrr_summary": convert_periods_to_string(mrr_by_month),
            "customer_metrics": convert_periods_to_string(cust_month),
            "marketing_funnel": convert_periods_to_string(events_month) if not events_month.empty else [],
            
            # Enhanced latest metrics
            "latest_metrics": {
                "mrr": float(mrr_by_month["mrr_total"].iloc[-1]),
                "arr": float(mrr_by_month["mrr_total"].iloc[-1] * 12),
                "active_customers": int(active_by_month["active_customers"].iloc[-1]),
                "latest_churn_rate": float(cust_month["churn_rate_pct"].iloc[-1]) if pd.notna(cust_month["churn_rate_pct"].iloc[-1]) else None,
                "latest_arpa": float(cust_month["arpa"].iloc[-1]) if pd.notna(cust_month["arpa"].iloc[-1]) else None,
                "recent_growth_rate": float(recent_growth) if recent_growth else None,
                "total_customers_acquired": int(customers.shape[0]) if not customers.empty else 0
            },
            
            # Revenue breakdown
            "revenue_analysis": {
                "plan_revenue_distribution": plan_revenue,
                "mrr_movements": convert_periods_to_string(mov_month) if not mov_month.empty else [],
                "average_monthly_growth": float(mrr_growth_rates.mean()) if not mrr_growth_rates.empty else None,
                "growth_volatility": float(mrr_growth_rates.std()) if not mrr_growth_rates.empty else None
            },
            
            # Customer insights
            "customer_insights": {
                "acquisition_trend": monthly_new_customers.to_dict() if not monthly_new_customers.empty else {},
                "churn_trend": monthly_churned.to_dict() if not monthly_churned.empty else {},
                "cohort_analysis": cohort_insights,
                "retention_metrics": retention_metrics
            },
            
            # Marketing performance
            "marketing_analysis": {
                "overall_cac": float(avg_cac) if avg_cac else None,
                "total_marketing_spend": float(total_spend),
                "total_conversions": int(total_conversions),
                "channel_performance": channel_performance,
                "conversion_funnel": {
                    "avg_visit_to_trial": float(events_month['visit_to_trial_rate'].mean()) if not events_month.empty and 'visit_to_trial_rate' in events_month else None,
                    "avg_trial_to_paid": float(events_month['trial_to_paid_rate'].mean()) if not events_month.empty and 'trial_to_paid_rate' in events_month else None
                }
            },
            
            # Business health
            "business_health": health_indicators,
            
            # Data context
            "data_periods": {
                "start_month": str(mrr_by_month["month"].min()),
                "end_month": str(mrr_by_month["month"].max()),
                "total_months": len(mrr_by_month),
                "data_completeness": {
                    "has_customer_data": not customers.empty,
                    "has_subscription_data": not subs.empty,
                    "has_marketing_data": not events.empty
                }
            },
            
            # Comparative benchmarks (you can adjust these based on your industry)
            "industry_context": {
                "typical_saas_churn_rate": "5-7% monthly for B2B SaaS",
                "healthy_growth_rate": "10-20% monthly for early stage",
                "good_ltv_cac_ratio": "3:1 or higher",
                "target_payback_period": "12-18 months"
            }
        }
        
        return context
        
    except Exception as e:
        st.error(f"Error preparing context: {str(e)}")
        return {
            "mrr_summary": convert_periods_to_string(mrr_by_month),
            "customer_metrics": convert_periods_to_string(cust_month),
            "latest_metrics": {
                "mrr": float(mrr_by_month["mrr_total"].iloc[-1]),
                "arr": float(mrr_by_month["mrr_total"].iloc[-1] * 12),
                "active_customers": int(active_by_month["active_customers"].iloc[-1])
            },
            "error": "Reduced context due to processing error"
        }
        
    except Exception as e:
        st.error(f"Error preparing context: {str(e)}")
        return {
            "mrr_summary": mrr_by_month.to_dict('records'),
            "customer_metrics": cust_month.to_dict('records'),
            "latest_metrics": {
                "mrr": float(mrr_by_month["mrr_total"].iloc[-1]),
                "arr": float(mrr_by_month["mrr_total"].iloc[-1] * 12),
                "active_customers": int(active_by_month["active_customers"].iloc[-1])
            },
            "error": "Reduced context due to processing error"
        }

def query_ai_chatbot(model, question, context):
    """Query the AI model with comprehensive business context"""
    prompt = f"""
    You are a senior SaaS business consultant. Analyze the data and provide CONCISE, actionable insights.

    BUSINESS DATA CONTEXT:
    {json.dumps(context, indent=2, default=str)}

    USER QUESTION: {question}

    RESPONSE FORMAT:
    - Start directly with key findings - NO greetings or introductions
    - Use bullet points for multiple insights
    - Include specific numbers from the data
    - Focus on actionable recommendations
    - Keep total response under 200 words
    - End with 2-3 prioritized actions
    - Use emojis for visual clarity (🔴 for problems, 💡 for solutions, 📊 for data)

    ANALYSIS PRIORITIES:
    1. Identify the most critical issue impacting the metric asked about
    2. Provide 2-3 specific data points supporting your analysis
    3. Give 2-3 immediate actionable recommendations
    4. Skip explanatory text and industry context unless directly relevant

    Be direct, data-driven, and actionable. No fluff.
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Sorry, I encountered an error analyzing your data: {str(e)}. Please check your API key and model configuration."

# ---------- UI ----------
st.set_page_config(page_title="KPI Dashboard", layout="wide")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🏦 Revenue & Growth",
    "👥 Customer Metrics",
    "💰 Value Metrics",
    "📈 Retention & Cohorts",
    "📊 Funnel & Marketing",
    "🌍 Segmentation"
])

# Initialize AI model
ai_model = configure_ai()

# Add chatbot sidebar
with st.sidebar:
    st.header("🤖 AI Business Assistant")
    
    if ai_model:
        # Chat interface
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
        
        # Chat input
        if question := st.chat_input("Ask about your business metrics..."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": question})
            
            # Display user message
            with st.chat_message("user"):
                st.write(question)
            
            # Get AI response
            with st.chat_message("assistant"):
                with st.spinner("Analyzing your data..."):
                    context = get_data_context()
                    response = query_ai_chatbot(ai_model, question, context)
                    st.write(response)
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Clear chat button
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.rerun()
            
        # Example questions
        st.subheader("💡 Example Questions")
        example_questions = [
            "What's our current MRR growth trend?",
            "How is our customer churn performing?",
            "What's our LTV/CAC ratio telling us?",
            "Which months had the best performance?",
            "How can we improve our retention?",
            "What's driving our revenue growth?"
        ]
        
        for eq in example_questions:
            if st.button(eq, key=f"eq_{hash(eq)}"):
                # Add to chat
                st.session_state.messages.append({"role": "user", "content": eq})
                context = get_data_context()
                response = query_ai_chatbot(ai_model, eq, context)
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.rerun()
    
    else:
        st.info("Configure your Gemini API key to enable the AI chatbot")
        st.markdown("""
        **Setup Instructions:**
        1. Get a free API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
        2. Create `.streamlit/secrets.toml` in your project root
        3. Add: `GEMINI_API_KEY = "your-api-key-here"`
        """)


with tab1:
    st.header("🏦 Revenue & Growth")
    st.divider()

    # Latest-run KPIs
    latest_period = mrr_by_month["month"].max()
    latest_mrr = float(mrr_by_month.loc[mrr_by_month["month"] == latest_period, "mrr_total"].iloc[0])
    prev_idx = mrr_by_month["month"] == latest_period
    prev_mrr_series = mrr_by_month["mrr_total"].shift(1)
    prev_mrr = float(prev_mrr_series[prev_idx].iloc[0]) if prev_idx.any() else None
    # Handle None / zero gracefully
    growth_latest = ((latest_mrr - prev_mrr) / prev_mrr * 100) if (prev_mrr and prev_mrr != 0) else None
    latest_arr = latest_mrr * 12

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("MRR (latest)", f"€{latest_mrr:,.0f}")
    with c2:
        st.metric("ARR (latest)", f"€{latest_arr:,.0f}")
    with c3:
        st.metric("Monthly Growth Rate (latest)", f"{growth_latest:+.2f}%" if growth_latest is not None else "—")

    st.subheader("MRR (Monthly Recurring Revenue)")
    avg_mrr = mrr_by_month["mrr_total"].mean()
    st.metric(label="Average MRR", value=f"€{avg_mrr:,.0f}")
    fig = px.line(mrr_by_month, x="month_str", y="mrr_total", markers=True, title="MRR Over Time", labels={"month_str":"Month","mrr_total":"MRR (€)"})
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("ARR (Annual Recurring Revenue)")
    arr_by_month = mrr_by_month.assign(ARR=mrr_by_month["mrr_total"] * 12)
    latest_arr = arr_by_month["ARR"].iloc[-1]
    st.metric(label="ARR (run-rate of latest month)", value=f"€{latest_arr:,.0f}")
    fig_arr = px.line(arr_by_month, x="month_str", y="ARR", markers=True, title="ARR Over Time", labels={"month_str":"Month","ARR":"ARR (€)"})
    st.plotly_chart(fig_arr, use_container_width=True)

    st.divider()
    st.subheader("MRR Movements (New, Expansion, Contraction, Churned)")
    plot_df = mov_month.copy()
    plot_df["month_str"] = plot_df["month"].astype(str)
    # Flip negative movements for divergent bar visual
    plot_long = (
        plot_df.assign(Contraction=-plot_df["contraction_mrr"], Churned=-plot_df["churned_mrr"], New=plot_df["new_mrr"], Expansion=plot_df["expansion_mrr"])
                .melt(id_vars="month", value_vars=["New","Expansion","Contraction","Churned"], var_name="Movement", value_name="Amount")
    )
    plot_long["month_str"] = plot_long["month"].astype(str)
    fig_mov = px.bar(plot_long, x="month_str", y="Amount", color="Movement", title="MRR Movements by Month", labels={"month_str":"Month","Amount":"MRR Movement (€)"})
    st.plotly_chart(fig_mov, use_container_width=True)

    st.divider()
    st.subheader("Growth Rate (MoM % change in MRR)")
    mrr_by_month_sorted = mrr_by_month.copy()
    mrr_by_month_sorted["growth_rate"] = mrr_by_month_sorted["mrr_total"].pct_change() * 100
    latest_growth = mrr_by_month_sorted["growth_rate"].iloc[-1]
    st.metric(label="Latest Growth Rate", value=f"{latest_growth:+.2f}%")
    fig_growth = px.line(mrr_by_month_sorted, x="month_str", y="growth_rate", markers=True, title="MRR Growth Rate (Month-over-Month)", labels={"month_str":"Month","growth_rate":"Growth Rate (%)"})
    st.plotly_chart(fig_growth, use_container_width=True)

    # Tabular summary (rounded for readability)
    mrr_month_tbl = mrr_by_month.rename(columns={"month":"month_per","mrr_total":"MRR (€)"}).copy()
    mrr_month_tbl["ARR (€)"] = mrr_month_tbl["MRR (€)"] * 12
    mrr_month_tbl["Growth Rate (%)"] = mrr_month_tbl["MRR (€)"].pct_change() * 100
    mov_renamed = mov_month.rename(columns={"month":"month_per","new_mrr":"New (€)","expansion_mrr":"Expansion (€)","contraction_mrr":"Contraction (€)","churned_mrr":"Churned (€)","net_new_mrr":"Net New (€)"})
    summary = mrr_month_tbl.merge(mov_renamed, on="month_per", how="left").fillna({"New (€)":0,"Expansion (€)":0,"Contraction (€)":0,"Churned (€)":0,"Net New (€)":0})
    summary_display = summary.copy()
    summary_display["Month"] = summary_display["month_per"].astype(str)
    summary_display = summary_display.drop(columns=["month_per"])
    summary_display = summary_display[["Month","MRR (€)","ARR (€)","Growth Rate (%)","New (€)","Expansion (€)","Contraction (€)","Churned (€)","Net New (€)"]]
    for col in ["MRR (€)","ARR (€)","New (€)","Expansion (€)","Contraction (€)","Churned (€)","Net New (€)"]:
        summary_display[col] = summary_display[col].round(2)
    summary_display["Growth Rate (%)"] = summary_display["Growth Rate (%)"].round(2)
    st.dataframe(summary_display, use_container_width=True)

with tab2:
    st.header("👥 Customer Metrics")
    st.divider()

    # New and churned logo counts by month
    new_by_month = (
        customers.groupby("signup_month")["customer_id"]
                 .nunique()
                 .reset_index()
                 .rename(columns={"signup_month":"month","customer_id":"new_customers"})
                 .sort_values("month")
    )
    churned_rows = cur_prev[(cur_prev["mrr_prev"] > 0) & (cur_prev["mrr"] == 0)]
    churned_by_month = churned_rows.groupby("month")["customer_id"].nunique().reset_index(name="churned_customers")

    mrr_by_month_2 = mrr_by_month.rename(columns={"mrr_total":"mrr"}).copy()

    # Combine into single customer-month frame
    cust_month = (
        active_by_month
        .merge(new_by_month, on="month", how="left")
        .merge(churned_by_month, on="month", how="left")
        .merge(mrr_by_month_2, on="month", how="left")
        .fillna({"new_customers":0, "churned_customers":0})
        .sort_values("month")
    )

    # Derived KPIs (guard NaNs)
    cust_month["active_prev"] = cust_month["active_customers"].shift(1)
    cust_month["churn_rate_pct"] = np.where(cust_month["active_prev"] > 0, cust_month["churned_customers"] / cust_month["active_prev"] * 100, np.nan)
    cust_month["arpa"] = np.where(cust_month["active_customers"] > 0, cust_month["mrr"] / cust_month["active_customers"], np.nan)
    churn_decimal = cust_month["churn_rate_pct"] / 100.0
    cust_month["lifetime_months"] = np.where(churn_decimal > 0, 1.0 / churn_decimal, np.nan)

    cust_month_plot = cust_month.copy()
    cust_month_plot["month_str"] = cust_month_plot["month"].astype(str)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Active Customers (latest)", int(cust_month["active_customers"].iloc[-1]))
    with col2:
        latest_churn = cust_month["churn_rate_pct"].iloc[-1]
        st.metric("Churn Rate (latest)", f"{latest_churn:.2f}%" if pd.notna(latest_churn) else "—")
    with col3:
        latest_arpa = cust_month["arpa"].iloc[-1]
        st.metric("ARPA (latest)", f"€{latest_arpa:,.2f}" if pd.notna(latest_arpa) else "—")

    st.subheader("Active / New / Churned Customers")
    fig_counts = px.line(
        cust_month_plot,
        x="month_str",
        y=["active_customers", "new_customers", "churned_customers"],
        markers=True,
        labels={"value": "Customers", "month_str": "Month"},
        title="Customer Counts Over Time"
    )

    # Human-friendly legend/hover
    pretty_cust_names = {
        "active_customers": "Active Customers",
        "new_customers": "New Customers",
        "churned_customers": "Churned Customers",
    }
    fig_counts.for_each_trace(
        lambda t: t.update(
            name=pretty_cust_names.get(t.name, t.name),
            legendgroup=pretty_cust_names.get(t.name, t.name),
            hovertemplate="%{y:.0f}<extra>" + pretty_cust_names.get(t.name, t.name) + "</extra>"
        )
    )
    st.plotly_chart(fig_counts, use_container_width=True)

    st.subheader("Churn Rate (%)")
    fig_churn = px.line(cust_month_plot, x="month_str", y="churn_rate_pct", markers=True, labels={"churn_rate_pct":"Churn Rate (%)","month_str":"Month"}, title="Customer Churn Rate (MoM)")
    st.plotly_chart(fig_churn, use_container_width=True)

    st.subheader("ARPA (€)")
    fig_arpa_only = px.line(cust_month_plot, x="month_str", y="arpa", markers=True, labels={"arpa":"ARPA (€)","month_str":"Month"}, title="Average Revenue per Account (ARPA)")
    fig_arpa_only.update_layout(yaxis_title="ARPA (€)")
    st.plotly_chart(fig_arpa_only, use_container_width=True)

    st.subheader("Estimated Customer Lifetime (months)")
    fig_lifetime_only = px.line(cust_month_plot, x="month_str", y="lifetime_months", markers=True, labels={"lifetime_months":"Lifetime (months)","month_str":"Month"}, title="Estimated Customer Lifetime")
    fig_lifetime_only.update_layout(yaxis_title="Months")
    st.plotly_chart(fig_lifetime_only, use_container_width=True)

    st.dataframe(
        cust_month.rename(columns={
            "month":"Month",
            "active_customers":"Active",
            "new_customers":"New",
            "churned_customers":"Churned",
            "churn_rate_pct":"Churn Rate (%)",
            "arpa":"ARPA (€)",
            "lifetime_months":"Lifetime (months)",
            "mrr":"MRR (€)"
        }).assign(**{"Churn Rate (%)": lambda d: d["Churn Rate (%)"].round(2)}),
        use_container_width=True
    )

with tab3:
    st.header("💰 Value Metrics")
    st.divider()

    # Keep GM user-adjustable for sensitivity
    gross_margin_pct = st.slider("Assumed Gross Margin (%)", 40, 99, 80, step=1)
    gm = gross_margin_pct / 100.0

    kpi_base = active_by_month.merge(mrr_by_month.rename(columns={"mrr_total":"mrr"}), on="month", how="left")
    kpi_base["arpa"] = np.where(kpi_base["active_customers"] > 0, kpi_base["mrr"] / kpi_base["active_customers"], np.nan)

    # Logo churn count for churn rate
    churned_by_month_v = ((cur_prev["mrr_prev"] > 0) & (cur_prev["mrr"] == 0)).groupby(cur_prev["month"]).sum().reset_index(name="churned")
    kpi_base = kpi_base.merge(churned_by_month_v, on="month", how="left").fillna({"churned":0})
    kpi_base["active_prev"] = kpi_base["active_customers"].shift(1)
    kpi_base["churn_rate"] = np.where(kpi_base["active_prev"] > 0, kpi_base["churned"] / kpi_base["active_prev"], np.nan)

    # Join CAC from marketing
    mkt_month = events_month[["month","cac"]].copy()
    vals = kpi_base.merge(mkt_month, on="month", how="left")

    # Standard SaaS approximations
    vals["ltv"] = np.where((vals["arpa"].notna()) & (vals["churn_rate"] > 0), vals["arpa"] * gm / vals["churn_rate"], np.nan)
    vals["ltv_cac"] = np.where((vals["ltv"].notna()) & (vals["cac"] > 0), vals["ltv"] / vals["cac"], np.nan)
    vals["payback_months"] = np.where((vals["cac"] > 0) & (vals["arpa"] > 0), vals["cac"] / vals["arpa"], np.nan)

    # Latest snapshot
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
    vals_plot["month_str"] = vals_plot["month"].astype(str)

    st.subheader("CAC Over Time (All Channels)")
    fig_cac = px.line(vals_plot, x="month_str", y="cac", markers=True, title="CAC Over Time", labels={"month_str":"Month","cac":"CAC (€)"})
    st.plotly_chart(fig_cac, use_container_width=True)

    st.subheader("CAC by Channel")
    plot_cac = events_month_channel_gl.copy()
    plot_cac["month_str"] = plot_cac["month"].astype(str)
    fig_cac_ch_all = px.bar(plot_cac, x="month_str", y="cac", color="channel", barmode="group", title="CAC by Channel Over Time", labels={"month_str":"Month","cac":"CAC (€)","channel":"Channel"})
    fig_cac_ch_all.update_layout(xaxis={"type":"category","categoryorder":"category ascending"})
    st.plotly_chart(fig_cac_ch_all, use_container_width=True)

    st.subheader("LTV Over Time")
    fig_ltv_only = px.line(vals_plot, x="month_str", y="ltv", markers=True, title="Customer Lifetime Value (LTV)", labels={"month_str":"Month","ltv":"LTV (€)"})
    st.plotly_chart(fig_ltv_only, use_container_width=True)

    st.subheader("Payback Period Over Time")
    fig_payback_only = px.line(vals_plot, x="month_str", y="payback_months", markers=True, title="Payback Period (months)", labels={"month_str":"Month","payback_months":"Months"})
    st.plotly_chart(fig_payback_only, use_container_width=True)

    st.subheader("LTV/CAC Ratio Over Time")
    fig_ratio = px.line(vals_plot, x="month_str", y="ltv_cac", markers=True, title="LTV/CAC Ratio", labels={"month_str":"Month","ltv_cac":"Ratio"})
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
    st.divider()

    # Revenue retention already calculated - just create plot version
    mov_plot = mov_ret.copy()
    mov_plot["month_str"] = mov_plot["month"].astype(str)

    st.subheader("Revenue Retention Over Time (GRR & NRR)")
    fig_ret = px.line(
        mov_plot,
        x="month_str",
        y=["grr", "nrr"],
        markers=True,
        labels={"value": "Retention (%)", "month_str": "Month"},
        title="Gross Revenue Retention (GRR) and Net Revenue Retention (NRR)"
    )
    # Clarify legend/hover
    pretty_ret_names = {
        "grr": "GRR (Gross Revenue Retention)",
        "nrr": "NRR (Net Revenue Retention)",
    }
    fig_ret.for_each_trace(
        lambda t: t.update(
            name=pretty_ret_names.get(t.name, t.name),
            legendgroup=pretty_ret_names.get(t.name, t.name),
            hovertemplate="%{y:.2f}%<extra>" + pretty_ret_names.get(t.name, t.name) + "</extra>"
        )
    )
    st.plotly_chart(fig_ret, use_container_width=True)

    # Cohort (logo) retention heatmap
    active = cm.copy()
    active["is_active"] = active["mrr"] > 0
    active = active.merge(customers[["customer_id","cohort"]], on="customer_id", how="left")
    cohort_active = (
        active[active["is_active"]]
        .groupby(["cohort","month"], as_index=False)["customer_id"]
        .nunique()
        .rename(columns={"customer_id":"active_customers"})
    )
    cohort_sizes = customers.groupby("cohort", as_index=False)["customer_id"].nunique().rename(columns={"customer_id":"cohort_size"})
    coh_piv = cohort_active.pivot_table(index="cohort", columns="month", values="active_customers", fill_value=0)
    cs = cohort_sizes.set_index("cohort").reindex(coh_piv.index)["cohort_size"]
    retention = coh_piv.divide(cs.replace(0, np.nan), axis=0) * 100
    observed_months_sorted = sorted(list(VALID_MONTHS))
    retention = retention.reindex(columns=observed_months_sorted)
    retention_plot = retention.copy()
    retention_plot.index = retention_plot.index.astype(str)
    retention_plot.columns = retention_plot.columns.astype(str)
    fig_heat = px.imshow(retention_plot, labels=dict(x="Month", y="Cohort (Signup Month)", color="Retention (%)"), color_continuous_scale="Blues", aspect="auto")
    fig_heat.update_xaxes(side="top")
    st.plotly_chart(fig_heat, use_container_width=True)

    st.subheader("Monthly Retention Summary")
    summary_disp = mov_ret.copy()
    summary_disp["Month"] = summary_disp["month"].astype(str)
    summary_disp = summary_disp.drop(columns=["month"])
    summary_disp = summary_disp.rename(columns={"base_mrr":"Base MRR (€)","churn":"Churn (€)","contraction":"Contraction (€)","expansion":"Expansion (€)","grr":"GRR (%)","nrr":"NRR (%)"})
    summary_disp[["GRR (%)","NRR (%)"]] = summary_disp[["GRR (%)","NRR (%)"]].round(2)
    st.dataframe(summary_disp, use_container_width=True)

with tab5:
    st.header("📊 Funnel & Marketing")
    st.divider()

    # Stage-to-stage conversion rates
    funnel = events_month.copy()
    funnel["visit_to_trial_rate"] = np.where(funnel["visits"] > 0, funnel["trials"] / funnel["visits"] * 100, np.nan)
    funnel["trial_to_paid_rate"] = np.where(funnel["trials"] > 0, funnel["conversions"] / funnel["trials"] * 100, np.nan)
    funnel["overall_conv_rate"] = np.where(funnel["visits"] > 0, funnel["conversions"] / funnel["visits"] * 100, np.nan)

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
    fplot["month_str"] = fplot["month"].astype(str)

    nice_names = {
        "visit_to_trial_rate": "Visits → Trials",
        "trial_to_paid_rate": "Trials → Paid",
        "overall_conv_rate": "Overall Conversion",
    }

    conv_long = fplot.melt(
        id_vars="month_str",
        value_vars=list(nice_names.keys()),
        var_name="Metric",
        value_name="Rate",
    )
    conv_long["Metric"] = conv_long["Metric"].map(nice_names)

    fig_conv_bar = px.bar(
        conv_long,
        x="month_str",
        y="Rate",
        color="Metric",
        barmode="group",
        title="Funnel Conversion Rates (Grouped Bar Chart)",
        labels={"Rate": "Rate (%)", "month_str": "Month", "Metric": "Metric"},
        category_orders={"Metric": list(nice_names.values())}
    )
    fig_conv_bar.update_layout(
        xaxis={"type": "category", "categoryorder": "category ascending"}
    )
    st.plotly_chart(fig_conv_bar, use_container_width=True)

    st.subheader("Spend per Channel (All Months)")
    spend_ch = events_month_channel[["month","channel","spend"]].copy()
    spend_ch["month_str"] = spend_ch["month"].astype(str)
    fig_spend = px.bar(spend_ch, x="month_str", y="spend", color="channel", barmode="group", title="Marketing Spend by Channel Over Time", labels={"spend":"Spend (€)","month_str":"Month","channel":"Channel"})
    fig_spend.update_layout(xaxis={"type":"category","categoryorder":"category ascending"})
    st.plotly_chart(fig_spend, use_container_width=True)

    # Wide summary table joining spend to funnel
    spend_wide = spend_ch.pivot_table(index="month", columns="channel", values="spend", fill_value=0)
    summary = (
        funnel[["month","visits","trials","conversions","visit_to_trial_rate","trial_to_paid_rate","overall_conv_rate"]]
        .merge(spend_wide.add_prefix("spend_"), left_on="month", right_index=True, how="left")
        .sort_values("month")
    )
    summary_disp = summary.copy()
    summary_disp["Month"] = summary_disp["month"].astype(str)
    summary_disp = summary_disp.drop(columns=["month"])
    for col in ["visit_to_trial_rate","trial_to_paid_rate","overall_conv_rate"]:
        summary_disp[col] = summary_disp[col].round(2)
    st.subheader("Monthly Funnel & Marketing Summary")
    st.dataframe(summary_disp, use_container_width=True)

with tab6:
    st.header("🌍 Segmentation")
    st.divider()

    st.subheader("MRR by Plan (monthly)")
    mrr_plan = subs.groupby(["month","plan"], as_index=False)["mrr"].sum().sort_values("month")
    mrr_plan["month_str"] = mrr_plan["month"].astype(str)
    fig_plan = px.bar(mrr_plan, x="month_str", y="mrr", color="plan", barmode="group", title="MRR by Plan Over Time", labels={"mrr":"MRR (€)","month_str":"Month","plan":"Plan"})
    fig_plan.update_layout(xaxis={"type":"category","categoryorder":"category ascending"})
    st.plotly_chart(fig_plan, use_container_width=True)

    if "country" in subs_seg.columns:
        st.subheader("MRR by Country (monthly)")
        mrr_country = subs_seg.groupby(["month","country"], as_index=False)["mrr"].sum().sort_values("month")
        mrr_country["month_str"] = mrr_country["month"].astype(str)
        fig_country = px.bar(mrr_country, x="month_str", y="mrr", color="country", barmode="group", title="MRR by Country Over Time", labels={"mrr":"MRR (€)","month_str":"Month","country":"Country"})
        fig_country.update_layout(xaxis={"type":"category","categoryorder":"category ascending"})
        st.plotly_chart(fig_country, use_container_width=True)
    else:
        st.info("`country` column not found in customers.csv — skipping MRR by Country.")

    if CHANNEL_COL is not None:
        st.subheader("MRR by Acquisition Channel (monthly)")
        mrr_channel = subs_seg.groupby(["month", CHANNEL_COL], as_index=False)["mrr"].sum().sort_values("month")
        mrr_channel = mrr_channel.rename(columns={CHANNEL_COL:"channel"})
        mrr_channel["month_str"] = mrr_channel["month"].astype(str)
        fig_channel = px.bar(mrr_channel, x="month_str", y="mrr", color="channel", barmode="group", title="MRR by Acquisition Channel Over Time", labels={"mrr":"MRR (€)","month_str":"Month","channel":"Channel"})
        fig_channel.update_layout(xaxis={"type":"category","categoryorder":"category ascending"})
        st.plotly_chart(fig_channel, use_container_width=True)
    else:
        st.info("No acquisition channel column (`channel` or `acquisition_channel`) — skipping MRR by Channel.")

    st.divider()
    st.subheader("Churn by Segment (monthly)")

    # Aggregate per customer-month then build lagged comparison for churn
    agg_dict = {"mrr":"sum","plan":"last"}
    if "country" in subs_seg.columns:
        agg_dict["country"] = "last"
    if CHANNEL_COL is not None:
        agg_dict[CHANNEL_COL] = "last"

    per_cust = subs_seg.groupby(["customer_id","month"], as_index=False).agg(agg_dict).sort_values(["customer_id","month"])

    # Prepare previous-period columns (names dynamic by availability)
    rename_prev = {"mrr":"mrr_prev","plan":"plan_prev"}
    if "country" in per_cust.columns:
        rename_prev["country"] = "country_prev"
    if CHANNEL_COL is not None:
        rename_prev[CHANNEL_COL] = f"{CHANNEL_COL}_prev"

    prev_seg = per_cust.rename(columns=rename_prev).copy()
    prev_seg["month"] = prev_seg["month"] + 1  # align t with t-1

    cur_prev_seg = per_cust.merge(prev_seg, on=["customer_id","month"], how="outer")
    min_m, max_m = subs_seg["month"].min(), subs_seg["month"].max()
    cur_prev_seg = cur_prev_seg[(cur_prev_seg["month"] >= min_m) & (cur_prev_seg["month"] <= max_m)]
    cur_prev_seg[["mrr","mrr_prev"]] = cur_prev_seg[["mrr","mrr_prev"]].fillna(0)

    # Segment options constrained by available columns
    seg_options = ["plan"]
    if "country" in customers.columns:
        seg_options.append("country")
    if CHANNEL_COL is not None:
        seg_options.append(CHANNEL_COL)

    seg_choice = st.selectbox(
        "Choose segment dimension",
        options=seg_options,
        index=0,
        format_func=lambda x: {"plan":"Plan","country":"Country", CHANNEL_COL:"Channel"}.get(x, x)
    )
    seg_prev_col = {"plan":"plan_prev","country":"country_prev"}.get(seg_choice, f"{CHANNEL_COL}_prev")

    # Only consider prior active to compute churn by segment
    cur_prev_seg = cur_prev_seg[cur_prev_seg["mrr_prev"] > 0]
    cur_prev_seg = cur_prev_seg.dropna(subset=[seg_prev_col])
    cur_prev_seg["churned_flag"] = (cur_prev_seg["mrr_prev"] > 0) & (cur_prev_seg["mrr"] == 0)

    churn_seg = (
        cur_prev_seg.groupby(["month", seg_prev_col], as_index=False)
                    .agg(active_prev=("mrr_prev", lambda s: (s > 0).sum()),
                         churned=("churned_flag", "sum"))
                    .rename(columns={seg_prev_col:"segment"})
                    .sort_values("month")
    )
    churn_seg["churn_rate_pct"] = np.where(churn_seg["active_prev"] > 0, churn_seg["churned"] / churn_seg["active_prev"] * 100, np.nan)

    churn_plot = churn_seg.copy()
    churn_plot["month_str"] = churn_plot["month"].astype(str)
    fig_churn_seg = px.line(churn_plot, x="month_str", y="churn_rate_pct", color="segment", markers=True, title=f"Churn Rate by {'Plan' if seg_choice=='plan' else ('Country' if seg_choice=='country' else 'Channel')}", labels={"churn_rate_pct":"Churn Rate (%)","month_str":"Month","segment":"Segment"})
    fig_churn_seg.update_layout(yaxis_title="Churn Rate (%)")
    st.plotly_chart(fig_churn_seg, use_container_width=True)

    st.dataframe(
        churn_seg.rename(columns={
            "month":"Month",
            "segment":"Segment",
            "active_prev":"Active (prev)",
            "churned":"Churned",
            "churn_rate_pct":"Churn Rate (%)"
        }).assign(**{"Churn Rate (%)": lambda d: d["Churn Rate (%)"].round(2)}),
        use_container_width=True
    )
