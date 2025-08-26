# 📊 SaaS KPI Dashboard

An interactive **Streamlit** app for monitoring core SaaS metrics from your own CSVs. It turns raw **subscriptions**, **customers**, and **marketing events** into clean charts and tables for **Revenue & Growth, Customer Metrics, Value Metrics, Retention & Cohorts, Funnel & Marketing, and Segmentation**.

---

## ✨ What you get

### 🏦 Revenue & Growth
- **MRR** and **ARR** time series
- **MRR Movements**: New, Expansion, Contraction, Churned, and Net New MRR
- **MoM Growth Rate**
- Monthly summary table

### 👥 Customer Metrics
- **Active / New / Churned customers**
- **Churn Rate (MoM)**
- **ARPA** (Average Revenue per Account)
- **Estimated Customer Lifetime** (in months)

### 💰 Value Metrics
- **CAC** (overall + by channel)
- **LTV**, **LTV/CAC**, **Payback Period**
- Metrics computed from your **gross margin slider**

### 📈 Retention & Cohorts
- **GRR (Gross Revenue Retention)** and **NRR (Net Revenue Retention)**
- **Cohort Retention Heatmap** (logos-based)
- **Monthly Retention Summary** table

### 📊 Funnel & Marketing
- **Conversion rates**: Visits → Trials, Trials → Paid, Overall
- **Spend by Channel** over time
- Monthly **Funnel & Marketing Summary** table

### 🌍 Segmentation
- **MRR by Plan, Country, and Acquisition Channel**
- **Churn by Segment** (Plan/Country/Channel) using previous-month attribution

---

## 🧱 Repository structure

```
├── app/
│   └── app.py                  # Main Streamlit app
├── data/
│   ├── customers.csv           # Customer signup & attributes
│   ├── subscriptions.csv       # Monthly MRR by customer/plan
│   └── events_marketing.csv    # Visits, trials, conversions, spend by channel
└── README.md
```

---

## 📄 Data contracts

Your CSVs must include the columns below (extra columns are OK).

### `customers.csv`
- `customer_id` (string/int)
- `signup_date` (ISO date)
- Optional: `country`
- Optional: `channel` **or** `acquisition_channel`

### `subscriptions.csv`
- `customer_id`
- `plan` (e.g., Basic / Pro / Enterprise)
- `period_start`
- `period_end`
- `mrr` (numeric, monthly recurring revenue for that row)

### `events_marketing.csv`
- `date` (ISO date)
- `channel` (e.g., Google Ads, LinkedIn Ads, Organic, Referral)
- `visits` (int)
- `trials` (int)
- `conversions` (int)
- `spend` (numeric)

---

## ▶️ Quick start

1. Clone and enter the repo

```bash
git clone https://github.com/yourusername/kpi-dashboard.git
cd kpi-dashboard
```

2. Create a virtual environment (recommended)

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

3. Install deps

```bash
pip install -r requirements.txt
```

4. Put your CSVs in `./data/` with the column names above.

5. Run the app

```bash
streamlit run app/app.py
```

The app opens at **http://localhost:8501**.

---

## 📦 Requirements

`requirements.txt` (minimum):

```
streamlit
pandas
numpy
plotly
```

---

## 🧠 How metrics are computed (high level)

- **Month keys**: all time series are keyed by `Period[M]` for robust joins.
- **MRR Movements**:
  - Compare each customer’s `mrr` this month vs last (`mrr_prev`) to classify **New**, **Expansion**, **Contraction**, **Churned**.
- **GRR / NRR**:
  - From customers with **positive prior MRR** (base), apply churn/contraction/expansion and compute:
    - `GRR = (Base − Churn − Contraction) / Base`
    - `NRR = (Base − Churn − Contraction + Expansion) / Base`
- **CAC / LTV**:
  - `CAC = Spend / Conversions` per month (and by channel when applicable)
  - `ARPA = MRR / Active Customers`
  - `Churn Rate` uses prior-month active base
  - `LTV = (ARPA * GrossMargin) / ChurnRate`
  - `Payback = CAC / ARPA`
- **Funnel**:
  - `Visits → Trials`, `Trials → Paid`, `Overall` expressed in **%** by month.
- **Segmentation churn**:
  - Attributes churn to the **previous month’s segment** (plan/country/channel) to avoid leakage.

---

## 🧩 Configuration notes

- Acquisition channel column is auto-detected:
  - `channel` → else `acquisition_channel` → else channel-based views are skipped with a friendly message.
- CAC by channel currently highlights `["Google Ads", "LinkedIn Ads"]`. Adjust `CHANNELS_KEEP` in `app.py`.
- All charts are **Plotly** for hover and legend control; legends and tooltips are prettified.

---

## 🔌 Optional AI assistant (roadmap)

You can add a Q&A assistant to answer natural language questions like:
> “What was NRR last quarter?”  
> “Show MRR growth vs CAC for the last 6 months.”

Implementation sketch:
- Add a new **“🤖 Ask the Data”** tab.
- Summarize current filtered DataFrames to structured text.
- Send the summaries + user question to your LLM (e.g., OpenAI) with guardrails.
- Render results and optionally generate ad-hoc Plotly charts.

(Requires adding `openai` and `OPENAI_API_KEY`.)

---

## 🧪 Tips & troubleshooting

- If a chart shows a future/nonexistent month, ensure data joins are filtered to **observed months only** (the app already guards with `VALID_MONTHS`).
- If “MRR by Country/Channel” is missing, check that the expected columns exist in `customers.csv`.
- Ensure dates parse correctly; files must use ISO‐like formats (YYYY-MM-DD).

---

## 🙌 Acknowledgements

Thanks to the open-source community behind **Streamlit, Pandas, NumPy, and Plotly**.
