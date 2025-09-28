# 📊 SaaS KPI Dashboard with AI Assistant

An interactive **Streamlit** app for monitoring core SaaS metrics from your own CSVs. It turns raw **subscriptions**, **customers**, and **marketing events** into clean charts and tables for **Revenue & Growth, Customer Metrics, Value Metrics, Retention & Cohorts, Funnel & Marketing, and Segmentation**.

**🤖 NEW:** Built-in AI assistant powered by Google Gemini for natural language insights and recommendations.

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

### 🤖 AI Business Assistant
- **Natural language queries** about your business metrics
- **Data-driven insights** and actionable recommendations
- **Contextual analysis** based on your actual performance data
- **Example questions** to get started quickly

---

## 🧱 Repository structure

```
├── app/
│   └── app.py                  # Main Streamlit app with AI integration
├── data/
│   ├── customers.csv           # Customer signup & attributes
│   ├── subscriptions.csv       # Monthly MRR by customer/plan
│   └── events_marketing.csv    # Visits, trials, conversions, spend by channel
├── .streamlit/
│   └── secrets.toml            # API keys
├── requirements.txt            # Python dependencies
└── README.md
```

---

## 🚀 Quick start

### 1. Clone and setup environment

```bash
git clone https://github.com/yourusername/kpi-dashboard.git
cd kpi-dashboard
```

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

```bash
pip install -r requirements.txt
```

### 2. Configure AI assistant (optional but recommended)

**Get your free Google Gemini API key:**
1. Visit [Google AI Studio](https://aistudio.google.com/u/0/api-keys)
2. Sign in with your Google account
3. Create a new API key
4. Copy the key (starts with `AI...`)

**Add API key to your project:**
1. Edit `secrets.toml` file inside `.streamlit/` folder
2. Add your API key:

```toml
# .streamlit/secrets.toml
GEMINI_API_KEY = "your-api-key-here"
```

**⚠️ Important:** Add `.streamlit/secrets.toml` to your `.gitignore` to keep your API key private!

### 3. Add your data

Put your CSVs in `./data/` with the required column names (see [Data contracts](#-data-contracts) below).

### 4. Run the app

```bash
streamlit run app/app.py
```

The app opens at **http://localhost:8501**.

---

## 🤖 Using the AI Assistant

### Features
- **Sidebar chat interface** for real-time conversations
- **Contextual analysis** of your business data
- **Actionable recommendations** based on actual performance
- **Example questions** to guide your analysis

### Example queries
- `"What's our current MRR growth trend?"`
- `"How is our customer churn performing?"`
- `"What's our LTV/CAC ratio telling us?"`
- `"Which months had the best performance?"`
- `"How can we improve our retention?"`
- `"What's driving our revenue growth?"`
- `"Como aumentar os lucros?"` (Portuguese supported)

### AI Response format
- **Direct insights** with specific numbers from your data
- **Problem identification** with 🔴 icons
- **Solution recommendations** with 💡 icons
- **Data points** with 📊 icons
- **Prioritized action items** for immediate implementation

### Without AI key
The dashboard works fully without an API key - you just won't have the AI assistant. You'll see a setup guide in the sidebar.


---

## 📄 Data contracts

Your CSVs must include the columns below (extra columns are OK).

### `customers.csv`
```csv
customer_id,signup_date,country,channel
CUST001,2024-01-15,Portugal,Google Ads
CUST002,2024-01-20,Spain,Organic
```
- `customer_id` (string/int) - **required**
- `signup_date` (ISO date: YYYY-MM-DD) - **required**
- `country` (optional) - enables country segmentation
- `channel` **or** `acquisition_channel` (optional) - enables channel analysis

### `subscriptions.csv`
```csv
customer_id,plan,period_start,period_end,mrr
CUST001,Pro,2024-01-01,2024-01-31,49.99
CUST001,Pro,2024-02-01,2024-02-29,49.99
```
- `customer_id` - **required**
- `plan` (e.g., Basic/Pro/Enterprise) - **required**
- `period_start` (ISO date) - **required**
- `period_end` (ISO date) - **required**
- `mrr` (numeric, monthly recurring revenue) - **required**

### `events_marketing.csv`
```csv
date,channel,visits,trials,conversions,spend
2024-01-01,Google Ads,1000,100,10,500.00
2024-01-01,LinkedIn Ads,500,50,5,800.00
```
- `date` (ISO date) - **required**
- `channel` (e.g., Google Ads, LinkedIn Ads, Organic, Referral) - **required**
- `visits` (int) - **required**
- `trials` (int) - **required**
- `conversions` (int) - **required**
- `spend` (numeric) - **required**

---

## 🧠 How metrics are computed

- **Month keys**: All time series use `Period[M]` for robust joins
- **MRR Movements**: Compare customer MRR month-over-month to classify New, Expansion, Contraction, Churned
- **GRR / NRR**: From customers with positive prior MRR:
  - `GRR = (Base − Churn − Contraction) / Base × 100`
  - `NRR = (Base − Churn − Contraction + Expansion) / Base × 100`
- **CAC / LTV**: 
  - `CAC = Spend / Conversions` (per month and by channel)
  - `ARPA = MRR / Active Customers`
  - `LTV = (ARPA × GrossMargin) / ChurnRate`
  - `Payback = CAC / ARPA`
- **Funnel**: Conversion rates calculated as percentages by month
- **Segmentation churn**: Attributes churn to previous month's segment to avoid leakage

---

## 🔧 Configuration & customization

### Channel focus
CAC by channel highlights `["Google Ads", "LinkedIn Ads"]` by default. 
Modify `CHANNELS_KEEP` in `app.py` to change this:

```python
CHANNELS_KEEP = ["Google Ads", "Facebook Ads", "LinkedIn Ads"]
```

### AI model settings
The app uses `gemini-2.5-flash` by default. To change models, modify:

```python
model = genai.GenerativeModel('gemini-pro')  # Alternative model
```

### Gross margin
Default gross margin slider ranges 40-99% (default: 80%). Adjust in the Value Metrics tab.

---

## 🔒 Security & privacy

- **API keys**: Never commit `.streamlit/secrets.toml` to version control
- **Data privacy**: Your CSV data stays local - only aggregated metrics are sent to AI
- **API usage**: Gemini has generous free tiers, but monitor your usage
- **Local processing**: All calculations happen locally; AI only receives summaries

---

## 🧪 Troubleshooting

### Common issues

**"Add your Gemini API key" warning**
- Edit `.streamlit/secrets.toml` with your API key
- Ensure file is in project root
- Check API key format starts with `AIza`

**Charts show future/missing months**
- Ensure date columns use ISO format (YYYY-MM-DD)
- Check for data quality issues in CSV files

**"Country/Channel segmentation missing"**
- Verify column names in `customers.csv`
- Use `channel` or `acquisition_channel` for channel data

**AI responses seem generic**
- Ensure your data covers multiple months for trend analysis
- Try more specific questions about your metrics

---

## 🙌 Acknowledgements

- **Streamlit** - Amazing web app framework
- **Google Gemini** - Powerful AI capabilities
- **Plotly** - Interactive visualizations
- **Pandas & NumPy** - Data processing foundation
