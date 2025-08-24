# Junitec KPI Copilot — Dataset & Guia de Arranque

Este repositório contém **dados sintéticos (18 meses)** e um **guia base** para construíres um MVP de Dashboard de KPIs para Startups/PMEs com um **Copiloto de AI** (Q&A sobre os dados).

## 📦 Conteúdo
- `customers.csv` — clientes, data de registo, plano inicial, país, canal de aquisição.
- `subscriptions.csv` — *snapshot mensal por cliente* com o plano ativo, MRR e período.
- `events_marketing.csv` — agregados mensais por canal (visitas, trials, conversões, spend).
- `prompt_copiloto.md` — prompt afinado para ligares o teu módulo de AI.

Período coberto: **Mar/2024 → Ago/2025** (18 meses).  
Domínio fictício: SaaS B2B com planos **Basic/Pro/Enterprise**.

## 🧠 KPIs recomendados
- **MRR** (Monthly Recurring Revenue) total e por *movimentos* (New/Expansion/Contraction/Churned).
- **Logo churn** (clientes perdidos), **Revenue churn** (MRR perdido).
- **Cohorts de retenção** (por mês de signup).
- **CAC por canal** = `spend / conversions`.
- **LTV (simples)** = `ARPA × margem_bruta / churn_rate`.
- **LTV/CAC** como razão de eficiência.

> Dica: Mantém as funções de métricas **puras** e testáveis (ex.: `metrics.py`), e a UI no `app.py` (Streamlit).

## 🚀 Quickstart (CSV + Streamlit)
1. Cria ambiente:
   ```bash
   python -m venv .venv && source .venv/bin/activate  # (Linux/macOS)
   # Windows: .venv\Scripts\activate
   pip install streamlit pandas plotly pydantic python-dateutil
   ```
2. Estrutura de pastas sugerida:
   ```
   app.py
   /services/metrics.py
   /services/ai.py
   /data/customers.csv
   /data/subscriptions.csv
   /data/events_marketing.csv
   ```
3. Carrega dados no `app.py` (exemplo mínimo):
   ```python
   import streamlit as st, pandas as pd
   customers = pd.read_csv("data/customers.csv", parse_dates=["signup_date"])
   subs = pd.read_csv("data/subscriptions.csv", parse_dates=["period_start","period_end"])
   events = pd.read_csv("data/events_marketing.csv", parse_dates=["date"])
   st.title("KPI Copilot — Junitec")
   st.write("MRR (amostra):", subs.groupby(subs["period_start"].dt.to_period("M"))["mrr"].sum().tail())
   ```
4. Correr:
   ```bash
   streamlit run app.py
   ```

## 🧩 Esquemas dos ficheiros
### `customers.csv`
| coluna | tipo | exemplo |
|---|---|---|
| customer_id | string | C0001 |
| signup_date | date(YYYY-MM-DD) | 2024-03-14 |
| plan_at_signup | enum(Basic, Pro, Enterprise) | Pro |
| country | enum(PT, ES, FR, DE, UK) | PT |
| acquisition_channel | enum(Organic, Google Ads, LinkedIn Ads, Referral) | Google Ads |

### `subscriptions.csv`  *(1 linha por cliente por mês ativo)*
| coluna | tipo | exemplo |
|---|---|---|
| subscription_id | string | C0001-00 |
| customer_id | string | C0001 |
| period_start | date | 2024-03-01 |
| period_end | date | 2024-03-31 |
| plan | enum(Basic, Pro, Enterprise) | Pro |
| mrr | float | 55.0 |
| is_active | int(0/1) | 1 |

### `events_marketing.csv` *(agregado mensal por canal)*
| coluna | tipo | exemplo |
|---|---|---|
| date | date | 2024-03-01 |
| channel | enum(Organic, Google Ads, LinkedIn Ads, Referral) | LinkedIn Ads |
| visits | int | 920 |
| trials | int | 260 |
| conversions | int | 12 |
| spend | float | 1840.50 |

## 🧪 Testes
- Garante fórmulas estáveis com `pytest` (ex.: `test_mrr_movements.py`, `test_churn.py`).
- Usa dados pequenos de amostra para validar antes de carregar tudo.

## 🧱 Roadmap (sugestão)
- Conectores reais: Stripe, GA4, HubSpot.
- Alertas proativos (email/Slack) em caso de anomalias.
- Q&A com grounding nos números do mês e *playbooks* de ação.
- Multi-tenant para uso em consultoria (templates por setor).

---

© 2025 — Dataset sintético gerado para fins de demonstração.
