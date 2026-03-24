# AI Data Analyst Dashboard

An AI-powered analytics dashboard that converts CSV files into interactive dashboards with insights, KPIs, charts, and a secure chat interface.

## Quick Start

### 1. Install

```bash
git clone <your-repo-url>
cd ai_dashboard
pip install -r requirements.txt
```

### 2. Configure API keys (optional)

You can run without keys (rule-based fallback will be used), but AI features are better with keys.

Option A: `.env`
- `GROQ_API_KEY=...`
- `CLAUDE_API_KEY=...` (optional)

Option B: Streamlit Secrets
- `GROQ_API_KEY`
- `CLAUDE_API_KEY`

### 3. Run locally

```bash
streamlit run app.py
```

---

## Features

- CSV upload with validation and preprocessing
- Data profiling with rich column metadata
- AI analysis using Groq and optional Anthropic fallback
- Consent-gated sample rows for privacy
- KPI cards with smart aggregations
- Plotly charts with correlation heatmap
- Anomaly detection using Z-score + IQR
- Dynamic sidebar filters
- Secure chat with AST validation and a hard timeout
- Export insights, filtered data, and summary stats

---

## AI Model Fallback Order

The app attempts models in this order (configurable in `utils/config.py`):

1. `openai/gpt-oss-120b`
2. `openai/gpt-oss-20b`
3. `qwen/qwen3-32b`
4. `openai/gpt-oss-safeguard-20b`
5. `claude-3-5-sonnet-20241022` (Anthropic, if key provided)

---

## Project Structure

```
ai_dashboard/
+-- app.py                    # Main Streamlit entry point
+-- modules/
¦   +-- data_loader.py        # CSV loading and preprocessing
¦   +-- profiler.py           # Data type detection and stats
¦   +-- ai_engine.py          # Groq/Anthropic analysis with fallback
¦   +-- analysis_verifier.py  # Deterministic repair of AI output
¦   +-- chart_generator.py    # Plotly chart generation
¦   +-- kpi_generator.py      # KPI computation
¦   +-- insights_generator.py # Insight generation
¦   +-- chat_engine.py        # Safe pandas code execution
¦   +-- anomaly_detector.py   # Z-score + IQR detection
+-- components/
¦   +-- dashboard.py          # Insights and data explorer
¦   +-- charts.py             # Chart renderer
¦   +-- kpis.py               # KPI card renderer
+-- utils/
¦   +-- config.py             # Central configuration
¦   +-- helpers.py            # CSS, logging, utilities
+-- sample_data/
¦   +-- ecommerce.csv
¦   +-- hr_data.csv
¦   +-- stocks.csv
+-- .streamlit/config.toml    # Streamlit theme config
+-- requirements.txt
```

---

## Deploy to Streamlit Cloud

1. Push to GitHub
2. Go to Streamlit Cloud and connect your repo
3. Set `app.py` as the entry point
4. Add `GROQ_API_KEY` (and optionally `CLAUDE_API_KEY`) in Secrets

---

## Chat Examples

Try asking:
- "What is the average revenue by region?"
- "Show me the top 5 products by sales"
- "How many missing values are there?"
- "Which category has the highest return rate?"
- "Show rows where rating is less than 3"

---

## Limitations

- Max file size: 20MB
- Max rows: 100,000 (larger files are sampled)
- Groq/Anthropic APIs can rate-limit or timeout
- Chat code execution is sandboxed and time-limited
