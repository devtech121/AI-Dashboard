# 🧠 AI Data Analyst Dashboard

An AI-powered analytics dashboard that automatically converts CSV files into interactive dashboards with insights, KPIs, charts, and a chat interface.

## 🚀 Quick Start

### 1. Clone and setup

```bash
git clone <your-repo-url>
cd ai_dashboard
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and add your HuggingFace API token
# Get free token at: https://huggingface.co/settings/tokens
```

### 3. Run locally

```bash
streamlit run app.py
```

---

## 🔑 HuggingFace API Token

The app uses the **HuggingFace Inference API** (free tier) for AI analysis.

1. Sign up at [huggingface.co](https://huggingface.co)
2. Go to Settings → Access Tokens
3. Create a new token (read access is sufficient)
4. Add it to `.env` or paste it in the app sidebar

**Without a token:** The app still works using rule-based fallback for KPIs, charts, and insights.

---

## 🧩 Features

| Feature | Description |
|---|---|
| 📤 CSV Upload | Drag & drop, auto-validates format and size |
| 🔬 Data Profiling | Auto-detects column types, stats, missing values |
| 🤖 AI Analysis | HuggingFace LLMs with 4-model fallback chain |
| 📊 KPI Cards | Auto-generated metrics with smart aggregations |
| 📈 Charts | 5–8 Plotly charts auto-selected by data type |
| 💬 Chat | Ask questions → LLM generates pandas code → executes safely |
| 🚨 Anomalies | Z-score + IQR statistical outlier detection |
| 🔍 Filters | Dynamic sidebar filters for any dataset |
| 📥 Export | Download insights report, filtered data, stats |
| 🎨 Dark Mode | Modern dark UI with purple/cyan accent theme |

---

## 🤖 AI Model Fallback Chain

The app tries models in this order, automatically falling back on failure:

1. `mistralai/Mistral-7B-Instruct-v0.2` ← default
2. `mistralai/Mixtral-8x7B-Instruct-v0.1`
3. `HuggingFaceH4/zephyr-7b-beta`
4. `google/flan-t5-large`

---

## 📁 Project Structure

```
ai_dashboard/
├── app.py                    # Main Streamlit entry point
├── modules/
│   ├── data_loader.py        # CSV loading & validation
│   ├── profiler.py           # Data type detection & stats
│   ├── ai_engine.py          # HuggingFace API + fallback
│   ├── chart_generator.py    # Plotly chart generation
│   ├── kpi_generator.py      # KPI computation
│   ├── insights_generator.py # Insight generation
│   ├── chat_engine.py        # Code-execution Q&A
│   └── anomaly_detector.py   # Z-score + IQR detection
├── components/
│   ├── dashboard.py          # Insights & data explorer
│   ├── charts.py             # Chart renderer
│   └── kpis.py               # KPI card renderer
├── utils/
│   ├── config.py             # Central configuration
│   └── helpers.py            # CSS, logging, utilities
├── sample_data/
│   ├── ecommerce.csv         # 500-row e-commerce dataset
│   ├── hr_data.csv           # 300-row HR analytics dataset
│   └── stocks.csv            # 2,600-row stock prices dataset
├── .streamlit/
│   └── config.toml           # Streamlit dark theme config
├── requirements.txt
├── .env.example
└── README.md
```

---

## ☁️ Deploy to Streamlit Cloud

1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo and set `app.py` as the entry point
4. Add `HF_API_TOKEN` in the Secrets section

---

## 💬 Chat Examples

Try asking:
- "What is the average revenue by region?"
- "Show me the top 5 products by sales"
- "How many missing values are there?"
- "Which category has the highest return rate?"
- "Show rows where rating is less than 3"

---

## ⚠️ Limitations

- Max file size: 20MB
- Max rows: 100,000 (larger files are sampled)
- HuggingFace free API has rate limits — be patient between requests
- Chat code execution is sandboxed and safe but limited to pandas/numpy ops
