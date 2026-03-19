"""
Utility Helpers
Logging, CSS injection, and misc utility functions
"""

import logging
import streamlit as st
from utils.config import AppConfig


def setup_logging() -> logging.Logger:
    """Configure application logging."""
    logging.basicConfig(
        level=getattr(logging, AppConfig.LOG_LEVEL, logging.INFO),
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
        ]
    )
    return logging.getLogger("ai_dashboard")


def format_number(value) -> str:
    """Format numbers for display."""
    if value is None:
        return "N/A"
    try:
        v = float(value)
        if abs(v) >= 1_000_000_000:
            return f"{v / 1_000_000_000:.2f}B"
        elif abs(v) >= 1_000_000:
            return f"{v / 1_000_000:.2f}M"
        elif abs(v) >= 1_000:
            return f"{v / 1_000:.1f}K"
        elif v == int(v):
            return f"{int(v):,}"
        else:
            return f"{v:,.2f}"
    except (TypeError, ValueError):
        return str(value)


def is_id_column(col_name: str) -> bool:
    """Detect if a column is likely an ID/key column."""
    col_lower = col_name.lower().strip()
    for keyword in AppConfig.ID_COLUMN_KEYWORDS:
        if col_lower == keyword or col_lower.endswith(f"_{keyword}") or col_lower.startswith(f"{keyword}_"):
            return True
    return False


def truncate_text(text: str, max_chars: int = 100) -> str:
    """Truncate long text with ellipsis."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars - 3] + "..."


def safe_divide(numerator, denominator, default=0):
    """Safe division avoiding ZeroDivisionError."""
    try:
        if denominator == 0:
            return default
        return numerator / denominator
    except Exception:
        return default


def apply_custom_css():
    """Inject custom CSS for the dark modern theme."""
    st.markdown("""
    <style>
    /* ── Google Fonts ── */
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    /* ── Root Variables ── */
    :root {
        --bg-primary: #0a0a0f;
        --bg-secondary: #111118;
        --bg-card: #16161f;
        --bg-card-hover: #1e1e2a;
        --border: #2a2a3a;
        --border-accent: #6366f1;
        --text-primary: #f0f0ff;
        --text-secondary: #8888aa;
        --text-muted: #555570;
        --accent-purple: #6366f1;
        --accent-blue: #06b6d4;
        --accent-green: #10b981;
        --accent-orange: #f59e0b;
        --accent-red: #ef4444;
        --accent-pink: #ec4899;
        --gradient-hero: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #06b6d4 100%);
        --gradient-card: linear-gradient(135deg, #16161f 0%, #1a1a27 100%);
        --shadow-card: 0 4px 24px rgba(99, 102, 241, 0.08);
        --shadow-hover: 0 8px 40px rgba(99, 102, 241, 0.2);
        --radius: 12px;
        --radius-lg: 20px;
    }

    /* ── Global Reset ── */
    .stApp {
        background: var(--bg-primary) !important;
        font-family: 'Space Grotesk', sans-serif !important;
        color: var(--text-primary) !important;
    }

    /* ── Hide Streamlit Defaults ── */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    .stDeployButton { display: none; }
    header[data-testid="stHeader"] { background: transparent !important; }

    /* ── Main Container ── */
    .main .block-container {
        padding: 1rem 2rem 3rem !important;
        max-width: 1400px !important;
    }

    /* ── Header ── */
    .app-header {
        background: var(--gradient-card);
        border: 1px solid var(--border);
        border-radius: var(--radius-lg);
        padding: 2rem 2.5rem;
        margin-bottom: 1.5rem;
        position: relative;
        overflow: hidden;
    }
    .app-header::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; bottom: 0;
        background: linear-gradient(135deg, rgba(99,102,241,0.05) 0%, rgba(6,182,212,0.05) 100%);
        pointer-events: none;
    }
    .header-content { position: relative; z-index: 1; }
    .header-title {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin-bottom: 0.5rem;
    }
    .brain-icon { font-size: 2rem; }
    .header-title h1 {
        font-size: 2rem !important;
        font-weight: 700 !important;
        background: var(--gradient-hero);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0 !important;
        padding: 0 !important;
    }
    .header-subtitle {
        color: var(--text-secondary);
        font-size: 1rem;
        margin: 0;
        font-weight: 400;
    }

    /* ── Upload Section ── */
    .upload-section {
        margin: 1rem 0;
    }
    .upload-hint {
        display: flex;
        gap: 1rem;
        justify-content: center;
        color: var(--text-muted);
        font-size: 0.8rem;
        margin-top: 0.5rem;
    }

    /* ── Feature Cards ── */
    .feature-card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 1.5rem;
        text-align: center;
        transition: all 0.3s ease;
        height: 100%;
    }
    .feature-card:hover {
        border-color: var(--border-accent);
        box-shadow: var(--shadow-hover);
        transform: translateY(-2px);
    }
    .feature-icon {
        font-size: 2rem;
        margin-bottom: 0.75rem;
    }
    .feature-card h3 {
        font-size: 1rem !important;
        font-weight: 600 !important;
        color: var(--text-primary) !important;
        margin: 0.5rem 0 !important;
    }
    .feature-card p {
        font-size: 0.85rem;
        color: var(--text-secondary);
        line-height: 1.5;
        margin: 0;
    }

    /* ── KPI Cards ── */
    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }
    .kpi-card {
        background: var(--gradient-card);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 1.25rem;
        position: relative;
        overflow: hidden;
        transition: all 0.3s ease;
    }
    .kpi-card:hover {
        border-color: var(--border-accent);
        box-shadow: var(--shadow-hover);
        transform: translateY(-1px);
    }
    .kpi-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0;
        width: 4px; height: 100%;
        border-radius: 2px 0 0 2px;
    }
    .kpi-card.purple::before { background: var(--accent-purple); }
    .kpi-card.blue::before   { background: var(--accent-blue); }
    .kpi-card.green::before  { background: var(--accent-green); }
    .kpi-card.orange::before { background: var(--accent-orange); }
    .kpi-card.red::before    { background: var(--accent-red); }
    .kpi-card.pink::before   { background: var(--accent-pink); }

    .kpi-label {
        font-size: 0.72rem;
        font-weight: 500;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.4rem;
    }
    .kpi-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: var(--text-primary);
        font-family: 'JetBrains Mono', monospace;
        line-height: 1;
    }
    .kpi-sub {
        font-size: 0.72rem;
        color: var(--text-muted);
        margin-top: 0.3rem;
    }
    .kpi-icon {
        position: absolute;
        top: 1rem; right: 1rem;
        font-size: 1.5rem;
        opacity: 0.4;
    }

    /* ── Insights ── */
    .insight-card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-left: 3px solid var(--accent-purple);
        border-radius: var(--radius);
        padding: 1rem 1.25rem;
        margin: 0.5rem 0;
        font-size: 0.9rem;
        color: var(--text-secondary);
        line-height: 1.6;
        transition: all 0.2s ease;
    }
    .insight-card:hover {
        border-left-color: var(--accent-blue);
        background: var(--bg-card-hover);
        color: var(--text-primary);
    }

    /* ── Anomaly Cards ── */
    .anomaly-card {
        background: rgba(239, 68, 68, 0.05);
        border: 1px solid rgba(239, 68, 68, 0.2);
        border-radius: var(--radius);
        padding: 0.75rem 1rem;
        margin: 0.25rem 0;
        font-size: 0.85rem;
        color: #fca5a5;
    }

    /* ── Chat ── */
    .stChatMessage {
        background: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius) !important;
    }
    .stChatInputContainer {
        background: var(--bg-card) !important;
        border: 1px solid var(--border-accent) !important;
        border-radius: var(--radius) !important;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: var(--bg-secondary) !important;
        border-right: 1px solid var(--border) !important;
    }
    [data-testid="stSidebar"] .sidebar-section {
        padding: 1rem;
    }

    /* ── Streamlit Overrides ── */
    .stSelectbox > div > div {
        background: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
        color: var(--text-primary) !important;
        border-radius: 8px !important;
    }
    .stTextInput > div > div > input {
        background: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
        color: var(--text-primary) !important;
        border-radius: 8px !important;
    }
    .stButton > button {
        background: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
        color: var(--text-primary) !important;
        border-radius: 8px !important;
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 500 !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button:hover {
        border-color: var(--border-accent) !important;
        box-shadow: var(--shadow-card) !important;
        transform: translateY(-1px) !important;
    }
    .stButton > button[kind="primary"] {
        background: var(--accent-purple) !important;
        border-color: var(--accent-purple) !important;
    }
    .stDataFrame {
        background: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius) !important;
    }
    .stMetric {
        background: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
        padding: 0.75rem 1rem !important;
    }
    [data-testid="metric-container"] {
        background: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
        padding: 0.75rem !important;
    }
    .stExpander {
        background: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius) !important;
    }
    .stAlert {
        border-radius: var(--radius) !important;
    }
    h1, h2, h3 {
        font-family: 'Space Grotesk', sans-serif !important;
        color: var(--text-primary) !important;
        font-weight: 600 !important;
    }
    h3 { font-size: 1.1rem !important; }

    /* ── Plotly Charts ── */
    .stPlotlyChart {
        background: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius) !important;
        overflow: hidden;
    }

    /* ── File Uploader ── */
    [data-testid="stFileUploader"] {
        background: var(--bg-card) !important;
        border: 2px dashed var(--border) !important;
        border-radius: var(--radius-lg) !important;
        padding: 2rem !important;
        transition: all 0.3s ease !important;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: var(--border-accent) !important;
        box-shadow: var(--shadow-card) !important;
    }

    /* ── Progress Bar ── */
    .stProgress > div > div {
        background: var(--gradient-hero) !important;
        border-radius: 4px !important;
    }

    /* ── Spinner ── */
    .stSpinner > div {
        border-color: var(--accent-purple) !important;
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        background: var(--bg-card) !important;
        border-radius: 8px !important;
        padding: 4px !important;
    }
    .stTabs [data-baseweb="tab"] {
        color: var(--text-secondary) !important;
        border-radius: 6px !important;
    }
    .stTabs [aria-selected="true"] {
        background: var(--accent-purple) !important;
        color: white !important;
    }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: var(--bg-secondary); }
    ::-webkit-scrollbar-thumb {
        background: var(--border);
        border-radius: 3px;
    }
    ::-webkit-scrollbar-thumb:hover { background: var(--border-accent); }
    </style>
    """, unsafe_allow_html=True)
