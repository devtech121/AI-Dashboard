"""
Application Configuration
Centralized config for AI Data Analyst Dashboard
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env for local development (harmless no-op on Streamlit Cloud)
load_dotenv()

BASE_DIR = Path(__file__).parent.parent
def get_groq_key() -> str:
    """
    Resolve the Groq API key at RUNTIME (not import time).
 
    Priority order:
      1. AppConfig.GROQ_API_KEY  — set by sidebar input at runtime
      2. st.secrets["GROQ_API_KEY"] — Streamlit Cloud secrets
      3. os.environ / .env file  — local development
    """
    # 1. Sidebar may have set this directly on the class
    runtime_key = getattr(AppConfig, "_runtime_key", "")
    if runtime_key:
        return runtime_key
 
    try:
        import streamlit as st
        key = st.secrets["GROQ_API_KEY"]
        if key:
            return key
    except Exception:
        pass
 
    # 3. Environment variable / .env file
    return os.getenv("GROQ_API_KEY", "")


class AppConfig:
    """Central application configuration."""

    # ── File Upload ────────────────────────────────────────────
    MAX_FILE_SIZE_MB: int = 20
    MAX_FILE_SIZE_BYTES: int = MAX_FILE_SIZE_MB * 1024 * 1024
    MAX_ROWS: int = 100_000
    ALLOWED_EXTENSIONS: list = ["csv"]

    # ── Groq API ───────────────────────────────────────────────
    # DO NOT set this to os.getenv() here — it would be read at
    # import time before st.secrets is ready on Streamlit Cloud.
    # Use get_groq_key() everywhere instead.
    # The sidebar writes to _runtime_key to override at runtime.
    _runtime_key: str = ""
    GROQ_TIMEOUT: int = 30

    @classmethod
    def set_groq_key(cls, key: str):
        """Called by sidebar when user types in a key."""
        cls._runtime_key = key.strip()

    @classmethod
    def get_groq_key(cls) -> str:
        """Alias so AIEngine can call AppConfig.get_groq_key()."""
        return get_groq_key()

    # ── Groq Model Options ─────────────────────────────────────
    MODELS: dict = {
        "openai/gpt-oss-120b": {
            "name": "GPT-OSS 120B ⭐ Best",
            "max_tokens": 4096,
            "temperature": 0.2,
        },
        "openai/gpt-oss-20b": {
            "name": "GPT-OSS 20B ⚡ Fast",
            "max_tokens": 4096,
            "temperature": 0.2,
        },
        "qwen/qwen3-32b": {
            "name": "Qwen3 32B ⚖️ Balanced",
            "max_tokens": 4096,
            "temperature": 0.2,
        },
        "openai/gpt-oss-safeguard-20b": {
            "name": "GPT-OSS Safeguard 20B 🛡️",
            "max_tokens": 4096,
            "temperature": 0.2,
        },
    }

    DEFAULT_MODEL: str = "openai/gpt-oss-120b"
    FALLBACK_ORDER: list = list(MODELS.keys())

    # ── Data Profiling ─────────────────────────────────────────
    SAMPLE_SIZE_FOR_AI: int = 5
    CATEGORICAL_THRESHOLD: int = 50
    HIGH_CARDINALITY_THRESHOLD: float = 0.9

    # ── Chart Generation ───────────────────────────────────────
    MIN_CHARTS: int = 5
    MAX_CHARTS: int = 8
    CHART_COLOR_SEQUENCE: list = [
        "#6366f1", "#8b5cf6", "#06b6d4", "#10b981",
        "#f59e0b", "#ef4444", "#ec4899", "#14b8a6"
    ]
    CHART_TEMPLATE: str = "plotly_dark"
    CHART_FONT_FAMILY: str = "Inter, sans-serif"

    # ── Anomaly Detection ──────────────────────────────────────
    Z_SCORE_THRESHOLD: float = 3.0
    IQR_MULTIPLIER: float = 1.5

    # ── KPI Generation ─────────────────────────────────────────
    MAX_KPIS: int = 8
    ID_COLUMN_KEYWORDS: list = ["id", "uuid", "key", "index", "code", "ref", "no", "num"]

    # ── Chat Engine ────────────────────────────────────────────
    CHAT_MAX_TOKENS: int = 512
    CHAT_TEMPERATURE: float = 0.1
    SAFE_PANDAS_OPS: list = [
        "df", "pd", "np", "len", "sum", "min", "max",
        "mean", "groupby", "sort_values", "value_counts",
        "describe", "head", "tail", "shape", "columns",
        "dtypes", "isnull", "notnull", "fillna", "dropna",
        "merge", "concat", "pivot_table", "crosstab",
        "str", "dt", "apply", "lambda", "round", "abs",
        "count", "unique", "nunique", "corr", "std",
        "var", "median", "mode", "quantile", "cumsum",
        "pct_change", "rolling", "resample", "reset_index",
        "set_index", "rename", "drop", "filter", "query",
        "astype", "copy", "sample", "nlargest", "nsmallest",
        "between", "isin", "where", "mask", "clip",
    ]

    # ── Logging ────────────────────────────────────────────────
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE: str = str(BASE_DIR / "app.log")

    # ── Paths ──────────────────────────────────────────────────
    SAMPLE_DATA_DIR: Path = BASE_DIR / "sample_data"
    # ── Paths ──────────────────────────────────────────────────
    SAMPLE_DATA_DIR: Path = BASE_DIR / "sample_data"
