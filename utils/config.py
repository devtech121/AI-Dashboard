"""
Application Configuration
Centralized config for AI Data Analyst Dashboard
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

BASE_DIR = Path(__file__).parent.parent


class AppConfig:
    """Central application configuration."""

    # ── File Upload ────────────────────────────────────────────
    MAX_FILE_SIZE_MB: int = 20
    MAX_FILE_SIZE_BYTES: int = MAX_FILE_SIZE_MB * 1024 * 1024
    MAX_ROWS: int = 100_000
    ALLOWED_EXTENSIONS: list = ["csv"]

    # ── Groq API ───────────────────────────────────────────────
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_TIMEOUT: int = 30  # seconds per request

    # ── Groq Model Options ─────────────────────────────────────
    MODELS: dict = {
        "openai/gpt-oss-120b": {
            "name": "GPT-OSS 120B (Best)",
            "max_tokens": 1024,
            "temperature": 0.3,
        },
        "openai/gpt-oss-20b": {
            "name": "GPT-OSS 20B (Fast)",
            "max_tokens": 1024,
            "temperature": 0.3,
        },
        "qwen/qwen3-32b": {
            "name": "Qwen3 32B (Balanced)",
            "max_tokens": 1024,
            "temperature": 0.3,
        },
        "openai/gpt-oss-safeguard-20b": {
            "name": "GPT-OSS Safeguard 20B",
            "max_tokens": 1024,
            "temperature": 0.3,
        },
    }

    DEFAULT_MODEL: str = "openai/gpt-oss-120b"
    FALLBACK_ORDER: list = list(MODELS.keys())

    # ── Data Profiling ─────────────────────────────────────────
    SAMPLE_SIZE_FOR_AI: int = 5          # rows to show AI as sample
    CATEGORICAL_THRESHOLD: int = 50      # max unique values to treat as categorical
    HIGH_CARDINALITY_THRESHOLD: int = 0.9  # if unique_ratio > this → skip AI chart

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