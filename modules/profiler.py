"""
Data Profiler Module
Generates comprehensive metadata and statistics for a DataFrame
"""

import logging
import numpy as np
import pandas as pd
import streamlit as st
from utils.config import AppConfig
from utils.helpers import is_id_column

logger = logging.getLogger("ai_dashboard.profiler")


class DataProfiler:
    """Profiles a DataFrame, detecting column types and computing statistics."""

    @st.cache_data(show_spinner=False)
    def profile(_self, df: pd.DataFrame) -> dict:
        """
        Generate full data profile.
        
        Returns dict with:
          - row_count, column_count
          - columns: list of column metadata dicts
          - numerical_cols, categorical_cols, datetime_cols, text_cols
          - missing_summary, duplicate_count
          - sample_data, correlation_matrix
        """
        logger.info(f"Profiling dataset: {df.shape}")

        columns = []
        for col in df.columns:
            col_meta = _self._profile_column(df, col)
            columns.append(col_meta)

        num_cols      = [c["name"] for c in columns if c["type"] == "numerical"]
        cat_cols      = [c["name"] for c in columns if c["type"] == "categorical"]
        dt_cols       = [c["name"] for c in columns if c["type"] == "datetime"]
        text_cols     = [c["name"] for c in columns if c["type"] == "text"]
        id_cols       = [c["name"] for c in columns if c.get("is_id", False)]

        # Missing summary
        missing = df.isnull().sum()
        missing_summary = {
            col: {
                "count": int(missing[col]),
                "pct": round(missing[col] / max(len(df), 1) * 100, 2)
            }
            for col in df.columns if missing[col] > 0
        }

        # Correlation matrix (numerical only)
        corr_matrix = None
        if len(num_cols) >= 2:
            try:
                corr_matrix = df[num_cols].corr().round(3).to_dict()
            except Exception:
                pass

        # Sample rows for AI
        sample = df.head(AppConfig.SAMPLE_SIZE_FOR_AI).to_dict(orient="records")

        profile = {
            "row_count": len(df),
            "column_count": len(df.columns),
            "columns": columns,
            "numerical_cols": num_cols,
            "categorical_cols": cat_cols,
            "datetime_cols": dt_cols,
            "text_cols": text_cols,
            "id_cols": id_cols,
            "missing_summary": missing_summary,
            "duplicate_count": int(df.duplicated().sum()),
            "sample_data": sample,
            "correlation_matrix": corr_matrix,
            "memory_mb": round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2),
        }

        logger.info(
            f"Profile complete — num={len(num_cols)}, cat={len(cat_cols)}, "
            f"dt={len(dt_cols)}, text={len(text_cols)}"
        )
        return profile

    # ── Private Methods ────────────────────────────────────────

    def _profile_column(self, df: pd.DataFrame, col: str) -> dict:
        """Generate metadata for a single column."""
        series = df[col]
        dtype = str(series.dtype)
        n = len(series)
        n_null = int(series.isnull().sum())
        n_unique = int(series.nunique())

        col_type = self._detect_type(series, col)

        meta = {
            "name": col,
            "dtype": dtype,
            "type": col_type,
            "n_null": n_null,
            "null_pct": round(n_null / max(n, 1) * 100, 2),
            "n_unique": n_unique,
            "unique_ratio": round(n_unique / max(n, 1), 4),
            "is_id": is_id_column(col) or (col_type == "numerical" and n_unique == n),
        }

        # Type-specific stats
        if col_type == "numerical":
            meta.update(self._numeric_stats(series))
        elif col_type == "categorical":
            meta.update(self._categorical_stats(series))
        elif col_type == "datetime":
            meta.update(self._datetime_stats(series))
        elif col_type == "text":
            meta.update(self._text_stats(series))

        return meta

    def _detect_type(self, series: pd.Series, col_name: str) -> str:
        """Detect semantic column type."""
        dtype = series.dtype

        # Explicit datetime
        if pd.api.types.is_datetime64_any_dtype(dtype):
            return "datetime"

        # Explicit numeric
        if pd.api.types.is_numeric_dtype(dtype):
            n_unique = series.nunique()
            n = len(series)
            # Very few unique values → treat as categorical
            if n_unique <= 10 and not is_id_column(col_name):
                return "categorical"
            return "numerical"

        # Object / string
        if dtype == object:
            # Try datetime
            try:
                sample = series.dropna().head(50)
                pd.to_datetime(sample, infer_datetime_format=True, errors="raise")
                return "datetime"
            except Exception:
                pass

            n_unique = series.nunique()
            n = len(series)
            avg_len = series.dropna().astype(str).str.len().mean() if len(series) > 0 else 0

            if avg_len > 50:
                return "text"
            if n_unique <= AppConfig.CATEGORICAL_THRESHOLD or (n_unique / max(n, 1)) < 0.5:
                return "categorical"
            return "text"

        return "categorical"

    def _numeric_stats(self, series: pd.Series) -> dict:
        """Compute statistics for numerical columns."""
        clean = series.dropna()
        if len(clean) == 0:
            return {}
        return {
            "min": float(clean.min()),
            "max": float(clean.max()),
            "mean": float(clean.mean()),
            "median": float(clean.median()),
            "std": float(clean.std()),
            "sum": float(clean.sum()),
            "q25": float(clean.quantile(0.25)),
            "q75": float(clean.quantile(0.75)),
        }

    def _categorical_stats(self, series: pd.Series) -> dict:
        """Compute statistics for categorical columns."""
        vc = series.value_counts()
        return {
            "top_values": vc.head(10).to_dict(),
            "mode": str(vc.index[0]) if len(vc) > 0 else None,
            "mode_freq": int(vc.iloc[0]) if len(vc) > 0 else 0,
        }

    def _datetime_stats(self, series: pd.Series) -> dict:
        """Compute statistics for datetime columns."""
        try:
            dt = pd.to_datetime(series, errors="coerce")
            clean = dt.dropna()
            if len(clean) == 0:
                return {}
            return {
                "min_date": str(clean.min().date()),
                "max_date": str(clean.max().date()),
                "date_range_days": (clean.max() - clean.min()).days,
            }
        except Exception:
            return {}

    def _text_stats(self, series: pd.Series) -> dict:
        """Compute statistics for text columns."""
        clean = series.dropna().astype(str)
        if len(clean) == 0:
            return {}
        lengths = clean.str.len()
        return {
            "avg_length": round(float(lengths.mean()), 1),
            "max_length": int(lengths.max()),
            "sample_values": clean.head(3).tolist(),
        }
