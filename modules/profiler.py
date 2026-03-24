"""
Data Profiler Module
Generates comprehensive metadata and statistics for a DataFrame.
Improvements:
  - Binary columns (0/1, Yes/No) correctly typed as categorical
  - Richer column metadata passed to AI (skew, top-value %, etc.)
  - Smarter ID column detection
"""

import logging
import numpy as np
import pandas as pd
from utils.config import AppConfig
from utils.helpers import is_id_column

logger = logging.getLogger("ai_dashboard.profiler")

# Canonical binary value pairs
BINARY_PAIRS = [
    {0, 1}, {"yes", "no"}, {"true", "false"},
    {"male", "female"}, {"m", "f"},
    {"y", "n"}, {"positive", "negative"},
]


class DataProfiler:
    """Profiles a DataFrame, detecting column types and computing statistics.

    Not decorated with @st.cache_data to avoid cross-user leakage; the caller stores results per session.
    """

    def profile(self, df: pd.DataFrame) -> dict:
        logger.info(f"Profiling dataset: {df.shape}")
        columns = [self._profile_column(df, col) for col in df.columns]

        num_cols  = [c["name"] for c in columns if c["type"] == "numerical"]
        cat_cols  = [c["name"] for c in columns if c["type"] == "categorical"]
        dt_cols   = [c["name"] for c in columns if c["type"] == "datetime"]
        text_cols = [c["name"] for c in columns if c["type"] == "text"]
        bin_cols  = [c["name"] for c in columns if c.get("is_binary")]
        id_cols   = [c["name"] for c in columns if c.get("is_id")]

        missing = df.isnull().sum()
        missing_summary = {
            col: {"count": int(missing[col]), "pct": round(missing[col] / max(len(df), 1) * 100, 2)}
            for col in df.columns if missing[col] > 0
        }

        corr_matrix = None
        usable_num = [c for c in num_cols if c not in id_cols]
        if len(usable_num) >= 2:
            try:
                corr_matrix = df[usable_num].corr().round(3).to_dict()
            except Exception:
                pass

        profile = {
            "row_count": len(df),
            "column_count": len(df.columns),
            "columns": columns,
            "numerical_cols": num_cols,
            "categorical_cols": cat_cols,
            "datetime_cols": dt_cols,
            "text_cols": text_cols,
            "binary_cols": bin_cols,
            "id_cols": id_cols,
            "missing_summary": missing_summary,
            "duplicate_count": int(df.duplicated().sum()),
            "sample_data": df.head(AppConfig.SAMPLE_SIZE_FOR_AI).to_dict(orient="records"),
            "correlation_matrix": corr_matrix,
            "memory_mb": round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2),
        }

        logger.info(
            f"Profile complete — num={len(num_cols)}, cat={len(cat_cols)}, "
            f"dt={len(dt_cols)}, binary={len(bin_cols)}, text={len(text_cols)}"
        )
        return profile

    # ── Private ────────────────────────────────────────────────

    def _profile_column(self, df: pd.DataFrame, col: str) -> dict:
        series = df[col]
        dtype = str(series.dtype)
        n = len(series)
        n_null = int(series.isnull().sum())
        n_unique = int(series.nunique())

        is_binary = self._is_binary(series)
        col_type = self._detect_type(series, col, is_binary)

        meta = {
            "name": col,
            "dtype": dtype,
            "type": col_type,
            "n_null": n_null,
            "null_pct": round(n_null / max(n, 1) * 100, 2),
            "n_unique": n_unique,
            "unique_ratio": round(n_unique / max(n, 1), 4),
            "is_id": is_id_column(col) and col_type == "numerical",
            "is_binary": is_binary,
        }

        if col_type == "numerical":
            meta.update(self._numeric_stats(series))
        elif col_type == "categorical":
            meta.update(self._categorical_stats(series))
        elif col_type == "datetime":
            meta.update(self._datetime_stats(series))
        elif col_type == "text":
            meta.update(self._text_stats(series))

        return meta

    def _is_binary(self, series: pd.Series) -> bool:
        """True if column is a binary flag (0/1, yes/no, etc.)."""
        clean = series.dropna()
        if len(clean) == 0:
            return False
        unique_vals = set(clean.astype(str).str.lower().unique())
        if len(unique_vals) != 2:
            # Also check raw values for 0/1
            try:
                raw = set(clean.unique())
                if raw in BINARY_PAIRS or raw == {0, 1}:
                    return True
            except Exception:
                pass
            return False
        return unique_vals in BINARY_PAIRS or unique_vals == {"0", "1"}

    def _detect_type(self, series: pd.Series, col_name: str, is_binary: bool = False) -> str:
        """Detect semantic column type with binary awareness."""
        dtype = series.dtype

        # Explicit datetime
        if pd.api.types.is_datetime64_any_dtype(dtype):
            return "datetime"

        # Binary columns → always categorical regardless of dtype
        if is_binary:
            return "categorical"

        # Explicit numeric
        if pd.api.types.is_numeric_dtype(dtype):
            n_unique = series.nunique()
            # Few unique values AND not an ID → categorical
            if n_unique <= 8 and not is_id_column(col_name):
                return "categorical"
            # Looks like a year column
            if col_name.lower() in ("year", "yr", "model_year", "birth_year"):
                return "categorical" if n_unique <= 20 else "numerical"
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
            avg_len = series.dropna().astype(str).str.len().mean() if n > 0 else 0

            if avg_len > 60:
                return "text"
            # High cardinality object → text
            if n_unique / max(n, 1) > 0.6 and n_unique > 50:
                return "text"
            return "categorical"

        return "categorical"

    def _numeric_stats(self, series: pd.Series) -> dict:
        clean = series.dropna()
        if len(clean) == 0:
            return {}
        try:
            skew = float(clean.skew())
        except Exception:
            skew = 0.0
        return {
            "min": float(clean.min()),
            "max": float(clean.max()),
            "mean": float(clean.mean()),
            "median": float(clean.median()),
            "std": float(clean.std()),
            "sum": float(clean.sum()),
            "q25": float(clean.quantile(0.25)),
            "q75": float(clean.quantile(0.75)),
            "skew": round(skew, 2),
        }

    def _categorical_stats(self, series: pd.Series) -> dict:
        vc = series.value_counts()
        n = len(series)
        top_values_with_pct = {
            str(k): {"count": int(v), "pct": round(v / max(n, 1) * 100, 1)}
            for k, v in vc.head(10).items()
        }
        return {
            "top_values": {str(k): int(v) for k, v in vc.head(10).items()},
            "top_values_pct": top_values_with_pct,
            "mode": str(vc.index[0]) if len(vc) > 0 else None,
            "mode_pct": round(vc.iloc[0] / max(n, 1) * 100, 1) if len(vc) > 0 else 0,
        }

    def _datetime_stats(self, series: pd.Series) -> dict:
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
        clean = series.dropna().astype(str)
        if len(clean) == 0:
            return {}
        lengths = clean.str.len()
        return {
            "avg_length": round(float(lengths.mean()), 1),
            "max_length": int(lengths.max()),
            "sample_values": clean.head(3).tolist(),
        }
