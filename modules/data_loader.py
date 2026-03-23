"""
Data Loader Module
Handles CSV file loading, validation, and initial preprocessing.
Includes currency/percentage string cleaning so columns like
"$15,000" and "50,000 mi" are correctly parsed as numerics.
"""

import io
import re
import logging
import pandas as pd
import numpy as np
from utils.config import AppConfig

logger = logging.getLogger("ai_dashboard.data_loader")


class DataLoader:
    """Loads and validates CSV files with preprocessing."""

    def load(self, uploaded_file) -> pd.DataFrame:
        self._validate_file(uploaded_file)
        df = self._read_csv(uploaded_file)
        df = self._preprocess(df)
        logger.info(f"Loaded dataset: {df.shape[0]} rows × {df.shape[1]} columns")
        return df

    # ── Private ────────────────────────────────────────────────

    def _validate_file(self, uploaded_file):
        size = getattr(uploaded_file, "size", None)
        if size is None:
            content = uploaded_file.read()
            if hasattr(uploaded_file, "seek"):
                uploaded_file.seek(0)
            size = len(content)
        if size > AppConfig.MAX_FILE_SIZE_BYTES:
            raise ValueError(
                f"File too large: {size/1024/1024:.1f}MB. Max: {AppConfig.MAX_FILE_SIZE_MB}MB"
            )
        name = getattr(uploaded_file, "name", "file.csv")
        ext = name.rsplit(".", 1)[-1].lower()
        if ext not in AppConfig.ALLOWED_EXTENSIONS:
            raise ValueError(f"Invalid file type: .{ext}. Only CSV files are supported.")

    def _read_csv(self, uploaded_file) -> pd.DataFrame:
        content = uploaded_file.read() if hasattr(uploaded_file, "read") else uploaded_file.getvalue()
        last_error = None
        for encoding in ["utf-8", "latin-1", "cp1252", "utf-8-sig"]:
            for sep in [",", ";", "\t", "|"]:
                try:
                    df = pd.read_csv(
                        io.BytesIO(content),
                        encoding=encoding,
                        sep=sep,
                        low_memory=False,
                        on_bad_lines="skip",
                    )
                    if df.shape[1] >= 2 and df.shape[0] >= 1:
                        logger.info(f"Read CSV with encoding={encoding}, sep={repr(sep)}")
                        return df
                except Exception as e:
                    last_error = e
        raise ValueError(f"Could not parse CSV file. Error: {last_error}")

    def _preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        if len(df) > AppConfig.MAX_ROWS:
            logger.warning(f"Dataset truncated to {AppConfig.MAX_ROWS:,} rows")
            df = df.sample(n=AppConfig.MAX_ROWS, random_state=42).reset_index(drop=True)

        df.columns = [self._clean_column_name(c) for c in df.columns]
        df = df.dropna(how="all").dropna(axis=1, how="all")

        n_before = len(df)
        df = df.drop_duplicates().reset_index(drop=True)
        if (removed := n_before - len(df)) > 0:
            logger.info(f"Removed {removed:,} duplicate rows")

        df = self._clean_currency_strings(df)
        df = self._infer_types(df)
        return df

    def _clean_column_name(self, name: str) -> str:
        name = str(name).strip()
        name = re.sub(r"\s+", "_", name)
        name = re.sub(r"[^\w]", "_", name)
        name = re.sub(r"_+", "_", name)
        return name.strip("_") or "col"

    @staticmethod
    def _parse_currency_value(v) -> float:
        """Parse a single currency/unit string to float. Returns nan on failure."""
        import re
        import numpy as np
        if v is None or (hasattr(v, '__class__') and v.__class__.__name__ == 'float' and v != v):
            return float('nan')
        s = str(v).strip()
        if not s or s.lower() in ('nan', 'none', 'null', 'n/a', ''):
            return float('nan')
        # Remove currency symbols — use replace to avoid regex escaping issues
        for sym in ('$', '£', '€', '¥'):
            s = s.replace(sym, '')
        s = s.strip()
        # Handle percentage
        is_pct = s.endswith('%')
        if is_pct:
            s = s[:-1].strip()
        # Remove unit suffixes
        s = re.sub(r'\s*(mi|km|mpg|hp|lb|lbs|kg|oz|ft|yrs?|yr)\s*$', '', s, flags=re.I).strip()
        # Handle k/M/B multiplier suffixes
        mult = 1
        if s and s[-1].lower() == 'k':
            mult = 1_000
            s = s[:-1]
        elif s and s[-1].lower() == 'm' and len(s) > 1:
            mult = 1_000_000
            s = s[:-1]
        elif s and s[-1].lower() == 'b' and len(s) > 1:
            mult = 1_000_000_000
            s = s[:-1]
        # Remove commas (thousands separator)
        s = s.replace(',', '').strip()
        try:
            val = float(s) * mult
            return val / 100.0 if is_pct else val
        except (ValueError, TypeError):
            return float('nan')

    def _clean_currency_strings(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert currency/unit strings to numeric BEFORE type inference.
        Handles: "$15,000" → 15000.0  |  "50,000 mi" → 50000.0
                 "45 mpg" → 45.0      |  "98%" → 0.98
                 "1.5k"  → 1500.0     |  "2.3M" → 2300000.0
        Only converts a column if ≥80% of non-null values parse successfully.
        """
        for col in df.columns:
            # Accept both legacy object dtype (kind='O') and pandas StringDtype
            try:
                is_string_col = (df[col].dtype.kind == 'O') or pd.api.types.is_string_dtype(df[col])
            except Exception:
                is_string_col = False
            if not is_string_col:
                continue

            sample = df[col].dropna()
            if len(sample) == 0:
                continue

            # Quick heuristic: does any value look like a currency/unit string?
            test = sample.head(50).astype(str)
            has_currency = any(c in ' '.join(test.tolist()) for c in ('$', '£', '€', '¥'))
            has_units    = test.str.contains(
                r'\d+\s*(?:mi|km|mpg|hp|lb|kg|oz|ft|yr|%)', regex=True, flags=re.I
            ).any()
            has_commas   = test.str.match(r'^\$?[\d,]+\.?\d*$').any()
            has_k_suffix = test.str.match(r'^\d+(\.\d+)?[kKmMbB]$').any()

            if not any([has_currency, has_units, has_commas, has_k_suffix]):
                continue

            # Test parse success rate on sample
            parsed = sample.astype(str).apply(self._parse_currency_value)
            success_rate = (parsed == parsed).sum() / len(sample)  # nan != nan

            if success_rate >= 0.80:
                logger.info(f"Cleaned currency/unit column '{col}' ({success_rate:.0%} parsed)")
                numeric_vals = df[col].astype(str).apply(self._parse_currency_value)
                df[col] = pd.to_numeric(numeric_vals, errors="coerce")

        return df

    def _infer_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Infer better dtypes for object columns."""
        for col in df.columns:
            try:
                is_string_col = (df[col].dtype.kind == 'O') or pd.api.types.is_string_dtype(df[col])
            except Exception:
                is_string_col = False
            if not is_string_col:
                continue
            # Try numeric
            try:
                numeric = pd.to_numeric(df[col], errors="coerce")
                if numeric.notna().sum() / max(len(df), 1) > 0.9:
                    df[col] = numeric
                    continue
            except Exception:
                pass
            # Try datetime
            try:
                dt = pd.to_datetime(df[col], errors="coerce", infer_datetime_format=True)
                if dt.notna().sum() / max(len(df), 1) > 0.8:
                    df[col] = dt
                    continue
            except Exception:
                pass
        return df