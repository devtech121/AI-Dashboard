"""
Data Loader Module
Handles CSV file loading, validation, and initial preprocessing
"""

import io
import logging
import pandas as pd
import numpy as np
from utils.config import AppConfig

logger = logging.getLogger("ai_dashboard.data_loader")


class DataLoader:
    """Loads and validates CSV files with preprocessing."""

    def load(self, uploaded_file) -> pd.DataFrame:
        """
        Load a CSV file from an uploaded file object.
        
        Args:
            uploaded_file: Streamlit UploadedFile or file-like object
            
        Returns:
            Validated and preprocessed DataFrame
        """
        self._validate_file(uploaded_file)

        df = self._read_csv(uploaded_file)
        df = self._preprocess(df)

        logger.info(f"Loaded dataset: {df.shape[0]} rows × {df.shape[1]} columns")
        return df

    # ── Private Methods ────────────────────────────────────────

    def _validate_file(self, uploaded_file):
        """Validate file size and type."""
        # Check file size
        size = getattr(uploaded_file, "size", None)
        if size is None:
            content = uploaded_file.read()
            uploaded_file.seek(0) if hasattr(uploaded_file, "seek") else None
            size = len(content)

        if size > AppConfig.MAX_FILE_SIZE_BYTES:
            raise ValueError(
                f"File too large: {size / 1024 / 1024:.1f}MB. "
                f"Maximum allowed: {AppConfig.MAX_FILE_SIZE_MB}MB"
            )

        # Check extension
        name = getattr(uploaded_file, "name", "file.csv")
        ext = name.rsplit(".", 1)[-1].lower()
        if ext not in AppConfig.ALLOWED_EXTENSIONS:
            raise ValueError(
                f"Invalid file type: .{ext}. "
                f"Only {', '.join(AppConfig.ALLOWED_EXTENSIONS)} files are supported."
            )

    def _read_csv(self, uploaded_file) -> pd.DataFrame:
        """Read CSV with multiple encoding fallbacks."""
        content = uploaded_file.read() if hasattr(uploaded_file, "read") else uploaded_file.getvalue()

        encodings = ["utf-8", "latin-1", "cp1252", "utf-8-sig"]
        separators = [",", ";", "\t", "|"]
        last_error = None

        for encoding in encodings:
            for sep in separators:
                try:
                    df = pd.read_csv(
                        io.BytesIO(content),
                        encoding=encoding,
                        sep=sep,
                        low_memory=False,
                        on_bad_lines="skip",
                    )
                    # Must have at least 2 columns and 1 row
                    if df.shape[1] >= 2 and df.shape[0] >= 1:
                        logger.info(f"Read CSV with encoding={encoding}, sep={repr(sep)}")
                        return df
                except Exception as e:
                    last_error = e
                    continue

        raise ValueError(f"Could not parse CSV file. Please ensure it's a valid CSV. Error: {last_error}")

    def _preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess the dataframe."""
        # Enforce row limit
        if len(df) > AppConfig.MAX_ROWS:
            logger.warning(f"Dataset truncated from {len(df):,} to {AppConfig.MAX_ROWS:,} rows")
            df = df.sample(n=AppConfig.MAX_ROWS, random_state=42).reset_index(drop=True)

        # Clean column names
        df.columns = [self._clean_column_name(c) for c in df.columns]

        # Remove fully empty rows and columns
        df = df.dropna(how="all")
        df = df.dropna(axis=1, how="all")

        # Remove duplicate rows
        n_before = len(df)
        df = df.drop_duplicates().reset_index(drop=True)
        n_removed = n_before - len(df)
        if n_removed > 0:
            logger.info(f"Removed {n_removed:,} duplicate rows")

        # Infer better dtypes
        df = self._infer_types(df)

        return df

    def _clean_column_name(self, name: str) -> str:
        """Normalize column names."""
        import re
        name = str(name).strip()
        name = re.sub(r"\s+", "_", name)
        name = re.sub(r"[^\w]", "_", name)
        name = re.sub(r"_+", "_", name)
        name = name.strip("_")
        return name if name else "col"

    def _infer_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Try to infer better data types for object columns."""
        for col in df.columns:
            if df[col].dtype == object:
                # Try numeric
                try:
                    numeric = pd.to_numeric(df[col], errors="coerce")
                    non_null_ratio = numeric.notna().sum() / max(len(df), 1)
                    if non_null_ratio > 0.9:
                        df[col] = numeric
                        continue
                except Exception:
                    pass

                # Try datetime
                try:
                    dt = pd.to_datetime(df[col], errors="coerce", infer_datetime_format=True)
                    non_null_ratio = dt.notna().sum() / max(len(df), 1)
                    if non_null_ratio > 0.8:
                        df[col] = dt
                        continue
                except Exception:
                    pass

        return df
